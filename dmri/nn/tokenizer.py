from abc import abstractmethod
from collections.abc import Sequence
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Tuple,
)

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from probjax.utils.typing import Array, ArrayLike, DTypeLike, PrecisionLike, RngKey

from dmri.simulators import MultiCompartment
from dmri.utils.transform import eps_mask


class Tokenizer(nnx.Module):
    def __call__(self, *args: Any, **kwds: Any) -> Array:
        return self.encode(*args, **kwds)

    @abstractmethod
    def encode(self, *args: Any, **kwargs: Any) -> Array:
        pass

    @abstractmethod
    def decode(self, *args: Any, **kwargs: Any) -> Array:
        pass


class ScalarTokenizer(Tokenizer):
    def __init__(
        self,
        num_nodes: int,
        rngs: nnx.Rngs,
        value_dim: int = 20,
        id_dim: int = 20,
        cond_dim: int = 10,
        context_dim: Optional[int] = None,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike | None = None,
        precision: PrecisionLike | None = None,
        preferred_element_type: DTypeLike | None = None,
    ) -> None:
        self.model_dim = value_dim + id_dim + cond_dim
        self.num_nodes = num_nodes
        self.value_dim = value_dim
        self.id_dim = id_dim
        self.cond_dim = cond_dim

        linear_kwargs: dict[str, Any] = {}
        for name, value in (
            ("dtype", dtype),
            ("param_dtype", param_dtype),
            ("precision", precision),
            ("preferred_element_type", preferred_element_type),
        ):
            if value is not None:
                linear_kwargs[name] = value
        self._linear_kwargs = linear_kwargs

        self.embed_id = nnx.Embed(num_nodes, self.id_dim, rngs=rngs)
        self.embed_value = nnx.Linear(
            1, self.value_dim, rngs=rngs, **self._linear_kwargs
        )
        if cond_dim > 0:
            self.condition_token = nnx.Param(
                0.01 * jax.random.normal(rngs.next(), (1, cond_dim))
            )
        if context_dim is not None:
            self.context_embed = nnx.Linear(
                context_dim, self.model_dim, rngs=rngs, **self._linear_kwargs
            )
        self.context_dim = context_dim

        self.outlayer = nnx.Linear(self.model_dim, 1, rngs=rngs, **self._linear_kwargs)

    def encode(
        self,
        x: ArrayLike,
        node_ids: ArrayLike,
        condition_mask: ArrayLike,
        context: Optional[ArrayLike] = None,
    ) -> Array:
        node_embed = self.embed_id(node_ids)
        value_embed = self.embed_value(x)

        if self.cond_dim > 0:
            condition_embed = self.condition_token.value * condition_mask[..., None]
            input_embed = jnp.concatenate(
                [node_embed, value_embed, condition_embed], axis=-1
            )
        else:
            input_embed = jnp.concatenate([node_embed, value_embed], axis=-1)

        if context is not None and self.context_dim is not None:
            context_embed = self.context_embed(context)
            input_embed += context_embed

        return input_embed

    def decode(
        self,
        h: ArrayLike,
        node_ids: ArrayLike,
        condition_mask: ArrayLike,
        context: Optional[ArrayLike] = None,
    ) -> Array:
        return self.outlayer(h)


class StructuredTokenizer(Tokenizer):
    def __init__(
        self,
        dims_by_id: list[int],
        *,
        value_dim: int = 32,
        id_dim: int = 32,
        cond_dim: int = 0,
        encode_nets: Optional[nnx.List[nnx.Module]] = None,
        decode_nets: Optional[nnx.List[nnx.Module]] = None,
        rngs: nnx.Rngs,
    ):
        self.dims_by_id = dims_by_id
        self.num_nodes = len(dims_by_id)
        if encode_nets is None:
            encode_nets = nnx.List([
                nnx.Linear(d, value_dim, rngs=rngs) for d in dims_by_id
            ])
        if decode_nets is None:
            dim_token = id_dim + value_dim + cond_dim
            decode_nets = nnx.List([
                nnx.Linear(dim_token, d, rngs=rngs) for d in dims_by_id
            ])

        self.encode_nets = encode_nets
        self.decode_nets = decode_nets
        self.embed_id = nnx.Embed(
            self.num_nodes,
            id_dim,
            rngs=rngs,
            embedding_init=nnx.initializers.orthogonal(),
        )

        if cond_dim > 0:
            self.condition_token = nnx.Param(
                0.01 * jax.random.normal(rngs.next(), (1, cond_dim))
            )

    def __call__(
        self,
        x: ArrayLike,
        node_ids: Sequence[int],
        condition_mask: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> Array:
        # These needs to be done outside of jax and recompiled on changes
        dims = np.asarray([self.dims_by_id[i] for i in node_ids], dtype=np.int32)
        split_dims = np.cumsum(dims)[:-1]
        # This will now be fully jax
        x_split = jnp.split(x, split_dims, axis=-1)
        net_subs = [self.encode_nets[i] for i in node_ids]
        val_embeddings = jax.tree_util.tree_map(
            lambda x, net: net(x)[..., None, :], x_split, net_subs
        )
        val_embedding = jnp.concatenate(val_embeddings, axis=-2)

        ids = self.embed_id(jnp.array(node_ids, dtype=jnp.int32))
        while len(ids.shape) < len(val_embedding.shape):
            ids = ids[None, ...]
            ids = jnp.repeat(ids, val_embedding.shape[0], axis=0)

        embeddings = jnp.concatenate([val_embedding, ids], axis=-1)

        return embeddings

    def decode(
        self,
        h: ArrayLike,
        node_ids: Sequence[int],
        condition_mask: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> Array:
        hs = jnp.split(h, h.shape[-2], axis=-2)
        net_subs = [self.decode_nets[i] for i in node_ids]
        x = jax.tree_util.tree_map(lambda x, net: net(x), hs, net_subs)
        out = jnp.concatenate(x, axis=-1)
        out = jnp.squeeze(out, axis=-2)
        return out


class DMRITokenizer(Tokenizer):
    """
    A tokenizer for dMRI multi-compartment model configurations. It handles embedding
    and decoding of model types, noise types, and associated parameters.
    """

    def __init__(
        self,
        simulator: type[MultiCompartment],
        *,
        token_dim: int = 64,
        theta_encode_nets: Optional[nnx.List[nnx.Module | None]] = None,
        theta_decode_nets: Optional[nnx.List[nnx.Module | None]] = None,
        init_component_embeddings: Callable | None = None,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike | None = None,
        precision: PrecisionLike | None = None,
        preferred_element_type: DTypeLike | None = None,
        rngs: nnx.Rngs,
    ) -> None:
        """
        Initializes a DMRITokenizer instance.

        Args:
            simulator (type[MultiCompartment]): The multi-compartment simulator class which
                defines model_types, noise_types, etc.
            rngs (Any): Random number generator(s) used for parameter initialization.
            token_dim (int, optional): Dimension of tokens for embedding. Defaults to 64.
            theta_encode_nets (Optional[List[nnx.Module]], optional): Encoding modules for parameters. Defaults to None.
            theta_decode_nets (Optional[List[nnx.Module]], optional): Decoding modules for parameters. Defaults to None.
        """
        self.simulator = nnx.Intermediate(simulator)
        self.num_models = len(simulator.model_types)
        self.num_noises = len(simulator.noise_types)
        self.params_dims = tuple(simulator.split_idx())

        self.model_indices: Tuple[int, ...] = tuple(range(self.num_models))
        self.noise_indices: Tuple[int, ...] = tuple(range(self.num_noises))
        self._model_class_labels = [cls.__name__ for cls in simulator.model_types]
        linear_kwargs: dict[str, Any] = {}
        for name, value in (
            ("dtype", dtype),
            ("param_dtype", param_dtype),
            ("precision", precision),
            ("preferred_element_type", preferred_element_type),
        ):
            if value is not None:
                linear_kwargs[name] = value
        self._linear_kwargs = linear_kwargs

        self.embed_idx = nnx.Embed(
            rngs=rngs,
            num_embeddings=len(simulator.model_types) + len(simulator.noise_types),
            features=token_dim,
            embedding_init=self._init_class_embeddings
            if init_component_embeddings is None
            else init_component_embeddings,
        )
        self.embed_fraction = nnx.Linear(
            len(simulator.model_types), token_dim, rngs=rngs, **self._linear_kwargs
        )
        if simulator.shared_parameter_type is not None:
            self.shared_parameter_embed = nnx.Linear(
                simulator.shared_parameter_type.theta_dim,
                token_dim,
                rngs=rngs,
                **self._linear_kwargs,
            )
            self.shared_parameter_decode = nnx.Linear(
                token_dim,
                simulator.shared_parameter_type.theta_dim,
                rngs=rngs,
                **self._linear_kwargs,
            )
            self.shared_parameter_idx = nnx.Embed(
                rngs=rngs, num_embeddings=1, features=token_dim
            )

        # Default to linear layers
        if theta_encode_nets is None:
            theta_encode_nets = nnx.List([
                nnx.Linear(d, token_dim, rngs=rngs, **self._linear_kwargs)
                if d > 0
                else None
                for d in self.params_dims
            ])
        if theta_decode_nets is None:
            theta_decode_nets = nnx.List([
                nnx.Linear(
                    token_dim,
                    d,
                    rngs=rngs,
                    kernel_init=nnx.initializers.zeros,
                    **self._linear_kwargs,
                )
                if d > 0
                else None
                for d in self.params_dims
            ])
        self.theta_encode_nets = theta_encode_nets
        self.theta_decode_nets = theta_decode_nets

    def _init_class_embeddings(
        self, key: RngKey, shape: Tuple[int, ...], dtype: jnp.dtype = jnp.float32
    ) -> Array:
        """Custom initializer that makes embeddings for same classes identical
        but orthogonal between different classes.

        Args:
            key: PRNG key
            shape: Shape of embeddings (num_embeddings, embedding_dim)
            dtype: Data type of embeddings
        """
        # Get unique class names for available model types
        class_names = self._model_class_labels
        num_classes = len(class_names)

        # Initialize orthogonal embeddings for each unique class
        class_embeddings = jax.random.orthogonal(
            key, n=shape[1], shape=(num_classes,), m=1
        ).squeeze(axis=-1)

        # Create full embedding matrix by mapping class embeddings to all indices
        embeddings = jnp.zeros(shape, dtype=dtype)
        for idx, name in enumerate(class_names):
            embeddings = embeddings.at[idx, ...].set(class_embeddings[idx])

        return embeddings

    def encode(
        self,
        theta: Optional[ArrayLike] = None,
        model_mask: Optional[ArrayLike] = None,
        tokens_cfg: Optional[ArrayLike] = None,
        alpha_prior: Optional[ArrayLike] = None,
        model_idx: Optional[Sequence[int]] = None,
        noise_idx: Optional[Sequence[int]] = None,
    ) -> Array:
        if model_idx is None:
            model_idx = self.model_indices
        if noise_idx is None:
            noise_idx = self.noise_indices
        if model_mask is None:
            model_mask = jnp.ones(
                len(self.model_indices) + len(self.noise_indices), dtype=jnp.bool_
            )

        if tokens_cfg is None:
            tokens_cfg = self.embed_cfgs(
                model_mask,
                alpha_prior,
                model_idx=model_idx,
                noise_idx=noise_idx,
            )
        # We assume that the provided tokens_cfg is already in the correct shape
        if theta is not None:
            tokens = self.embed_theta(
                theta,
                tokens_cfg,
                model_idx=model_idx,
                noise_idx=noise_idx,
                model_mask=model_mask,
            )
        else:
            tokens = tokens_cfg
        return tokens

    def decode(
        self,
        tokens: ArrayLike,
        model_idx: Optional[Sequence[int]] = None,
        noise_idx: Optional[Sequence[int]] = None,
        **kwargs: Any,
    ) -> Array:
        if model_idx is None:
            model_idx = self.model_indices
        if noise_idx is None:
            noise_idx = self.noise_indices

        theta = self.decode_theta(tokens, model_idx, noise_idx, **kwargs)
        return theta

    def get_indices_with_params(
        self, model_idx: Sequence[int], noise_idx: Sequence[int]
    ) -> List[int]:
        """
        Returns a list of model and noise indices corresponding to the provided model and noise types.
        """
        indices: List[int] = []
        for idx in model_idx:
            if self.simulator.value.model_types[idx].theta_dim > 0:
                indices.append(idx)
        offset = self.num_models
        for idx in noise_idx:
            if self.simulator.value.noise_types[idx].theta_dim > 0:
                indices.append(offset + idx)
        return indices

    @staticmethod
    def theta_fraction_mask(model_mask: ArrayLike) -> Array:
        return jnp.ones(model_mask.shape[:-1] + (1,), dtype=jnp.bool_)

    def theta_mask(
        self,
        model_mask: ArrayLike,
        model_types: Optional[Sequence[type]] = None,
        noise_types: Optional[Sequence[type]] = None,
    ) -> Array:
        if model_types is None:
            model_types = tuple(self.simulator.value.model_types)
        if noise_types is None:
            noise_types = tuple(self.simulator.value.noise_types)

        # First need to find active model fractions
        model_component_mask = model_mask[..., : len(model_types)]
        active_thetas = eps_mask(model_component_mask)

        # Next is the shared parameters which are always active
        if self.simulator.value.shared_parameter_type is not None:
            theta_dim = self.simulator.value.shared_parameter_type.theta_dim
            active_thetas = jnp.concatenate(
                [
                    active_thetas,
                    jnp.ones(model_mask.shape[:-1] + (theta_dim,), dtype=jnp.bool_),
                ],
                axis=-1,
            )

        # Next are the model parameters, if model_mask is true it should be multiplied by the dimension of the parameter
        for i in model_idx:
            active_thetas = jnp.concatenate(
                [active_thetas]
                + [model_mask[..., i, None]] * self.simulator.value.model_types[i].theta_dim,
                axis=-1,
            )

        # Next are the noise parameters, if noise_mask is true it should be multiplied by the dimension of the parameter
        for idx in noise_idx:
            active_thetas = jnp.concatenate(
                [active_thetas]
                + [model_mask[..., self.num_models + idx, None]]
                * self.simulator.value.noise_types[idx].theta_dim,
                axis=-1,
            )

        return active_thetas

    def theta_token_mask(
        self,
        model_mask: ArrayLike,
        model_idx: Optional[Sequence[int]] = None,
        noise_idx: Optional[Sequence[int]] = None,
    ) -> Array:
        if model_idx is None:
            model_idx = self.model_indices
        if noise_idx is None:
            noise_idx = self.noise_indices

        idx_with_params = [
            i for i in model_idx if self.simulator.value.model_types[i].theta_dim > 0
        ]
        idx_with_params_noise = [
            self.num_models + i
            for i in noise_idx
            if self.simulator.value.noise_types[i].theta_dim > 0
        ]
        theta_fraction_mask = self.theta_fraction_mask(
            model_mask[..., : self.num_models]
        )

        # Create a mask for shared parameters - always true (1) since they're global
        # The shared parameter should be represented as a single token after the fractions
        if self.simulator.value.shared_parameter_type is not None:
            # Create a single token mask for shared parameters
            shared_param_mask = jnp.ones(
                model_mask.shape[:-1] + (1,),  # Just one token for shared parameters
                dtype=jnp.bool_,
            )
            # Combine fraction mask, shared parameter mask, and model parameter mask
            theta_mask = jnp.concatenate(
                [
                    theta_fraction_mask,
                    shared_param_mask,
                    model_mask[..., idx_with_params],
                    model_mask[..., idx_with_params_noise],
                ],
                axis=-1,
            )
        else:
            # If no shared parameters, just combine fraction mask and model parameter mask
            theta_mask = jnp.concatenate(
                [
                    theta_fraction_mask,
                    model_mask[..., idx_with_params],
                    model_mask[..., idx_with_params_noise],
                ],
                axis=-1,
            )

        return theta_mask

    def embed_cfgs(
        self,
        model_mask: ArrayLike,
        alpha_prior: Optional[ArrayLike] = None,
        model_idx: Optional[Sequence[int]] = None,
        noise_idx: Optional[Sequence[int]] = None,
    ) -> Array:
        """
        Embeds the configuration of model and noise types into tokens.

        Token Structure:
            The output tokens have the following structure:

            1. alpha_token (B, 1, token_dim):
               - Represents the prior fractions for model components
               - Position: First token in the sequence
               - Shape: (batch_dims..., 1, token_dim)

            2. idx_tokens (B, T, token_dim):
               - Represents the embedded indices for each model and noise component
               - Value: If the component mask is True the token will be the embedded
                 index, if the component mask is False the token will be zero
               - Position: Follows the alpha_token
               - Shape: (batch_dims..., T, token_dim) where T is the number of components
               - Components that are not active (masked out) will have zero token values

            The final output is a concatenation of these tokens along the second-to-last axis,
            resulting in a tensor of shape (batch_dims..., 1+T, token_dim).

        Args:
            model_mask (ArrayLike): A binary mask indicating active model components.
            alpha_prior (Optional[ArrayLike]): Prior fractions for model components.
            model_idx (Optional[Sequence[int]]): Indices of model components.
            noise_idx (Optional[Sequence[int]]): Indices of noise components.

        Returns:
            ArrayLike: The embedded tokens for each model/noise component.
        """
        if model_idx is None:
            model_idx = self.model_indices
        if noise_idx is None:
            noise_idx = self.noise_indices
        if alpha_prior is None:
            alpha_prior = self.simulator.value.fraction_prior

        idx = tuple(int(i) for i in model_idx) + tuple(
            self.num_models + int(i) for i in noise_idx
        )
        assert len(idx) == model_mask.shape[-1], (
            f"model_mask shape last axis {model_mask.shape} does not match the number of model components {len(idx)}"
        )

        *batch_dims, T = model_mask.shape
        # Broadcast alpha_prior to the batch dims
        alpha_prior = jnp.broadcast_to(alpha_prior, batch_dims + [self.num_models])

        # Get the fraction prior token, which will always be in the beginning
        alpha_token = self.embed_fraction(alpha_prior)[
            ..., None, :
        ]  # (B, 1, token_dim)
        # Get the component tokens
        idx = jnp.array(idx, dtype=jnp.int32)
        idx_tokens = self.embed_idx(idx)  # (T, token_dim)
        for _ in range(len(batch_dims)):
            idx_tokens = idx_tokens[None, ...]  # (1, T, token_dim)
        # Components that are not active will have zero token
        idx_tokens = idx_tokens * model_mask[..., None]  # (B, T, token_dim)
        # Combine the tokens
        tokens = jnp.concatenate([alpha_token, idx_tokens], axis=-2)
        return tokens

    def embed_theta(
        self,
        theta: ArrayLike,
        tokens_cfg: ArrayLike,
        model_idx: Optional[Sequence[int]] = None,
        noise_idx: Optional[Sequence[int]] = None,
        model_mask: Optional[ArrayLike] = None,
    ) -> Array:
        """
        Embeds the continuous parameter vector theta into token representation.

        Args:
            theta (ArrayLike): The parameters of each model/noise component.
            tokens_cfg (ArrayLike): Configuration tokens returned by embed_cfgs.
            model_types (Optional[List[type]]): List of model types.
            noise_types (Optional[List[type]]): List of noise types.

        Returns:
            ArrayLike: The token representation augmented with encoded parameters.

        Token Structure:
            The output tokens have the following structure:

            1. val_tokens (B, N, token_dim):
               - Represents the encoded parameter values for each component
               - Shape: (batch_dims..., N, token_dim) where N is the number of components with parameters
               - Each token corresponds to the parameters of a specific model or noise component
               - Components without parameters are excluded from the output

            2. tokens_cfg (B, N, token_dim):
               - Configuration tokens from embed_cfgs, filtered to match the components with parameters
               - Shape: (batch_dims..., N, token_dim)

            The final output is the sum of val_tokens and tokens_cfg, resulting in a tensor of shape
            (batch_dims..., N, token_dim). This combines the parameter information with the component
            identity information in a single token representation.

        Processing Steps:
            1. The theta vector is split into components based on the parameter dimensions of each model/noise type
            2. Each component's parameters are encoded using the corresponding encoding network
            3. Components without parameters are filtered out
            4. The encoded parameters are concatenated along the second-to-last axis
            5. The configuration tokens are filtered to match only the components with parameters
            6. The final tokens are the sum of the encoded parameters and the filtered configuration tokens
        """
        if model_idx is None:
            model_idx = self.model_indices
        if noise_idx is None:
            noise_idx = self.noise_indices

        model_types_all = self.simulator.value.model_types
        noise_types_all = self.simulator.value.noise_types

        model_idx_with_params = [
            int(i) for i in model_idx if model_types_all[i].theta_dim > 0
        ]
        noise_idx_with_params = [
            int(i) for i in noise_idx if noise_types_all[i].theta_dim > 0
        ]

        idx = tuple(model_idx_with_params) + tuple(
            self.num_models + i for i in noise_idx_with_params
        )

        # First dim -> Model fractions
        # Second dim -> Shared parameters (if any)
        # Other dims -> Component parameters
        dims_per_component = [self.params_dims[0]]  # Model fractions
        offset = 1
        if self.simulator.value.shared_parameter_type is not None:
            dims_per_component.append(self.params_dims[1])  # Shared parameters
            offset += 1
        dims_per_component += [self.params_dims[i + offset] for i in idx]

        assert sum(dims_per_component) == theta.shape[-1], (
            f"theta shape last axis {theta.shape} does not match the number of model components {sum(dims_per_component)}"
        )

        # Split theta into components
        dims = np.asarray(dims_per_component, dtype=np.int32)
        split_dims = np.cumsum(dims)[:-1]
        theta_split = jnp.split(theta, split_dims, axis=-1)

        # Get the val embeddings
        nets_encode = [self.theta_encode_nets[0]]  # First is for fractions
        offset = 1
        if self.simulator.value.shared_parameter_type is not None:
            nets_encode += [self.theta_encode_nets[1]]  # Global shared parameters
            offset += 1
        nets_encode += [
            self.theta_encode_nets[i + offset] for i in idx
        ]  # Component parameters
        val_embeddings = jax.tree_util.tree_map(
            lambda x, net: net(x)[..., None, :],
            theta_split,
            nets_encode,
        )

        val_tokens = jnp.concatenate(val_embeddings, axis=-2)

        # Add configuration tokens for shared parameters
        tokens_cfg_fractions = tokens_cfg[..., :1, :]
        tokens_cfg_models = tokens_cfg[..., 1:, :]
        indices = self.get_indices_with_params(model_idx, noise_idx)

        token_cfg_models_with_params = tokens_cfg_models[..., indices, :]
        if self.simulator.value.shared_parameter_type is not None:
            shared_index_array = jnp.zeros(
                tokens_cfg.shape[:-2] + (1,), dtype=jnp.int32
            )
            shared_parameter_token = self.shared_parameter_idx(shared_index_array)
            tokens_cfg = jnp.concatenate(
                [
                    tokens_cfg_fractions,
                    shared_parameter_token,
                    token_cfg_models_with_params,
                ],
                axis=-2,
            )
        else:
            tokens_cfg = jnp.concatenate(
                [
                    tokens_cfg_fractions,
                    token_cfg_models_with_params,
                ],
                axis=-2,
            )
        # Combine the tokens
        tokens = val_tokens + tokens_cfg

        return tokens

    def decode_theta(
        self,
        tokens: ArrayLike,
        model_idx: Optional[Sequence[int]] = None,
        noise_idx: Optional[Sequence[int]] = None,
        model_mask: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> Array:
        """Decode tokens back into the continuous parameter vector theta."""
        del model_mask, kwargs

        if model_idx is None:
            model_idx = self.model_indices
        if noise_idx is None:
            noise_idx = self.noise_indices

        model_types_all = self.simulator.value.model_types
        noise_types_all = self.simulator.value.noise_types

        model_idx_with_params = [
            int(i) for i in model_idx if model_types_all[i].theta_dim > 0
        ]
        noise_idx_with_params = [
            int(i) for i in noise_idx if noise_types_all[i].theta_dim > 0
        ]

        tokens_split = jnp.split(tokens, tokens.shape[-2], axis=-2)

        net_subs: List[nnx.Module] = [self.theta_decode_nets[0]]
        offset = 1
        if self.simulator.value.shared_parameter_type is not None:
            net_subs.append(self.theta_decode_nets[1])
            offset += 1

        for i in model_idx_with_params:
            net_subs.append(self.theta_decode_nets[i + offset])
        for i in noise_idx_with_params:
            net_subs.append(self.theta_decode_nets[self.num_models + i + offset])

        assert len(net_subs) == len(tokens_split), (
            f"Number of networks ({len(net_subs)}) does not match number of tokens ({len(tokens_split)})"
        )

        x = jax.tree_util.tree_map(
            lambda token, net: net(token), tokens_split, net_subs
        )
        out = jnp.concatenate(x, axis=-1)
        out = jnp.squeeze(out, axis=-2)
        return out


class DMRITokenizerPP(DMRITokenizer):
    def __init__(
        self,
        simulator,
        rngs,
        token_dim=64,
        theta_encode_nets=None,
        theta_decode_nets=None,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike | None = None,
        precision: PrecisionLike | None = None,
        preferred_element_type: DTypeLike | None = None,
    ):
        super().__init__(
            simulator,
            token_dim=token_dim,
            theta_encode_nets=theta_encode_nets,
            theta_decode_nets=theta_decode_nets,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            preferred_element_type=preferred_element_type,
            rngs=rngs,
        )
        # Override the first linear layer for the fraction prior
        # Shared linear value embedding applied to all fractions
        self.theta_encode_nets[0] = nnx.Linear(
            1, token_dim, rngs=rngs, **self._linear_kwargs
        )
        self.theta_decode_nets[0] = nnx.Linear(
            token_dim,
            1,
            rngs=rngs,
            kernel_init=nnx.initializers.zeros,
            **self._linear_kwargs,
        )
        # Embedding to distinguish between model and noise components
        self.fraction_embed = nnx.Embed(
            len(self.simulator.value.model_types) - 1,
            token_dim,
            rngs=rngs,
            embedding_init=nnx.initializers.orthogonal(),
        )

    @staticmethod
    def theta_fraction_mask(model_mask: ArrayLike) -> Array:
        """
        Creates a mask for the model fractions.
        """
        _eps_mask = eps_mask
        for _ in range(model_mask.ndim - 1):
            _eps_mask = jax.vmap(_eps_mask)
        mask_fractions = _eps_mask(model_mask)
        return mask_fractions

    def _create_fraction_tokens(
        self,
        theta_fractions: ArrayLike,
        model_idx: Tuple[int, ...],
        theta_fraction_mask: ArrayLike,
    ) -> Array:
        """
        Creates fraction tokens by combining model type embeddings with fraction values.

        Args:
            theta_fractions: The fraction values for each model component.
            model_idx: Indices of the model types.
            theta_fraction_mask: Mask indicating which model components are active.

        Returns:
            jnp.ndarray: The combined fraction tokens with shape (batch_dims..., num_models_with_params, token_dim).
        """
        # Get the fraction embedding for each model type
        fraction_id = self.fraction_embed(jnp.array(model_idx[:-1], dtype=jnp.int32))

        # Create a batch-compatible version of fraction_id
        # We need to match the batch dimensions of theta_fraction_mask
        batch_shape = theta_fraction_mask.shape[
            :-1
        ]  # Get all dimensions except the last one

        # Create a tensor with the right shape for fraction_id_batch
        # The shape should be (batch_dims..., num_models_with_params, token_dim)
        fraction_id_batch = jnp.zeros(
            batch_shape + (len(model_idx[:-1]), fraction_id.shape[-1]),
            dtype=fraction_id.dtype,
        )

        # Copy the fraction_id values to each batch position
        for i in range(len(model_idx[:-1])):
            fraction_id_batch = fraction_id_batch.at[..., i, :].set(fraction_id[i])

        # Now multiply with the mask
        fraction_tokens = fraction_id_batch * theta_fraction_mask[..., None]

        # Add the value embedding
        fraction_val = self.theta_encode_nets[0](theta_fractions[..., None])
        fraction_tokens = fraction_tokens + fraction_val

        return fraction_tokens

    def embed_theta(
        self,
        theta: ArrayLike,
        tokens_cfg: ArrayLike,
        model_idx: Optional[Sequence[int]] = None,
        noise_idx: Optional[Sequence[int]] = None,
        model_mask: Optional[ArrayLike] = None,
    ) -> Array:
        if model_idx is None:
            model_idx = self.model_indices
        if noise_idx is None:
            noise_idx = self.noise_indices

        model_types_all = self.simulator.value.model_types
        noise_types_all = self.simulator.value.noise_types

        model_idx_with_params = [
            int(i) for i in model_idx if model_types_all[i].theta_dim > 0
        ]
        noise_idx_with_params = [
            int(i) for i in noise_idx if noise_types_all[i].theta_dim > 0
        ]

        idx = tuple(model_idx_with_params) + tuple(
            self.num_models + i for i in noise_idx_with_params
        )

        # First dim -> Model fractions
        # Second dim -> Shared parameters (if any)
        # Other dims -> Component parameters
        dims_per_component = [self.params_dims[0]]  # Model fractions
        offset = 1
        # Add shared parameter dimensions if they exist
        if self.simulator.value.shared_parameter_type is not None:
            dims_per_component.append(self.params_dims[1])  # Shared parameters
            offset += 1
        # Add component parameter dimensions
        dims_per_component += [self.params_dims[i + offset] for i in idx]

        assert sum(dims_per_component) == theta.shape[-1], (
            f"theta shape last axis {theta.shape} does not match the number of model components {sum(dims_per_component)}"
        )

        # Split theta into components
        dims = np.asarray(dims_per_component, dtype=np.int32)
        split_dims = np.cumsum(dims)[:-1]
        theta_split = jnp.split(theta, split_dims, axis=-1)

        # Create fraction tokens
        theta_fractions = theta_split[0]
        model_component_mask = model_mask[..., : self.num_models]
        theta_fraction_mask = self.theta_fraction_mask(model_component_mask)
        fraction_tokens = self._create_fraction_tokens(
            theta_fractions, tuple(int(i) for i in model_idx), theta_fraction_mask
        )

        # Handle shared parameters if they exist
        if self.simulator.value.shared_parameter_type is not None:
            theta_shared = theta_split[1]
            shared_tokens = self.theta_encode_nets[1](theta_shared[..., None, :])
            theta_models = theta_split[2:]
        else:
            theta_models = theta_split[1:]

        # Create a list of networks for encoding
        encode_nets = []
        offset = 1
        if self.simulator.value.shared_parameter_type is not None:
            offset += 1

        encode_nets += [self.theta_encode_nets[i + offset] for i in idx]
        # Get embeddings for components with parameters
        val_embeddings = jax.tree_util.tree_map(
            lambda x, net: net(x)[..., None, :],
            theta_models,
            encode_nets,
        )
        val_embeddings = jnp.concatenate(val_embeddings, axis=-2)
        val_tokens_cfg = tokens_cfg[..., 1:, :]
        indices_with_params = self.get_indices_with_params(model_idx, noise_idx)

        model_tokens = val_tokens_cfg[..., indices_with_params, :] + val_embeddings
        # Combine the tokens
        if self.simulator.value.shared_parameter_type is not None:
            shared_index_array = jnp.zeros(
                shared_tokens.shape[:-1], dtype=jnp.int32
            )
            shared_tokens_idx = self.shared_parameter_idx(shared_index_array)
            shared_tokens = shared_tokens_idx + shared_tokens
            theta_tokens = jnp.concatenate(
                [fraction_tokens, shared_tokens, model_tokens], axis=-2
            )
        else:
            theta_tokens = jnp.concatenate([fraction_tokens, model_tokens], axis=-2)

        return theta_tokens

    def decode_theta(
        self,
        tokens: ArrayLike,
        model_idx: Optional[Sequence[int]] = None,
        noise_idx: Optional[Sequence[int]] = None,
        model_mask: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> Array:
        del model_mask, kwargs

        if model_idx is None:
            model_idx = self.model_indices
        if noise_idx is None:
            noise_idx = self.noise_indices

        model_types_all = self.simulator.value.model_types
        noise_types_all = self.simulator.value.noise_types

        model_idx_with_params = [
            int(i) for i in model_idx if model_types_all[i].theta_dim > 0
        ]
        noise_idx_with_params = [
            int(i) for i in noise_idx if noise_types_all[i].theta_dim > 0
        ]

        tokens_split = jnp.split(tokens, tokens.shape[-2], axis=-2)

        net_subs: List[nnx.Module] = []

        num_fractions = len(model_idx) - 1
        for _ in range(num_fractions):
            net_subs.append(self.theta_decode_nets[0])

        offset = 1
        if self.simulator.value.shared_parameter_type is not None:
            net_subs.append(self.theta_decode_nets[1])
            offset += 1

        for i in model_idx_with_params:
            net_subs.append(self.theta_decode_nets[i + offset])
        for i in noise_idx_with_params:
            net_subs.append(self.theta_decode_nets[self.num_models + i + offset])

        assert len(net_subs) == len(tokens_split), (
            f"Number of networks ({len(net_subs)}) does not match number of tokens ({len(tokens_split)})"
        )

        x = jax.tree_util.tree_map(
            lambda token, net: net(token), tokens_split, net_subs
        )
        out = jnp.concatenate(x, axis=-1)
        out = jnp.squeeze(out, axis=-2)
        return out
