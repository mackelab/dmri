from abc import abstractmethod
from collections import defaultdict
from copy import deepcopy
from functools import cache
from typing import Any, List, Optional, Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.typing import ArrayLike

from dmri.simulators import MultiCompartment
from dmri.utils.transform import dirichlet_to_normal


def map_classes_to_indices(class_list: list, start_idx: int = 0):
    """
    Returns a dictionary mapping each class object in `class_list`
    to a list of distinct integer indices corresponding to all of
    its occurrences in `class_list`.
    """
    mapping = defaultdict(list)

    for idx, cls in enumerate(class_list):
        # Classes make problems with tree_flatten
        mapping[cls.__name__].append(idx + start_idx)

    return dict(mapping)  # convert defaultdict back to a normal dict


class Tokenizer(nnx.Module, experimental_pytree=True):
    def __call__(self, *args, **kwds):
        return self.encode(*args, **kwds)

    @abstractmethod
    def encode(self, *args, **kwargs):
        pass

    @abstractmethod
    def decode(self, *args, **kwargs):
        pass


class ScalarTokenizer(Tokenizer):
    def __init__(
        self, num_nodes, rngs, value_dim=20, id_dim=20, cond_dim=10, context_dim=None
    ):
        self.model_dim = value_dim + id_dim + cond_dim
        self.num_nodes = num_nodes
        self.value_dim = value_dim
        self.id_dim = id_dim
        self.cond_dim = cond_dim

        self.embed_id = nnx.Embed(num_nodes, self.id_dim, rngs=rngs)
        self.embed_value = nnx.Linear(1, self.value_dim, rngs=rngs)
        if cond_dim > 0:
            self.condition_token = nnx.Param(
                0.01 * jax.random.normal(rngs.next(), (1, cond_dim))
            )
        if context_dim is not None:
            self.context_embed = nnx.Linear(context_dim, self.model_dim, rngs=rngs)
        self.context_dim = context_dim

        self.outlayer = nnx.Linear(self.model_dim, 1, rngs=rngs)

    def encode(self, x, node_ids, condition_mask, context=None):
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

    def decode(self, h, node_ids, condition_mask, context=None):
        return self.outlayer(h)


class StructuredTokenizer(Tokenizer):
    def __init__(
        self,
        dims_by_id: list[int],
        value_dim: int = 32,
        id_dim: int = 32,
        cond_dim: int = 0,
        encode_nets: Optional[list[nnx.Module]] = None,
        decode_nets: Optional[list[nnx.Module]] = None,
        rngs: nnx.RngStream = None,
    ):
        self.dims_by_id = dims_by_id
        self.num_nodes = len(dims_by_id)
        if encode_nets is None:
            encode_nets = [nnx.Linear(d, value_dim, rngs=rngs) for d in dims_by_id]
        if decode_nets is None:
            dim_token = id_dim + value_dim + cond_dim
            decode_nets = [nnx.Linear(dim_token, d, rngs=rngs) for d in dims_by_id]

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

    def __call__(self, x, node_ids, condition_mask=None, **kwargs):
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

    def decode(self, h, node_ids, condition_mask=None, **kwargs):
        hs = jnp.split(h, h.shape[-2], axis=-2)
        net_subs = [self.decode_nets[i] for i in node_ids]
        x = jax.tree_util.tree_map(lambda x, net: net(x), hs, net_subs)
        out = jnp.concatenate(x, axis=-1)
        out = jnp.squeeze(out, axis=-2)
        return out


class DMRITokenizer(Tokenizer, experimental_pytree=True):
    """
    A tokenizer for dMRI multi-compartment model configurations. It handles embedding
    and decoding of model types, noise types, and associated parameters.
    """

    def __init__(
        self,
        simulator: type[MultiCompartment],
        rngs: Any,
        token_dim: int = 64,
        theta_encode_nets: Optional[list[nnx.Module]] = None,
        theta_decode_nets: Optional[list[nnx.Module]] = None,
        init_component_embeddings: Callable = nnx.initializers.orthogonal(),
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

        self.model_types_to_idx = nnx.Intermediate(
            map_classes_to_indices(self.simulator.value.model_types)
        )
        self.noise_types_to_idx = nnx.Intermediate(
            map_classes_to_indices(
                self.simulator.value.noise_types,
                start_idx=len(self.simulator.value.model_types),
            )
        )
        self.embed_idx = nnx.Embed(
            rngs=rngs,
            num_embeddings=len(simulator.model_types) + len(simulator.noise_types),
            features=token_dim,
            embedding_init=init_component_embeddings,
        )
        self.embed_fraction = nnx.Linear(
            len(simulator.model_types), token_dim, rngs=rngs
        )

        # Default to linear layers
        if theta_encode_nets is None:
            theta_encode_nets = [
                nnx.Linear(d, token_dim, rngs=rngs) for d in self.params_dims
            ]
        if theta_decode_nets is None:
            theta_decode_nets = [
                nnx.Linear(token_dim, d, rngs=rngs) for d in self.params_dims
            ]
        self.theta_encode_nets = theta_encode_nets
        self.theta_decode_nets = theta_decode_nets

    def encode(
        self,
        theta: Optional[ArrayLike] = None,
        model_mask: Optional[ArrayLike] = None,
        tokens_cfg: Optional[ArrayLike] = None,
        alpha_prior: Optional[ArrayLike] = None,
        model_types: Optional[list[type]] = None,
        noise_types: Optional[list[type]] = None,
    ):
        if model_types is None:
            model_types = tuple(self.simulator.value.model_types)
        if noise_types is None:
            noise_types = tuple(self.simulator.value.noise_types)

        if model_mask is None:
            assert tokens_cfg is not None, "model_mask or tokens_cfg must be provided"

        if tokens_cfg is None:
            tokens_cfg = self.embed_cfgs(
                model_mask,
                alpha_prior,
                model_types=model_types,
                noise_types=noise_types,
            )
        # We assume that the provided tokens_cfg is already in the correct shape
        if theta is not None:
            tokens = self.embed_theta(theta, tokens_cfg, model_types, noise_types)
        else:
            tokens = tokens_cfg
        return tokens

    def decode(
        self,
        tokens: ArrayLike,
        model_types: Optional[list[type]] = None,
        noise_types: Optional[list[type]] = None,
        **kwargs,
    ):
        if model_types is None:
            model_types = tuple(self.simulator.value.model_types)
        if noise_types is None:
            noise_types = tuple(self.simulator.value.noise_types)

        theta = self.decode_theta(tokens, model_types, noise_types)
        return theta

    @cache
    def get_model_idx(self, model_types: list[type]) -> list[int]:
        """
        Returns a list of model indices corresponding to the provided model types.
        """
        model_types_to_idx = deepcopy(self.model_types_to_idx.value)
        return tuple(
            [model_types_to_idx[model.__name__].pop(0) for model in model_types]
        )

    @cache
    def get_noise_idx(self, noise_types: List[type]) -> List[int]:
        """
        Returns a list of noise indices corresponding to the provided noise types.
        """
        noise_types_to_idx = deepcopy(self.noise_types_to_idx.value)
        return tuple(
            [noise_types_to_idx[noise.__name__].pop(0) for noise in noise_types]
        )

    @cache
    def get_idx(self, model_types: List[type], noise_types: List[type]) -> List[int]:
        """
        Returns a list of model and noise indices corresponding to the provided model and noise types.
        """
        model_idx = self.get_model_idx(model_types)
        noise_idx = self.get_noise_idx(noise_types)
        return model_idx + noise_idx

    def embed_cfgs(
        self,
        model_mask: ArrayLike,
        alpha_prior: Optional[ArrayLike] = None,
        model_types: Optional[list[type]] = None,
        noise_types: Optional[list[type]] = None,
    ) -> ArrayLike:
        """
        Embeds the configuration of model and noise types into tokens.

        Args:
            model_mask (ArrayLike): A binary mask indicating active model components.
            alpha_prior (Optional[ArrayLike]): Prior fractions for model components.
            model_types (Optional[List[type]]): List of model types.
            noise_types (Optional[List[type]]): List of noise types.

        Returns:
            ArrayLike: The embedded tokens for each model/noise component.
        """
        if model_types is None:
            model_types = tuple(self.simulator.value.model_types)
        if noise_types is None:
            noise_types = tuple(self.simulator.value.noise_types)
        if alpha_prior is None:
            alpha_prior = self.simulator.value.fraction_prior

        idx = self.get_idx(model_types, noise_types)
        print(idx)
        assert len(idx) == model_mask.shape[-1], (
            f"model_mask shape last axis {model_mask.shape} does not match the number of model components {len(idx)}"
        )

        *batch_dims, T = model_mask.shape
        # Broadcast alpha_prior to the batch dims
        alpha_prior = jnp.broadcast_to(
            alpha_prior, batch_dims + [len(self.simulator.model_types)]
        )

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
        model_types: Optional[list[type]] = None,
        noise_types: Optional[list[type]] = None,
        model_mask: Optional[ArrayLike] = None,
    ) -> ArrayLike:
        """
        Embeds the continuous parameter vector theta into token representation.

        Args:
            theta (ArrayLike): The parameters of each model/noise component.
            tokens_cfg (ArrayLike): Configuration tokens returned by embed_cfgs.
            model_types (Optional[List[type]]): List of model types.
            noise_types (Optional[List[type]]): List of noise types.

        Returns:
            ArrayLike: The token representation augmented with encoded parameters.
        """
        if model_types is None:
            model_types = tuple(self.simulator.value.model_types)
        if noise_types is None:
            noise_types = tuple(self.simulator.value.noise_types)

        model_idx = self.get_model_idx(model_types)
        noise_idx = self.get_noise_idx(noise_types)
        idx = model_idx + noise_idx

        # First dim -> Model fractions
        # Other dims -> Component parameters
        dims_per_component = [self.params_dims[0]]
        dims_per_component += [
            self.params_dims[i + 1]
            for i in idx  # 0 is the fraction prior
        ]

        assert sum(dims_per_component) == theta.shape[-1], (
            f"theta shape last axis {theta.shape} does not match the number of model components {sum(dims_per_component)}"
        )

        # Split theta into components
        dims = np.asarray(dims_per_component, dtype=np.int32)
        split_dims = np.cumsum(dims)[:-1]
        theta_split = jnp.split(theta, split_dims, axis=-1)
        # Get the val embeddings
        val_embeddings = jax.tree_util.tree_map(
            lambda x, net: net(x)[..., None, :], theta_split, self.theta_encode_nets
        )
        val_tokens = jnp.concatenate(val_embeddings, axis=-2)

        # Combine the tokens
        tokens = val_tokens + tokens_cfg

        return tokens

    def decode_theta(
        self,
        tokens: ArrayLike,
        model_types: Optional[List[type]] = None,
        noise_types: Optional[List[type]] = None,
        model_mask: Optional[ArrayLike] = None,
        **kwargs,
    ) -> ArrayLike:
        """
        Decodes the tokens back into the continuous parameter vector theta.

        Args:
            tokens (ArrayLike): The token representation that includes the embedded parameters.
            model_types (Optional[List[type]]): List of model types.
            noise_types (Optional[List[type]]): List of noise types.

        Returns:
            ArrayLike: The decoded parameter vector.
        """
        del model_mask

        if model_types is None:
            model_types = tuple(self.simulator.value.model_types)
        if noise_types is None:
            noise_types = tuple(self.simulator.value.noise_types)

        model_idx = self.get_model_idx(model_types)
        noise_idx = self.get_noise_idx(noise_types)
        idx = model_idx + noise_idx

        tokens_split = jnp.split(tokens, tokens.shape[-2], axis=-2)
        net_subs = [self.theta_decode_nets[0]] + [
            self.theta_decode_nets[i + 1] for i in idx
        ]
        x = jax.tree_util.tree_map(lambda x, net: net(x), tokens_split, net_subs)
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
    ):
        super().__init__(
            simulator,
            rngs,
            token_dim,
            theta_encode_nets,
            theta_decode_nets,
            init_component_embeddings=self._init_class_embeddings,
        )
        # Override the first linear layer for the fraction prior
        # Shared linear value embedding applied to all fractions
        self.theta_encode_nets[0] = nnx.Linear(1, token_dim, rngs=rngs)
        self.theta_decode_nets[0] = nnx.Linear(token_dim, 1, rngs=rngs)
        # Embedding to distinguish between model and noise components
        self.fraction_embed = nnx.Embed(
            len(self.simulator.value.model_types) - 1,
            token_dim,
            rngs=rngs,
            embedding_init=nnx.initializers.orthogonal(),
        )

    def _init_class_embeddings(self, key, shape, dtype=jnp.float32):
        """Custom initializer that makes embeddings for same classes identical
        but orthogonal between different classes.

        Args:
            key: PRNG key
            shape: Shape of embeddings (num_embeddings, embedding_dim)
            dtype: Data type of embeddings
        """
        # Get unique class indices from model_types_to_idx
        class_to_indices = self.model_types_to_idx.value
        num_classes = len(class_to_indices)

        # Initialize orthogonal embeddings for each unique class
        class_embeddings = jax.random.orthogonal(
            key, n=shape[1], shape=(num_classes,), m=1
        ).squeeze(axis=-1)

        # Create full embedding matrix by mapping class embeddings to all indices
        embeddings = jnp.zeros(shape, dtype=dtype)
        for i, (_, indices) in enumerate(class_to_indices.items()):
            embeddings = embeddings.at[tuple(indices), ...].set(class_embeddings[i])

        return embeddings

    @staticmethod
    def transform_model_to_theta_mask(model_mask):
        eps = dirichlet_to_normal(
            jnp.ones(model_mask.shape[0]),
            jnp.ones(model_mask.shape[0]) / model_mask.shape[0],
            model_mask,
        )
        return eps != 0

    def embed_theta(
        self, theta, tokens_cfg, model_types=None, noise_types=None, model_mask=None
    ):
        if model_types is None:
            model_types = tuple(self.simulator.value.model_types)
        if noise_types is None:
            noise_types = tuple(self.simulator.value.noise_types)

        model_idx = self.get_model_idx(model_types)
        noise_idx = self.get_noise_idx(noise_types)
        idx = model_idx + noise_idx

        # First dim -> Model fractions
        # Other dims -> Component parameters
        dims_per_component = [self.params_dims[0]]
        dims_per_component += [
            self.params_dims[i + 1]
            for i in idx  # 0 is the fraction prior
        ]

        assert sum(dims_per_component) == theta.shape[-1], (
            f"theta shape last axis {theta.shape} does not match the number of model components {sum(dims_per_component)}"
        )

        # Split theta into components
        dims = np.asarray(dims_per_component, dtype=np.int32)
        split_dims = np.cumsum(dims)[:-1]
        theta_split = jnp.split(theta, split_dims, axis=-1)

        # Get model components mask
        theta_fractions = theta_split[0]
        model_component_mask = model_mask[..., : len(model_types)]
        theta_fraction_mask = self.transform_model_to_theta_mask(model_component_mask)
        # For present models get the fraction embedding
        fraction_id = self.fraction_embed(jnp.array(model_idx[:-1], dtype=jnp.int32))
        fraction_val = self.theta_encode_nets[0](theta_fractions[..., None])
        fraction_tokens = fraction_id * theta_fraction_mask[..., None]
        fraction_tokens = fraction_tokens + fraction_val

        # Get the val embeddings
        theta_models = theta_split[1:]
        val_embeddings = jax.tree_util.tree_map(
            lambda x, net: net(x)[..., None, :],
            theta_models,
            self.theta_encode_nets[1:],
        )
        val_tokens = jnp.concatenate(val_embeddings, axis=-2)
        val_tokens = val_tokens + tokens_cfg[..., 1:, :]
        # Combine the tokens
        tokens = jnp.concatenate([fraction_tokens, val_tokens], axis=-2)

        return tokens

    def decode_theta(self, tokens, model_types=None, noise_types=None, model_mask=None):
        if model_types is None:
            model_types = tuple(self.simulator.value.model_types)
        if noise_types is None:
            noise_types = tuple(self.simulator.value.noise_types)

        model_idx = self.get_model_idx(model_types)
        noise_idx = self.get_noise_idx(noise_types)
        idx = model_idx + noise_idx

        tokens_split = jnp.split(tokens, tokens.shape[-2], axis=-2)
        net_subs = [self.theta_decode_nets[0]] * (len(model_types) - 1) + [
            self.theta_decode_nets[i + 1] for i in idx
        ]
        x = jax.tree_util.tree_map(lambda x, net: net(x), tokens_split, net_subs)
        out = jnp.concatenate(x, axis=-1)
        out = jnp.squeeze(out, axis=-2)
        return out
