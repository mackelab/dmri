from dataclasses import dataclass
from functools import partial
from typing import Any, Optional

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from probjax.nn import CausalMask, GaussianFourierEmbedding, Transformer
from probjax.nn.layers.attention import flex_attention
from probjax.utils.typing import Array, ArrayLike, DTypeLike, PrecisionLike, RngKey

from dmri.nn.tokenizer import Tokenizer


@dataclass
class DMRIModelSelectionConfig:
    num_layers: int = 4
    num_heads: int = 4
    widening_factor: int = 3
    attn_size: int = 16
    dropout_rate: float = 0.0
    prior_params_embed_dim: int = 0
    mask_prior_dim: Optional[int] = None
    kv_in_features: Optional[int] = None
    use_flash_attention: bool = False
    use_flash_cross_attention: bool = False
    normalize_qk_attn: bool = False
    normalize_qk_cross_attn: bool = False
    dtype: DTypeLike | None = None
    param_dtype: DTypeLike | None = None
    precision: PrecisionLike | None = None
    preferred_element_type: DTypeLike | None = None


@dataclass
class DMRIModelSelectionAmortizedPriorConfig(DMRIModelSelectionConfig):
    prior_params_embed_dim: int = 64  # Context dimension embedding
    mask_prior_dim: Optional[int] = 1  # Scalar mask probability


class BinaryAutoregressiveDecoder(nnx.Module):
    model_dim: int = 64
    num_heads: int = 4
    num_layers: int = 4
    widening_factor: int = 2
    attn_size: int = 16

    def __init__(
        self,
        rngs: nnx.Rngs,
        model_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        widening_factor: int = 4,
        attn_size: int = 16,
        dropout_rate: float = 0.0,
        prior_params_embed_dim: int = 0,
        mask_prior_dim: Optional[int] = None,
        additional_context_dim: int = 0,
        kv_in_features: Optional[int] = None,
        enable_cross_attention: bool = True,
        use_flash_attention: bool = False,
        use_flash_cross_attention: bool = False,
        normalize_qk_attn: bool = False,
        normalize_qk_cross_attn: bool = False,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike | None = None,
        precision: PrecisionLike | None = None,
        preferred_element_type: DTypeLike | None = None,
    ):
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.widening_factor = widening_factor
        self.attn_size = attn_size
        self.prior_params_embed_dim = prior_params_embed_dim
        self.additional_context_dim = additional_context_dim
        self.total_context_dim = (
            self.prior_params_embed_dim + self.additional_context_dim
        )
        context_dim = self.total_context_dim if self.total_context_dim > 0 else None
        self.context_dim = context_dim
        self.use_flash_attention = use_flash_attention

        precision_kwargs: dict[str, Any] = {
            name: value
            for name, value in (
                ("dtype", dtype),
                ("param_dtype", param_dtype),
                ("precision", precision),
                ("preferred_element_type", preferred_element_type),
            )
            if value is not None
        }

        attn_fn = flex_attention if use_flash_attention else None
        cross_attn_fn = flex_attention if use_flash_cross_attention else None

        if self.prior_params_embed_dim > 0:
            if mask_prior_dim is None:
                raise ValueError(
                    "mask_prior_dim must be provided when prior_params_embed_dim > 0."
                )
            self.mask_prior_dim = mask_prior_dim
            self.mask_prior_embed = GaussianFourierEmbedding(
                mask_prior_dim,
                self.prior_params_embed_dim,
                rngs=rngs,
                **precision_kwargs,  # type: ignore[arg-type]
            )
        else:
            self.mask_prior_dim = None
            self.mask_prior_embed = None

        self.kv_in_features = (
            kv_in_features if kv_in_features is not None else model_dim
        )

        self.transformer = Transformer(
            model_dim,
            self.num_heads,
            self.num_layers,
            self.attn_size,
            kv_in_features=self.kv_in_features,
            context_dim=context_dim,
            widening_factor=self.widening_factor,
            rngs=rngs,
            dropout_rate=dropout_rate,
            attention_fn=attn_fn,
            cross_attention_fn=cross_attn_fn,
            enable_cross_attention=enable_cross_attention,
            normalize_qk_attn=normalize_qk_attn,
            normalize_qk_cross_attn=normalize_qk_cross_attn,
            **precision_kwargs,
        )
        self.out_norm = nnx.LayerNorm(model_dim, rngs=rngs)
        self.output = nnx.Linear(
            model_dim,
            1,
            rngs=rngs,
            **precision_kwargs,
        )

    def __call__(
        self,
        model_mask: Array,
        tokenizer: Tokenizer,
        mask_prior: Optional[Array] = None,
        additional_context: Optional[Array] = None,
        y: Optional[Array] = None,
        mask: Optional[Array] = None,
        decode: bool = False,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> Array:
        input_tokens = self._encode_model_mask(model_mask, tokenizer, **kwargs)
        batch_shape = input_tokens.shape[:-2]
        context_vec = self._prepare_context(
            tuple(batch_shape),
            input_tokens.dtype,
            mask_prior=mask_prior,
            additional_context=additional_context,
        )
        # Autoregressive mask constrained
        output_tokens = self._forward_tokens(
            input_tokens,
            y,
            context=context_vec,
            attention_mask=mask,
            decode=decode,
            deterministic=deterministic,
        )
        output_tokens = self.out_norm(output_tokens)
        # Reduce to logits
        logits = self.output(output_tokens)
        # Remove the first "padding" token output
        return logits[..., :-1, 0]

    def _encode_model_mask(
        self, model_mask: ArrayLike, tokenizer: Tokenizer, **kwargs: Any
    ) -> Array:
        input_tokens = tokenizer.encode(model_mask=model_mask, **kwargs)
        *_, _, model_dim = input_tokens.shape

        assert model_dim == self.model_dim, (
            f"Token dim mismatch, is {model_dim}, expected {self.model_dim}"
        )
        return input_tokens

    def _prepare_context(
        self,
        batch_shape: tuple[Any, ...],
        dtype: jnp.dtype,
        mask_prior: Optional[Array],
        additional_context: Optional[Array],
    ) -> Optional[Array]:
        batch_shape = tuple(batch_shape)
        context_parts = []

        if self.prior_params_embed_dim > 0:
            if mask_prior is None:
                raise ValueError(
                    "mask_prior must be provided for prior-parameter context."
                )
            mask_prior_arr = jnp.asarray(mask_prior, dtype=dtype)

            expected_shape = batch_shape + (self.mask_prior_dim,)

            # Allow callers to omit the trailing singleton dimension.
            if mask_prior_arr.ndim == len(batch_shape):
                mask_prior_arr = mask_prior_arr[..., None]

            # Make sure the final dimension can align with the configured context size.
            if mask_prior_arr.shape[-1] != self.mask_prior_dim:
                if mask_prior_arr.shape[-1] == 1:
                    mask_prior_arr = jnp.broadcast_to(
                        mask_prior_arr,
                        mask_prior_arr.shape[:-1] + (self.mask_prior_dim,),
                    )
                else:
                    raise ValueError(
                        "Mask prior dimensionality does not match configured mask_prior_dim."
                    )

            # Broadcast leading dimensions if needed (e.g. scalar or single batch prior).
            try:
                mask_prior_arr = jnp.broadcast_to(mask_prior_arr, expected_shape)
            except ValueError as err:
                raise ValueError(
                    "Mask prior batch shape does not match tokens."
                ) from err

            if self.mask_prior_embed is None:
                raise ValueError("Mask prior embedding is not initialized.")
            context_parts.append(self.mask_prior_embed(mask_prior_arr))
        elif mask_prior is not None:
            raise ValueError(
                "Mask prior provided but prior_params_embed_dim is set to 0."
            )

        if self.additional_context_dim > 0:
            if additional_context is None:
                additional_context_arr = jnp.zeros(
                    batch_shape + (self.additional_context_dim,),
                    dtype=dtype,
                )
            else:
                additional_context_arr = jnp.asarray(additional_context, dtype=dtype)
                if additional_context_arr.shape[:-1] != batch_shape:
                    raise ValueError(
                        "Additional context batch shape does not match tokens."
                    )
                if additional_context_arr.shape[-1] != self.additional_context_dim:
                    raise ValueError(
                        "Additional context dimensionality does not match configuration."
                    )
            context_parts.append(additional_context_arr)
        elif additional_context is not None:
            raise ValueError(
                "Additional context provided but additional_context_dim is 0."
            )

        if not context_parts:
            return None
        return jnp.concatenate(context_parts, axis=-1)

    def _forward_tokens(
        self,
        input_tokens: Array,
        y: Optional[Array],
        context: Optional[Array] = None,
        attention_mask: Optional[Array] = None,
        decode: bool = False,
        deterministic: bool = False,
    ) -> Array:
        *_, seq_len, _ = input_tokens.shape

        # Autoregressive mask constrained
        if not self.use_flash_attention:
            base_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
            if attention_mask is not None:
                base_mask = base_mask & attention_mask
        else:
            base_mask = CausalMask()
            if attention_mask is not None:
                raise NotImplementedError("Not supported with flash attention")

        batch_shape = input_tokens.shape[:-2]
        if self.total_context_dim > 0:
            if context is None:
                raise ValueError("Context expected but not provided.")
            context = jnp.asarray(context, dtype=input_tokens.dtype)
            if context.shape[:-1] != batch_shape:
                raise ValueError("Context batch shape does not match inputs.")
            if context.shape[-1] != self.total_context_dim:
                raise ValueError(
                    "Context dimension mismatch for autoregressive decoder."
                )
        elif context is not None:
            raise ValueError("Context provided but no context dimension configured.")

        if context is not None:
            context = context[..., None, :]

        x = self.transformer(
            input_tokens,
            y,
            y,
            context=context,
            mask=base_mask,
            deterministic=deterministic,
            decode=decode,
        )
        return x

    def loss_fn(
        self,
        model_mask: ArrayLike,
        tokenizer: Tokenizer,
        y: Array,
        rng: Optional[RngKey] = None,
        permute_order: bool = False,
        label_smoothing: float = 0.0,
        mask_prior: Optional[Array] = None,
        additional_context: Optional[Array] = None,
        tokens_cfg: Optional[Array] = None,
        **kwargs: Any,
    ) -> Array:
        if permute_order:
            assert rng is not None, "rng must be provided if permute_order is True"

        if tokens_cfg is None:
            input_tokens = self._encode_model_mask(model_mask, tokenizer, **kwargs)
            *batch_shape, seq_len, _ = input_tokens.shape
        else:
            input_tokens = tokens_cfg
            *batch_shape, seq_len, _ = input_tokens.shape

        if permute_order:
            assert rng is not None, "rng must be provided if permute_order is True"
            elements = jnp.arange(seq_len - 1)  # First element is padding token
            def rand_perm(key):
                return jax.random.permutation(key, elements)
            batch_orders = jax.vmap(rand_perm)(
                jax.random.split(rng, batch_shape)
            )

            # Input tokens should be permuted, except the first element of dim -2
            tokens_except_first = input_tokens[..., 1:, :]

            # Create a function to permute a single batch element
            def permute_batch_element(tokens, order):
                return tokens[order]

            # Apply permutation to each batch element
            tokens_except_first = jax.vmap(permute_batch_element)(
                tokens_except_first, batch_orders
            )
            input_tokens = input_tokens.at[..., 1:, :].set(tokens_except_first)

            # Target should be permuted
            model_mask = jax.vmap(permute_batch_element)(model_mask, batch_orders)

        if label_smoothing > 0.0:
            model_mask = model_mask * (1.0 - label_smoothing) + 0.5 * label_smoothing

        context_vec = self._prepare_context(
            tuple(batch_shape),
            input_tokens.dtype,
            mask_prior=mask_prior,
            additional_context=additional_context,
        )

        # AR next token prediction
        output_tokens = self._forward_tokens(
            input_tokens,
            y,
            context=context_vec,
            **kwargs,
        )
        model_mask_logits = self.output(output_tokens)
        model_mask_logits = model_mask_logits[..., :-1, 0]

        return jnp.mean(
            optax.sigmoid_binary_cross_entropy(model_mask_logits, model_mask).sum(-1)
        )

    def sample(
        self,
        key,
        tokenizer,
        y,
        dim,
        mask_prior: Optional[Array] = None,
        additional_context: Optional[Array] = None,
    ):
        return naive_autoregressive_decoding(
            self,
            key,
            tokenizer,
            y,
            dim,
            mask_prior=mask_prior,
            additional_context=additional_context,
        )

    def log_prob(
        self,
        model_mask: Array,
        tokenizer: Tokenizer,
        y: Array,
        mask_prior: Optional[Array] = None,
        additional_context: Optional[Array] = None,
        **kwargs: Any,
    ) -> Array:
        model_mask_logits = self(
            model_mask,
            tokenizer,
            y=y,
            mask_prior=mask_prior,
            additional_context=additional_context,
            **kwargs,
        )
        # Correct Bernoulli log probability is negative binary cross entropy
        bernoulli_log_prob = -optax.sigmoid_binary_cross_entropy(
            model_mask_logits, model_mask
        )
        return jnp.sum(bernoulli_log_prob, axis=-1)


@partial(jax.jit, static_argnums=(2, 4))
def naive_autoregressive_decoding(
    model: BinaryAutoregressiveDecoder,
    key: RngKey,
    tokenizer: Tokenizer,
    y: Array,
    dim: int,
    mask_prior: Optional[Array] = None,
    additional_context: Optional[Array] = None,
) -> Array:
    x = jnp.zeros((dim,), dtype=jnp.bool_)

    def scan_fn(carry: tuple[Array, int], k: RngKey) -> tuple[tuple[Array, int], None]:
        x, i = carry
        logits = model(
            x.astype(jnp.int32),
            tokenizer,
            y=y,
            mask_prior=mask_prior,
            additional_context=additional_context,
        )
        p_i = jax.nn.sigmoid(logits[i])

        x_i = jax.random.bernoulli(k, p_i)
        x = x.at[i].set(x_i)
        return (x, i + 1), None

    keys = jax.random.split(key, (dim,))
    x, _ = jax.lax.scan(scan_fn, (x, 0), keys)

    return x[0]
