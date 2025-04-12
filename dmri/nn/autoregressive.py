from functools import partial
from typing import Optional
from dmri.nn.tokenizer import Tokenizer
import jax
import jax.numpy as jnp

from flax import nnx

from probjax.nn import Transformer

import optax
from dataclasses import dataclass


@dataclass
class DMRIModelSelectionConfig:
    num_layers: int = 4
    num_heads: int = 4
    widening_factor: int = 3
    attn_size: int = 16
    dropout_rate: float | None = None
    context_dim = None


@dataclass
class DMRIModelSelectionAmortizedPriorConfig:
    num_layers: int = 4
    num_heads: int = 4
    widening_factor: int = 3
    attn_size: int = 16
    dropout_rate: float | None = None
    context_dim: int = 64  # Context dimension embedding
    mask_prior_dim: int = 1  # Scalar mask probability


class BinaryAutoregressiveDecoder(nnx.Module, experimental_pytree=True):
    model_dim: int = 64
    num_heads: int = 4
    num_layers: int = 4
    widening_factor: int = 2
    attn_size: int = 16

    def __init__(
        self,
        rngs,
        model_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        widening_factor: int = 4,
        attn_size: int = 16,
        dropout_rate: int = None,
        context_dim: Optional[int] = None,
        enable_cross_attention: bool = True,
    ):
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.widening_factor = widening_factor
        self.attn_size = attn_size
        self.context_dim = context_dim

        # Why not use a shared tokenizer with the model mask?
        self.transformer = Transformer(
            model_dim,
            self.num_heads,
            self.num_layers,
            self.attn_size,
            context_dim=context_dim,
            widening_factor=self.widening_factor,
            rngs=rngs,
            dropout_rate=dropout_rate,
            enable_cross_attention=enable_cross_attention,
        )
        self.output = nnx.Linear(model_dim, 1, rngs=rngs)

    def __call__(
        self,
        model_mask,
        tokenizer: Tokenizer,
        context=None,
        y=None,
        mask=None,
        decode=False,
        deterministic=False,
        **kwargs,
    ):
        input_tokens = self._encode_model_mask(model_mask, tokenizer, **kwargs)
        # Autoregressive mask constrained
        output_tokens = self._forward_tokens(
            input_tokens,
            y,
            context=context,
            mask=mask,
            decode=decode,
            deterministic=deterministic,
        )
        # Reduce to logits
        logits = self.output(output_tokens)
        # Remove the first "padding" token output
        return logits[..., 1:, 0]

    def _encode_model_mask(self, model_mask, tokenizer, **kwargs):
        input_tokens = tokenizer.encode(model_mask=model_mask, **kwargs)
        *_, _, model_dim = input_tokens.shape

        assert model_dim == self.model_dim, (
            f"Token dim mismatch, is {model_dim}, expected {self.model_dim}"
        )
        return input_tokens

    def _forward_tokens(
        self,
        input_tokens,
        y,
        context=None,
        mask=None,
        decode=False,
        deterministic=False,
    ):
        *_, seq_len, _ = input_tokens.shape

        # Autoregressive mask constrained
        base_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        if mask is not None:
            base_mask = base_mask & mask

        if context is not None:
            context = context[..., None, :]

        x = self.transformer(
            input_tokens,
            y,
            y,
            context=context,
            mask=mask,
            deterministic=deterministic,
            decode=decode,
        )
        return x

    def loss_fn(
        self,
        model_mask,
        tokenizer,
        y,
        rng=None,
        permute_order=False,
        context=None,
        tokens_cfg=None,
        **kwargs,
    ):
        if permute_order:
            assert rng is not None, "rng must be provided if permute_order is True"

        if tokens_cfg is None:
            input_tokens = self._encode_model_mask(model_mask, tokenizer, **kwargs)
            *batch_shape, seq_len, _ = input_tokens.shape
        else:
            input_tokens = tokens_cfg
            *batch_shape, seq_len, _ = input_tokens.shape

        if permute_order:
            elements = jnp.arange(seq_len - 1)  # First element is padding token
            batch_orders = jax.vmap(lambda k: jax.random.permutation(k, elements))(
                jax.random.split(rng, int(jnp.prod(jnp.array(batch_shape))))
            ).reshape(batch_shape + [-1])

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

        # AR next token prediction
        output_tokens = self._forward_tokens(
            input_tokens,
            y,
            context=context,
            **kwargs,
        )
        model_mask_logits = self.output(output_tokens)
        # Remove the first "padding" token output
        model_mask_logits = model_mask_logits[..., 1:, 0]

        return jnp.mean(
            optax.sigmoid_binary_cross_entropy(model_mask_logits, model_mask).sum(-1)
        )

    def sample(self, key, tokenizer, y, dim, context=None):
        return naive_autoregressive_decoding(
            self, key, tokenizer, y, dim, context=context
        )

    def log_prob(self, model_mask, tokenizer, y, **kwargs):
        model_mask_logits = self(model_mask, tokenizer, y=y, **kwargs)
        # Correct Bernoulli log probability is negative binary cross entropy
        bernoulli_log_prob = -optax.sigmoid_binary_cross_entropy(model_mask_logits, model_mask)
        return jnp.sum(bernoulli_log_prob, axis=-1)


@partial(
    jax.jit,
    static_argnums=(
        2,
        4,
    ),
)
def naive_autoregressive_decoding(model, key, tokenizer, y, dim, context=None):
    x = jnp.zeros((dim,), dtype=jnp.bool_)

    def scan_fn(carry, k):
        x, i = carry
        logits = model(x.astype(jnp.int32), tokenizer, y=y, context=context)
        p_i = jax.nn.sigmoid(logits[i])

        x_i = jax.random.bernoulli(k, p_i)
        x = x.at[i].set(x_i)
        return (x, i + 1), None

    keys = jax.random.split(key, (dim,))
    x, _ = jax.lax.scan(scan_fn, (x, 0), keys)

    return x[0]
