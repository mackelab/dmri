import jax
import jax.numpy as jnp

from flax import nnx

from probjax.nn import GaussianFourierEmbedding, Transformer
from probjax.nn.loss_fn.denoising import build_time_dependent_denoising_loss
from probjax.nn.nets.diffusion_model import EDM
from probjax.nn import MLP


class ScalarTokenizer(nnx.Module, experimental_pytree=True):
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

    def __call__(self, x, node_ids, condition_mask, context=None):
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

    def decode(self, h):
        return self.outlayer(h)


class VectorTokenizer(nnx.Module, experimental_pytree=True):
    def __init__(self, num_nodes, dims, rngs, value_dim=20, id_dim=20, cond_dim=10):
        self.model_dim = value_dim + id_dim + cond_dim
        self.num_nodes = num_nodes
        self.max_dim = max(dims)
        self.dims = list(dims)
        self.value_dim = value_dim
        self.id_dim = id_dim
        self.cond_dim = cond_dim

        self.embed_id = nnx.Embed(num_nodes, self.id_dim, rngs=rngs)
        self.embed_value = nnx.Linear(self.max_dim, self.value_dim, rngs=rngs)
        self.outlayer = nnx.Linear(self.model_dim, self.max_dim, rngs=rngs)
        if cond_dim > 0:
            self.condition_token = nnx.Param(
                0.01 * jax.random.normal(rngs.next(), (1, cond_dim))
            )

    def __call__(self, x, node_ids, condition_mask):
        node_embed = self.embed_id(node_ids)
        # Pad all inputs to max_dim
        x = jax.tree_map(
            lambda x: jnp.pad(
                x, [(0, 0)] * (x.ndim - 1) + [(0, self.max_dim - x.shape[-1])]
            ),
            x,
        )
        x = jnp.concatenate(x, axis=-2)

        value_embed = self.embed_value(x)

        if self.cond_dim > 0:
            condition_embed = self.condition_token.value * condition_mask[..., None]
            input_embed = jnp.concatenate(
                [node_embed, value_embed, condition_embed], axis=-1
            )
        else:
            input_embed = jnp.concatenate([node_embed, value_embed], axis=-1)
        return input_embed

    def decode(self, h):
        out = self.outlayer(h)
        out_split = jnp.split(out, self.num_nodes, axis=-2)
        out = jax.tree_map(lambda x, d: x[..., :d], out_split, self.dims)
        return out


class Simformer(nnx.Module, experimental_pytree=True):
    model_dim: int = 50
    condition_dim: int = 10
    num_heads: int = 4
    num_layers: int = 4
    widening_factor: int = 2
    attn_size: int = 10

    def __init__(
        self,
        tokenizer,
        rngs,
        model_dim=50,
        condition_dim=10,
        num_heads=4,
        num_layers=4,
        attn_size=10,
        widening_factor=2,
    ) -> None:
        self.model_dim = model_dim
        self.condition_dim = condition_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.attn_size = attn_size
        self.widening_factor = widening_factor

        self.tokenizer = tokenizer
        self.time_embedding = GaussianFourierEmbedding(1, self.model_dim, rngs=rngs)
        self.transformer = Transformer(
            self.model_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            attn_size=self.attn_size,
            rngs=rngs,
            context_dim=self.model_dim,
        )

    def __call__(
        self,
        t,
        x,
        node_ids,
        condition_mask,
        *args,
        context=None,
        attention_mask=None,
        **kwargs,
    ):
        time_embed = self.time_embedding(t)
        input_embed = self.tokenizer(x, node_ids, condition_mask, context=context)

        while time_embed.ndim < input_embed.ndim:
            time_embed = time_embed[None, ...]

        output = self.transformer(input_embed, context=time_embed, mask=attention_mask)
        output = self.tokenizer.decode(output)
        return output


class ModelIdentificationDecoder(nnx.Module, experimental_pytree=True):
    model_dim: int = 50
    num_heads: int = 4
    num_layers: int = 2
    widening_factor: int = 2
    attn_size: int = 10

    def __init__(
        self,
        dim,
        context_dim,
        rngs,
        model_dim=50,
        num_heads=5,
        num_layers=2,
        widening_factor=2,
        attn_size=10,
        dropout_rate=None,
        context_net=None,
    ):
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.widening_factor = widening_factor
        self.attn_size = attn_size

        self.embed = nnx.Embed(2, dim, rngs=rngs)
        self.transformer = Transformer(
            dim,
            self.num_heads,
            self.num_heads,
            self.attn_size,
            context_dim=self.model_dim,
            widening_factor=self.widening_factor,
            rngs=rngs,
            dropout_rate=dropout_rate,
        )
        self.output = nnx.Linear(dim, 1, rngs=rngs)
        if context_net is None:
            self.context_embedding = MLP([context_dim, 2 * dim, dim], rngs=rngs)
        else:
            self.context_embedding = context_net

    def __call__(
        self, x, context, decode=False, deterministic=False, return_latents=False
    ):
        # Autoregressive mask
        x = self.embed(x.astype(jnp.int32))
        context = self.context_embedding(context)
        start_token = jnp.ones_like(x[..., :1, :])
        x = jnp.concatenate([start_token, x], axis=-2)
        mask = jnp.tril(jnp.ones((x.shape[-2], x.shape[-2])))
        context = context[..., None, :]
        x = self.transformer(
            x, context=context, mask=mask, deterministic=deterministic, decode=decode
        )
        logits = self.output(x)
        if return_latents:
            return logits[..., :-1, 0], x[..., :-1, :]
        return logits[..., :-1, 0]
