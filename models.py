from functools import partial
import jax
import jax.numpy as jnp
import numpy as np

from flax import nnx

from probjax.nn import GaussianFourierEmbedding, Transformer
from probjax.nn.loss_fn.denoising import build_time_dependent_denoising_loss
from probjax.nn.nets.diffusion_model import EDM
from probjax.nn import MLP


def random_split_like_tree(rng_key, target=None, treedef=None):
    if treedef is None:
        treedef = jax.tree_structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree_util.tree_unflatten(treedef, keys)


def tree_random_normal_like(rng_key, target):
    keys_tree = random_split_like_tree(rng_key, target)
    return jax.tree_util.tree_map(
        lambda l, k: jax.random.normal(k, l.shape, l.dtype),
        target,
        keys_tree,
    )


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

    def decode(self, h, node_ids):
        return self.outlayer(h)


class PyTreeTokenizer(nnx.Module, experimental_pytree=True):
    def __init__(
        self,
        dims_by_id,
        encode_nets,
        decode_nets,
        id_dim=50,
        cond_dim=10,
        context_net=None,
        rngs=None,
    ):
        self.dims_by_id = dims_by_id
        self.encode_nets = encode_nets
        self.decode_nets = decode_nets
        self.embed_id = nnx.Embed(len(self.encode_nets), id_dim, rngs=rngs)
        self.context_net = context_net

    def __call__(self, x, node_ids, condition_mask=None, **kwargs):
        dims = np.asarray([self.dims_by_id[i] for i in node_ids])
        split_dims = np.cumsum(dims)[:-1]
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
        context_embed=None,
    ) -> None:
        self.model_dim = model_dim
        self.condition_dim = condition_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.attn_size = attn_size
        self.widening_factor = widening_factor

        self.tokenizer = tokenizer
        self.time_embedding = GaussianFourierEmbedding(
            1, self.model_dim // 2, rngs=rngs
        )
        self.transformer = Transformer(
            self.model_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            attn_size=self.attn_size,
            rngs=rngs,
            context_dim=self.model_dim,
        )
        self.context_embed = context_embed

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
        input_embed = self.tokenizer(x, node_ids, condition_mask)

        while time_embed.ndim < input_embed.ndim:
            time_embed = time_embed[..., None, :]

        context = (
            self.context_embed(context) if self.context_embed is not None else None
        )
        context = context[..., None, :]
        print(time_embed.shape, context.shape)
        context = jnp.concatenate([time_embed, context], axis=-1)
        output = self.transformer(input_embed, context=context, mask=attention_mask)
        output = self.tokenizer.decode(output, node_ids)
        return output


class ModelIdentificationDecoder(nnx.Module, experimental_pytree=True):
    model_dim: int = 50
    num_heads: int = 4
    num_layers: int = 2
    widening_factor: int = 2
    attn_size: int = 10

    def __init__(
        self,
        model_dim,
        context_dim,
        rngs,
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

        self.embed = nnx.Embed(2, model_dim, rngs=rngs)
        self.transformer = Transformer(
            model_dim,
            self.num_heads,
            self.num_heads,
            self.attn_size,
            context_dim=model_dim,
            widening_factor=self.widening_factor,
            rngs=rngs,
            dropout_rate=dropout_rate,
        )
        self.output = nnx.Linear(model_dim, 1, rngs=rngs)
        if context_net is None:
            self.context_embedding = MLP(
                [context_dim, 2 * model_dim, model_dim], rngs=rngs
            )
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

    def sample(self, key, context, dim):
        return naive_autoregressive_decoding(self, key, context, dim)


@partial(jax.jit, static_argnums=(3,))
def naive_autoregressive_decoding(model, key, context, dim):
    x = jnp.zeros((dim,), dtype=jnp.bool_)

    def scan_fn(carry, k):
        x, i = carry
        logits = model(x.astype(jnp.int32), context)
        p_i = jax.nn.sigmoid(logits[i])

        x_i = jax.random.bernoulli(k, p_i)
        x = x.at[i].set(x_i)
        return (x, i + 1), None

    keys = jax.random.split(key, (dim,))
    x, _ = jax.lax.scan(scan_fn, (x, 0), keys)

    return x[0]
