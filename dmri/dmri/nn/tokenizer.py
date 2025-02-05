from functools import partial
from typing import List, Optional
import jax
import jax.numpy as jnp
import numpy as np

from flax import nnx

from probjax.nn import GaussianFourierEmbedding, Transformer
from probjax.nn.loss_fn.denoising import build_time_dependent_denoising_loss
from probjax.nn.nets.denoising_diffusion_model import EDM
from probjax.nn import MLP
from probjax.nn.utils import AffineFuse


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


class StructuredTokenizer(nnx.Module, experimental_pytree=True):
    def __init__(
        self,
        dims_by_id: List[int],
        id_dim: int = 32,
        value_dim: int = 32,
        cond_dim: int = 0,
        encode_nets: Optional[List[nnx.Module]] = None,
        decode_nets: Optional[List[nnx.Module]] = None,
        rngs=None,
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
