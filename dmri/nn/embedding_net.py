from dataclasses import dataclass
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from probjax.nn import GaussianFourierEmbedding, Transformer
from probjax.nn.layers.attention import flex_attention
from probjax.utils.typing import Array, ArrayLike, DTypeLike, PrecisionLike

from dmri.simulators.acquisition_scheme import (
    AcquisitionScheme,
    ssfp_acquisition_scheme,
)


def soft_squash(signal, tau=0.5):
    # signal: arbitrary real
    # tau: temperature controls how sharp the transition is
    return jax.nn.sigmoid((signal - 0.5) / tau)


@dataclass
class DMRIEmbeddingConfig:
    num_layers: int = 3
    num_heads: int = 4
    widening_factor: int = 2
    use_flash_attention: bool = False
    attn_size: int = 16
    dropout_rate: float = 0.0
    bvals_embed_dim: int = 3
    signals_embed_dim: int = 3
    bvec_repeats: int = 1
    y_seq_dim: Optional[int] = None
    y_glob_dim: Optional[int] = None
    min_bval: float = 0.0
    max_bval: float = 4000.0
    min_signal: float = 0.0
    max_signal: float = 1.0
    log_transform_signals: bool = False
    embed_signals: str = "repeat"
    embed_bvals: str = "fourier"
    dtype: DTypeLike | None = None
    param_dtype: DTypeLike | None = None
    precision: PrecisionLike | None = None
    preferred_element_type: DTypeLike | None = None
    use_global_summary_token: bool = False
    global_summary_bins: int = 8
    out_norm: bool = False
    reduce_factor: int = 1


@dataclass
class GroupedDMRIEmbeddingConfig(DMRIEmbeddingConfig):
    group_size: int = 4


class BvalBvecSignalEmbeddingNet(nnx.Module):
    model_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    widening_factor: int = 2
    attn_size: int = 16

    def __init__(
        self,
        rngs: nnx.Rngs,
        model_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        widening_factor: int = 2,
        attn_size: int = 16,
        dropout_rate: float = 0.0,
        bvals_embed_dim: int = 3,
        signals_embed_dim: int = 3,
        bvec_repeats: int = 1,
        y_seq_dim: Optional[int] = None,
        y_glob_dim: Optional[int] = None,
        min_bval: float = 0.0,
        max_bval: float = 4000.0,
        min_signal: float = 0.0,
        max_signal: float = 1.0,
        log_transform_signals: bool = False,
        use_flash_attention: bool = False,
        embed_signals: str = "repeat",
        embed_bvals: str = "fourier",
        use_global_summary_token: bool = False,
        global_summary_bins: int = 8,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike | None = None,
        precision: PrecisionLike | None = None,
        preferred_element_type: DTypeLike | None = None,
        out_norm: bool = False,
        reduce_factor: int = 1,
    ) -> None:
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.widening_factor = widening_factor
        self.attn_size = attn_size
        self.log_transform_signals = log_transform_signals
        self.bvec_repeats = bvec_repeats
        self.min_bval = min_bval
        self.max_bval = max_bval
        self.min_signal = min_signal
        self.max_signal = max_signal
        self.use_global_summary_token = use_global_summary_token
        self.global_summary_bins = global_summary_bins
        self.out_norm = out_norm
        self.reduce_factor = reduce_factor
        self.model_dim = model_dim // reduce_factor

        precision_kwargs = {
            "dtype": dtype,
            "param_dtype": param_dtype,
            "precision": precision,
            "preferred_element_type": preferred_element_type,
        }
        self.initial_layer = nnx.Linear(
            bvals_embed_dim + signals_embed_dim + 3 * bvec_repeats,
            model_dim,
            rngs=rngs,
            **precision_kwargs,
        )
        # Bval Embedding
        if embed_bvals == "fourier":
            self.embed_bvals = GaussianFourierEmbedding(
                1, bvals_embed_dim, rngs=rngs, **precision_kwargs
            )  # type: ignore[assignment]
        elif embed_bvals == "linear":
            self.embed_bvals = nnx.Linear(
                1, bvals_embed_dim, rngs=rngs, **precision_kwargs
            )
        else:
            raise ValueError(f"Invalid embed_bvals: {embed_bvals}")

        # Signal embedding
        if embed_signals == "repeat":
            self.embed_signals = lambda x: jnp.repeat(x, signals_embed_dim, axis=-1)
        elif embed_signals == "fourier":
            self.embed_signals = GaussianFourierEmbedding(  # type: ignore[assignment]
                1, signals_embed_dim, rngs=rngs, **precision_kwargs
            )
        elif embed_signals == "linear":
            self.embed_signals = nnx.Linear(
                1, signals_embed_dim, rngs=rngs, **precision_kwargs
            )
        else:
            raise ValueError(f"Invalid embed_signals: {embed_signals}")

        if use_flash_attention:
            attention_fn = flex_attention
        else:
            attention_fn = None

        self.transformer = Transformer(
            model_dim,
            self.num_heads,
            self.num_layers,
            self.attn_size,
            widening_factor=self.widening_factor,
            rngs=rngs,
            dropout_rate=dropout_rate,
            attention_fn=attention_fn,
            **precision_kwargs,
        )

        if self.global_summary_bins <= 0:
            raise ValueError("global_summary_bins must be a positive integer.")

        self.global_summary_feature_dim = 3 * self.global_summary_bins + 12
        if self.use_global_summary_token:
            self.global_summary_projection = nnx.Linear(
                self.global_summary_feature_dim,
                model_dim,
                rngs=rngs,
                **precision_kwargs,
            )
            self.global_summary_outnorm = nnx.LayerNorm(model_dim, rngs=rngs)
        else:
            self.global_summary_projection = None

        if y_seq_dim is not None:
            self.y_seq_dim = y_seq_dim
            self.output_seq_layer = nnx.Linear(
                model_dim, y_seq_dim, rngs=rngs, **precision_kwargs
            )
            self.out_norm_seq = (
                nnx.LayerNorm(y_seq_dim, rngs=rngs) if out_norm else lambda x: x
            )
        else:
            self.y_seq_dim = model_dim
            self.output_seq_layer = lambda x: x
            self.out_norm_seq = (
                nnx.LayerNorm(model_dim, rngs=rngs) if out_norm else lambda x: x
            )

        if y_glob_dim is not None:
            self.y_glob_dim = y_glob_dim
            self.output_glob_layer = nnx.Linear(
                model_dim, y_glob_dim, rngs=rngs, **precision_kwargs
            )
            self.out_norm_glob = (
                nnx.LayerNorm(y_glob_dim, rngs=rngs) if out_norm else lambda x: x
            )
        else:
            self.y_glob_dim = model_dim
            self.output_glob_layer = lambda x: x
            self.out_norm_glob = (
                nnx.LayerNorm(model_dim, rngs=rngs) if out_norm else lambda x: x
            )

    def transform_bvals(self, bvals: ArrayLike) -> Array:
        if self.max_bval <= self.min_bval:
            raise ValueError(
                "max_bval must be greater than min_bval for normalization."
            )
        bvals = jnp.asarray(bvals)
        bvals = jnp.clip(bvals, a_min=self.min_bval, a_max=self.max_bval)
        bvals = (bvals - self.min_bval) / (self.max_bval - self.min_bval)
        return bvals * 2.0 - 1.0

    def transform_signals(self, signals: ArrayLike) -> Array:
        if self.max_signal <= self.min_signal:
            raise ValueError(
                "max_signal must be greater than min_signal for normalization."
            )
        signals = jnp.asarray(signals)
        normalized = (signals - self.min_signal) / (self.max_signal - self.min_signal)
        normalized = soft_squash(normalized)
        transformed = self.min_signal + normalized * (self.max_signal - self.min_signal)
        if self.log_transform_signals:
            transformed = jnp.log(jnp.clip(transformed, a_min=1e-8))
        return transformed

    def global_summary_token(self, bvals, bvecs, signals) -> Array:
        if not self.use_global_summary_token or self.global_summary_projection is None:
            raise ValueError("Global summary token requested but not configured.")

        signals_arr = jnp.asarray(signals)
        bvals_arr = jnp.asarray(bvals)
        bvecs_arr = jnp.asarray(bvecs)
        dtype = signals_arr.dtype

        min_b = jnp.asarray(self.min_bval, dtype=dtype)
        max_b = jnp.asarray(self.max_bval, dtype=dtype)
        max_b = jnp.where(max_b > min_b, max_b, min_b + jnp.asarray(1.0, dtype=dtype))

        bin_width = (max_b - min_b) / self.global_summary_bins
        bin_width = jnp.where(bin_width > 0, bin_width, jnp.asarray(1.0, dtype=dtype))
        clipped_bvals = jnp.clip(bvals_arr, min_b, max_b)
        relative_position = (clipped_bvals - min_b) / bin_width
        bin_idx = jnp.floor(relative_position).astype(jnp.int32)
        bin_idx = jnp.clip(bin_idx, 0, self.global_summary_bins - 1)

        one_hot = jax.nn.one_hot(
            bin_idx, self.global_summary_bins, axis=-1, dtype=dtype
        )
        counts = jnp.sum(one_hot, axis=-2).astype(dtype)
        token_count = signals_arr.shape[-1]
        total = jnp.maximum(
            jnp.asarray(token_count, dtype=dtype), jnp.asarray(1.0, dtype=dtype)
        )
        density = counts / total

        sum_signal = jnp.sum(one_hot * signals_arr[..., None], axis=-2)
        sum_sq_signal = jnp.sum(one_hot * (signals_arr[..., None] ** 2), axis=-2)
        counts_safe = jnp.where(counts > 0, counts, jnp.ones_like(counts))
        mean_signal = sum_signal / counts_safe
        variance_signal = jnp.maximum(sum_sq_signal / counts_safe - mean_signal**2, 0.0)
        mean_signal = jnp.where(counts > 0, mean_signal, jnp.zeros_like(mean_signal))
        variance_signal = jnp.where(
            counts > 0, variance_signal, jnp.zeros_like(variance_signal)
        )

        bvecs_arr = jnp.asarray(bvecs_arr, dtype=dtype)
        bvec_mean = jnp.mean(bvecs_arr, axis=-2)
        bvec_std = jnp.std(bvecs_arr, axis=-2)

        signal_mean = jnp.mean(signals_arr, axis=-1, keepdims=True)
        signal_std = jnp.std(signals_arr, axis=-1, keepdims=True)
        signal_min = jnp.min(signals_arr, axis=-1, keepdims=True)
        signal_max = jnp.max(signals_arr, axis=-1, keepdims=True)

        bval_mean = jnp.mean(bvals_arr, axis=-1, keepdims=True)
        bval_std = jnp.std(bvals_arr, axis=-1, keepdims=True)

        summary_features = jnp.concatenate(
            [
                density,
                mean_signal,
                variance_signal,
                bvec_mean,
                bvec_std,
                signal_mean,
                signal_std,
                signal_min,
                signal_max,
                bval_mean,
                bval_std,
            ],
            axis=-1,
        )

        return self.global_summary_projection(summary_features)

    def __call__(
        self,
        acq: AcquisitionScheme,
        x: ArrayLike,
        deterministic: bool | None = None,
        decode: bool = False,
    ) -> tuple[Array | None, Array]:
        raw_bvals = acq.bvals
        raw_bvecs = acq.bvecs
        summary_token = None
        if self.use_global_summary_token:
            summary_token = self.global_summary_token(raw_bvals, raw_bvecs, x)
        using_summary = summary_token is not None

        bvals = self.transform_bvals(raw_bvals)
        signals = self.transform_signals(x)
        bvals = self.embed_bvals(bvals[..., None])
        signals = self.embed_signals(signals[..., None])
        bvecs = jnp.asarray(raw_bvecs)
        bvecs = jnp.repeat(bvecs, self.bvec_repeats, axis=-1)
        data = jnp.concatenate([bvals, signals, bvecs], axis=-1)
        tokens = self.initial_layer(data)
        if using_summary:
            expanded_summary = summary_token.reshape(
                summary_token.shape[:-1] + (1, summary_token.shape[-1])
            )
            tokens = jnp.concatenate([expanded_summary, tokens], axis=-2)
        out_tokens = self.transformer(
            tokens, deterministic=deterministic, decode=decode
        )
        if using_summary:
            global_summary = out_tokens[..., 0, :]
            sequence_tokens = out_tokens[..., 1:, :]
            global_summary = self.output_glob_layer(
                self.global_summary_outnorm(global_summary)
            )
            global_summary = self.out_norm_glob(global_summary)
            sequence_tokens = self.output_seq_layer(sequence_tokens)
            sequence_tokens = self.out_norm_seq(sequence_tokens)
        else:
            global_summary = None
            sequence_tokens = out_tokens
            sequence_tokens = self.output_seq_layer(sequence_tokens)
            sequence_tokens = self.out_norm_seq(sequence_tokens)
        return global_summary, sequence_tokens


class GroupedBvalBvecSignalEmbeddingNet(BvalBvecSignalEmbeddingNet):
    def __init__(
        self,
        rngs: nnx.Rngs,
        model_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        widening_factor: int = 2,
        attn_size: int = 16,
        dropout_rate: float = 0.0,
        bvals_embed_dim: int = 3,
        signals_embed_dim: int = 3,
        bvec_repeats: int = 1,
        y_seq_dim: Optional[int] = None,
        y_glob_dim: Optional[int] = None,
        min_bval: float = 0.0,
        max_bval: float = 4000.0,
        min_signal: float = 0.0,
        max_signal: float = 1.0,
        log_transform_signals: bool = False,
        use_flash_attention: bool = False,
        embed_signals: str = "repeat",
        embed_bvals: str = "fourier",
        use_global_summary_token: bool = False,
        global_summary_bins: int = 8,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike | None = None,
        precision: PrecisionLike | None = None,
        preferred_element_type: DTypeLike | None = None,
        out_norm: bool = False,
        group_size: int = 6,
    ) -> None:
        super().__init__(
            rngs=rngs,
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            widening_factor=widening_factor,
            attn_size=attn_size,
            dropout_rate=dropout_rate,
            bvals_embed_dim=bvals_embed_dim,
            signals_embed_dim=signals_embed_dim,
            bvec_repeats=bvec_repeats,
            y_seq_dim=y_seq_dim,
            y_glob_dim=y_glob_dim,
            min_bval=min_bval,
            max_bval=max_bval,
            min_signal=min_signal,
            max_signal=max_signal,
            log_transform_signals=log_transform_signals,
            use_flash_attention=use_flash_attention,
            embed_signals=embed_signals,
            embed_bvals=embed_bvals,
            use_global_summary_token=use_global_summary_token,
            global_summary_bins=global_summary_bins,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            preferred_element_type=preferred_element_type,
            out_norm=out_norm,
        )
        if group_size <= 0:
            raise ValueError("group_size must be a positive integer.")
        self.group_size = int(group_size)

    def _sort_measurements(
        self, bvals: ArrayLike, bvecs: ArrayLike, signals: ArrayLike
    ) -> tuple[Array, Array, Array]:
        bvals_arr = jnp.asarray(bvals)
        sort_idx = jnp.argsort(bvals_arr, axis=-1)
        signals_arr = jnp.asarray(signals)
        sorted_bvals = jnp.take_along_axis(bvals_arr, sort_idx, axis=-1)
        sorted_signals = jnp.take_along_axis(signals_arr, sort_idx, axis=-1)
        bvecs_arr = jnp.asarray(bvecs)
        expanded_idx = jnp.expand_dims(sort_idx, axis=-1)
        expanded_idx = jnp.broadcast_to(expanded_idx, bvecs_arr.shape)
        sorted_bvecs = jnp.take_along_axis(bvecs_arr, expanded_idx, axis=-2)
        return sorted_bvals, sorted_bvecs, sorted_signals

    def _group_tokens(self, tokens: Array) -> Array:
        seq_len = tokens.shape[-2]
        if seq_len == 0:
            raise ValueError("Cannot group tokens when no diffusion measurements exist.")
        pad_len = (-seq_len) % self.group_size
        mask = jnp.ones(tokens.shape[:-1], dtype=tokens.dtype)
        if pad_len:
            pad_width = [(0, 0)] * tokens.ndim
            pad_width[-2] = (0, pad_len)
            tokens = jnp.pad(tokens, pad_width)
            mask_pad = [(0, 0)] * mask.ndim
            mask_pad[-1] = (0, pad_len)
            mask = jnp.pad(mask, mask_pad)
        mask = mask[..., None]
        seq_len = tokens.shape[-2]
        num_groups = seq_len // self.group_size
        new_shape = tokens.shape[:-2] + (num_groups, self.group_size, tokens.shape[-1])
        tokens = tokens.reshape(new_shape)
        mask = mask.reshape(mask.shape[:-2] + (num_groups, self.group_size, 1))
        weighted = tokens * mask
        counts = jnp.clip(mask.sum(axis=-2), a_min=1.0)
        return weighted.sum(axis=-2) / counts

    def __call__(
        self,
        acq: AcquisitionScheme,
        x: ArrayLike,
        deterministic: bool | None = None,
        decode: bool = False,
    ) -> tuple[Array | None, Array]:
        raw_bvals = acq.bvals
        raw_bvecs = acq.bvecs
        signals = x
        sorted_bvals, sorted_bvecs, sorted_signals = self._sort_measurements(
            raw_bvals, raw_bvecs, signals
        )

        summary_token = None
        if self.use_global_summary_token:
            summary_token = self.global_summary_token(
                sorted_bvals, sorted_bvecs, sorted_signals
            )
        using_summary = summary_token is not None

        bvals = self.transform_bvals(sorted_bvals)
        signals = self.transform_signals(sorted_signals)
        bvals = self.embed_bvals(bvals[..., None])
        signals = self.embed_signals(signals[..., None])
        bvecs = jnp.asarray(sorted_bvecs)
        bvecs = jnp.repeat(bvecs, self.bvec_repeats, axis=-1)
        data = jnp.concatenate([bvals, signals, bvecs], axis=-1)
        tokens = self.initial_layer(data)
        tokens = self._group_tokens(tokens)
        if using_summary:
            expanded_summary = summary_token.reshape(
                summary_token.shape[:-1] + (1, summary_token.shape[-1])
            )
            tokens = jnp.concatenate([expanded_summary, tokens], axis=-2)
        out_tokens = self.transformer(
            tokens, deterministic=deterministic, decode=decode
        )
        if using_summary:
            global_summary = out_tokens[..., 0, :]
            sequence_tokens = out_tokens[..., 1:, :]
            global_summary = self.output_glob_layer(
                self.global_summary_outnorm(global_summary)
            )
            global_summary = self.out_norm_glob(global_summary)
            sequence_tokens = self.output_seq_layer(sequence_tokens)
            sequence_tokens = self.out_norm_seq(sequence_tokens)
        else:
            global_summary = None
            sequence_tokens = out_tokens
            sequence_tokens = self.output_seq_layer(sequence_tokens)
            sequence_tokens = self.out_norm_seq(sequence_tokens)
        return global_summary, sequence_tokens


@dataclass
class SSFPEmbeddingNetConfig:
    num_layers: int = 3
    num_heads: int = 4
    widening_factor: int = 2
    use_flash_attention: bool = False
    attn_size: int = 16
    dropout_rate: float = 0.0
    use_flash_attention: bool = False
    dtype: DTypeLike | None = None
    param_dtype: DTypeLike | None = None
    precision: PrecisionLike | None = None
    preferred_element_type: DTypeLike | None = None
    use_global_summary_token: bool = False
    global_summary_bins: int = 8
    y_seq_dim: Optional[int] = None
    y_glob_dim: Optional[int] = None


class SSFPEmbeddingNet(nnx.Module):
    model_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    widening_factor: int = 2
    attn_size: int = 16

    def __init__(
        self,
        rngs: nnx.Rngs,
        model_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        widening_factor: int = 2,
        attn_size: int = 16,
        dropout_rate: float = 0.0,
        use_flash_attention: bool = False,
        use_global_summary_token: bool = False,
        global_summary_bins: int = 8,
        y_seq_dim: Optional[int] = None,
        y_glob_dim: Optional[int] = None,
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
        self.use_global_summary_token = use_global_summary_token
        self.global_summary_bins = global_summary_bins

        if self.global_summary_bins <= 0:
            raise ValueError("global_summary_bins must be a positive integer.")

        linear_kwargs: dict[str, Any] = {}
        transformer_kwargs: dict[str, Any] = {}
        for name, value in (
            ("dtype", dtype),
            ("param_dtype", param_dtype),
            ("precision", precision),
            ("preferred_element_type", preferred_element_type),
        ):
            if value is not None:
                linear_kwargs[name] = value
                transformer_kwargs[name] = value

        scalar_embed_dim = self.model_dim // 3
        signal_embed_dim = self.model_dim // 3
        bvec_embed_dim = self.model_dim - scalar_embed_dim - signal_embed_dim
        self.embed_scalars = GaussianFourierEmbedding(7, scalar_embed_dim, rngs=rngs)
        self.embed_signals = GaussianFourierEmbedding(1, signal_embed_dim, rngs=rngs)

        # Repeat bvecs
        def _repeat_bvecs(x: ArrayLike) -> Array:
            arr: Array = jnp.asarray(x)
            repeated = jnp.repeat(arr[..., None], bvec_embed_dim, axis=-1)
            return repeated.reshape(*arr.shape[:-1], -1)[..., :bvec_embed_dim]

        self.embed_bvecs: Callable[[ArrayLike], Array] = _repeat_bvecs

        if use_flash_attention:
            attention_fn = flex_attention
        else:
            attention_fn = None

        self.transformer = Transformer(
            model_dim,
            self.num_heads,
            self.num_layers,
            self.attn_size,
            widening_factor=self.widening_factor,
            rngs=rngs,
            dropout_rate=dropout_rate,
            attention_fn=attention_fn,
            **transformer_kwargs,
        )

        self.global_summary_feature_dim = 3 * self.global_summary_bins + 18
        if self.use_global_summary_token:
            self.global_summary_projection = nnx.Linear(
                self.global_summary_feature_dim,
                model_dim,
                rngs=rngs,
                **linear_kwargs,
            )
        else:
            self.global_summary_projection = None

        if y_seq_dim is not None:
            self.y_seq_dim = y_seq_dim
            self.output_seq_layer = nnx.Linear(
                model_dim, y_seq_dim, rngs=rngs, **linear_kwargs
            )
        else:
            self.y_seq_dim = model_dim
            self.output_seq_layer = lambda x: x

        if y_glob_dim is not None:
            self.y_glob_dim = y_glob_dim
            self.output_glob_layer = nnx.Linear(
                model_dim, y_glob_dim, rngs=rngs, **linear_kwargs
            )
        else:
            self.y_glob_dim = model_dim
            self.output_glob_layer = lambda x: x

    def global_summary_token(
        self, acq: ssfp_acquisition_scheme, signals: ArrayLike
    ) -> Array:
        if not self.use_global_summary_token or self.global_summary_projection is None:
            raise ValueError("Global summary token requested but not configured.")

        signals_arr = jnp.asarray(signals)
        dtype = signals_arr.dtype
        grad = jnp.asarray(acq.diffGradAmps, dtype=dtype)

        min_grad = jnp.min(grad, axis=-1, keepdims=True)
        max_grad = jnp.max(grad, axis=-1, keepdims=True)
        span_grad = jnp.maximum(max_grad - min_grad, jnp.asarray(1e-6, dtype=dtype))
        bin_width = span_grad / jnp.asarray(self.global_summary_bins, dtype=dtype)
        bin_width = jnp.where(bin_width > 0, bin_width, jnp.ones_like(bin_width))
        relative_position = (grad - min_grad) / bin_width
        bin_idx = jnp.floor(relative_position).astype(jnp.int32)
        bin_idx = jnp.clip(bin_idx, 0, self.global_summary_bins - 1)

        one_hot = jax.nn.one_hot(
            bin_idx, self.global_summary_bins, axis=-1, dtype=dtype
        )
        counts = jnp.sum(one_hot, axis=-2).astype(dtype)
        token_count = signals_arr.shape[-1]
        total = jnp.maximum(
            jnp.asarray(token_count, dtype=dtype), jnp.asarray(1.0, dtype=dtype)
        )
        density = counts / total

        sum_signal = jnp.sum(one_hot * signals_arr[..., None], axis=-2)
        sum_sq_signal = jnp.sum(one_hot * (signals_arr[..., None] ** 2), axis=-2)
        counts_safe = jnp.where(counts > 0, counts, jnp.ones_like(counts))
        mean_signal = sum_signal / counts_safe
        variance_signal = jnp.maximum(sum_sq_signal / counts_safe - mean_signal**2, 0.0)
        mean_signal = jnp.where(counts > 0, mean_signal, jnp.zeros_like(mean_signal))
        variance_signal = jnp.where(
            counts > 0, variance_signal, jnp.zeros_like(variance_signal)
        )

        def _mean_std(x: ArrayLike) -> tuple[Array, Array]:
            arr = jnp.asarray(x, dtype=dtype)
            return (
                jnp.mean(arr, axis=-1, keepdims=True),
                jnp.std(arr, axis=-1, keepdims=True),
            )

        stats = []
        for field in (
            acq.T1,
            acq.T2,
            acq.B1,
            acq.diffGradAmps,
            acq.flipAngles,
            acq.TRs,
            acq.diffGradDur,
        ):
            mean_val, std_val = _mean_std(field)
            stats.extend([mean_val, std_val])

        signal_mean = jnp.mean(signals_arr, axis=-1, keepdims=True)
        signal_std = jnp.std(signals_arr, axis=-1, keepdims=True)
        signal_min = jnp.min(signals_arr, axis=-1, keepdims=True)
        signal_max = jnp.max(signals_arr, axis=-1, keepdims=True)

        summary_features = jnp.concatenate(
            [
                density,
                mean_signal,
                variance_signal,
                signal_mean,
                signal_std,
                signal_min,
                signal_max,
                *stats,
            ],
            axis=-1,
        )

        return self.global_summary_projection(summary_features)

    def __call__(
        self,
        acq: ssfp_acquisition_scheme,
        signals: ArrayLike,
        deterministic: bool | None = None,
        decode: bool = False,
    ) -> tuple[Array | None, Array]:
        summary_token = None
        if self.use_global_summary_token:
            summary_token = self.global_summary_token(acq, signals)
        using_summary = summary_token is not None
        signals = jnp.asarray(signals)

        T1 = acq.T1
        T2 = acq.T2
        B1 = acq.B1
        diffGradAmps = acq.diffGradAmps
        flipAngles = acq.flipAngles
        TRs = acq.TRs
        diffGradDur = acq.diffGradDur
        scalar = jnp.stack(
            [T1, T2, B1, diffGradAmps, flipAngles, TRs, diffGradDur], axis=-1
        )
        scalar = self.embed_scalars(scalar)
        signal = self.embed_signals(signals[..., None])
        bvecs = self.embed_bvecs(acq.bvecs)
        tokens = jnp.concatenate([scalar, signal, bvecs], axis=-1)
        if using_summary:
            expanded_summary = summary_token.reshape(
                summary_token.shape[:-1] + (1, summary_token.shape[-1])
            )
            tokens = jnp.concatenate([expanded_summary, tokens], axis=-2)
        out_tokens = self.transformer(
            tokens, deterministic=deterministic, decode=decode
        )
        if using_summary:
            global_summary = out_tokens[..., 0, :]
            sequence_tokens = out_tokens[..., 1:, :]
            global_summary = self.output_glob_layer(global_summary)
            sequence_tokens = self.output_seq_layer(sequence_tokens)
        else:
            global_summary = None
            sequence_tokens = out_tokens
            sequence_tokens = self.output_seq_layer(sequence_tokens)
        return global_summary, sequence_tokens
