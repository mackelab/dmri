from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn


def soft_squash(signal: Tensor, tau: float = 0.5) -> Tensor:
    return torch.sigmoid((signal - 0.5) / tau)


class GaussianFourierFeatures(nn.Module):
    """Torch equivalent of a Gaussian Fourier scalar embedding."""

    def __init__(self, out_dim: int, scale: float = 1.0) -> None:
        super().__init__()
        if out_dim <= 0:
            raise ValueError("out_dim must be positive.")
        self.out_dim = int(out_dim)
        self.num_frequencies = max(1, math.ceil(out_dim / 2))
        frequencies = torch.randn(self.num_frequencies) * float(scale)
        self.register_buffer("frequencies", frequencies)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != 1:
            raise ValueError("GaussianFourierFeatures expects a trailing singleton dimension.")
        phases = 2.0 * math.pi * x * self.frequencies
        embedding = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)
        return embedding[..., : self.out_dim]


class RepeatScalarEmbedding(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        if out_dim <= 0:
            raise ValueError("out_dim must be positive.")
        self.out_dim = int(out_dim)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != 1:
            raise ValueError("RepeatScalarEmbedding expects a trailing singleton dimension.")
        return x.repeat_interleave(self.out_dim, dim=-1)


@dataclass
class SBIEmbeddingNetConfig:
    model_dim: int = 128
    output_dim: int = 128
    num_cls_tokens: int = 2
    num_layers: int = 4
    num_heads: int = 8
    widening_factor: int = 4
    dropout_rate: float = 0.0
    bvals_embed_dim: int = 16
    signals_embed_dim: int = 16
    bvec_repeats: int = 4
    min_bval: float = 0.0
    max_bval: float = 4000.0
    min_signal: float = 0.0
    max_signal: float = 1.0
    soft_squash_signals: bool = True
    log_transform_signals: bool = False
    embed_bvals: str = "fourier"
    embed_signals: str = "repeat"
    fourier_scale: float = 1.0


class SBIEmbeddingNet(nn.Module):
    """Pure-Torch dMRI encoder with multiple learnable CLS summary tokens."""

    def __init__(
        self,
        model_dim: int = 128,
        output_dim: int = 128,
        num_cls_tokens: int = 2,
        num_layers: int = 4,
        num_heads: int = 8,
        widening_factor: int = 8,
        dropout_rate: float = 0.0,
        bvals_embed_dim: int = 16,
        signals_embed_dim: int = 16,
        bvec_repeats: int = 4,
        min_bval: float = 0.0,
        max_bval: float = 4000.0,
        min_signal: float = 0.0,
        max_signal: float = 1.0,
        soft_squash_signals: bool = True,
        log_transform_signals: bool = False,
        embed_bvals: str = "fourier",
        embed_signals: str = "repeat",
        fourier_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if model_dim <= 0:
            raise ValueError("model_dim must be positive.")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        if num_cls_tokens <= 0:
            raise ValueError("num_cls_tokens must be positive.")
        if bvec_repeats <= 0:
            raise ValueError("bvec_repeats must be positive.")
        if max_bval <= min_bval:
            raise ValueError("max_bval must be greater than min_bval.")
        if max_signal <= min_signal:
            raise ValueError("max_signal must be greater than min_signal.")

        self.model_dim = int(model_dim)
        self.output_dim = int(output_dim)
        self.num_cls_tokens = int(num_cls_tokens)
        self.bvec_repeats = int(bvec_repeats)
        self.min_bval = float(min_bval)
        self.max_bval = float(max_bval)
        self.min_signal = float(min_signal)
        self.max_signal = float(max_signal)
        self.soft_squash_signals = bool(soft_squash_signals)
        self.log_transform_signals = bool(log_transform_signals)

        if embed_bvals == "fourier":
            self.embed_bvals = GaussianFourierFeatures(
                bvals_embed_dim,
                scale=fourier_scale,
            )
        elif embed_bvals == "linear":
            self.embed_bvals = nn.Linear(1, bvals_embed_dim)
        else:
            raise ValueError(f"Unsupported b-value embedding: {embed_bvals}.")

        if embed_signals == "repeat":
            self.embed_signals = RepeatScalarEmbedding(signals_embed_dim)
        elif embed_signals == "fourier":
            self.embed_signals = GaussianFourierFeatures(
                signals_embed_dim,
                scale=fourier_scale,
            )
        elif embed_signals == "linear":
            self.embed_signals = nn.Linear(1, signals_embed_dim)
        else:
            raise ValueError(f"Unsupported signal embedding: {embed_signals}.")

        token_dim = bvals_embed_dim + signals_embed_dim + 3 * self.bvec_repeats
        self.input_projection = nn.Linear(token_dim, self.model_dim)
        self.cls_tokens = nn.Parameter(
            torch.zeros(1, self.num_cls_tokens, self.model_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=num_heads,
            dim_feedforward=self.model_dim * widening_factor,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.output_norm = nn.LayerNorm(self.model_dim)
        self.output_projection = nn.Linear(
            self.model_dim * self.num_cls_tokens,
            self.output_dim,
        )

        nn.init.normal_(self.cls_tokens, std=0.02)

    @classmethod
    def from_config(cls, config: SBIEmbeddingNetConfig) -> "SBIEmbeddingNet":
        return cls(**config.__dict__)

    def transform_bvals(self, bvals: Tensor) -> Tensor:
        bvals = torch.clamp(bvals, min=self.min_bval, max=self.max_bval)
        bvals = (bvals - self.min_bval) / (self.max_bval - self.min_bval)
        return bvals * 2.0 - 1.0

    def transform_signals(self, signals: Tensor) -> Tensor:
        normalized = (signals - self.min_signal) / (self.max_signal - self.min_signal)
        if self.soft_squash_signals:
            normalized = soft_squash(normalized)
        transformed = self.min_signal + normalized * (self.max_signal - self.min_signal)
        if self.log_transform_signals:
            transformed = torch.log(torch.clamp(transformed, min=1e-8))
        return transformed

    def _extract_acq(self, acq: Any) -> tuple[Any, Any]:
        if isinstance(acq, Mapping):
            if "bvals" not in acq or "bvecs" not in acq:
                raise KeyError("acq mapping must contain 'bvals' and 'bvecs'.")
            return acq["bvals"], acq["bvecs"]
        if not hasattr(acq, "bvals") or not hasattr(acq, "bvecs"):
            raise TypeError("acq must be a mapping or an object with bvals/bvecs.")
        return acq.bvals, acq.bvecs

    def _prepare_inputs(
        self,
        x: Tensor | Mapping[str, Any],
        acq: Any | None,
    ) -> tuple[Tensor, Tensor, Tensor, bool]:
        if isinstance(x, Mapping):
            batch = x
            if acq is None:
                acq = batch.get("acq")
            if "x" in batch:
                signals = batch["x"]
            elif "signals" in batch:
                signals = batch["signals"]
            else:
                raise KeyError("Input mapping must contain 'x' or 'signals'.")
        else:
            signals = x

        if acq is None:
            raise ValueError("acq must be provided when forward is not called with a batch mapping.")

        bvals_raw, bvecs_raw = self._extract_acq(acq)
        device = signals.device if isinstance(signals, Tensor) else None
        signals = torch.as_tensor(signals, dtype=torch.float32, device=device)
        bvals = torch.as_tensor(bvals_raw, dtype=torch.float32, device=signals.device)
        bvecs = torch.as_tensor(bvecs_raw, dtype=torch.float32, device=signals.device)

        squeezed = False
        if signals.ndim == 1:
            signals = signals.unsqueeze(0)
            squeezed = True
        elif signals.ndim == 3 and signals.shape[-1] == 1:
            signals = signals.squeeze(-1)

        if bvals.ndim == 1:
            bvals = bvals.unsqueeze(0)
        if bvecs.ndim == 2:
            bvecs = bvecs.unsqueeze(0)

        if signals.ndim != 2:
            raise ValueError("signals must have shape [seq] or [batch, seq].")
        if bvals.shape != signals.shape:
            raise ValueError("bvals and signals must have the same shape.")
        if bvecs.shape[:-1] != signals.shape or bvecs.shape[-1] != 3:
            raise ValueError("bvecs must have shape [seq, 3] or [batch, seq, 3].")

        return signals, bvals, bvecs, squeezed

    def _tokenize(self, signals: Tensor, bvals: Tensor, bvecs: Tensor) -> Tensor:
        bvals_features = self.embed_bvals(self.transform_bvals(bvals).unsqueeze(-1))
        signal_features = self.embed_signals(
            self.transform_signals(signals).unsqueeze(-1)
        )
        bvec_features = bvecs.repeat_interleave(self.bvec_repeats, dim=-1)
        token_features = torch.cat(
            [bvals_features, signal_features, bvec_features],
            dim=-1,
        )
        return self.input_projection(token_features)

    def forward(
        self,
        x: Tensor | Mapping[str, Any],
        acq: Any | None = None,
        *,
        return_tokens: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        signals, bvals, bvecs, squeezed = self._prepare_inputs(x, acq)
        tokens = self._tokenize(signals, bvals, bvecs)

        cls_tokens = self.cls_tokens.expand(tokens.shape[0], -1, -1)
        encoded = self.transformer(torch.cat([cls_tokens, tokens], dim=1))

        cls_summary = self.output_norm(encoded[:, : self.num_cls_tokens])
        summary = self.output_projection(
            cls_summary.reshape(cls_summary.shape[0], -1)
        )
        token_outputs = encoded[:, self.num_cls_tokens :]

        if squeezed:
            summary = summary[0]
            token_outputs = token_outputs[0]

        if return_tokens:
            return summary, token_outputs
        return summary
