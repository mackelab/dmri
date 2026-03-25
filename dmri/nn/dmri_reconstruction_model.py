import copy
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Optional, cast
import jax
import jax.numpy as jnp
from flax import nnx
from probjax.utils.typing import Array, ArrayLike, DTypeLike, PrecisionLike, RngKey

from dmri.simulators import MultiCompartment
from dmri.simulators.acquisition_scheme import (
    acquisition_scheme,
    ssfp_acquisition_scheme,
)

from .autoregressive import (
    BinaryAutoregressiveDecoder,
    DMRIModelSelectionAmortizedPriorConfig,
    DMRIModelSelectionConfig,
)
from .embedding_net import (
    BvalBvecSignalEmbeddingNet,
    DMRIEmbeddingConfig,
    GroupedBvalBvecSignalEmbeddingNet,
    SSFPEmbeddingNet,
    SSFPEmbeddingNetConfig,
)
from .simformer import DMRIThetaInferenceConfig, EDMSimformer
from .tokenizer import DMRITokenizer, DMRITokenizerPP, DMRITokenizerPPP

EmbeddingModule = (
    BvalBvecSignalEmbeddingNet | GroupedBvalBvecSignalEmbeddingNet | SSFPEmbeddingNet
)
TokenizerType = type[DMRITokenizer]
AcquisitionSchemeLike = acquisition_scheme | ssfp_acquisition_scheme


class EvidenceDiagnostics(NamedTuple):
    ess: Array
    max_weight_share: Array
    num_finite: Array


def _logmeanexp_and_delta_se(
    log_w: Array,
) -> tuple[Array, Array, EvidenceDiagnostics]:
    """Return log-mean-exp, delta-method SE, and basic diagnostics."""
    finite = jnp.isfinite(log_w)
    lw = jnp.where(finite, log_w, -jnp.inf)

    lw_max = jnp.max(lw)
    all_invalid = ~jnp.isfinite(lw_max)

    shifted = lw - lw_max
    w = jnp.exp(shifted)

    n = lw.shape[0]
    n_f = jnp.sum(finite).astype(lw.dtype)

    mean_w = jnp.mean(w)
    mean_w2 = jnp.mean(w * w)
    var_w = jnp.maximum(mean_w2 - mean_w * mean_w, 0.0)

    se_log = jnp.sqrt(var_w / jnp.asarray(n, lw.dtype)) / jnp.maximum(mean_w, 1e-30)

    log_mean = lw_max + jnp.log(jnp.maximum(mean_w, 1e-300))

    sum_w = jnp.sum(w)
    ess = (sum_w * sum_w) / jnp.maximum(jnp.sum(w * w), 1e-30)
    max_share = jnp.max(w) / jnp.maximum(sum_w, 1e-30)

    log_mean = jnp.where(all_invalid, -jnp.inf, log_mean)
    se_log = jnp.where(all_invalid, jnp.inf, se_log)
    diag = EvidenceDiagnostics(
        ess=jnp.where(all_invalid, 0.0, ess),
        max_weight_share=jnp.where(all_invalid, 1.0, max_share),
        num_finite=n_f,
    )
    return log_mean, se_log, diag


def _update_logw_moments(
    state: tuple[Array, Array, Array, Array],
    log_w: Array,
) -> tuple[Array, Array, Array, Array]:
    """Update streaming log-weight moments for log-mean-exp + SE."""
    m, s1, s2, num_finite = state
    finite = jnp.isfinite(log_w)
    any_finite = jnp.any(finite)
    lw = jnp.where(finite, log_w, -jnp.inf)
    batch_max = jnp.max(lw)

    shifted = jnp.where(any_finite, lw - batch_max, 0.0)
    batch_s1 = jnp.where(any_finite, jnp.sum(jnp.exp(shifted)), 0.0)
    batch_s2 = jnp.where(any_finite, jnp.sum(jnp.exp(2.0 * shifted)), 0.0)

    prev_finite = jnp.isfinite(m)
    new_m = jnp.maximum(m, batch_max)
    rescale_old = jnp.where(prev_finite, jnp.exp(m - new_m), 0.0)
    rescale_new = jnp.where(any_finite, jnp.exp(batch_max - new_m), 0.0)
    rescale_old2 = jnp.where(prev_finite, jnp.exp(2.0 * (m - new_m)), 0.0)
    rescale_new2 = jnp.where(any_finite, jnp.exp(2.0 * (batch_max - new_m)), 0.0)

    new_s1 = s1 * rescale_old + batch_s1 * rescale_new
    new_s2 = s2 * rescale_old2 + batch_s2 * rescale_new2
    new_num_finite = num_finite + jnp.sum(finite).astype(num_finite.dtype)
    return new_m, new_s1, new_s2, new_num_finite


def _finalize_logw_moments(
    state: tuple[Array, Array, Array, Array],
    n: int | Array,
) -> tuple[Array, Array, EvidenceDiagnostics]:
    """Finalize log-mean-exp and delta-method SE from streaming moments."""
    m, s1, s2, num_finite = state
    n_arr = jnp.asarray(n, dtype=s1.dtype)
    all_invalid = num_finite <= 0

    mean_w = s1 / jnp.maximum(n_arr, 1.0)
    mean_w2 = s2 / jnp.maximum(n_arr, 1.0)
    var_w = jnp.maximum(mean_w2 - mean_w * mean_w, 0.0)
    se_log = jnp.sqrt(var_w / jnp.maximum(n_arr, 1.0)) / jnp.maximum(mean_w, 1e-30)

    log_mean = m + jnp.log(jnp.maximum(mean_w, 1e-300))
    log_mean = jnp.where(all_invalid, -jnp.inf, log_mean)
    se_log = jnp.where(all_invalid, jnp.inf, se_log)

    ess = (s1 * s1) / jnp.maximum(s2, 1e-30)
    max_share = 1.0 / jnp.maximum(s1, 1e-30)
    diag = EvidenceDiagnostics(
        ess=jnp.where(all_invalid, 0.0, ess),
        max_weight_share=jnp.where(all_invalid, 1.0, max_share),
        num_finite=num_finite,
    )
    return log_mean, se_log, diag


def _logsumexp(x: Array) -> Array:
    m = jnp.max(x)
    return m + jnp.log(jnp.sum(jnp.exp(x - m)))


@dataclass
class DMRIInferenceModelConfig:
    simulator: type[MultiCompartment]
    model_dim: int = 64
    use_attention_mask: bool = True
    inference_loss_type: str = "v"
    dtype: DTypeLike | None = None
    param_dtype: DTypeLike | None = None
    precision: PrecisionLike | None = None
    preferred_element_type: DTypeLike | None = None
    tokenizer_cls: TokenizerType = DMRITokenizer
    embedding_cls: type[EmbeddingModule] = BvalBvecSignalEmbeddingNet
    embedding_cfg: Any = field(default_factory=DMRIEmbeddingConfig)
    model_selection_cfg: Any = field(default_factory=DMRIModelSelectionConfig)
    theta_inference_cfg: DMRIThetaInferenceConfig = field(
        default_factory=DMRIThetaInferenceConfig
    )


@dataclass
class DMRIInferenceModelConfigMaskPriorAmortized(DMRIInferenceModelConfig):
    model_selection_cfg: Any = field(
        default_factory=DMRIModelSelectionAmortizedPriorConfig
    )


@dataclass
class DMRIInferenceModelConfigMaskPriorAmortizedPP(
    DMRIInferenceModelConfigMaskPriorAmortized
):
    tokenizer_cls: TokenizerType = DMRITokenizerPP
    embedding_cls: type[EmbeddingModule] = BvalBvecSignalEmbeddingNet
    embedding_cfg: Any = field(default_factory=DMRIEmbeddingConfig)

@dataclass
class DMRIInferenceModelConfigMaskPriorAmortizedPPP(
    DMRIInferenceModelConfigMaskPriorAmortized
):
    tokenizer_cls: TokenizerType = DMRITokenizerPPP
    embedding_cls: type[EmbeddingModule] = BvalBvecSignalEmbeddingNet
    embedding_cfg: Any = field(default_factory=DMRIEmbeddingConfig)


@dataclass
class SSFPInferenceModelConfig(DMRIInferenceModelConfig):
    tokenizer_cls: TokenizerType = DMRITokenizerPP
    embedding_cls: type[EmbeddingModule] = SSFPEmbeddingNet
    embedding_cfg: Any = field(default_factory=SSFPEmbeddingNetConfig)


class DMRIInferenceModel(nnx.Module):
    def __init__(self, cfg: DMRIInferenceModelConfig, rngs: nnx.Rngs) -> None:
        self.cfg: DMRIInferenceModelConfig = cfg
        self.precision_fields: tuple[str, ...] = (
            "dtype",
            "param_dtype",
            "precision",
            "preferred_element_type",
        )
        self._precision_defaults: dict[str, Any] = {}
        self._update_precision_defaults()
        self._initialize_modules(rngs)

    def __call__(
        self,
        model_mask: Array,
        theta: Array,
        x: Array,
        acq: AcquisitionSchemeLike,
        mask_prior: Optional[Array] = None,
        alpha_prior: Optional[Array] = None,
        model_idx: list[int] | None = None,
        noise_idx: list[int] | None = None,
        t: ArrayLike | None = None,
    ) -> tuple[Array, Array]:
        # Embed model configuration
        batch_shape = model_mask.shape[:-1]
        tokens_cfg, y_ctx, y_seq, mask_prior = self.embed_inputs(
            model_mask,
            x,
            acq,
            mask_prior=mask_prior,
            alpha_prior=alpha_prior,
            model_idx=model_idx,
            noise_idx=noise_idx,
        )

        # Mode compartment prediction
        model_mask_logits = self.model_decoder(
            model_mask,
            self.tokenizer,
            y=y_seq,
            tokens_cfg=tokens_cfg,
            mask_prior=mask_prior,
            additional_context=y_ctx,
        )

        # Get theta predictions
        if t is None:
            t = jnp.ones(batch_shape + (1,)) * 0.001
        # Mask out non-selected models
        attention_mask = self.marginalization_mask(
            model_mask, model_idx=model_idx, noise_idx=noise_idx
        )
        theta_pred = cast(
            Array,
            self.inference_decoder(
                t,
                theta,
                self.tokenizer,
                y=y_seq,
                tokens_cfg=tokens_cfg,
                attention_mask=attention_mask,
                model_mask=model_mask,
                context=y_ctx,
            ),
        )

        return model_mask_logits, theta_pred

    # ------------------------------------------------------------------
    # Internal helpers

    def _update_precision_defaults(self) -> None:
        self._precision_defaults = {
            key: getattr(self.cfg, key, None)
            for key in self.precision_fields
            if getattr(self.cfg, key, None) is not None
        }

    def _config_kwargs(self, source_cfg: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            key: value
            for key, value in vars(source_cfg).items()
            if value is not None and key not in self.precision_fields
        }
        for key, value in self._precision_defaults.items():
            kwargs.setdefault(key, value)
        for key in self.precision_fields:
            explicit_value = getattr(source_cfg, key, None)
            if explicit_value is not None:
                kwargs[key] = explicit_value
        return kwargs

    def _initialize_modules(self, rngs: nnx.Rngs) -> None:
        cfg = self.cfg

        embedding_kwargs = self._config_kwargs(cfg.embedding_cfg)
        self.encoder = cfg.embedding_cls(
            rngs,
            model_dim=cfg.model_dim,
            **embedding_kwargs,
        )
        reduce_factor = getattr(cfg.embedding_cfg, "reduce_factor", 1) or 1
        default_dim = cfg.model_dim // reduce_factor
        self.y_seq_dim = getattr(self.encoder, "y_seq_dim", default_dim)
        self.use_y_ctx = bool(getattr(self.encoder, "use_global_summary_token", False))
        self.y_glob_dim = getattr(self.encoder, "y_glob_dim", default_dim)
        self.y_ctx_dim = self.y_glob_dim if self.use_y_ctx else 0

        self.tokenizer = cfg.tokenizer_cls(
            cfg.simulator,
            token_dim=cfg.model_dim,
            rngs=rngs,
        )

        selection_cfg = cfg.model_selection_cfg
        selection_kwargs = self._config_kwargs(selection_cfg)
        prior_params_embed_dim = selection_kwargs.get("prior_params_embed_dim", 0)
        self.requires_mask_prior = prior_params_embed_dim > 0
        selection_kwargs["additional_context_dim"] = self.y_ctx_dim
        selection_kwargs["kv_in_features"] = self.y_seq_dim

        self.model_decoder = BinaryAutoregressiveDecoder(
            rngs,
            model_dim=cfg.model_dim,
            **selection_kwargs,
        )
        self.mask_prior_dim = self.model_decoder.mask_prior_dim

        theta_cfg = cfg.theta_inference_cfg
        theta_kwargs = self._config_kwargs(theta_cfg)
        theta_kwargs["additional_context_dim"] = self.y_ctx_dim
        theta_kwargs["kv_in_features"] = self.y_seq_dim
        self.inference_decoder = EDMSimformer(
            rngs=rngs,
            model_dim=cfg.model_dim,
            loss_type=cfg.inference_loss_type,
            **theta_kwargs,
        )

    def set_precision(
        self,
        *,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike | None = None,
        precision: PrecisionLike | None = None,
        preferred_element_type: DTypeLike | None = None,
        use_flash_attention: bool | None = None,
        use_flash_cross_attention: bool | None = None,
    ) -> None:
        """Reinitialize the model with updated precision/attention settings.

        Args:
            dtype: Optional override for computation dtype.
            param_dtype: Optional override for parameter dtype.
            precision: Optional override for XLA matmul precision.
            preferred_element_type: Optional override for matmul element type.
            use_flash_attention: Optional override applied to every sub-config
                that exposes a ``use_flash_attention`` flag.
            use_flash_cross_attention: Optional override applied to every
                sub-config exposing ``use_flash_cross_attention``.

        Note:
            Calling this method discards the current parameter values because
            all modules are rebuilt from scratch.
        """

        updated_cfg = copy.deepcopy(self.cfg)
        precision_updates = {
            "dtype": dtype,
            "param_dtype": param_dtype,
            "precision": precision,
            "preferred_element_type": preferred_element_type,
        }

        nested_cfgs = (
            updated_cfg.embedding_cfg,
            updated_cfg.model_selection_cfg,
            updated_cfg.theta_inference_cfg,
        )

        for key, value in precision_updates.items():
            if value is None:
                continue
            setattr(updated_cfg, key, value)
            for nested in nested_cfgs:
                if hasattr(nested, key):
                    setattr(nested, key, value)

        def _maybe_set_flag(target: Any, attr: str, value: bool | None) -> None:
            if value is not None and hasattr(target, attr):
                setattr(target, attr, value)

        for nested in nested_cfgs:
            _maybe_set_flag(nested, "use_flash_attention", use_flash_attention)
            _maybe_set_flag(
                nested, "use_flash_cross_attention", use_flash_cross_attention
            )

        self.cfg = updated_cfg
        self._update_precision_defaults()
        self._initialize_modules(nnx.Rngs(0))

    def _encode_observations(
        self,
        acq: AcquisitionSchemeLike,
        x: ArrayLike,
        deterministic: bool | None = None,
        decode: bool = False,
    ) -> tuple[Array | None, Array]:
        if isinstance(
            self.encoder,
            (BvalBvecSignalEmbeddingNet, GroupedBvalBvecSignalEmbeddingNet),
        ):
            if not isinstance(acq, acquisition_scheme):
                raise TypeError(
                    "Expected a diffusion acquisition scheme for the diffusion encoder."
                )
            return self.encoder(
                acq=acq, x=x, deterministic=deterministic, decode=decode
            )
        if isinstance(self.encoder, SSFPEmbeddingNet):
            if not isinstance(acq, ssfp_acquisition_scheme):
                raise TypeError(
                    "Expected an SSFP acquisition scheme for the SSFP encoder."
                )
            return self.encoder(acq, x, deterministic=deterministic, decode=decode)
        raise TypeError(f"Unsupported encoder type: {type(self.encoder)!r}")

    def embed_inputs(
        self,
        model_mask: Array,
        x: Array,
        acq: AcquisitionSchemeLike,
        mask_prior: Optional[Array] = None,
        alpha_prior: Optional[Array] = None,
        model_idx: Optional[list[int]] = None,
        noise_idx: Optional[list[int]] = None,
    ) -> tuple[Array, Optional[Array], Array, Optional[Array]]:
        # Embed model configuration
        tokens_cfg = self.tokenizer.embed_cfgs(
            model_mask,
            alpha_prior,
            model_idx=model_idx,
            noise_idx=noise_idx,
        )
        # Embed observations and acquisition parameters
        y_ctx, y = self._encode_observations(acq, x)
        return tokens_cfg, y_ctx, y, mask_prior

    def theta_mask(
        self,
        model_mask: Array,
        model_idx: Optional[list[int]] = None,
        noise_idx: Optional[list[int]] = None,
    ) -> Array:
        theta_token_mask = self.tokenizer.theta_token_mask(
            model_mask, model_idx=model_idx, noise_idx=noise_idx
        )
        return theta_token_mask

    def marginalization_mask(
        self,
        model_mask: Array,
        model_idx: Optional[list[int]] = None,
        noise_idx: Optional[list[int]] = None,
    ) -> Array:
        with jax.ensure_compile_time_eval():
            _model_mask_extended = self.tokenizer.theta_token_mask(
                model_mask, model_idx=model_idx, noise_idx=noise_idx
            )
            attention_mask = (
                _model_mask_extended[..., None, :] & _model_mask_extended[..., :, None]
            )
            attention_mask = attention_mask | jnp.eye(
                _model_mask_extended.shape[-1], dtype=bool
            )
            return attention_mask

    def loss_fn(
        self,
        rng: RngKey,
        model_mask: Array,
        theta: Array,
        x: Array,
        acq: AcquisitionSchemeLike,
        mask_prior: Array | None = None,
        alpha_prior: Array | None = None,
        target_score: Array | None = None,
        model_idx: Optional[list[int]] = None,
        noise_idx: Optional[list[int]] = None,
        permute_order: bool = False,
        use_loss_mask: bool = False,
        weight_by_complexity: bool = False,
        label_smoothing: float = 0.0,
        cut_off_tsm: float = 0.1,
    ) -> Array:
        # Embed model configuration
        tokens_cfg = self.tokenizer.embed_cfgs(
            model_mask, alpha_prior, model_idx=model_idx, noise_idx=noise_idx
        )

        y_ctx, y = self._encode_observations(acq, x)

        if permute_order:
            rng, permute_rng = jax.random.split(rng)
        else:
            permute_rng = rng
        model_mask_loss = self.model_decoder.loss_fn(
            model_mask,
            self.tokenizer,
            y=y,
            tokens_cfg=tokens_cfg,
            permute_order=permute_order,
            rng=permute_rng,
            mask_prior=mask_prior,
            additional_context=y_ctx,
            label_smoothing=label_smoothing,
        )
        if self.cfg.use_attention_mask:
            attention_mask = self.marginalization_mask(model_mask)
        else:
            attention_mask = None

        if use_loss_mask:
            loss_mask = jax.vmap(self.tokenizer.simulator.theta_mask)(model_mask)
        else:
            loss_mask = None

        theta_loss = self.inference_decoder.loss(
            rng,
            theta,
            self.tokenizer,
            y=y,
            model_mask=model_mask,
            tokens_cfg=tokens_cfg,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            target_score=target_score,
            weight_by_complexity=weight_by_complexity,
            cut_off_tsm=cut_off_tsm,
            context=y_ctx,
        )
        theta_loss /= jnp.sqrt(theta.shape[-1])

        return jnp.concatenate([model_mask_loss[None], theta_loss[None]])

    def sample_mask(
        self,
        rng: RngKey,
        acq: AcquisitionSchemeLike,
        x: Array,
        mask_prior: Array | None = None,
        temperature: float = 1.0,
    ) -> Array:
        # Update for different model configs
        mask_prior = jnp.asarray(mask_prior) if mask_prior is not None else None
        y_ctx, y = self._encode_observations(acq, x)
        model_mask = self.model_decoder.sample(
            rng,
            tokenizer=self.tokenizer,
            y=y,
            dim=self.tokenizer.num_models + self.tokenizer.num_noises,
            mask_prior=mask_prior,
            additional_context=y_ctx,
            temperature=temperature,
        )
        return model_mask

    def log_prob_mask(
        self,
        model_mask: Array,
        acq: AcquisitionSchemeLike,
        x: Array,
        mask_prior: Array | None = None,
    ) -> Array:
        mask_prior_arr = jnp.asarray(mask_prior) if mask_prior is not None else None
        y_ctx, y = self._encode_observations(acq, x)
        log_prob = self.model_decoder.log_prob(
            model_mask,
            self.tokenizer,
            y=y,
            mask_prior=mask_prior_arr,
            additional_context=y_ctx,
        )
        return log_prob

    def sample_and_log_prob_mask(
        self,
        rng: RngKey,
        acq: AcquisitionSchemeLike,
        x: Array,
        mask_prior: Array | None = None,
        temperature: float = 1.0,
    ) -> tuple[Array, Array]:
        mask_prior_arr = jnp.asarray(mask_prior) if mask_prior is not None else None
        y_ctx, y = self._encode_observations(acq, x)
        model_mask, log_prob = self.model_decoder.sample_and_log_prob(
            rng,
            tokenizer=self.tokenizer,
            y=y,
            dim=self.tokenizer.num_models + self.tokenizer.num_noises,
            mask_prior=mask_prior_arr,
            additional_context=y_ctx,
            temperature=temperature,
        )
        return model_mask, log_prob

    def sample_theta(
        self,
        rng: RngKey,
        acq: AcquisitionSchemeLike,
        x: Array,
        model_mask: Array,
        sample_method: str = "ode",
        num_steps: int = 64,
        last_euler_step: bool = True,
        t_min: float | None = None,
        t_max: float | None = None,
        temperature: float = 1.0,
    ) -> Array:
        y_ctx, y = self._encode_observations(acq, x)
        tokens_cfg = self.tokenizer.embed_cfgs(model_mask)

        attention_mask = self.marginalization_mask(model_mask)

        # diffusion sampling
        theta = self.inference_decoder.sample(
            rng,
            y=y,
            tokenizer=self.tokenizer,
            dim=self.cfg.simulator.theta_dim,
            tokens_cfg=tokens_cfg,
            attention_mask=attention_mask,
            model_mask=model_mask,
            sample_method=sample_method,
            context=y_ctx,
            t_max=t_max,
            t_min=t_min,
            num_steps=num_steps,
            last_euler_step=last_euler_step,
            temperature=temperature,
        )

        return theta

    def log_prob_theta(
        self,
        theta: Array,
        acq: AcquisitionSchemeLike,
        x: Array,
        model_mask: Array,
        num_steps: int = 64,
        t_min: float | None = None,
        t_max: float | None = None,
    ) -> Array:
        y_ctx, y = self._encode_observations(acq, x)
        tokens_cfg = self.tokenizer.embed_cfgs(model_mask)

        attention_mask = self.marginalization_mask(model_mask)

        log_prob = self.inference_decoder.log_prob(
            theta,
            y=y,
            tokenizer=self.tokenizer,
            tokens_cfg=tokens_cfg,
            attention_mask=attention_mask,
            model_mask=model_mask,
            context=y_ctx,
            num_steps=num_steps,
            t_max=t_max,
            t_min=t_min,
        )

        return log_prob

    def sample_and_log_prob_theta(
        self,
        rng: RngKey,
        acq: AcquisitionSchemeLike,
        x: Array,
        model_mask: Array,
        sample_method: str = "ode",
        num_steps: int = 64,
        last_euler_step: bool = True,
        t_min: float | None = None,
        t_max: float | None = None,
    ) -> tuple[Array, Array]:
        y_ctx, y = self._encode_observations(acq, x)
        tokens_cfg = self.tokenizer.embed_cfgs(model_mask)

        attention_mask = self.marginalization_mask(model_mask)
        theta, log_prob = self.inference_decoder.sample_and_log_prob(
            rng,
            y=y,
            tokenizer=self.tokenizer,
            dim=self.cfg.simulator.theta_dim,
            tokens_cfg=tokens_cfg,
            attention_mask=attention_mask,
            model_mask=model_mask,
            sample_method=sample_method,
            context=y_ctx,
            t_max=t_max,
            t_min=t_min,
            num_steps=num_steps,
            last_euler_step=last_euler_step,
        )

        return theta, log_prob

    def estimate_evidence(
        self,
        rng: RngKey,
        acq: AcquisitionSchemeLike,
        x: Array,
        model_mask: Array,
        num_samples: int = 128,
        num_steps: int = 128,
        last_euler_step: bool = True,
        t_min: float | None = None,
        t_max: float | None = None,
        estimator: str = "importance",
        defensive_eps: float = 0.0,
        bridge_iters: int = 30,
        bridge_alpha: float = 0.5,
        batch_size: int = 1024,
    ) -> tuple[Array, Array]:
        """Return (log_evidence, std_log_evidence) for the chosen estimator.

        Estimators:
          - "importance": defensive importance sampling using q (and optional prior mix)
          - "bridge": Meng–Wong bridge sampling using q-samples

        batch_size limits memory by chunking importance sampling.
        """
        y_ctx, y = self._encode_observations(acq, x)
        tokens_cfg = self.tokenizer.embed_cfgs(model_mask)
        attention_mask = self.marginalization_mask(model_mask)
        theta_mask = self.cfg.simulator.theta_mask(model_mask)

        num_samples_int = int(num_samples)
        if num_samples_int <= 0:
            raise ValueError("num_samples must be positive.")
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        batch_size = min(batch_size, num_samples_int)

        rngs = jax.random.split(rng, num_samples_int)
        log_prior_const = -0.5 * jnp.log(2.0 * jnp.pi)

        def sample_q_once(key: RngKey) -> tuple[Array, Array]:
            return self.inference_decoder.sample_and_log_prob(
                key,
                y=y,
                tokenizer=self.tokenizer,
                dim=self.cfg.simulator.theta_dim,
                tokens_cfg=tokens_cfg,
                attention_mask=attention_mask,
                model_mask=model_mask,
                context=y_ctx,
                t_max=t_max,
                t_min=t_min,
                num_steps=num_steps,
                last_euler_step=last_euler_step,
            )

        def log_likelihood(theta: Array) -> Array:
            simulator = self.cfg.simulator.from_theta(theta, model_mask=model_mask)
            ll = simulator.log_likelihood(acq, x)
            ll = jnp.sum(ll)
            return jnp.where(jnp.isfinite(ll), ll, -jnp.inf)

        def log_prior(theta: Array) -> Array:
            logp = -0.5 * jnp.square(theta) + log_prior_const
            logp = logp * theta_mask.astype(theta.dtype)
            return jnp.sum(logp, axis=-1)

        v_log_likelihood = jax.vmap(log_likelihood)
        v_log_prior = jax.vmap(log_prior)

        estimator_key = estimator.lower()
        if estimator_key in (
            "harmonic_mean",
            "hm",
            "generalized_harmonic_mean",
            "ghm",
        ):
            estimator_key = "bridge"
        if estimator_key in ("importance", "importance_sampling", "is"):
            if defensive_eps > 0.0 and not hasattr(self.inference_decoder, "log_prob"):
                raise ValueError(
                    "defensive_eps>0 requires inference_decoder.log_prob(theta, ...)"
                )

            if defensive_eps <= 0.0:
                def log_w_batch(keys_batch: Array) -> Array:
                    thetas, log_q = jax.vmap(sample_q_once)(keys_batch)
                    ll = v_log_likelihood(thetas)
                    lp = v_log_prior(thetas)
                    return ll + lp - log_q

                state = (
                    jnp.asarray(-jnp.inf, dtype=y.dtype),
                    jnp.asarray(0.0, dtype=y.dtype),
                    jnp.asarray(0.0, dtype=y.dtype),
                    jnp.asarray(0.0, dtype=y.dtype),
                )
                for start in range(0, num_samples_int, batch_size):
                    log_w = log_w_batch(rngs[start : start + batch_size])
                    state = _update_logw_moments(state, log_w)
            else:
                eps = defensive_eps

                def sample_once(key: RngKey) -> tuple[Array, Array]:
                    key_u, key_p, key_q = jax.random.split(key, 3)
                    u = jax.random.uniform(key_u, ())

                    def sample_from_prior() -> tuple[Array, Array]:
                        theta = jax.random.normal(
                            key_p, (self.cfg.simulator.theta_dim,), dtype=y.dtype
                        )
                        theta = theta * theta_mask.astype(theta.dtype)
                        log_q = self.inference_decoder.log_prob(
                            theta,
                            y=y,
                            tokenizer=self.tokenizer,
                            tokens_cfg=tokens_cfg,
                            attention_mask=attention_mask,
                            model_mask=model_mask,
                            context=y_ctx,
                            t_max=t_max,
                            t_min=t_min,
                            num_steps=num_steps,
                        )
                        return theta, log_q

                    def sample_from_q() -> tuple[Array, Array]:
                        return sample_q_once(key_q)

                    return jax.lax.cond(u < eps, sample_from_prior, sample_from_q)

                def log_w_batch(keys_batch: Array) -> Array:
                    thetas, log_q = jax.vmap(sample_once)(keys_batch)
                    ll = v_log_likelihood(thetas)
                    lp = v_log_prior(thetas)
                    log_f = ll + lp
                    log_qdef = jnp.logaddexp(
                        jnp.log1p(-eps) + log_q,
                        jnp.log(eps) + lp,
                    )
                    return log_f - log_qdef

                state = (
                    jnp.asarray(-jnp.inf, dtype=y.dtype),
                    jnp.asarray(0.0, dtype=y.dtype),
                    jnp.asarray(0.0, dtype=y.dtype),
                    jnp.asarray(0.0, dtype=y.dtype),
                )
                for start in range(0, num_samples_int, batch_size):
                    log_w = log_w_batch(rngs[start : start + batch_size])
                    state = _update_logw_moments(state, log_w)

            logZ, se_logZ, _ = _finalize_logw_moments(state, num_samples_int)
            return logZ, se_logZ
        if estimator_key in ("bridge", "bridge_sampling", "bs"):
            thetas, log_q = jax.lax.map(sample_q_once, rngs, batch_size=8192)
            ll = v_log_likelihood(thetas)
            lp = v_log_prior(thetas)
            log_f = ll + lp

            logZ, _, _ = _logmeanexp_and_delta_se(log_f - log_q)
            alpha = bridge_alpha

            def one_iter(logZ_curr: Array) -> Array:
                log_denom = jnp.logaddexp(
                    jnp.log(alpha) + log_f,
                    jnp.log1p(-alpha) + logZ_curr + log_q,
                )
                log_num_terms = log_f - log_denom
                num = jnp.asarray(log_num_terms.shape[0], dtype=log_num_terms.dtype)
                log_num = _logsumexp(log_num_terms) - jnp.log(num)
                log_den_terms = log_q - log_denom
                den = jnp.asarray(log_den_terms.shape[0], dtype=log_den_terms.dtype)
                log_den = _logsumexp(log_den_terms) - jnp.log(den)
                return log_num - log_den

            def body(_, logZ_curr: Array) -> Array:
                return one_iter(logZ_curr)

            logZ = jax.lax.fori_loop(0, bridge_iters, body, logZ)

            log_denom = jnp.logaddexp(
                jnp.log(alpha) + log_f,
                jnp.log1p(-alpha) + logZ + log_q,
            )
            log_a = log_f - log_denom
            _, se_logZ, _ = _logmeanexp_and_delta_se(log_a)
            return logZ, se_logZ
        raise ValueError(f"Unknown evidence estimator: {estimator}")

    def score_theta(
        self,
        theta: Array,
        acq: AcquisitionSchemeLike,
        x: ArrayLike,
        model_mask: Array,
        t: Optional[ArrayLike] = None,
    ) -> Array:
        if t is None:
            t = jnp.ones((1,)) * 0.01
        else:
            t = jnp.asarray(t)

        y_ctx, y = self._encode_observations(acq, x)
        tokens_cfg = self.tokenizer.embed_cfgs(model_mask)

        attention_mask = self.marginalization_mask(model_mask)

        score = cast(
            Array,
            self.inference_decoder.score(
                t,
                theta,
                y=y,
                tokenizer=self.tokenizer,
                tokens_cfg=tokens_cfg,
                attention_mask=attention_mask,
                model_mask=model_mask,
                context=y_ctx,
            ),
        )

        return score
