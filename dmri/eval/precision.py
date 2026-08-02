"""Evaluation-time numeric precision selection.

The model config already carries ``dtype`` / ``param_dtype`` / ``precision`` /
``preferred_element_type``, which :func:`dmri.train.build_model.build_model`
forwards into every submodule. Because a checkpoint is *rebuilt* from its stored
config before its parameters are restored, choosing a precision at evaluation
time is just a matter of patching that config first — no layer needs to change.

Half precision applies to the network's matmul inputs only. Parameters stay
float32 (they are small; the memory goes into activations), and
``preferred_element_type: float32`` keeps the dot *outputs* in float32, so the
diffusion sampler, the correctors and the export never see half precision.
"""

from omegaconf import OmegaConf, open_dict

FP32 = "fp32"

#: Config overrides applied to ``cfg.model`` for each selectable precision.
#:
#: No explicit ``precision`` dot-algorithm preset: it adds nothing over
#: ``preferred_element_type``, and ``F16_F16_F32`` is rejected by the attention
#: kernels on Ampere ("Unsupported dot precision algorithm").
PRECISION_PRESETS = {
    FP32: {},
    "bf16": {
        "dtype": "bfloat16",
        "param_dtype": "float32",
        "preferred_element_type": "float32",
    },
    "fp16": {
        "dtype": "float16",
        "param_dtype": "float32",
        "preferred_element_type": "float32",
    },
}

PRECISION_CHOICES = tuple(PRECISION_PRESETS)


def normalize_precision(precision):
    """Validate a precision name, mapping None/empty to ``fp32``."""
    if precision in (None, "", "null", "None"):
        return FP32
    name = str(precision).lower()
    aliases = {
        "float32": FP32,
        "f32": FP32,
        "single": FP32,
        "bfloat16": "bf16",
        "float16": "fp16",
        "f16": "fp16",
        "half": "fp16",
    }
    name = aliases.get(name, name)
    if name not in PRECISION_PRESETS:
        raise ValueError(
            f"Unknown precision '{precision}'. Choose one of {', '.join(PRECISION_CHOICES)}."
        )
    return name


def is_half(precision) -> bool:
    return normalize_precision(precision) != FP32


def apply_precision_to_cfg(cfg, precision, logger=None):
    """Patch ``cfg.model`` in place with the preset for ``precision``.

    Returns the normalized name. ``fp32`` is a no-op.
    """
    name = normalize_precision(precision)
    preset = PRECISION_PRESETS[name]
    if not preset:
        return name

    model_cfg = cfg.model if hasattr(cfg, "model") else cfg["model"]
    if OmegaConf.is_config(model_cfg):
        # Stored training configs are struct-mode and may lack these keys.
        with open_dict(model_cfg):
            for key, value in preset.items():
                model_cfg[key] = value
    else:
        for key, value in preset.items():
            model_cfg[key] = value

    if logger is not None:
        logger.info(
            "Running inference in %s (%s); parameters stay float32.",
            name,
            ", ".join(f"{k}={v}" for k, v in preset.items()),
        )
    return name
