from dataclasses import dataclass
from typing import Any

from flax import nnx
from huggingface_hub import HfApi

DEFAULT_REPO_ID = "manugloeck/dmri-pretrained"
DEFAULT_MODEL_NAME = "b3s_2_4_6_128"


@dataclass(frozen=True)
class PretrainedModel:
    model: Any
    simulator: Any
    config: Any
    step: int | None


def load_checkpoint(**kwargs):
    """Load the heavy checkpoint implementation only when a model is requested."""
    from dmri.train.utils import load_checkpoint as load

    return load(**kwargs)


def load_pretrained(
    model_name=DEFAULT_MODEL_NAME,
    repo_id=DEFAULT_REPO_ID,
    which="best",
    *,
    revision=None,
    cache_dir=None,
    token=None,
    local_files_only=False,
    precision=None,
) -> PretrainedModel:
    checkpoint, model, simulator = load_checkpoint(
        repo_id=repo_id,
        model_name=model_name,
        which=which,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
        precision=precision,
    )
    params = checkpoint.get("params_ema")
    if params is None:
        params = checkpoint["params"]
    nnx.update(model, params)
    model.eval()

    return PretrainedModel(
        model=model,
        simulator=simulator,
        config=getattr(model, "cfg", None),
        step=checkpoint.get("step"),
    )


def list_pretrained_models(
    repo_id=DEFAULT_REPO_ID,
    *,
    revision=None,
    token=None,
) -> list[str]:
    files = HfApi(token=token).list_repo_files(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
    )
    return sorted(
        path.split("/", 1)[0]
        for path in files
        if path.count("/") == 1 and path.endswith("/config.yaml")
    )
