"""Public package API.

Hub helpers are loaded lazily so lightweight entry points such as ``dmri.cli``
can configure the process before Flax and JAX load their native runtimes.
"""

from importlib import import_module

__all__ = [
    "DEFAULT_MODEL_NAME",
    "DEFAULT_REPO_ID",
    "PretrainedModel",
    "list_pretrained_models",
    "load_pretrained",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module("dmri.hub"), name)
