from functools import partial
from typing import Any

from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from dmri.simulators.acquisition_scheme import (
    random_advanced_reasearch_acquisition_scheme,
    random_clinical_acquisition,
    random_hardi_acquisition,
    random_hcp_acquisition,
    random_hcp_large_acquisition,
    random_ssfp_acquisition,
)
from dmri.simulators.multi_compartment import MultiCompartment

# This map is only a reader for frozen named configs. New acquisition factories
# are imported directly from their Hydra targets and require no registration.
_LEGACY_ACQUISITION_FACTORIES = {
    "clinical": random_clinical_acquisition,
    "hardi": random_hardi_acquisition,
    "advanced_research": random_advanced_reasearch_acquisition_scheme,
    "hcp": random_hcp_acquisition,
    "hcp_large": random_hcp_large_acquisition,
    "ssfp": random_ssfp_acquisition,
}


def _simulator_cfg(cfg):
    return cfg.simulator if hasattr(cfg, "simulator") else cfg


def resolve_simulator_model(cfg) -> type[MultiCompartment]:
    """Resolve the structural signal-model class from new or frozen config."""
    simulator_cfg = _simulator_cfg(cfg)
    model_class = simulator_cfg.get("model_class")
    legacy_name = OmegaConf.select(simulator_cfg, "sim_type.name")
    legacy_path = None
    if legacy_name is not None:
        legacy_path = (
            legacy_name
            if "." in legacy_name
            else f"dmri.simulators.models.{legacy_name}"
        )
    if model_class is None:
        model_class = legacy_path
    if model_class is None:
        raise ValueError(
            "simulator.model_class must be an importable MultiCompartment class"
        )

    resolved = get_class(str(model_class))
    if legacy_path is not None and get_class(str(legacy_path)) is not resolved:
        raise ValueError(
            "Conflicting simulator settings 'model_class' and 'sim_type.name'"
        )
    if not issubclass(resolved, MultiCompartment):
        raise TypeError(
            f"simulator.model_class {model_class!r} must subclass MultiCompartment"
        )
    return resolved


def simulator_model_config(cfg) -> dict[str, str]:
    """Return the minimal, portable simulator specification for an artifact."""
    model_class = resolve_simulator_model(cfg)
    return {"model_class": f"{model_class.__module__}.{model_class.__qualname__}"}


def _plain(value):
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _legacy_acquisition_factories(simulator_cfg):
    acquisition_cfg = simulator_cfg.get("acquisition_scheme")
    if acquisition_cfg is None:
        return None
    schemes = OmegaConf.select(acquisition_cfg, "params.schemes")
    if schemes is None:
        schemes = acquisition_cfg.get("schemes")
    if schemes is None:
        schemes = [
            {
                "name": acquisition_cfg.get("name"),
                "params": acquisition_cfg.get("params", {}),
            }
        ]

    factories = []
    for scheme in schemes:
        name = scheme.get("name")
        try:
            factory = _LEGACY_ACQUISITION_FACTORIES[name]
        except KeyError as error:
            raise ValueError(f"Unknown legacy acquisition scheme {name!r}") from error
        factories.append(partial(factory, **(_plain(scheme.get("params")) or {})))
    return factories


def resolve_acquisition_factories(cfg) -> list[Any]:
    """Resolve direct acquisition targets or frozen named acquisition configs."""
    simulator_cfg = _simulator_cfg(cfg)
    acquisitions = simulator_cfg.get("acquisitions")
    legacy_cfg = simulator_cfg.get("acquisition_scheme")
    if acquisitions is not None and legacy_cfg is not None:
        raise ValueError(
            "Conflicting simulator settings 'acquisitions' and 'acquisition_scheme'"
        )
    if acquisitions is None:
        legacy = _legacy_acquisition_factories(simulator_cfg)
        if legacy is not None:
            return legacy
        raise ValueError("simulator.acquisitions must contain at least one factory")

    factories = [instantiate(acquisition) for acquisition in acquisitions]
    if not factories or not all(callable(factory) for factory in factories):
        raise TypeError(
            "Every simulator acquisition must instantiate to a callable; "
            "set _partial_: true on acquisition functions"
        )
    return factories


def simulator_option(simulator_cfg, new_name, legacy_name, default=None):
    new_value = simulator_cfg.get(new_name)
    legacy_value = simulator_cfg.get(legacy_name)
    if new_value is not None and legacy_value is not None and new_value != legacy_value:
        raise ValueError(
            f"Conflicting simulator settings {new_name!r} and {legacy_name!r}"
        )
    if new_value is not None:
        return new_value
    if legacy_value is not None:
        return legacy_value
    return default


def mask_prior_overrides(simulator_cfg) -> dict[str, Any]:
    """Read generic overrides while retaining frozen prior field aliases."""
    overrides = dict(_plain(simulator_cfg.get("mask_prior")) or {})
    aliases = {
        "prior_mask_alpha": "alpha",
        "prior_mask_beta": "beta",
        "prior_mask_p0": "p0",
        "prior_mask_u_alpha": "u_alpha",
        "prior_mask_u_beta": "u_beta",
    }
    for old_name, new_name in aliases.items():
        legacy_value = simulator_cfg.get(old_name)
        if legacy_value is None:
            continue
        if new_name in overrides and overrides[new_name] != legacy_value:
            raise ValueError(
                f"Conflicting simulator mask-prior settings {old_name!r} and "
                f"mask_prior.{new_name}"
            )
        overrides.setdefault(new_name, legacy_value)
    return overrides
