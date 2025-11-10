import importlib

from flax import nnx
from omegaconf import DictConfig

from dmri.nn.dmri_reconstruction_model import (
    DMRIInferenceModel,
)


def build_model(cfg: DictConfig, sim_type):
    name_embed = cfg.model.embedding_net.name
    params_embed = cfg.model.embedding_net.params
    cls_embed_name = cfg.model.embedding_net.get("cls", None)
    name_model_selection_net = cfg.model.model_selection_net.name
    params_model_selection_net = cfg.model.model_selection_net.params
    name_inference_net = cfg.model.inference_net.name
    params_inference_net = cfg.model.inference_net.params
    module_embed = importlib.import_module("dmri.nn.dmri_reconstruction_model")

    cfg_embed = getattr(module_embed, name_embed)(**params_embed)
    embedding_cls = None
    if cls_embed_name is not None:
        embedding_cls = getattr(module_embed, cls_embed_name, None)
        if embedding_cls is None:
            raise ValueError(
                f"Unknown embedding class '{cls_embed_name}' requested in config."
            )
    cfg_model_selection_net = getattr(module_embed, name_model_selection_net)(
        **params_model_selection_net
    )
    cfg_inference_net = getattr(module_embed, name_inference_net)(
        **params_inference_net
    )
    name_model = cfg.model.name
    cfg_class = getattr(module_embed, name_model)
    precision_keys = (
        "dtype",
        "param_dtype",
        "precision",
        "preferred_element_type",
    )

    precision_kwargs = {
        key: cfg.model.get(key)
        for key in precision_keys
        if key in cfg.model and cfg.model.get(key) is not None
    }
    # Preserve legacy spelling if present.
    legacy_preferred = cfg.model.get("prefered_element_type", None)
    if (
        "preferred_element_type" not in precision_kwargs
        and legacy_preferred is not None
    ):
        precision_kwargs["preferred_element_type"] = legacy_preferred

    cfg_kwargs = dict(
        model_dim=cfg.model.model_dim,
        use_attention_mask=cfg.model.use_attention_mask,
        embedding_cfg=cfg_embed,
        model_selection_cfg=cfg_model_selection_net,
        theta_inference_cfg=cfg_inference_net,
        **precision_kwargs,
    )
    if embedding_cls is not None:
        cfg_kwargs["embedding_cls"] = embedding_cls
    cfg_m = cfg_class(
        sim_type,
        **cfg_kwargs,
    )
    model = DMRIInferenceModel(cfg_m, nnx.Rngs(cfg.model.init_seed))

    return model
