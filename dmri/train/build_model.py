from omegaconf import DictConfig
from dmri.nn.dmri_reconstruction_model import (
    DMRIInferenceModel,
)
from flax import nnx
import importlib


def build_model(cfg: DictConfig, sim_type):
    name_embed = cfg.model.embedding_net.name
    params_embed = cfg.model.embedding_net.params
    name_model_selection_net = cfg.model.model_selection_net.name
    params_model_selection_net = cfg.model.model_selection_net.params
    name_inference_net = cfg.model.inference_net.name
    params_inference_net = cfg.model.inference_net.params
    module_embed = importlib.import_module("dmri.nn.dmri_reconstruction_model")

    cfg_embed = getattr(module_embed, name_embed)(**params_embed)
    cfg_model_selection_net = getattr(module_embed, name_model_selection_net)(
        **params_model_selection_net
    )
    cfg_inference_net = getattr(module_embed, name_inference_net)(
        **params_inference_net
    )
    name_model = cfg.model.name
    cfg_class = getattr(module_embed, name_model)
    cfg_m = cfg_class(
        sim_type,
        model_dim=cfg.model.model_dim,
        embedding_cfg=cfg_embed,
        model_selection_cfg=cfg_model_selection_net,
        theta_inference_cfg=cfg_inference_net,
    )
    model = DMRIInferenceModel(cfg_m, nnx.Rngs(cfg.model.init_seed))

    return model
