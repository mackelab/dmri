from omegaconf import DictConfig, OmegaConf
from dmri.nn.dmri_reconstruction_model import (
    DMRIInferenceModel,
    DMRIInferenceModelConfig,
    DMRIInferenceModelConfigMaskPriorAmortized,
)
from flax import nnx


def build_model(cfg: DictConfig, sim_type):
    cfg_m = DMRIInferenceModelConfigMaskPriorAmortized(sim_type)
    model = DMRIInferenceModel(cfg_m, nnx.Rngs(cfg.model.init_seed))

    params = nnx.state(model, nnx.Param)

    return model, params
