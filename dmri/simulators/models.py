from __future__ import annotations

import jax.numpy as jnp
from jax.typing import ArrayLike

from dmri.simulators.local_signal_models import (
    Ball,
    BinghamStick,
    BinghamZeppelin,
    Dot,
    Dti,
    MultiShellStaticBall,
    MultiShellStaticStick,
    NoddiB,
    NoddiW,
    SandiB,
    SandiW,
    SSFPStaticBall,
    SSFPStaticStick,
    StaticBall,
    StaticStick,
    Stick,
    WatsonStick,
    WatsonZeppelin,
    Zeppelin,
)
from dmri.simulators.mask_prior import TotalParamPenalizedPrior
from dmri.simulators.multi_compartment import (
    MultiCompartment,
    SharedDiffusivity,
    SharedMultiShellDiffusivity,
    SharedMultiShellDiffusivityGammaPrior,
    SharedSSFPDiffusivity,
)
from dmri.simulators.noise_compartments import (
    BoundedGaussianNoise,
    BoundedRicianNoise,
    RicianNoiseSNR310,
    RicianNoiseSNR1020,
)
from dmri.utils.dmriutils import ssfp_signal_fn


class BallStickSharedDiffusivity(MultiCompartment):
    model_types = [StaticBall, StaticStick]
    noise_types = []
    fraction_prior = jnp.ones(2)
    shared_parameter_type = SharedDiffusivity


class BallStickSharedDiffusivity2(MultiCompartment):
    model_types = [StaticBall, StaticStick]
    noise_types = []
    fraction_prior = jnp.ones(2)
    shared_parameter_type = SharedDiffusivity


class Ball3StickSharedDiffusivity(MultiCompartment):
    model_types = [StaticBall, StaticStick, StaticStick, StaticStick]
    noise_types = [BoundedGaussianNoise]
    fraction_prior = jnp.array([3.5, 1.0, 0.3, 0.1])
    shared_parameter_type = SharedDiffusivity


class Ball3StickSharedDiffusivityTotalParamPenalizedPrior(MultiCompartment):
    model_types = [StaticBall, StaticStick, StaticStick, StaticStick]
    noise_types = [BoundedGaussianNoise]
    fraction_prior = jnp.array([3.5, 1.0, 0.3, 0.1])
    shared_parameter_type = SharedDiffusivity
    mask_prior_cls = TotalParamPenalizedPrior


class SSFPBall3StickSharedDiffusivity(MultiCompartment):
    model_types = [SSFPStaticBall, SSFPStaticStick, SSFPStaticStick, SSFPStaticStick]
    noise_types = [BoundedGaussianNoise]
    fraction_prior = jnp.array([3.5, 1.0, 0.3, 0.1])
    shared_parameter_type = SharedSSFPDiffusivity

    @staticmethod
    def normalizing_fn(acq, x: ArrayLike) -> ArrayLike:
        return (x - jnp.min(x)) / (jnp.max(x) - jnp.min(x))


class SSFPBall3StickSharedDiffusivityBetterNorm(MultiCompartment):
    model_types = [SSFPStaticBall, SSFPStaticStick, SSFPStaticStick, SSFPStaticStick]
    noise_types = [BoundedGaussianNoise]
    fraction_prior = jnp.array([3.5, 1.0, 0.3, 0.1])
    shared_parameter_type = SharedSSFPDiffusivity

    @staticmethod
    def pre_normalizing_fn(acq, x: ArrayLike) -> ArrayLike:
        ssfp_max = (
            ssfp_signal_fn(
                0.0,
                acq.qvals * 0,
                acq.E1,
                acq.E2,
                acq.sa,
                acq.ca,
                acq.TRs,
                acq.diffGradDur,
            )
            + 1e-5
        )

        return x / ssfp_max


class Ball3StickSharedDiffusivityUniformFraction(MultiCompartment):
    model_types = [StaticBall, StaticStick, StaticStick, StaticStick]
    noise_types = [BoundedGaussianNoise]
    fraction_prior = jnp.ones(4)
    shared_parameter_type = SharedDiffusivity


class MultiShellBall3StickSharedDiffusivity(MultiCompartment):
    model_types = [
        MultiShellStaticBall,
        MultiShellStaticStick,
        MultiShellStaticStick,
        MultiShellStaticStick,
    ]
    noise_types = [BoundedGaussianNoise]
    fraction_prior = jnp.array([3.5, 1.0, 0.3, 0.1])
    shared_parameter_type = SharedMultiShellDiffusivity


class MultiShellBall3StickSharedDiffusivityUniformFraction(MultiCompartment):
    model_types = [
        MultiShellStaticBall,
        MultiShellStaticStick,
        MultiShellStaticStick,
        MultiShellStaticStick,
    ]
    noise_types = [BoundedGaussianNoise]
    fraction_prior = jnp.ones(4)
    shared_parameter_type = SharedMultiShellDiffusivity


class MultiShellBall3StickSharedDiffusivityGammaPrior(MultiCompartment):
    model_types = [
        MultiShellStaticBall,
        MultiShellStaticStick,
        MultiShellStaticStick,
        MultiShellStaticStick,
    ]
    noise_types = [BoundedGaussianNoise]
    fraction_prior = jnp.array([3.5, 1.0, 0.3, 0.1])
    shared_parameter_type = SharedMultiShellDiffusivityGammaPrior


class BallStick(MultiCompartment):
    model_types = [Ball, Stick]
    noise_types = []
    fraction_prior = jnp.ones(2)


class Ball2Stick(MultiCompartment):
    model_types = [Ball, Stick, Stick]
    noise_types = []
    fraction_prior = jnp.ones(3)


class Ball3Stick(MultiCompartment):
    model_types = [Ball, Stick, Stick, Stick]
    noise_types = []
    fraction_prior = jnp.ones(4)


class Ball3StickNoise(MultiCompartment):
    model_types = [Ball, Stick, Stick, Stick]
    noise_types = [RicianNoiseSNR310, RicianNoiseSNR1020]
    fraction_prior = jnp.ones(4)


class BallStickZeppelinNoise(MultiCompartment):
    model_types = [Ball, Stick, Zeppelin]
    noise_types = [RicianNoiseSNR310, RicianNoiseSNR1020]
    fraction_prior = jnp.ones(3)


class Ball2Stick2Zeppelin2Dti(MultiCompartment):
    model_types = [
        Ball,
        Stick,
        Stick,
        Zeppelin,
        Zeppelin,
        Dti,
        Dti,
    ]
    noise_types = []
    fraction_prior = jnp.ones(7)


class Ball3Stick3ZeppelinNoise(MultiCompartment):
    model_types = [Ball] + 3 * [Stick] + 3 * [Zeppelin]
    noise_types = [
        RicianNoiseSNR310,
        RicianNoiseSNR1020,
    ]
    fraction_prior = jnp.ones(1 + 3 + 3)


class AllGaussianModels(MultiCompartment):
    model_types = [Ball] + 3 * [Stick] + 3 * [Zeppelin] + 3 * [Dti]
    noise_types = [
        BoundedGaussianNoise,
        BoundedRicianNoise,
    ]
    fraction_prior = jnp.ones(1 + 3 + 3 + 3)


class AllGaussianModelsParamCountPrior(MultiCompartment):
    model_types = [Ball] + 3 * [Stick] + 3 * [Zeppelin] + 3 * [Dti]
    noise_types = [
        BoundedGaussianNoise,
        BoundedRicianNoise,
    ]
    fraction_prior = jnp.ones(1 + 3 + 3 + 3)
    mask_prior_cls = TotalParamPenalizedPrior


class AllGaussianAndConvolvedModels(MultiCompartment):
    model_types = (
        [Ball]
        + 3 * [Stick]
        + 3 * [Zeppelin]
        + 3 * [Dti]
        + [Dot]
        + [WatsonStick]
        + [WatsonZeppelin]
        + [BinghamStick]
        + [BinghamZeppelin]
        + [NoddiB]
        + [NoddiW]
        + [SandiB]
        + [SandiW]
    )
    noise_types = [
        BoundedGaussianNoise,
    ] + [
        BoundedRicianNoise,
    ]
    fraction_prior = jnp.ones(1 + 3 + 3 + 3 + 9)
    mask_prior_cls = TotalParamPenalizedPrior
