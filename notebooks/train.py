# %%
import jax
import jax.numpy as jnp


import numpy as np
import matplotlib.pyplot as plt
from dmri.simulators import (
    Ball,
    Stick,
    Zeppelin,
    Dti,
    BallStick,
    Ball2Stick,
    AllGaussianModels,
)
from dmri.simulators.acquisition_scheme import (
    acquisition_scheme,
    random_hardi_acquisition,
    random_clinical_acquisition,
    random_advanced_reasearch_acquisition_scheme,
)
from dmri.simulators.multi_compartment import AllGaussianAndConvolvedModels
from flax import nnx


ball_stick = AllGaussianAndConvolvedModels.from_theta(
    np.random.randn(AllGaussianAndConvolvedModels.theta_dim)
)

# %%


import wandb
import random

wandb.init(
    # set the wandb project where this run will be logged
    project="dmri",
    # track hyperparameters and run metadata
    config={
        "architecture": "mask_amortized_3",
    },
)


def simulator_hardi(rng):
    rng0, rng1, rng2, rng3, rng4, rng5, rng6, rng7 = jax.random.split(rng, 8)
    acq = random_hardi_acquisition(rng0)
    theta = jax.random.normal(rng1, shape=(AllGaussianAndConvolvedModels.theta_dim,))
    p = jax.random.beta(rng2, 1.0, 1.0, shape=(1,))
    model_mask = jax.random.bernoulli(
        rng3, p=p, shape=(len(AllGaussianAndConvolvedModels.model_types))
    )
    # This needs to have only one entry
    model_noise_idx = jax.random.randint(
        rng4,
        minval=0,
        maxval=len(AllGaussianAndConvolvedModels.noise_types),
        shape=(1,),
    )
    model_noise_mask = jnp.zeros(
        len(AllGaussianAndConvolvedModels.noise_types), dtype=jnp.bool
    )
    model_noise_mask = model_noise_mask.at[model_noise_idx].set(True)
    model_mask = jnp.concatenate([model_mask, model_noise_mask], axis=-1)

    ball_stick = AllGaussianAndConvolvedModels.from_theta(theta, model_mask=model_mask)

    x_o = ball_stick.signal(acq, rng=rng5)
    return p, model_mask, theta, x_o, acq


def simulator_clinical(rng):
    rng0, rng1, rng2, rng3, rng4, rng5, rng6, rng7 = jax.random.split(rng, 8)
    acq = random_clinical_acquisition(rng0)
    theta = jax.random.normal(rng1, shape=(AllGaussianAndConvolvedModels.theta_dim,))
    p = jax.random.beta(rng2, 1.0, 1.0, shape=(1,))
    model_mask = jax.random.bernoulli(
        rng3, p=p, shape=(len(AllGaussianAndConvolvedModels.model_types))
    )
    # This needs to have only one entry
    model_noise_idx = jax.random.randint(
        rng4,
        minval=0,
        maxval=len(AllGaussianAndConvolvedModels.noise_types),
        shape=(1,),
    )
    model_noise_mask = jnp.zeros(
        len(AllGaussianAndConvolvedModels.noise_types), dtype=jnp.bool
    )
    model_noise_mask = model_noise_mask.at[model_noise_idx].set(True)
    model_mask = jnp.concatenate([model_mask, model_noise_mask], axis=-1)

    ball_stick = AllGaussianAndConvolvedModels.from_theta(theta, model_mask=model_mask)

    x_o = ball_stick.signal(acq, rng=rng5)
    return p, model_mask, theta, x_o, acq


# %%
p_mask, masks, theta, x_o, acq = jax.vmap(simulator_hardi)(
    jax.random.split(jax.random.PRNGKey(0), num=200)
)

plt.imshow(x_o, aspect="auto", vmin=0, vmax=1)
plt.colorbar()

# %%
from dmri.nn.dmri_reconstruction_model import (
    DMRIInferenceModel,
    DMRIInferenceModelConfig,
    DMRIInferenceModelConfigMaskPriorAmortized,
)

cfg = DMRIInferenceModelConfigMaskPriorAmortized(AllGaussianAndConvolvedModels)
cfg.model_dim = 64
cfg.model_selection_cfg.num_layers = 8
cfg.theta_inference_cfg.num_layers = 8

model = DMRIInferenceModel(cfg, nnx.Rngs(1))

params = nnx.state(model, nnx.Param)


# %%
import optax


optimizer = optax.chain(optax.adaptive_grad_clip(10.0), optax.adamw(5e-4))

opt_state = optimizer.init(params)


# %%


def loss_fn(params, data, rng):
    rng1, rng2 = jax.random.split(rng, 2)
    p_mask, mask, thetas, xs, bvals, bvecs = data[0]
    loss1 = model.loss_fn(
        params, rng1, mask, thetas, xs, bvals, bvecs, mask_prior=p_mask
    ).sum()
    p_mask, mask, thetas, xs, bvals, bvecs = data[1]
    loss2 = model.loss_fn(
        params, rng2, mask, thetas, xs, bvals, bvecs, mask_prior=p_mask
    ).sum()
    return 0.5 * (loss1 * loss2)


@jax.jit
def update(params, opt_state, data, rng):
    loss, grads = jax.value_and_grad(loss_fn)(params, data, rng)
    updates, opt_state = optimizer.update(grads, opt_state, params=params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss


# %%
batch_simulator = jax.jit(jax.vmap(simulator_hardi))

# %%
batch_simulator2 = jax.jit(jax.vmap(simulator_clinical))

# %%
p_mask, masks, thetas, x_os, acq = batch_simulator(
    jax.random.split(jax.random.key(521), 2**12)
)

# %%
key = jax.random.key(521)
for i in range(200):
    key, subkey1 = jax.random.split(key, 2)
    subkey1, subkey2 = jax.random.split(subkey1, 2)
    p_mask, masks, thetas, x_os, acq = batch_simulator(jax.random.split(subkey1, 2**12))
    p_mask2, masks2, thetas2, x_os2, acq2 = batch_simulator2(
        jax.random.split(subkey2, 2**13)
    )
    l = 0
    for j in range(100):
        key, subkey2 = jax.random.split(key, 2)
        idx = jax.random.randint(subkey2, (512,), 0, 2**12)
        idx2 = jax.random.randint(subkey2, (512,), 0, 2**13)
        (
            alphas_batch,
            masks_batch,
            thetas_batch,
            x_os_batch,
            bvals_batch,
            bvecs_batch,
        ) = (
            p_mask[idx],
            masks[idx],
            thetas[idx],
            x_os[idx],
            acq.bvals[idx],
            acq.bvecs[idx],
        )
        (
            alphas_batch2,
            masks_batch2,
            thetas_batch2,
            x_os_batch2,
            bvals_batch2,
            bvecs_batch2,
        ) = (
            p_mask2[idx2],
            masks2[idx2],
            thetas2[idx2],
            x_os2[idx2],
            acq2.bvals[idx2],
            acq2.bvecs[idx2],
        )
        data = (
            (
                alphas_batch,
                masks_batch,
                thetas_batch,
                x_os_batch,
                bvals_batch,
                bvecs_batch,
            ),
            (
                alphas_batch2,
                masks_batch2,
                thetas_batch2,
                x_os_batch2,
                bvals_batch2,
                bvecs_batch2,
            ),
        )
        params, opt_state, loss = update(params, opt_state, data, subkey2)
        l += loss
    wandb.log({"loss": l / 10})

import pickle

with open("params_big_large_mask1_2.pkl", "wb") as f:
    pickle.dump(params, f)

# Restart optimizer
opt_state = optimizer.init(params)
key = jax.random.key(522)
for i in range(200):
    key, subkey1 = jax.random.split(key, 2)

    p_mask, masks, thetas, x_os, acq = batch_simulator(jax.random.split(subkey1, 2**12))
    p_mask2, masks2, thetas2, x_os2, acq2 = batch_simulator2(
        jax.random.split(subkey1, 2**13)
    )
    l = 0
    for j in range(100):
        key, subkey2 = jax.random.split(key, 2)
        idx = jax.random.randint(subkey2, (512,), 0, 2**12)
        idx2 = jax.random.randint(subkey2, (512,), 0, 2**13)
        (
            alphas_batch,
            masks_batch,
            thetas_batch,
            x_os_batch,
            bvals_batch,
            bvecs_batch,
        ) = (
            p_mask[idx],
            masks[idx],
            thetas[idx],
            x_os[idx],
            acq.bvals[idx],
            acq.bvecs[idx],
        )
        (
            alphas_batch2,
            masks_batch2,
            thetas_batch2,
            x_os_batch2,
            bvals_batch2,
            bvecs_batch2,
        ) = (
            p_mask2[idx2],
            masks2[idx2],
            thetas2[idx2],
            x_os2[idx2],
            acq2.bvals[idx2],
            acq2.bvecs[idx2],
        )
        data = (
            (
                alphas_batch,
                masks_batch,
                thetas_batch,
                x_os_batch,
                bvals_batch,
                bvecs_batch,
            ),
            (
                alphas_batch2,
                masks_batch2,
                thetas_batch2,
                x_os_batch2,
                bvals_batch2,
                bvecs_batch2,
            ),
        )
        params, opt_state, loss = update(params, opt_state, data, subkey2)
        l += loss
    wandb.log({"loss": l / 10})

with open("params_big_large_mask2_2.pkl", "wb") as f:
    pickle.dump(params, f)
