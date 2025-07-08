from dmri.simulators.acquisition_scheme import ssfp_acquisition_scheme
import jax
import jax.numpy as jnp
import numpy as np


def select_models(
    cfg, key, data, logger, sim_type, model, acq, model_mask_samples=None
):
    """Select models based on the model mask"""
    num_comp = len(sim_type.model_types) + len(sim_type.noise_types)

    if cfg.model_selection.name == "none":
        return None
    else:
        model_mask = jnp.array(model_mask_samples, dtype=jnp.bool)

        nans_in_samples = jnp.isnan(model_mask).sum()
        logger.info(f"Number of NaNs in model_mask: {nans_in_samples}")
        model_mask = jnp.where(jnp.isnan(model_mask), True, model_mask)

        incorporate_models = cfg.model_selection.name

        if incorporate_models == "average":
            logger.info("Sampling parameters for each model")
            assert model_mask.ndim == 3, (
                "model_mask must be 3D if incorporate_models is ball3stick_average"
            )
            num_mask_samples = model_mask.shape[1]
            if num_mask_samples >= cfg.theta_sample.num_samples:
                # Select random num_mask_samples from model_mask
                if num_mask_samples > cfg.theta_sample.num_samples:
                    selected_mask_samples = jax.random.choice(
                        key, model_mask, (cfg.theta_sample.num_samples,), axis=1
                    )
                    model_mask = selected_mask_samples
            else:
                raise ValueError(
                    f"num_mask_samples ({num_mask_samples}) must be greater than or equal to num_samples ({cfg.theta_sample.num_samples})"
                )
            average_freq = jnp.mean(model_mask, axis=1).mean(0)
            logger.info(f"Average frequency of all models: {average_freq}")

        elif incorporate_models == "best":
            logger.info("Sampling best model only")
            # Select the most frequent mask
            feasible_models = jnp.array(
                cfg.model_selection.feasible_models, dtype=jnp.bool
            )
            p_mask = cfg.mask_sample.p_mask

            def eval_feasible_log_probs(x, idx):
                if isinstance(acq, ssfp_acquisition_scheme):
                    new_acq = acq.select(idx)
                else:
                    del idx
                    new_acq = acq

                model_logpmf = jax.vmap(
                    model.log_prob_mask, in_axes=(0, None, None, None)
                )(feasible_models, new_acq, x, jnp.array([p_mask]))
                return model_logpmf

            batch_size = 10_000
            models_selected = []
            for i in range(0, data.shape[0], batch_size):
                batch_data = data[i : i + batch_size]
                idx = jnp.arange(i, i + batch_size)
                batch_logpmf = jax.vmap(eval_feasible_log_probs, in_axes=(0, 0))(
                    batch_data, idx
                )
                batch_mask = batch_logpmf.argmax(axis=-1)
                batch_mask = np.array(feasible_models[batch_mask], dtype=np.bool)
                models_selected.append(batch_mask)
            model_mask = np.concatenate(models_selected, axis=0)

        assert model_mask.shape[0] == data.shape[0], (
            "model_mask must have the same number of voxels as the data"
        )
        assert model_mask.shape[-1] == num_comp, (
            "model_mask must have the same number of components as the model"
        )

    return model_mask
