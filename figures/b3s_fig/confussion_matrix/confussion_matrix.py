#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from functools import partial
from probjax.nn import MLP

from dmri.simulators.mask_prior import BetaBernoulliMaskPrior, TotalParamPenalizedPrior
from dmri.train.utils import load_cfg, load_checkpoint


def _parse_which(which: str):
    if which.isdigit():
        return int(which)
    return which


def _save_array(path: str | None, array: np.ndarray) -> None:
    if path is None:
        return
    out_path = Path(path)
    if out_path.suffix == ".csv":
        np.savetxt(out_path, array, delimiter=",")
    else:
        np.save(out_path, array)


def _select_params(checkpoint: dict, model, choice: str) -> str:
    params_ema = checkpoint.get("params_ema")
    params_raw = checkpoint.get("params")

    if choice == "ema":
        if params_ema is None:
            raise ValueError("Checkpoint does not contain EMA params.")
        nnx.update(model, params_ema)
        return "ema"
    if choice == "raw":
        if params_raw is None:
            raise ValueError("Checkpoint does not contain raw params.")
        nnx.update(model, params_raw)
        return "raw"

    if params_ema is not None:
        nnx.update(model, params_ema)
        return "ema"
    if params_raw is not None:
        nnx.update(model, params_raw)
        return "raw"
    raise ValueError("Checkpoint does not contain params.")


def _build_mask_prior(sim_type, cfg):
    mask_prior_overrides = {}
    if issubclass(sim_type.mask_prior_cls, BetaBernoulliMaskPrior):
        prior_mask_alpha = getattr(cfg.simulator, "prior_mask_alpha", None)
        prior_mask_beta = getattr(cfg.simulator, "prior_mask_beta", None)
        if prior_mask_alpha is not None:
            mask_prior_overrides["alpha"] = prior_mask_alpha
        if prior_mask_beta is not None:
            mask_prior_overrides["beta"] = prior_mask_beta
    elif issubclass(sim_type.mask_prior_cls, TotalParamPenalizedPrior):
        prior_mask_p0 = getattr(cfg.simulator, "prior_mask_p0", None)
        prior_mask_u_alpha = getattr(cfg.simulator, "prior_mask_u_alpha", None)
        prior_mask_u_beta = getattr(cfg.simulator, "prior_mask_u_beta", None)
        if prior_mask_p0 is not None:
            mask_prior_overrides["p0"] = prior_mask_p0
        if prior_mask_u_alpha is not None:
            mask_prior_overrides["u_alpha"] = prior_mask_u_alpha
        if prior_mask_u_beta is not None:
            mask_prior_overrides["u_beta"] = prior_mask_u_beta
        mask_prior_overrides["num_model_parameters"] = [
            mt.theta_dim for mt in sim_type.model_types
        ]
    return sim_type.create_mask_prior(**mask_prior_overrides)


def _build_dataset(
    sim_fn,
    num_samples: int,
    seed: int,
    sim_batch_size: int,
    support_masks: jax.Array | None = None,
    mask_prior=None,
    mask_prior_hyperparameter: jax.Array | None = None,
):
    rng = jax.random.PRNGKey(seed)
    remaining = num_samples

    xs_list = []
    masks_list = []
    priors_list = []
    acq_list = []

    if support_masks is not None:
        if mask_prior is None:
            raise ValueError("mask_prior must be provided when support_masks is set.")
        support_masks = jnp.asarray(support_masks, dtype=jnp.bool_)

        def sample_support(rng):
            rng_lambda, rng_mask, rng_sim = jax.random.split(rng, 3)
            hyper = jax.random.uniform(
                rng_lambda,
                shape=(mask_prior.mask_prior_dim,),
                minval=0.0,
                maxval=1.0,
            )
            log_probs = jax.vmap(lambda m: mask_prior.log_prob(m, hyper))(
                support_masks
            )
            idx = jax.random.categorical(rng_mask, log_probs)
            model_mask = support_masks[idx]
            return sim_fn(
                rng_sim, mask_prior_hyperparameter=hyper, model_mask=model_mask
            )

        sample_fn = jax.vmap(sample_support)
    else:
        if mask_prior_hyperparameter is None:
            sample_fn = jax.vmap(partial(sim_fn))
        else:
            hyper = jnp.asarray(mask_prior_hyperparameter)
            if hyper.ndim == 0:
                hyper = hyper[None]

            def sample_fixed(rng):
                return sim_fn(rng, mask_prior_hyperparameter=hyper)

            sample_fn = jax.vmap(sample_fixed)

    while remaining > 0:
        batch_size = min(sim_batch_size, remaining)
        rng, rng_batch = jax.random.split(rng)
        rngs = jax.random.split(rng_batch, batch_size)
        data_batch = sample_fn(rngs)
        xs_list.append(data_batch["x"])
        masks_list.append(data_batch["model_mask"])
        priors_list.append(data_batch["mask_prior"])
        acq_list.append(data_batch["acq"])
        remaining -= batch_size

    xs = jnp.concatenate(xs_list, axis=0)
    true_masks = jnp.concatenate(masks_list, axis=0)
    mask_priors = jnp.concatenate(priors_list, axis=0)
    acqs = jax.tree_util.tree_map(
        lambda *vals: jnp.concatenate(vals, axis=0), *acq_list
    )
    return xs, true_masks, mask_priors, acqs


def _compute_support_masks(true_masks: jax.Array, num_support_classes: int):
    true_masks_np = np.asarray(true_masks)
    unique_masks, unique_counts = np.unique(
        true_masks_np,
        axis=0,
        return_counts=True,
    )
    order = np.argsort(-unique_counts)
    support_masks_np = unique_masks[order[:num_support_classes]]
    support_counts = unique_counts[order[:num_support_classes]]
    support_masks = jnp.asarray(support_masks_np)

    support_index = {tuple(m.tolist()): i for i, m in enumerate(support_masks_np)}
    true_idx = np.array(
        [support_index.get(tuple(m.tolist()), -1) for m in true_masks_np],
        dtype=int,
    )
    valid = true_idx >= 0
    return support_masks, support_masks_np, support_counts, true_idx, valid


_special_token_abbr = {
    "NoddiW": r"No$_{\mathcal{W}}$",
    "NoddiB": r"No$_{\mathcal{B}}$",
    "SandiW": r"Sa$_W$",
    "SandiB": r"Sa$_B$",
    "WatsonStick": r"S$_{\mathcal{W}}$",
    "BinghamStick": r"S$_{\mathcal{B}}$",
    "WatsonZeppelin": r"Z$_{\mathcal{W}}$",
    "BinghamZeppelin": r"Z$_{\mathcal{B}}$",
}

_abbr_token = {
    "Ball": "B",
    "Zeppelin": "Z",
    "Dti": "T",
    "Tensor": "T",
    "Stick": "S",
    "Dot": "D",
    **_special_token_abbr,
}


def _strip_counts(label: str) -> str:
    return re.sub(r"\d+", "", label)


def _natural_sort_key(label: str):
    parts = re.split(r"(\d+)", label)
    return tuple(int(p) if p.isdigit() else p for p in parts if p)


def _model_type_name(model_type) -> str:
    if hasattr(model_type, "__name__"):
        return model_type.__name__
    name = str(model_type).split(".")[-1].replace("'", "")
    return name.rstrip(">")


def _normalize_model_token(name: str) -> str:
    for prefix in ("MultiShellStatic", "SSFPStatic", "Static"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _mask_model_tokens(mask: np.ndarray, model_types) -> list[str]:
    tokens = []
    for i, active in enumerate(mask[: len(model_types)]):
        if active:
            token = _normalize_model_token(_model_type_name(model_types[i]))
            tokens.append(token)
    return tokens


def _is_preselected_tokens(tokens: list[str]) -> bool:
    if not tokens:
        return False
    if tokens[0] != "Ball":
        return False
    if len(tokens) == 1:
        return True
    allowed = {"Stick", "Zeppelin", "Dti"}
    if any(t != tokens[1] for t in tokens[1:]):
        return False
    return tokens[1] in allowed


def _preselected_name_tokens(tokens: list[str]) -> str:
    if len(tokens) == 1:
        return "Ball"
    comp = tokens[1]
    count = len(tokens) - 1
    comp_abbr = {"Stick": "S", "Zeppelin": "Z", "Dti": "T"}
    return f"B{count}{comp_abbr.get(comp, comp[:1].upper())}"


def _shorthand_tokens(tokens: list[str]) -> str:
    out = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        j = i + 1
        while j < len(tokens) and tokens[j] == token:
            j += 1
        count = j - i
        base = _abbr_token.get(token, token[:1].upper())
        out.append(f"{base}{count if count > 1 else ''}")
        i = j
    return "".join(out)


def _noise_token_name(noise_type) -> str:
    name = _model_type_name(noise_type)
    if name.startswith("Bounded"):
        name = name[len("Bounded") :]
    if name.endswith("Noise"):
        name = name[: -len("Noise")]
    return name or _model_type_name(noise_type)


def _mask_noise_label(mask: np.ndarray, noise_types, num_models: int) -> str | None:
    if not noise_types:
        return None
    noise_mask = np.asarray(mask)[num_models : num_models + len(noise_types)]
    active = [idx for idx, on in enumerate(noise_mask) if on]
    if not active:
        return None
    labels = []
    for idx in active:
        name = _noise_token_name(noise_types[idx])
        if "Gaussian" in name:
            labels.append("G")
        elif "Rician" in name:
            labels.append("R")
        else:
            labels.append(name[:1].upper())
    return "".join(labels)


def _support_mask_labels(
    support_masks_np: np.ndarray, model_types, noise_types
) -> list[str]:
    num_models = len(model_types)
    num_noise = len(noise_types)
    include_noise = False
    if num_noise:
        noise_masks = support_masks_np[:, num_models : num_models + num_noise]
        include_noise = np.unique(noise_masks, axis=0).shape[0] > 1

    labels = []
    for mask in support_masks_np:
        tokens = _mask_model_tokens(mask, model_types)
        if not tokens:
            label = ""
        elif _is_preselected_tokens(tokens):
            label = _preselected_name_tokens(tokens)
        else:
            label = _shorthand_tokens(tokens)

        if include_noise:
            noise_label = _mask_noise_label(mask, noise_types, num_models)
            if noise_label and label:
                label = f"{label} | {noise_label}"
        labels.append(label)
    return labels


def _grouped_support_order(labels: list[str]) -> list[int]:
    def base_label(label: str) -> str:
        if not label:
            return ""
        return label.split(" | ", 1)[0]

    def sort_key(idx: int):
        label = labels[idx]
        base = base_label(label)
        group_key = _strip_counts(base)
        return (
            group_key,
            _natural_sort_key(base),
            _natural_sort_key(label),
            idx,
        )

    return sorted(range(len(labels)), key=sort_key)


def _reorder_support(
    support_masks_np: np.ndarray,
    support_labels: list[str],
    support_counts: np.ndarray,
    true_idx: np.ndarray,
    order: list[int],
):
    if not order:
        return support_masks_np, support_labels, support_counts, true_idx
    order_arr = np.asarray(order, dtype=int)
    reordered_masks = support_masks_np[order_arr]
    reordered_labels = [support_labels[i] for i in order_arr]
    reordered_counts = support_counts[order_arr]
    inverse = np.empty(len(order_arr), dtype=int)
    inverse[order_arr] = np.arange(len(order_arr))
    remapped_idx = np.where(true_idx >= 0, inverse[true_idx], -1)
    return reordered_masks, reordered_labels, reordered_counts, remapped_idx


def _is_b3s_simulator(sim_type) -> bool:
    tokens = [
        _normalize_model_token(_model_type_name(model_type))
        for model_type in sim_type.model_types
    ]
    return (
        len(tokens) == 4
        and tokens.count("Ball") == 1
        and tokens.count("Stick") == 3
    )


def _b3s_mask_label(model_mask: np.ndarray) -> str:
    if model_mask.size == 0:
        return "empty"
    sticks = int(model_mask[1:].sum()) if model_mask.size > 1 else 0
    if model_mask[0]:
        return f"B{sticks}S" if sticks > 0 else "B"
    return f"{sticks}S" if sticks > 0 else "empty"


def _b3s_support_masks(sim_type) -> tuple[np.ndarray, list[str]]:
    if not _is_b3s_simulator(sim_type):
        raise ValueError(
            "B3S support masks require exactly Ball + 3 Stick components. "
            "Use --support-mode auto for other models."
        )
    num_models = len(sim_type.model_types)
    num_noise = len(sim_type.noise_types)

    noise_mask = np.zeros(num_noise, dtype=bool)
    if num_noise:
        noise_mask[0] = True

    masks = []
    labels = []
    for bitmask in range(1, 1 << num_models):
        model_mask = np.array(
            [(bitmask >> i) & 1 for i in range(num_models)],
            dtype=bool,
        )
        if num_noise:
            full_mask = np.concatenate([model_mask, noise_mask], axis=0)
        else:
            full_mask = model_mask
        masks.append(full_mask)
        labels.append(_b3s_mask_label(model_mask))
    return _sort_b3s_support(np.asarray(masks), labels, num_models)


def _b3s_support_masks_from_prior(
    sim_type, seed: int, max_draws: int, mask_prior=None
) -> tuple[np.ndarray, list[str]]:
    if not _is_b3s_simulator(sim_type):
        raise ValueError(
            "B3S support masks require exactly Ball + 3 Stick components. "
            "Use --support-mode auto for other models."
        )
    num_models = len(sim_type.model_types)
    num_noise = len(sim_type.noise_types)
    prior = mask_prior if mask_prior is not None else sim_type.create_mask_prior()
    rng = jax.random.PRNGKey(seed)
    target_count = (1 << num_models) - 1
    found: dict[int, np.ndarray] = {}
    draws = 0
    noise_mask = np.zeros(num_noise, dtype=bool)
    if num_noise:
        noise_mask[0] = True

    while draws < max_draws and len(found) < target_count:
        draws += 1
        rng, rng_h, rng_m = jax.random.split(rng, 3)
        hyper = prior.sample_hyperparameters(rng_h)
        mask = prior.sample_model_mask(rng_m, hyper)
        mask_np = np.asarray(mask).astype(bool)
        model_mask = mask_np[:num_models]
        bitmask = int(sum((1 << i) for i, v in enumerate(model_mask) if v))
        if bitmask == 0:
            continue
        if bitmask not in found:
            if num_noise:
                full_mask = np.concatenate([model_mask, noise_mask], axis=0)
            else:
                full_mask = model_mask
            found[bitmask] = full_mask

    if len(found) < target_count:
        missing = [
            bitmask
            for bitmask in range(1, 1 << num_models)
            if bitmask not in found
        ]
        raise ValueError(
            "Failed to sample B3S support masks from prior after "
            f"{max_draws} draws. Missing bitmasks: {missing}."
        )

    ordered = [found[bitmask] for bitmask in range(1, 1 << num_models)]
    labels = [
        _b3s_mask_label(np.asarray(mask)[:num_models])
        for mask in ordered
    ]
    masks = ordered
    return _sort_b3s_support(np.asarray(masks), labels, num_models)


def _sort_b3s_support(
    masks: np.ndarray, labels: list[str], num_models: int
) -> tuple[np.ndarray, list[str]]:
    order = ["B", "1S", "2S", "3S", "B1S", "B2S", "B3S"]
    order_idx = {label: idx for idx, label in enumerate(order)}

    def sort_key(item):
        idx, mask = item
        label = labels[idx]
        model_mask = np.asarray(mask)[:num_models]
        bitmask = int(sum((1 << i) for i, v in enumerate(model_mask) if v))
        return (order_idx.get(label, len(order)), bitmask)

    indices = sorted(range(len(labels)), key=lambda i: sort_key((i, masks[i])))
    sorted_masks = masks[indices]
    sorted_labels = [labels[i] for i in indices]
    return sorted_masks, sorted_labels


def _label_by_support_masks(
    true_masks: jax.Array, support_masks_np: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    support_index = {tuple(m.tolist()): i for i, m in enumerate(support_masks_np)}
    true_masks_np = np.asarray(true_masks)
    true_idx = np.array(
        [support_index.get(tuple(m.tolist()), -1) for m in true_masks_np],
        dtype=int,
    )
    valid = true_idx >= 0
    return true_idx, valid


def _predict_indices(
    model,
    support_masks: jax.Array,
    acqs: jax.Array,
    xs: jax.Array,
    mask_priors: jax.Array,
    topk: int,
    pred_batch_size: int,
):
    def predict_topk(acq, x, p):
        log_probs = jax.vmap(lambda m: model.log_prob_mask(m, acq, x, p))(
            support_masks
        )
        topk_idx = jax.lax.top_k(log_probs, topk)[1]
        return topk_idx[0], topk_idx

    pred_idx, pred_topk = jax.lax.map(
        lambda args: predict_topk(args[0], args[1], args[2]),
        (acqs, xs, mask_priors),
        batch_size=pred_batch_size,
    )
    return np.asarray(pred_idx), np.asarray(pred_topk)


class _ClassifierBaseline(nnx.Module):
    def __init__(
        self,
        *,
        in_features: int,
        num_classes: int,
        mask_prior_dim: int,
        num_layers: int = 8,
        hidden_size: int = 512,
        rngs,
    ) -> None:
        super().__init__()
        self.bvec_linear = nnx.Linear(in_features * 3, hidden_size // 3, rngs=rngs)
        self.bval_linear = nnx.Linear(in_features, hidden_size // 3, rngs=rngs)
        self.signal_linear = nnx.Linear(
            in_features, hidden_size - 2 * (hidden_size // 3), rngs=rngs
        )
        self.lam_linear = nnx.Linear(mask_prior_dim, 64, rngs=rngs)
        self.mlp = MLP([hidden_size] * num_layers, rngs=rngs, context_dim=64)
        self.output_layer = nnx.Linear(hidden_size, num_classes, rngs=rngs)

    def __call__(self, data):
        acq = data["acq"]
        bvecs = acq.bvecs
        bvals = acq.bvals
        signal = data["x"]
        bvec_embed = self.bvec_linear(bvecs.reshape(bvecs.shape[:-2] + (-1,)))
        bval_embed = self.bval_linear(bvals)
        lam_embed = self.lam_linear(data["mask_prior"])
        signal_embed = self.signal_linear(signal)
        x = jnp.concatenate([bvec_embed, bval_embed, signal_embed], axis=-1)
        x = self.mlp(x, context=lam_embed)
        logits = self.output_layer(x)
        return logits


def _sanitize_tree(tree, *, clip: float | None = None):
    def sanitize(x):
        if isinstance(x, (jax.Array, np.ndarray)) and jnp.issubdtype(
            x.dtype, jnp.floating
        ):
            x = jnp.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
            if clip is not None:
                x = jnp.clip(x, -clip, clip)
        return x

    return jax.tree_util.tree_map(sanitize, tree)


def _build_batch_sampler(sim_fn, *, support_masks=None, mask_prior=None):
    if support_masks is not None:
        if mask_prior is None:
            raise ValueError("mask_prior must be provided when support_masks is set.")
        support_masks = jnp.asarray(support_masks, dtype=jnp.bool_)

        def sample_support(rng):
            rng_lambda, rng_mask, rng_sim = jax.random.split(rng, 3)
            hyper = jax.random.uniform(
                rng_lambda,
                shape=(mask_prior.mask_prior_dim,),
                minval=0.0,
                maxval=1.0,
            )
            log_probs = jax.vmap(lambda m: mask_prior.log_prob(m, hyper))(
                support_masks
            )
            idx = jax.random.categorical(rng_mask, log_probs)
            model_mask = support_masks[idx]
            return sim_fn(
                rng_sim, mask_prior_hyperparameter=hyper, model_mask=model_mask
            )

        sample_one = sample_support
    else:
        sample_one = partial(sim_fn)

    def sample_batch(rng, batch_size: int):
        rngs = jax.random.split(rng, batch_size)
        return jax.vmap(sample_one)(rngs)

    return sample_batch


def _train_classifier(
    sim_fn,
    support_masks: jax.Array,
    *,
    mask_prior_dim: int,
    num_steps: int,
    batch_size: int,
    lr: float,
    hidden_size: int,
    num_layers: int,
    seed: int,
    log_every: int,
    simulation_device,
    simulation_batch_size: int,
    buffer_size: int,
):
    rng = jax.random.PRNGKey(seed)
    rng, rng_init, rng_probe = jax.random.split(rng, 3)
    #probe_batch = sampler(rng_probe, 1)
    #in_features = int(probe_batch["acq"].bvals.shape[-1])
    in_features = 105

    num_classes = int(support_masks.shape[0])
    classifier = _ClassifierBaseline(
        in_features=in_features,
        num_classes=num_classes,
        mask_prior_dim=mask_prior_dim,
        num_layers=num_layers,
        hidden_size=hidden_size,
        rngs=nnx.Rngs(rng_init),
    )
    graphdef, params, state = nnx.split(classifier, nnx.Param, ...)

    optimizer = optax.chain(optax.adaptive_grad_clip(1.), optax.adam(lr))
    opt_state = optimizer.init(params)
    support_masks = jnp.asarray(support_masks, dtype=jnp.bool_)
    @jax.jit
    def mask_to_label(mask):
        matches = jnp.all(mask == support_masks, axis=-1)
        return jnp.argmax(matches)

    def loss_fn(params, batch):
        classifier = nnx.merge(graphdef, params, state)
        labels = jax.vmap(mask_to_label)(batch["model_mask"])
        logits = classifier(batch)
        logits = jnp.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
        return jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        )

    @jax.jit
    def update(params, opt_state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state
    print("Training classifier...")
    from dmri.train.dataset import SimulationDataset
    from probjax.nn import DataLoader
    dataset = SimulationDataset(
        sim_fn,
        rng=jax.random.PRNGKey(seed + 1),
        simulation_device=simulation_device,
        simulation_batch_size=simulation_batch_size,
        buffer_size=buffer_size,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    datastream = iter(dataloader)
    for step in range(num_steps):
        batch = _sanitize_tree(next(datastream))
        loss, params, opt_state = update(params, opt_state, batch)
        if log_every > 0 and (step + 1) % log_every == 0:
            print(f"classifier_step {step + 1}/{num_steps} loss={float(loss):.4f}")

    del datastream
    dataloader.close()
    del dataloader

    return graphdef, params, state


def _predict_classifier_indices(
    classifier,
    acqs: jax.Array,
    xs: jax.Array,
    mask_priors: jax.Array,
    topk: int,
    pred_batch_size: int,
):
    def predict_topk(acq, x, p):
        logits = classifier({"acq": acq, "x": x, "mask_prior": p})
        log_probs = jax.nn.log_softmax(logits)
        topk_idx = jax.lax.top_k(log_probs, topk)[1]
        return topk_idx[0], topk_idx

    pred_idx, pred_topk = jax.lax.map(
        lambda args: predict_topk(args[0], args[1], args[2]),
        (acqs, xs, mask_priors),
        batch_size=pred_batch_size,
    )
    return np.asarray(pred_idx), np.asarray(pred_topk)


def _confusion_matrices(true_idx: np.ndarray, pred_idx: np.ndarray, num_classes: int):
    counts = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(counts, (true_idx, pred_idx), 1)
    row_counts = counts.sum(axis=1, keepdims=True)
    conf = np.divide(
        counts,
        row_counts,
        out=np.zeros_like(counts, dtype=float),
        where=row_counts > 0,
    )
    return counts, conf


def _compute_metrics(
    counts: np.ndarray,
    conf: np.ndarray,
    pred_topk: np.ndarray,
    true_idx: np.ndarray,
    topk: int,
):
    total = int(counts.sum())
    num_classes = int(counts.shape[0])
    diag = np.diag(counts)
    row_counts = counts.sum(axis=1)
    col_counts = counts.sum(axis=0)

    top1_micro = float(diag.sum() / total) if total else 0.0
    recall_per_class = np.divide(
        diag,
        row_counts,
        out=np.zeros_like(diag, dtype=float),
        where=row_counts > 0,
    )
    top1_macro = float(recall_per_class.mean()) if num_classes else 0.0

    precision_per_class = np.divide(
        diag,
        col_counts,
        out=np.zeros_like(diag, dtype=float),
        where=col_counts > 0,
    )
    f1_per_class = np.divide(
        2 * precision_per_class * recall_per_class,
        precision_per_class + recall_per_class,
        out=np.zeros_like(precision_per_class),
        where=(precision_per_class + recall_per_class) > 0,
    )
    macro_precision = float(precision_per_class.mean()) if num_classes else 0.0
    macro_recall = float(recall_per_class.mean()) if num_classes else 0.0
    macro_f1 = float(f1_per_class.mean()) if num_classes else 0.0
    weighted_f1 = (
        float(np.average(f1_per_class, weights=row_counts)) if total else 0.0
    )

    topk_macro_hits = 0
    for i in range(num_classes):
        row = conf[i]
        topk_idx = np.argsort(row)[-topk:]
        if i in topk_idx:
            topk_macro_hits += 1
    topk_macro = float(topk_macro_hits / num_classes) if num_classes else 0.0

    topk_micro = float(
        np.mean((pred_topk == true_idx[:, None]).any(axis=1))
    ) if total else 0.0

    metrics = {
        "num_samples": total,
        "num_classes": num_classes,
        "top1_micro": top1_micro,
        "top1_macro": top1_macro,
        f"top{topk}_micro": topk_micro,
        f"top{topk}_macro": topk_macro,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "support_min": int(row_counts.min()) if num_classes else 0,
        "support_max": int(row_counts.max()) if num_classes else 0,
        "support_mean": float(row_counts.mean()) if num_classes else 0.0,
    }
    return metrics


def _print_metrics(metrics: dict) -> None:
    print(f"num_samples: {metrics['num_samples']}")
    print(f"num_classes: {metrics['num_classes']}")
    print(f"top1_micro: {metrics['top1_micro']:.6f}")
    print(f"top1_macro: {metrics['top1_macro']:.6f}")
    for key in sorted(
        k
        for k in metrics
        if k.startswith("top") and k.endswith("_micro") and not k.startswith("top1_")
    ):
        print(f"{key}: {metrics[key]:.6f}")
    for key in sorted(
        k
        for k in metrics
        if k.startswith("top") and k.endswith("_macro") and not k.startswith("top1_")
    ):
        print(f"{key}: {metrics[key]:.6f}")
    print(f"macro_precision: {metrics['macro_precision']:.6f}")
    print(f"macro_recall: {metrics['macro_recall']:.6f}")
    print(f"macro_f1: {metrics['macro_f1']:.6f}")
    print(f"weighted_f1: {metrics['weighted_f1']:.6f}")
    print(f"support_min: {metrics['support_min']}")
    print(f"support_max: {metrics['support_max']}")
    print(f"support_mean: {metrics['support_mean']:.2f}")


def _plot_confusion_matrix(
    conf: np.ndarray,
    labels: list[str],
    output_path: str,
    *,
    cmap: str = "Greys",
) -> None:
    import matplotlib.pyplot as plt

    num_classes = len(labels)
    side = max(2.0, 0.2 * num_classes)
    fig, ax = plt.subplots(figsize=(side, side))
    ax.imshow(conf, vmin=0.0, vmax=1.0, cmap=cmap, aspect="equal")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=9, rotation=90, ha="center")
    ax.set_yticklabels(labels, fontsize=9)
    ax.tick_params(axis="both", which="both", length=3, width=1.5)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    fig.tight_layout(pad=0.3)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate confusion matrix and classification metrics for a checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the checkpoint run directory (e.g., results/updated2_symbolic100).",
    )
    parser.add_argument(
        "--which",
        default="latest",
        help="Checkpoint selection: latest, best, or an integer step.",
    )
    parser.add_argument(
        "--params",
        default="auto",
        choices=("auto", "ema", "raw"),
        help="Which parameters to evaluate: ema, raw, or auto (prefer ema).",
    )
    parser.add_argument(
        "--support-mode",
        default="b3s",
        choices=("auto", "b3s", "b3s_prior"),
        help="How to choose support masks (auto from data or B1S/B2S/B3S presets).",
    )
    parser.add_argument(
        "--support-max-draws",
        type=int,
        default=5_000,
        help="Maximum draws when sampling B3S support masks from the prior.",
    )
    parser.add_argument(
        "--support-lambda",
        type=float,
        default=None,
        help="Fixed mask-prior hyperparameter (lambda/u) to use when building auto support.",
    )
    parser.add_argument(
        "--num-support-classes",
        type=int,
        default=100,
        help="Number of support classes (unique masks) to evaluate (auto mode only).",
    )
    parser.add_argument(
        "--num-per-class",
        type=int,
        default=50_000,
        help="Approximate number of samples per support class to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for dataset generation.",
    )
    parser.add_argument(
        "--sim-batch-size",
        type=int,
        default=50_000,
        help="Batch size for simulator sampling.",
    )
    parser.add_argument(
        "--pred-batch-size",
        type=int,
        default=5000,
        help="Batch size for prediction evaluation.",
    )
    parser.add_argument(
        "--simulator-index",
        type=int,
        default=0,
        help="Index of the simulator acquisition scheme to use.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Top-k accuracy to report.",
    )
    parser.add_argument(
        "--classifier",
        default="none",
        choices=("none", "b3s_mlp", "all_mlp", "all_conv_mlp"),
        help="Train and evaluate a baseline MLP classifier instead of log-prob scores.",
    )
    parser.add_argument(
        "--clf-steps",
        type=int,
        default=500_000,
        help="Number of training steps for the classifier.",
    )
    parser.add_argument(
        "--clf-batch-size",
        type=int,
        default=2048,
        help="Batch size for classifier training.",
    )
    parser.add_argument(
        "--clf-lr",
        type=float,
        default=5e-4,
        help="Learning rate for classifier training.",
    )
    parser.add_argument(
        "--clf-hidden-size",
        type=int,
        default=512,
        help="Hidden size for classifier MLP.",
    )
    parser.add_argument(
        "--clf-num-layers",
        type=int,
        default=8,
        help="Number of layers for classifier MLP.",
    )
    parser.add_argument(
        "--clf-log-every",
        type=int,
        default=500,
        help="Print classifier loss every N steps (0 to disable).",
    )
    parser.add_argument(
        "--save-figure",
        default=None,
        help="Optional path to save a confusion matrix figure (.svg, .png, ...).",
    )
    parser.add_argument(
        "--save-counts",
        default=None,
        help="Optional path to save raw confusion counts (.npy or .csv).",
    )
    parser.add_argument(
        "--save-matrix",
        default=None,
        help="Optional path to save normalized confusion matrix (.npy or .csv).",
    )
    parser.add_argument(
        "--save-metrics",
        default=None,
        help="Optional path to save metrics as JSON.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print metrics as JSON instead of plain text.",
    )
    args = parser.parse_args()

    checkpoint_path = os.path.abspath(args.checkpoint)
    which = _parse_which(args.which)
    checkpoint, model, simulator = load_checkpoint(checkpoint_path, which=which)
    chosen = _select_params(checkpoint, model, args.params)
    model.eval()

    sim_fn = simulator[args.simulator_index]
    sim_type = model.tokenizer.simulator
    num_support = args.num_support_classes
    support_labels: list[str] | None = None
    support_masks = None
    support_masks_np = None
    mask_prior = None
    use_classifier = args.classifier != "none"
    is_all_conv = args.classifier == "all_conv_mlp"
    support_lambda = args.support_lambda

    if args.support_mode == "auto" and support_lambda is None:
        ckpt_lower = checkpoint_path.lower()
        support_lambda = 0.4 if "conv" in ckpt_lower else 0.99
    support_lambda_hyper = (
        None
        if support_lambda is None
        else jnp.asarray([support_lambda], dtype=jnp.float32)
    )

    if args.classifier == "b3s_mlp" and args.support_mode not in ("b3s", "b3s_prior"):
        raise ValueError("b3s_mlp requires --support-mode b3s or b3s_prior.")

    if args.support_mode in ("b3s", "b3s_prior") or use_classifier:
        cfg = load_cfg(checkpoint_path)
        mask_prior = _build_mask_prior(sim_type, cfg)

    if args.support_mode == "b3s":
        support_masks_np, support_labels = _b3s_support_masks(sim_type)
        support_masks = jnp.asarray(support_masks_np)
        num_support = support_masks.shape[0]
    elif args.support_mode == "b3s_prior":
        support_masks_np, support_labels = _b3s_support_masks_from_prior(
            sim_type,
            args.seed,
            args.support_max_draws,
            mask_prior=mask_prior,
        )
        support_masks = jnp.asarray(support_masks_np)
        num_support = support_masks.shape[0]
    else:
        if args.num_support_classes <= 0:
            raise ValueError("num-support-classes must be > 0 in auto mode.")
        if use_classifier or support_lambda_hyper is not None:
            num_samples = num_support * args.num_per_class
            _, probe_true_masks, _, _ = _build_dataset(
                sim_fn,
                num_samples=num_samples,
                seed=args.seed,
                sim_batch_size=args.sim_batch_size,
                mask_prior_hyperparameter=support_lambda_hyper,
            )
            support_masks, support_masks_np, support_counts, true_idx, _valid = (
                _compute_support_masks(probe_true_masks, num_support_classes=num_support)
            )
            num_support = support_masks.shape[0]
            support_labels = _support_mask_labels(
                support_masks_np,
                model.tokenizer.simulator.model_types,
                model.tokenizer.simulator.noise_types,
            )
            order = _grouped_support_order(support_labels)
            support_masks_np, support_labels, support_counts, true_idx = _reorder_support(
                support_masks_np, support_labels, support_counts, true_idx, order
            )
            support_masks = jnp.asarray(support_masks_np)

    if use_classifier:
        sampler = _build_batch_sampler(
            sim_fn, support_masks=support_masks, mask_prior=mask_prior
        )
        train_batch_size = args.clf_batch_size
        simulation_device = jax.devices("cpu")[0]
        simulation_batch_size = 2048
        buffer_size = 2048 * 128
        if is_all_conv:
            gpus = jax.devices("gpu")
            if not gpus:
                raise RuntimeError("all_conv_mlp requires a GPU for simulation.")
            simulation_device = gpus[0]
            train_batch_size = min(train_batch_size, 1024)
            buffer_size = 2048 * 256
        graphdef, params, state = _train_classifier(
            sim_fn,
            support_masks,
            mask_prior_dim=mask_prior.mask_prior_dim,
            num_steps=args.clf_steps,
            batch_size=train_batch_size,
            lr=args.clf_lr,
            hidden_size=args.clf_hidden_size,
            num_layers=args.clf_num_layers,
            seed=args.seed,
            log_every=args.clf_log_every,
            simulation_device=simulation_device,
            simulation_batch_size=simulation_batch_size,
            buffer_size=buffer_size,
        )

    use_support_sampling = args.support_mode in ("b3s", "b3s_prior") or use_classifier
    num_samples = num_support * args.num_per_class
    eval_seed = args.seed + 1 if use_classifier and args.support_mode == "auto" else args.seed
    xs, true_masks, mask_priors, acqs = _build_dataset(
        sim_fn,
        num_samples=num_samples,
        seed=eval_seed,
        sim_batch_size=args.sim_batch_size,
        support_masks=support_masks if use_support_sampling else None,
        mask_prior=mask_prior if use_support_sampling else None,
    )

    if support_masks is not None:
        true_idx, valid = _label_by_support_masks(true_masks, support_masks_np)
        support_counts = np.bincount(true_idx[valid], minlength=num_support)
    else:
        support_masks, support_masks_np, support_counts, true_idx, valid = (
            _compute_support_masks(true_masks, num_support_classes=num_support)
        )
        num_support = support_masks.shape[0]
        support_labels = _support_mask_labels(
            support_masks_np,
            model.tokenizer.simulator.model_types,
            model.tokenizer.simulator.noise_types,
        )
        order = _grouped_support_order(support_labels)
        support_masks_np, support_labels, support_counts, true_idx = _reorder_support(
            support_masks_np, support_labels, support_counts, true_idx, order
        )
        support_masks = jnp.asarray(support_masks_np)

    topk = args.topk
    if topk > num_support:
        print(
            f"topk ({topk}) exceeds number of classes ({num_support}), "
            f"clamping to {num_support}."
        )
        topk = num_support
    xs = xs[valid]
    mask_priors = mask_priors[valid]
    acqs = jax.tree_util.tree_map(lambda x: x[valid], acqs)
    true_idx = true_idx[valid]

    if use_classifier:
        classifier = nnx.merge(graphdef, params, state)
        classifier.eval()
        pred_idx, pred_topk = _predict_classifier_indices(
            classifier,
            acqs,
            xs,
            mask_priors,
            topk=topk,
            pred_batch_size=args.pred_batch_size,
        )
    else:
        pred_idx, pred_topk = _predict_indices(
            model,
            support_masks,
            acqs,
            xs,
            mask_priors,
            topk=topk,
            pred_batch_size=args.pred_batch_size,
        )

    counts, conf = _confusion_matrices(
        true_idx=true_idx,
        pred_idx=pred_idx,
        num_classes=support_masks.shape[0],
    )
    metrics = _compute_metrics(
        counts=counts,
        conf=conf,
        pred_topk=pred_topk,
        true_idx=true_idx,
        topk=topk,
    )
    metrics["classifier"] = args.classifier
    if args.support_mode == "auto":
        metrics["support_lambda"] = support_lambda
    metrics["params_used"] = chosen
    metrics["support_counts"] = support_counts.tolist()
    metrics["support_labels"] = support_labels

    _save_array(args.save_counts, counts)
    _save_array(args.save_matrix, conf)
    if args.save_figure:
        _plot_confusion_matrix(conf, support_labels, args.save_figure)

    if args.save_metrics:
        Path(args.save_metrics).write_text(json.dumps(metrics, indent=2))

    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        print(f"params_used: {metrics['params_used']}")
        _print_metrics(metrics)


if __name__ == "__main__":
    main()
