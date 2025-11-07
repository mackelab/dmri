from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from jax.typing import ArrayLike


@dataclass(frozen=True)
class MaskPriorSample:
    """Container holding sampled hyperparameters and the resulting mask."""

    hyperparameters: jax.Array
    model_mask: jax.Array


class MaskPrior:
    """Base class for hierarchical priors over simulator model masks."""

    mask_prior_dim: int = 1

    def __init__(
        self,
        num_model_components: int,
        num_noise_components: int,
    ) -> None:
        if num_model_components <= 0:
            raise ValueError("num_model_components must be positive.")
        if num_noise_components < 0:
            raise ValueError("num_noise_components cannot be negative.")
        self.num_model_components = int(num_model_components)
        self.num_noise_components = int(num_noise_components)

    def sample(self, rng: jax.random.KeyArray) -> MaskPriorSample:
        """Sample hyperparameters and a mask conditioned on them."""
        rng_h, rng_mask = jax.random.split(rng)
        hyperparameters = self.sample_hyperparameters(rng_h)
        model_mask = self.sample_model_mask(rng_mask, hyperparameters)
        return MaskPriorSample(hyperparameters=hyperparameters, model_mask=model_mask)

    def log_prob(self, model_mask: ArrayLike, hyperparameters: ArrayLike) -> jax.Array:
        """Return log probability of a binary mask conditioned on hyperparameters."""
        mask = jnp.asarray(model_mask, dtype=jnp.bool_)
        model_mask_slice = mask[: self.num_model_components]
        noise_mask_slice = mask[self.num_model_components :]
        logp_models = self.log_prob_model_components(model_mask_slice, hyperparameters)
        logp_noise = self.log_prob_noise_components(noise_mask_slice, hyperparameters)
        return logp_models + logp_noise

    def prob(self, model_mask: ArrayLike, hyperparameters: ArrayLike) -> jax.Array:
        """Return probability of a binary mask conditioned on hyperparameters."""
        return jnp.exp(self.log_prob(model_mask, hyperparameters))

    def sample_model_mask(
        self,
        rng: jax.random.KeyArray,
        hyperparameters: ArrayLike,
    ) -> jax.Array:
        """Sample masks for model and noise components and concatenate them."""
        rng_models, rng_noise = jax.random.split(rng)
        model_mask = self.sample_model_components(rng_models, hyperparameters)
        noise_mask = self.sample_noise_components(rng_noise, hyperparameters)
        if self.num_noise_components == 0:
            return model_mask
        return jnp.concatenate([model_mask, noise_mask], axis=-1)

    def sample_noise_components(
        self,
        rng: jax.random.KeyArray,
        hyperparameters: ArrayLike,
    ) -> jax.Array:
        """Default noise prior: exactly one active component (if any exist)."""
        del hyperparameters
        if self.num_noise_components == 0:
            return jnp.zeros((0,), dtype=jnp.bool_)

        idx = jax.random.randint(
            rng,
            shape=(1,),
            minval=0,
            maxval=self.num_noise_components,
        )
        noise_mask = jnp.zeros((self.num_noise_components,), dtype=jnp.bool_)
        noise_mask = noise_mask.at[idx].set(True)
        return noise_mask

    def log_prob_noise_components(
        self,
        noise_mask: ArrayLike,
        hyperparameters: ArrayLike,
    ) -> jax.Array:
        del hyperparameters
        if self.num_noise_components == 0:
            return jnp.array(0.0)
        mask = jnp.asarray(noise_mask, dtype=jnp.bool_)
        num_active = jnp.sum(mask)
        log_uniform = -jnp.log(self.num_noise_components)
        return jnp.where(num_active == 1, log_uniform, -jnp.inf)

    def sample_model_components(
        self,
        rng: jax.random.KeyArray,
        hyperparameters: ArrayLike,
    ) -> jax.Array:
        """Override to draw model-component masks."""
        raise NotImplementedError

    def sample_hyperparameters(self, rng: jax.random.KeyArray) -> jax.Array:
        """Override to sample hyperparameters for the prior."""
        raise NotImplementedError

    def log_prob_model_components(
        self,
        model_mask: ArrayLike,
        hyperparameters: ArrayLike,
    ) -> jax.Array:
        """Override to evaluate log probability under the prior."""
        raise NotImplementedError


class BetaBernoulliMaskPrior(MaskPrior):
    """Independent Bernoulli masks driven by a shared Beta hyper-prior."""

    mask_prior_dim: int = 1

    def __init__(
        self,
        num_model_components: int,
        num_noise_components: int,
        alpha: float = 1.0,
        beta: float = 1.0,
        min_active_models: int = 1,
    ) -> None:
        super().__init__(num_model_components, num_noise_components)
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be positive.")
        if min_active_models < 0:
            raise ValueError("min_active_models cannot be negative.")
        if min_active_models > self.num_model_components:
            raise ValueError(
                "min_active_models cannot exceed available model components."
            )
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.min_active_models = int(min_active_models)
        if self.min_active_models > 0:
            combos = np.array(
                list(
                    itertools.combinations(
                        range(self.num_model_components),
                        self.min_active_models,
                    )
                ),
                dtype=np.int32,
            )
            if combos.size == 0:
                combos = np.zeros((1, 0), dtype=np.int32)
            forced_masks = np.zeros(
                (len(combos), self.num_model_components),
                dtype=np.bool_,
            )
            for idx, combo in enumerate(combos):
                forced_masks[idx, combo] = True
            self._forced_masks = jnp.asarray(forced_masks)
            self._num_forced_combos = forced_masks.shape[0]
        else:
            self._forced_masks = None
            self._num_forced_combos = 1

    def sample_hyperparameters(self, rng: jax.random.KeyArray) -> jax.Array:
        return jax.random.beta(
            rng,
            a=self.alpha,
            b=self.beta,
            shape=(self.mask_prior_dim,),
        )

    def sample_model_components(
        self,
        rng: jax.random.KeyArray,
        hyperparameters: ArrayLike,
    ) -> jax.Array:
        prob = jnp.clip(jnp.asarray(hyperparameters)[0], 0.0, 1.0)
        rng_mask, rng_force = jax.random.split(rng)
        base_mask = jax.random.bernoulli(
            rng_mask,
            p=prob,
            shape=(self.num_model_components,),
        )
        if self.min_active_models == 0:
            return base_mask.astype(jnp.bool_)

        # Force a subset of components to remain active.
        if self.min_active_models == 1:
            forced_idx = jax.random.randint(
                rng_force,
                shape=(1,),
                minval=0,
                maxval=self.num_model_components,
            )
        else:
            forced_idx = jax.random.choice(
                rng_force,
                self.num_model_components,
                shape=(self.min_active_models,),
                replace=False,
            )
        base_mask = base_mask.at[forced_idx].set(True)
        return base_mask.astype(jnp.bool_)

    def log_prob_model_components(
        self,
        model_mask: ArrayLike,
        hyperparameters: ArrayLike,
    ) -> jax.Array:
        mask = jnp.asarray(model_mask, dtype=jnp.bool_)
        prob = jnp.clip(jnp.asarray(hyperparameters)[0], 1e-6, 1 - 1e-6)
        log_on = jnp.log(prob)
        log_off = jnp.log1p(-prob)
        component_log = jnp.where(mask, log_on, log_off)

        if self.min_active_models == 0:
            return jnp.sum(component_log)

        forced_masks = self._forced_masks
        assert forced_masks is not None
        valid_combo = jnp.all(jnp.where(forced_masks, mask[None, :], True), axis=-1)
        masked_component_log = jnp.where(~forced_masks, component_log[None, :], 0.0)
        log_prob_cond = jnp.sum(masked_component_log, axis=-1)
        log_prob_cond = jnp.where(valid_combo, log_prob_cond, -jnp.inf)
        log_weight = -jnp.log(self._num_forced_combos)
        return logsumexp(log_prob_cond + log_weight)


class MarkovMaskPrior(MaskPrior):
    """Binary masks generated by a fixed-length Markov walk."""

    mask_prior_dim: int = 1

    def __init__(
        self,
        num_model_components: int,
        num_noise_components: int,
        transition_matrix: ArrayLike,
        start_prob: ArrayLike | None = None,
        walk_length: int | None = None,
    ) -> None:
        super().__init__(num_model_components, num_noise_components)
        tm = np.asarray(transition_matrix, dtype=np.float32)
        if tm.shape != (self.num_model_components, self.num_model_components):
            raise ValueError(
                "transition_matrix must have shape "
                f"({self.num_model_components}, {self.num_model_components})."
            )
        if np.any(tm < 0):
            raise ValueError("transition_matrix must be non-negative.")
        row_sums = tm.sum(axis=-1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized_tm = np.divide(
                tm,
                row_sums,
                out=np.full_like(tm, 1.0 / self.num_model_components),
                where=row_sums > 0,
            )
        self.transition_logits = jnp.log(jnp.asarray(normalized_tm) + 1e-12)

        if start_prob is None:
            start_prob = np.ones(self.num_model_components, dtype=np.float32)
        start_prob = np.asarray(start_prob, dtype=np.float32)
        if start_prob.shape != (self.num_model_components,):
            raise ValueError(
                f"start_prob must have shape ({self.num_model_components},)"
            )
        if np.any(start_prob < 0):
            raise ValueError("start_prob must be non-negative.")
        total_start = np.sum(start_prob)
        if total_start <= 0:
            raise ValueError("start_prob must sum to a positive value.")
        start_prob = start_prob / total_start
        self.start_logits = jnp.log(jnp.asarray(start_prob) + 1e-12)

        if walk_length is None:
            walk_length = self.num_model_components
        if walk_length <= 0:
            raise ValueError("walk_length must be positive.")
        self.walk_length = int(walk_length)

    def sample_hyperparameters(self, rng: jax.random.KeyArray) -> jax.Array:
        del rng
        return jnp.array([float(self.walk_length)], dtype=jnp.float32)

    def sample_model_components(
        self,
        rng: jax.random.KeyArray,
        hyperparameters: ArrayLike,
    ) -> jax.Array:
        # Clamp requested walk length
        walk_length = jnp.asarray(hyperparameters)[0]
        walk_length = jnp.clip(walk_length, 1, self.num_model_components).astype(
            jnp.int32
        )

        rng_start, rng_steps = jax.random.split(rng)
        start_idx = jax.random.categorical(rng_start, logits=self.start_logits)

        visited0 = jnp.zeros((self.num_model_components,), dtype=jnp.bool_)
        visited0 = visited0.at[start_idx].set(True)

        # Fixed maximum number of transitions for JIT: N-1
        max_transitions = self.num_model_components - 1
        step_keys = jax.random.split(rng_steps, max_transitions)

        very_neg = jnp.array(-1e9, dtype=self.transition_logits.dtype)

        def body(carry, key):
            current_idx, visited, steps_done = carry

            # Are we still allowed to take another step?
            need_more_steps = steps_done < (walk_length - 1)
            # Are there any unvisited nodes left?
            any_unvisited = jnp.any(~visited)

            def do_step(args):
                current_idx, visited, key = args
                # Mask visited nodes out of the transition logits
                row_logits = self.transition_logits[current_idx]
                masked_logits = jnp.where(visited, very_neg, row_logits)

                # If everything is masked (can happen if the row had prob only on visited nodes),
                # we fall back to a no-op by sampling current_idx again via cond below.
                # Detect availability:
                has_available = jnp.any(~jnp.isneginf(masked_logits)) & jnp.any(
                    masked_logits > very_neg * 0.5
                )

                def choose_next(_):
                    return jax.random.categorical(key, logits=masked_logits)

                def keep_current(_):
                    return current_idx

                next_idx = jax.lax.cond(
                    has_available, choose_next, keep_current, operand=None
                )

                visited = visited.at[next_idx].set(True)
                return next_idx, visited

            def no_step(args):
                current_idx, visited, _key = args
                return current_idx, visited

            take_step = need_more_steps & any_unvisited
            next_idx, new_visited = jax.lax.cond(
                take_step, do_step, no_step, operand=(current_idx, visited, key)
            )

            steps_done = steps_done + jnp.asarray(take_step, dtype=jnp.int32)
            return (next_idx, new_visited, steps_done), None

        (final_idx, visited, _), _ = jax.lax.scan(
            body,
            (start_idx, visited0, jnp.int32(0)),
            step_keys,
        )
        del final_idx
        return visited

    def log_prob_model_components(
        self,
        model_mask: ArrayLike,
        hyperparameters: ArrayLike,
    ) -> jax.Array:
        """
        Exact log-PMF of the visited set under a simple (no-revisit) Markov walk
        with length k = popcount(mask). Costs O(k^2 2^k).
        """
        mask = jnp.asarray(model_mask, dtype=jnp.bool_)
        # Clamp walk_length the same way as in sampling
        walk_length = jnp.asarray(hyperparameters)[0]
        walk_length = jnp.clip(walk_length, 1, self.num_model_components).astype(
            jnp.int32
        )

        k = jnp.int32(jnp.sum(mask))

        # If mask size doesn't match requested walk length, impossible under our generator
        # (we never revisit and we stop after exactly k steps).
        # Return -inf in that case.
        def impossible():
            return jnp.array(-jnp.inf, dtype=jnp.float32)

        def possible():
            # Indices of active components
            K = jnp.where(mask, size=self.num_model_components, fill_value=-1)[0]
            K = K[:k]  # take only the first k valid indices

            # Submatrix/logits restricted to K
            start_log = self.start_logits[K]  # (k,)
            trans_log = self.transition_logits[K[:, None], K[None, :]]  # (k, k)

            # DP over subsets: dp[subset, j] where subset in [0 .. 2^k-1], j in [0..k-1]
            # Initialize with -inf
            total_states = 1 << k
            dp = jnp.full((total_states, k), -jnp.inf, dtype=jnp.float32)

            # Base: subsets of size 1
            # For each j, subset = 1 << j
            base_rows = jnp.arange(k, dtype=jnp.int32)
            base_masks = 1 << base_rows
            dp = dp.at[base_masks, base_rows].set(start_log.astype(jnp.float32))

            # Iterate subset sizes from 2..k
            def subset_step(carry, size_s):
                dp = carry

                # Enumerate all subsets of this cardinality
                # We build all subset bitmasks of size `size_s`
                # Generate all bitmasks by brute force then filter by popcount.
                # (Efficient generation of k-combinations in JAX is nontrivial; this is clear and works for small k.)
                all_masks = jnp.arange(total_states, dtype=jnp.int32)
                popc = jnp.unpackbits(all_masks.view(jnp.uint8), axis=1)[:, -k:].sum(
                    axis=1
                )
                S_masks = all_masks[popc == size_s]

                def update_subset(dp, Smask):
                    # For each possible end j ∈ S:
                    js = jnp.arange(k, dtype=jnp.int32)
                    in_S = ((Smask >> js) & 1) == 1
                    js_in_S = js[in_S]

                    def update_end(dp, j):
                        Sj = Smask ^ (1 << j)  # S \ {j}
                        # predecessors i ∈ S\{j}
                        is_ = jnp.arange(k, dtype=jnp.int32)
                        in_Sj = ((Sj >> is_) & 1) == 1
                        preds = is_[in_Sj]

                        def agg_one(i):
                            return dp[Sj, i] + trans_log[i, j]

                        # If no predecessors (shouldn't happen for size>=2), keep -inf
                        val = jax.lax.cond(
                            preds.size > 0,
                            lambda _: jax.scipy.special.logsumexp(
                                jax.vmap(agg_one)(preds)
                            ),
                            lambda _: jnp.array(-jnp.inf, dtype=jnp.float32),
                            operand=None,
                        )
                        return dp.at[Smask, j].set(val)

                    # Reduce over all valid j
                    def body(dp, j):
                        return update_end(dp, j)

                    dp = jax.lax.fori_loop(
                        0, js_in_S.size, lambda idx, d: body(d, js_in_S[idx]), dp
                    )
                    return dp

                # Reduce over all subsets of this size
                def body(dp, idx):
                    return update_subset(dp, S_masks[idx])

                dp = jax.lax.fori_loop(0, S_masks.size, body, dp)
                return dp, None

                # end subset_step

            dp, _ = jax.lax.scan(subset_step, dp, jnp.arange(2, k + 1, dtype=jnp.int32))

            full_mask = (1 << k) - 1
            # Sum over last node
            logp = jax.scipy.special.logsumexp(dp[full_mask, :])
            return logp

        return jax.lax.cond(k == walk_length, possible, impossible)


def sample_mask_and_prior(
    mask_prior: MaskPrior,
    rng: jax.random.KeyArray,
) -> Tuple[jax.Array, jax.Array]:
    """Utility helper to sample `(mask_prior, mask)` pairs."""
    sample = mask_prior.sample(rng)
    return sample.hyperparameters, sample.model_mask
