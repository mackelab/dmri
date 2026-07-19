import logging
import math
import os
from typing import Any, Optional

import orbax.checkpoint as ocp

try:
    from absl import logging as absl_logging

    absl_logging.set_verbosity(absl_logging.ERROR)
    logging.getLogger("absl").setLevel(logging.ERROR)
except Exception:
    absl_logging = None
    logging.getLogger("absl").setLevel(logging.ERROR)


class CheckpointManager:
    def __init__(
        self,
        ckpt_dir: str,
        max_to_keep: int = 10,
        keep_best: bool = True,
        recovery_threshold: float | None = float("inf"),
        continue_training: bool = False,
    ) -> None:
        self.ckpt_dir = ckpt_dir
        self.best_ckpt_dir = os.path.join(ckpt_dir, "best")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.best_ckpt_dir, exist_ok=True)

        opts = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            create=not continue_training,
        )
        self.manager = ocp.CheckpointManager(self.ckpt_dir, options=opts)

        best_opts = ocp.CheckpointManagerOptions(
            max_to_keep=1,
            create=not continue_training,
        )
        self.best_manager = ocp.CheckpointManager(self.best_ckpt_dir, options=best_opts)

        self.keep_best = keep_best
        self.best_val_loss = float("inf")
        self.recovery_threshold = (
            float("inf") if recovery_threshold is None else recovery_threshold
        )
        self.prev_metric: Optional[float] = None

        if continue_training:
            latest = self.manager.latest_step()
            if latest is not None:
                logging.info(f"Continuing from step {latest}")
            else:
                logging.info("No existing checkpoints found; starting fresh.")

    def _build_save_args(
        self,
        *,
        step: int,
        params: Any,
        optimizer_state: Any | None,
        loss: float,
        val_loss: Optional[float] = None,
        params_ema: Any = None,
        model_state: Any = None,
        ema_state: Any = None,
        rng: Any = None,
    ) -> ocp.args.Composite:
        items: dict[str, ocp.args.CheckpointArgs] = {
            "params": ocp.args.StandardSave(params),  # type: ignore
            "optimizer_state": ocp.args.StandardSave(optimizer_state),  # type: ignore
            "step": ocp.args.JsonSave(step),  # type: ignore
            "loss": ocp.args.JsonSave(loss),  # type: ignore
        }
        if val_loss is not None:
            items["val_loss"] = ocp.args.JsonSave(val_loss)  # type: ignore
        if params_ema is not None:
            items["params_ema"] = ocp.args.StandardSave(params_ema)  # type: ignore
        if model_state is not None:
            items["model_state"] = ocp.args.StandardSave(model_state)  # type: ignore
        if ema_state is not None:
            items["ema_state"] = ocp.args.StandardSave(ema_state)  # type: ignore
        if rng is not None:
            items["rng"] = ocp.args.ArraySave(rng)  # type: ignore
        return ocp.args.Composite(**items)

    def _build_restore_args(
        self,
        *,
        params: Any,
        optimizer_state: Any,
        params_ema: Any = None,
        model_state: Any = None,
        ema_state: Any = None,
        rng: Any = None,
        partial_restore: bool = False,
    ) -> ocp.args.Composite:
        def pytree_restore(item: Any) -> ocp.args.CheckpointArgs:
            if not partial_restore:
                return ocp.args.StandardRestore(item)  # type: ignore
            restore_args = ocp.checkpoint_utils.construct_restore_args(item)
            return ocp.args.PyTreeRestore(
                item=item,
                restore_args=restore_args,
                partial_restore=True,
            )

        items: dict[str, ocp.args.CheckpointArgs] = {
            "params": pytree_restore(params),
            "step": ocp.args.JsonRestore(),
            "loss": ocp.args.JsonRestore(),
        }
        if optimizer_state is not None:
            items["optimizer_state"] = pytree_restore(optimizer_state)
        if params_ema is not None:
            items["params_ema"] = pytree_restore(params_ema)
        if model_state is not None:
            items["model_state"] = pytree_restore(model_state)
        if ema_state is not None:
            items["ema_state"] = pytree_restore(ema_state)
        if rng is not None:
            items["rng"] = ocp.args.ArrayRestore(rng)  # type: ignore
        return ocp.args.Composite(**items)

    def save(
        self,
        step: int,
        params: Any,
        optimizer_state: Any | None,
        loss: float = float("inf"),
        val_loss: Optional[float] = None,
        write_standard: bool = True,
        params_ema: Any = None,
        model_state: Any = None,
        ema_state: Any = None,
        rng: Any = None,
        partial_restore: bool = False,
    ) -> None:
        args = self._build_save_args(
            step=step,
            params=params,
            optimizer_state=optimizer_state,
            loss=loss,
            val_loss=val_loss,
            params_ema=params_ema,
            model_state=model_state,
            ema_state=ema_state,
            rng=rng,
            partial_restore=partial_restore,
        )
        if write_standard:
            self.manager.save(step, args=args)
        if self.keep_best and val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_manager.save(step, args=args)

    def restore(
        self,
        step: Optional[int],
        params: Any,
        optimizer_state: Any,
        from_best: bool = False,
        params_ema: Any = None,
        model_state: Any = None,
        ema_state: Any = None,
        rng: Any = None,
        partial_restore: bool = False,
    ):
        manager = self.best_manager if from_best else self.manager
        target_step = manager.latest_step() if step is None else step
        if target_step is None:
            logging.warning("No checkpoint found to restore.")
            return None

        args = self._build_restore_args(
            params=params,
            optimizer_state=optimizer_state,
            params_ema=params_ema,
            model_state=model_state,
            ema_state=ema_state,
            rng=rng,
            partial_restore=partial_restore,
        )
        return manager.restore(target_step, args=args)

    def get_latest_step(self) -> Optional[int]:
        return self.manager.latest_step()

    def should_recover(
        self, current_metric: Optional[float], recovery_threshold: float | None = None
    ) -> bool:
        if current_metric is None:
            return False
        recovery_threshold = (
            self.recovery_threshold
            if recovery_threshold is None
            else recovery_threshold
        )
        if not math.isfinite(recovery_threshold) or recovery_threshold <= 1.0:
            if math.isfinite(current_metric):
                self.prev_metric = current_metric
            return False
        if not math.isfinite(current_metric):
            return True
        if self.prev_metric is None:
            self.prev_metric = current_metric
            return False

        trigger_value = self.prev_metric + abs(self.prev_metric) * (
            recovery_threshold - 1.0
        )
        if current_metric > trigger_value:
            return True

        self.prev_metric = current_metric
        return False

    def wait_until_finished(self) -> None:
        self.manager.wait_until_finished()
        self.best_manager.wait_until_finished()
