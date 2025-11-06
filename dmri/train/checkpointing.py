import logging
import os
from typing import Any, Optional

import numpy as np
import orbax.checkpoint as ocp


class CheckpointManager:
    def __init__(
        self,
        ckpt_dir: str,
        max_to_keep: int = 10,
        keep_best: bool = True,
        recovery_threshold: float = float("inf"),
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
        self.best_loss = float("inf")
        self.recovery_threshold = recovery_threshold
        self.prev_loss: Optional[float] = None

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
        optimizer_state: Any,
        loss: float,
        params_ema: Any = None,
        model_state: Any = None,
        ema_state: Any = None,
        rng: Any = None,
    ) -> ocp.args.Composite:
        items: dict[str, ocp.args.CheckpointArgs] = {
            "params": ocp.args.StandardSave(params),
            "optimizer_state": ocp.args.StandardSave(optimizer_state),
            "step": ocp.args.JsonSave(step),
            "loss": ocp.args.JsonSave(loss),
        }
        if params_ema is not None:
            items["params_ema"] = ocp.args.StandardSave(params_ema)
        if model_state is not None:
            items["model_state"] = ocp.args.StandardSave(model_state)
        if ema_state is not None:
            items["ema_state"] = ocp.args.StandardSave(ema_state)
        if rng is not None:
            items["rng"] = ocp.args.ArraySave(rng)
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
    ) -> ocp.args.Composite:
        items: dict[str, ocp.args.CheckpointArgs] = {
            "params": ocp.args.StandardRestore(params),
            "optimizer_state": ocp.args.StandardRestore(optimizer_state),
            "step": ocp.args.JsonRestore(),
            "loss": ocp.args.JsonRestore(),
        }
        if params_ema is not None:
            items["params_ema"] = ocp.args.StandardRestore(params_ema)
        if model_state is not None:
            items["model_state"] = ocp.args.StandardRestore(model_state)
        if ema_state is not None:
            items["ema_state"] = ocp.args.StandardRestore(ema_state)
        if rng is not None:
            items["rng"] = ocp.args.ArrayRestore(rng)
        return ocp.args.Composite(**items)

    def save(
        self,
        step: int,
        params: Any,
        optimizer_state: Any,
        loss: float = float("inf"),
        params_ema: Any = None,
        model_state: Any = None,
        ema_state: Any = None,
        rng: Any = None,
    ) -> None:
        args = self._build_save_args(
            step=step,
            params=params,
            optimizer_state=optimizer_state,
            loss=loss,
            params_ema=params_ema,
            model_state=model_state,
            ema_state=ema_state,
            rng=rng,
        )
        self.manager.save(step, args=args)
        if self.keep_best and loss < self.best_loss:
            self.best_loss = loss
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
        )
        return manager.restore(target_step, args=args)

    def get_latest_step(self) -> Optional[int]:
        return self.manager.latest_step()

    def should_recover(self, current_loss: float) -> bool:
        if self.prev_loss is None:
            self.prev_loss = current_loss
            return False

        if (
            current_loss > self.recovery_threshold
            or np.isnan(current_loss)
            or np.isinf(current_loss)
        ):
            self.prev_loss = current_loss
            return True

        if current_loss > self.prev_loss * 2:
            self.prev_loss = current_loss
            return True

        self.prev_loss = current_loss
        return False

    def wait_until_finished(self) -> None:
        self.manager.wait_until_finished()
        self.best_manager.wait_until_finished()
