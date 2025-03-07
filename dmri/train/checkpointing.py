import os
from typing import Any, Dict
import orbax.checkpoint as ocp
from flax.training import orbax_utils
import numpy as np


class CheckpointManager:
    def __init__(
        self,
        ckpt_dir: str,
        max_to_keep: int = 3,
        keep_best: bool = True,
        recovery_threshold: float = float("inf"),
        continue_training: bool = False,
    ):
        """
        Initialize the checkpoint manager.

        Args:
            ckpt_dir: Directory to store checkpoints
            max_to_keep: Maximum number of checkpoints to keep
            keep_best: Whether to keep the best model separately
            recovery_threshold: Loss threshold for automatic recovery
            continue_training: Whether to continue training from existing checkpoints
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        self.ckpt_dir = ckpt_dir
        self.best_ckpt_dir = os.path.join(ckpt_dir, "best")
        os.makedirs(self.best_ckpt_dir, exist_ok=True)
        self.continue_training = continue_training

        # Create checkpointer
        self.checkpointer = ocp.PyTreeCheckpointer()

        # Create checkpoint manager options - updated to new API
        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            create=not continue_training,  # Don't delete existing checkpoints if continuing
        )

        # Create checkpoint managers with updated API
        self.manager = ocp.CheckpointManager(
            directory=self.ckpt_dir,
            checkpointers={"checkpoint": self.checkpointer},
            options=options,
        )

        best_options = ocp.CheckpointManagerOptions(
            max_to_keep=1, create=not continue_training
        )
        self.best_manager = ocp.CheckpointManager(
            directory=self.best_ckpt_dir,
            checkpointers={"checkpoint": self.checkpointer},
            options=best_options,
        )

        # For tracking best model
        self.keep_best = keep_best
        self.best_loss = float("inf")

        # For recovery
        self.recovery_threshold = recovery_threshold
        self.prev_loss = None

        # If continuing training, load best loss from existing checkpoint
        if continue_training:
            self._load_training_state()

    def _load_training_state(self):
        """Load training state when continuing training"""
        # Check if best checkpoint exists and load best loss
        if self.keep_best and self.best_manager.latest_step() is not None:
            best_ckpt = self.restore(from_best=True)
            if best_ckpt and "metrics" in best_ckpt and "loss" in best_ckpt["metrics"]:
                self.best_loss = best_ckpt["metrics"]["loss"]
                print(f"Continuing training with best loss: {self.best_loss:.4f}")

    def save(
        self,
        step: int,
        model: Any,
        params: Any,
        optimizer_state: Any,
        metrics: Dict = None,
    ):
        """
        Save a checkpoint.

        Args:
            step: Current training step
            model: The model object
            params: Model parameters
            optimizer_state: State of the optimizer
            metrics: Dictionary with metrics like loss, accuracy etc.
        """
        # Save regular checkpoint
        ckpt = {"params": params, "optimizer_state": optimizer_state, "step": step}

        # Add metrics if available
        if metrics:
            ckpt["metrics"] = metrics

        save_args = orbax_utils.save_args_from_target(ckpt)

        # Updated save API
        self.manager.save(
            step,
            {"checkpoint": ckpt},
            save_kwargs={"checkpoint": {"save_args": save_args}},
        )

        # Save best model if needed
        if self.keep_best and metrics and "loss" in metrics:
            current_loss = metrics["loss"]
            if current_loss < self.best_loss:
                print(
                    f"New best model with loss: {current_loss:.4f} (previous: {self.best_loss:.4f})"
                )
                self.best_loss = current_loss
                self.best_manager.save(
                    0,
                    {"checkpoint": ckpt},
                    save_kwargs={"checkpoint": {"save_args": save_args}},
                )

    def restore(self, step: int = None, from_best: bool = False):
        """
        Restore a checkpoint.

        Args:
            step: Step to restore from, or None for latest
            from_best: Whether to restore from best checkpoint

        Returns:
            The restored checkpoint
        """
        manager = self.best_manager if from_best else self.manager

        if step is None:
            step = manager.latest_step()

        if step is None:
            print("No checkpoint found to restore.")
            return None

        # Updated restore API
        restored = manager.restore(step, items={"checkpoint": None})
        return restored["checkpoint"] if "checkpoint" in restored else None

    def get_latest_step(self):
        """
        Get the latest checkpoint step. Useful when continuing training.

        Returns:
            The latest step or 0 if no checkpoints exist
        """
        latest = self.manager.latest_step()
        return latest if latest is not None else 0

    def should_recover(self, current_loss):
        """
        Check if recovery is needed based on loss value.

        Args:
            current_loss: Current training loss

        Returns:
            Boolean indicating if recovery is needed
        """
        if self.prev_loss is None:
            self.prev_loss = current_loss
            return False

        # Check if loss exceeds threshold or has exploded
        if (
            current_loss > self.recovery_threshold
            or np.isnan(current_loss)
            or np.isinf(current_loss)
        ):
            print(f"Loss explosion detected ({current_loss}). Triggering recovery.")
            self.prev_loss = current_loss
            return True

        # Check for sudden large increases
        if current_loss > self.prev_loss * 2:  # Loss doubled
            print(
                f"Sudden loss increase detected: {self.prev_loss:.4f} -> {current_loss:.4f}. Triggering recovery."
            )
            self.prev_loss = current_loss
            return True

        self.prev_loss = current_loss
        return False
