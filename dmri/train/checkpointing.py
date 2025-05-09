import logging
import os
from typing import Any

import numpy as np
import orbax.checkpoint as ocp
from flax.training import orbax_utils


class CheckpointManager:
    def __init__(
        self,
        ckpt_dir: str,
        max_to_keep: int = 5,
        keep_best: bool = True,
        recovery_threshold: float = float("inf"),
        continue_training: bool = False,
        use_async: bool = True,
    ):
        """
        Initialize the checkpoint manager.

        Args:
            ckpt_dir: Directory to store checkpoints
            max_to_keep: Maximum number of checkpoints to keep
            keep_best: Whether to keep the best model separately
            recovery_threshold: Loss threshold for automatic recovery
            continue_training: Whether to continue training from existing checkpoints
            use_async: Whether to use AsyncCheckpointer for non-blocking checkpoints
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        self.ckpt_dir = ckpt_dir
        self.best_ckpt_dir = os.path.join(ckpt_dir, "best")
        os.makedirs(self.best_ckpt_dir, exist_ok=True)
        self.continue_training = continue_training

        # Create checkpointer with proper handler
        handler = ocp.PyTreeCheckpointHandler()
        self.checkpointer = ocp.Checkpointer(handler)
        if use_async:
            try:
                # Use async configuration if available
                self.checkpointer = ocp.AsyncCheckpointer(handler)
                logging.info("Using async checkpoint handler")
            except (ImportError, AttributeError) as e:
                logging.warning(
                    f"AsyncCheckpointHandler not available: {e}. Using standard PyTreeCheckpointHandler instead."
                )

        # Create checkpoint manager options
        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            create=not continue_training,  # Don't delete existing checkpoints if continuing
        )

        # Create checkpoint managers
        self.manager = ocp.CheckpointManager(
            self.ckpt_dir,
            self.checkpointer,
            options=options,
        )

        # Best checkpoint manager
        best_options = ocp.CheckpointManagerOptions(
            max_to_keep=1, create=not continue_training
        )
        self.best_manager = ocp.CheckpointManager(
            self.best_ckpt_dir,
            self.checkpointer,
            options=best_options,
        )

        # For tracking best model
        self.keep_best = keep_best
        self.best_loss = float("inf")

        # For recovery
        self.recovery_threshold = recovery_threshold
        self.prev_loss = None

        if self.continue_training:
            self.latest_step = self.manager.latest_step() or 0
            logging.info(f"Continuing from step {self.latest_step}")

    def save(
        self,
        step: int,
        params: Any,
        optimizer_state: Any,
        loss: float = None,
        params_ema: Any = None,
    ):
        """
        Save a checkpoint.

        Args:
            step: Current training step
            params: Model parameters
            optimizer_state: State of the optimizer
            loss: Current training loss (optional)
            params_ema: EMA model parameters (optional)
        """
        # Save regular checkpoint
        if loss is None:
            loss = float("inf")
        ckpt = {
            "params": params,
            "optimizer_state": optimizer_state,
            "step": step,
            "loss": loss,
        }

        # Include EMA params if provided
        if params_ema is not None:
            ckpt["params_ema"] = params_ema

        save_args = orbax_utils.save_args_from_target(ckpt)

        # Save with progress reporting
        self.manager.save(step, ckpt, save_kwargs={"save_args": save_args})
        logging.info(f"Saved checkpoint at step {step}")

        # Save best model if needed
        if self.keep_best and loss is not None and loss < self.best_loss:
            logging.info(f"New best model with loss {loss} (previous {self.best_loss})")
            self.best_loss = loss
            self.best_manager.save(0, ckpt, save_kwargs={"save_args": save_args})

    def restore(
        self,
        step: int,
        params: Any,
        optimizer_state: Any,
        from_best: bool = False,
        params_ema: Any = None,
    ):
        """
        Restore a checkpoint.

        Args:
            step: Step to restore from. If None, restores the latest checkpoint.
            params: Existing parameters structure to match against
            optimizer_state: Existing optimizer state structure to match against
            from_best: Whether to restore from the best checkpoint
            params_ema: Existing EMA parameters structure to match against (optional)

        Returns:
            The restored checkpoint dict or None if no checkpoint exists
        """
        manager = self.best_manager if from_best else self.manager

        # Determine which step to restore
        if step is None:
            step = manager.latest_step()
            if step is None:
                logging.warning("No checkpoint found to restore.")
                return None

        # Create a skeleton abstract structure for the checkpoint
        abstract_ckpt = {
            "params": params,
            "optimizer_state": optimizer_state,  # This will be fully restored
            "step": step,  # This will be fully restored
            "loss": float("inf"),  # This will be fully restored
        }

        # Include EMA params in the abstract structure if provided
        if params_ema is not None:
            abstract_ckpt["params_ema"] = params_ema

        # Construct restore args from the abstract checkpoint structure
        restore_args = orbax_utils.restore_args_from_target(abstract_ckpt)

        # Restore the checkpoint
        restored_ckpt = manager.restore(
            step, abstract_ckpt, restore_kwargs={"restore_args": restore_args}
        )
        return restored_ckpt

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

    def wait_until_finished(self):
        """
        Wait for any pending async checkpoint operations to complete.
        This should be called before exiting to ensure all checkpoints are saved.
        """
        if hasattr(self.checkpointer, "wait_until_finished"):
            logging.info("Waiting for async checkpoint operations to complete...")
            self.checkpointer.wait_until_finished()
            logging.info("All async checkpoint operations completed")
