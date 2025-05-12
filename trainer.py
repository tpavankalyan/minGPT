import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import os
import logging

# Assuming GPT model and GPTConfig are imported from mingpt.model
# from mingpt.model import GPT, GPTConfig
# from config import TrainingConfig, GlobalConfig # Import specific config classes for type hinting

logger = logging.getLogger(__name__)

class Trainer:
    """
    Handles the training and evaluation loop for a language model.
    Manages optimization, mixed precision, gradient accumulation, logging, and checkpointing.
    """
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 optimizer: torch.optim.Optimizer, loss_fn: nn.Module,
                 global_config, # Accepts GlobalConfig for unified access
                 lr_scheduler=None):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.config = global_config.training # Access training specific configurations
        self.output_dir = global_config.output_dir
        self.device = global_config.device
        self.best_val_loss = float('inf') # To track the best validation loss for saving
        self.epoch = 0 # Current epoch counter, updated in train() method

        self.model.to(self.device)

        # Initialize GradScaler for mixed precision training if enabled
        self.scaler = GradScaler() if self.config.mixed_precision else None

        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Using mixed precision: {self.config.mixed_precision}")
        logger.info(f"Effective batch size (per optimizer step): {self.config.batch_size * self.config.gradient_accumulation_steps}")

    def _run_epoch(self, data_loader: DataLoader, is_train: bool):
        """
        Runs a single epoch of training or validation.

        Args:
            data_loader (DataLoader): The DataLoader for the current phase (train/val).
            is_train (bool): True for training phase, False for validation.

        Returns:
            float: The average loss for the epoch.
        """
        self.model.train(is_train) # Set model to train or eval mode
        total_loss = 0.0
        num_batches = len(data_loader)

        desc = f"Epoch {self.epoch + 1} (Training)" if is_train else f"Epoch {self.epoch + 1} (Validation)"

        # Wrap with tqdm for a visually informative progress bar
        for i, batch in enumerate(tqdm(data_loader, desc=desc, dynamic_ncols=True)):
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)

            # Use autocast for automatic mixed precision if enabled
            with autocast(enabled=self.config.mixed_precision):
                logits, _ = self.model(x)
                # Reshape logits for CrossEntropyLoss: (N, C) where N is total items, C is num classes
                # For token prediction, logits are (batch_size * sequence_length, vocab_size),
                # targets are (batch_size * sequence_length)
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / self.config.gradient_accumulation_steps # Scale loss for accumulation

            if is_train:
                # Perform backward pass
                if self.scaler:
                    self.scaler.scale(loss).backward() # Scale loss for mixed precision
                else:
                    loss.backward()

                # Perform optimizer step only after accumulating gradients for N steps
                if (i + 1) % self.config.gradient_accumulation_steps == 0:
                    # Clip gradients if specified (important for stable training, especially with AMP)
                    if self.config.clip_grad_norm > 0:
                        if self.scaler:
                            self.scaler.unscale_(self.optimizer) # Must unscale gradients before clipping
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)

                    # Optimizer step
                    if self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update() # Update scaler for next iteration
                    else:
                        self.optimizer.step()

                    self.optimizer.zero_grad(set_to_none=True) # More efficient than zero_grad()

                    # Learning rate scheduler step (if configured to step per batch)
                    if self.lr_scheduler and getattr(self.lr_scheduler, 'step_per_batch', False):
                        self.lr_scheduler.step()

                # Log batch loss to WandB (unscale `loss` to get the actual loss value)
                current_loss_item = loss.item() * self.config.gradient_accumulation_steps
                total_loss += current_loss_item

                if (i + 1) % self.config.log_interval == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    wandb.log({
                        'train/batch_loss': current_loss_item,
                        'train/learning_rate': current_lr,
                        'global_step': self._get_global_step(i)
                    })
                    logger.debug(f"Global Step {self._get_global_step(i)} | Batch Loss: {current_loss_item:.4f} | LR: {current_lr:.6f}")
            else:
                # For validation, simply accumulate the loss
                total_loss += loss.item() * self.config.gradient_accumulation_steps # Unscale for actual loss value

        avg_loss = total_loss / num_batches
        return avg_loss

    def _get_global_step(self, batch_idx: int) -> int:
        """Calculates the global step for logging purposes."""
        # Adjusted calculation for gradient accumulation: global step increments only on optimizer steps
        optimizer_steps_in_epoch = (batch_idx + 1) // self.config.gradient_accumulation_steps
        return self.epoch * (len(self.train_loader) // self.config.gradient_accumulation_steps) + optimizer_steps_in_epoch

    def train(self):
        """
        Main training loop. Iterates through epochs, runs training and validation phases,
        logs metrics to WandB, and saves model checkpoints.
        """
        logger.info("Starting training loop...")

        for epoch in range(self.config.max_epochs):
            self.epoch = epoch # Update current epoch for use in _run_epoch and logging
            logger.info(f"--- Epoch {epoch + 1}/{self.config.max_epochs} ---")

            # Training phase
            train_loss = self._run_epoch(self.train_loader, is_train=True)
            logger.info(f"Epoch {epoch + 1} Training Loss: {train_loss:.4f}")
            wandb.log({'train/epoch_loss': train_loss, 'epoch': epoch + 1})

            # Learning rate scheduler step (if configured to step per epoch)
            if self.lr_scheduler and not getattr(self.lr_scheduler, 'step_per_batch', False):
                # Note: If using ReduceLROnPlateau, you might pass val_loss here
                self.lr_scheduler.step() # Or self.lr_scheduler.step(val_loss)

            # Validation phase
            if (epoch + 1) % self.config.eval_interval == 0:
                val_loss = self._run_epoch(self.val_loader, is_train=False)
                logger.info(f"Epoch {epoch + 1} Validation Loss: {val_loss:.4f}")
                wandb.log({'val/epoch_loss': val_loss, 'epoch': epoch + 1})

                # Checkpointing logic
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    logger.info(f"Validation loss improved to {self.best_val_loss:.4f}. Saving best model checkpoint.")
                    self._save_checkpoint(epoch + 1, is_best=True)
                else:
                    logger.info(f"Validation loss did not improve. Current best: {self.best_val_loss:.4f}")

                if (epoch + 1) % self.config.save_interval == 0:
                    self._save_checkpoint(epoch + 1, is_best=False) # Save periodic checkpoint

        logger.info("Training complete!")
        self._save_checkpoint(self.config.max_epochs, is_final=True) # Save final model state

        if wandb.run:
            wandb.finish() # End the WandB run gracefully


    def _save_checkpoint(self, epoch: int, is_best: bool = False, is_final: bool = False):
        """
        Saves the model, optimizer, and training state for resumption.

        Args:
            epoch (int): The current epoch number.
            is_best (bool): True if this is the best model so far based on validation loss.
            is_final (bool): True if this is the final checkpoint after training completes.
        """
        checkpoint_name = f"epoch_{epoch}"
        if is_best:
            checkpoint_name += "_best"
        elif is_final:
            checkpoint_name += "_final"

        checkpoint_path = os.path.join(self.output_dir, f"{checkpoint_name}.pth")

        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'training_config': self.config.__dict__ # Save training config for reproducibility
        }

        torch.save(state, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

        # Upload checkpoint to WandB as an artifact
        if wandb.run:
            # Log as an artifact for better versioning and management
            artifact = wandb.Artifact(name=f"{wandb.run.name}_model", type="model")
            artifact.add_file(checkpoint_path, name=f"model_{checkpoint_name}.pth")

            # Add aliases for easy identification
            aliases = ["latest"]
            if is_best:
                aliases.append("best")
            if is_final:
                aliases.append("final")

            wandb.run.log_artifact(artifact, aliases=aliases)
