import os
import torch
import torch.nn as nn
import numpy as np
import logging
import wandb
from datetime import datetime
import math # Needed for cosine annealing LR scheduler
import sys

# Add the parent directory of mingpt to the Python path
# This assumes your directory structure is like:
# project_root/
# ├── main.py
# ├── trainer.py
# ├── dataset.py
# ├── config.py
# └── mingpt/
#     ├── __init__.py
#     ├── model.py
#     ├── utils.py # Assuming CfgNode is here
#     └── bpe_new.py # Assuming BPETokenizerReasoning is here

# Get the directory of the current script (assuming this script is in project_root)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory (project_root) to the sys.path
# Adjust this path if your structure is different
project_root_dir = current_dir # Assuming main.py is in project_root, and mingpt is a subdir
sys.path.insert(0, project_root_dir)


# Imports from local project modules
from config import GlobalConfig # All configurations are nested under GlobalConfig
from dataset import PackedReasoningDataset, collate_batch_packed, load_and_prepare_datasets

# Import GPT and CN (CfgNode)
try:
    from mingpt.model import GPT
    # Assuming CfgNode is in mingpt.utils
    from mingpt.utils import CfgNode as CN
    # Assuming BPETokenizerReasoning is in mingpt.bpe_new
    from mingpt.bpe_new import BPETokenizerReasoning
except ImportError as e:
    print(f"Error importing mingpt components: {e}")
    print("Please ensure that the 'mingpt' directory is correctly placed and accessible in your Python path.")
    print("Current sys.path:", sys.path)
    sys.exit(1)

from trainer import Trainer # Import Trainer after other components

# Configure global logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_seed(seed: int):
    """Set seeds for reproducibility across PyTorch, NumPy, and standard random."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # For reproducible results, these settings can sometimes impact performance slightly
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    """
    Main function to set up and run the GPT training process.
    """
    # --- 1. Configuration Loading ---
    global_config = GlobalConfig() # Loads default configuration

    # --- Optional: Override config values here, e.g., from command-line arguments ---
    # Example: global_config.training.max_epochs = 100
    # Example: global_config.model.model_type = 'gpt-base'
    # Example: global_config.wandb.mode = 'disabled' # To disable WandB logging

    # Create a unique run name for WandB using current timestamp
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    global_config.wandb.name = f"{global_config.wandb.name}-{current_time}"

    # Set random seeds for reproducibility
    setup_seed(global_config.training.seed)
    logger.info(f"Using device: {global_config.device}")
    logger.info(f"Random seed set to: {global_config.training.seed}")

    # --- 2. Weights & Biases Initialization ---
    if global_config.wandb.mode != 'disabled':
        wandb.init(
            project=global_config.wandb.project,
            name=global_config.wandb.name,
            entity=global_config.wandb.entity,
            mode=global_config.wandb.mode,
            group=global_config.wandb.group,
            tags=global_config.wandb.tags,
            config={
                'data': global_config.data.__dict__, # Log all config parameters to WandB
                'model': global_config.model.__dict__,
                'training': global_config.training.__dict__
            }
        )
        logger.info(f"Initialized WandB run: {wandb.run.url}")
    else:
        logger.info("WandB logging is disabled for this run.")

    # --- 3. Tokenizer and Dataset Preparation ---
    logger.info("Initializing tokenizer...")
    try:
        # Ensure BPETokenizerReasoning is accessible in your Python environment
        tokenizer = BPETokenizerReasoning()
        logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    except ImportError:
        logger.error("Could not import BPETokenizerReasoning. Please ensure 'mingpt.bpe_new' is correctly set up and accessible in your Python path.")
        return
    except Exception as e:
        logger.error(f"Failed to initialize tokenizer: {e}")
        return

    logger.info("Loading and preparing datasets...")
    try:
        train_dataset, val_dataset = load_and_prepare_datasets(
            data_config=global_config.data,
            tokenizer=tokenizer
        )
    except RuntimeError as e: # Catch custom RuntimeError from load_and_prepare_datasets
        logger.error(f"Error during dataset loading or preparation: {e}")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred during dataset loading: {e}")
        return

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=global_config.training.batch_size,
        shuffle=True,
        num_workers=global_config.training.num_workers,
        collate_fn=collate_batch_packed,
        pin_memory=True # Enables faster data transfer to GPU
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=global_config.training.batch_size,
        shuffle=False, # No need to shuffle validation data
        num_workers=global_config.training.num_workers,
        collate_fn=collate_batch_packed,
        pin_memory=True
    )
    logger.info(f"Train dataset size (packed examples): {len(train_dataset)}")
    logger.info(f"Validation dataset size (packed examples): {len(val_dataset)}")
    logger.info(f"Train loader batches: {len(train_loader)}")
    logger.info(f"Validation loader batches: {len(val_loader)}")

    # --- 4. GPT Model Initialization ---
    logger.info(f"Initializing GPT model: {global_config.model.model_type}...")

    # Correctly create the model configuration using CN and GPT.get_default_config
    model_config = GPT.get_default_config()

    # Set parameters from the GlobalConfig dataclass onto the CN object
    model_config.vocab_size = tokenizer.vocab_size
    model_config.block_size = global_config.data.block_size
    model_config.model_type = global_config.model.model_type # This will trigger merging in GPT.__init__

    # If you are NOT using a predefined model_type from model.py
    # AND you have specified n_layer, n_head, n_embd in your GlobalConfig,
    # you would set them explicitly here:
    # if global_config.model.model_type is None:
    #     model_config.n_layer = global_config.model.n_layer
    #     model_config.n_head = global_config.model.n_head
    #     model_config.n_embd = global_config.model.n_embd
    # Note: The current model.py GPT.__init__ handles merging predefined types.
    # If you want to use custom layer/head/embd counts *without* a predefined type,
    # ensure your GlobalConfig.ModelConfig correctly sets those, and the logic above
    # or within GPT.__init__ handles it. The provided model.py seems to expect
    # either model_type OR n_layer, n_head, n_embd to be set, not necessarily both
    # if model_type is used. The current setup where main.py sets model_type,
    # vocab_size, and block_size on the default config should work with your model.py.


    model = GPT(model_config) # Pass the CN object to the GPT constructor
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    # --- 5. Optimizer Setup ---
    # Use the configure_optimizers method from the GPT model
    optimizer = model.configure_optimizers(global_config.training)

    # --- 6. Learning Rate Scheduler Setup (Cosine Decay with Warmup) ---
    # Calculate total number of optimizer steps for cosine decay
    # This is the number of times optimizer.step() will be called
    # Need to handle cases where train_loader is empty
    steps_per_epoch = len(train_loader) // global_config.training.gradient_accumulation_steps
    if steps_per_epoch == 0:
        logger.warning("Steps per epoch is 0. Check batch size and gradient accumulation steps relative to dataset size.")
        total_optimizer_steps = 0
    else:
        total_optimizer_steps = steps_per_epoch * global_config.training.max_epochs


    lr_scheduler = None
    if total_optimizer_steps > 0:
        if global_config.training.warmup_steps > 0:
            def lr_lambda(current_step):
                if current_step < global_config.training.warmup_steps:
                    return float(current_step) / float(max(1, global_config.training.warmup_steps))
                # Apply cosine decay after warmup period
                progress = float(current_step - global_config.training.warmup_steps) / float(max(1, total_optimizer_steps - global_config.training.warmup_steps))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            setattr(lr_scheduler, 'step_per_batch', True) # Custom flag to indicate this scheduler steps per batch
            logger.info(f"Using LambdaLR scheduler with {global_config.training.warmup_steps} warmup steps.")
        else:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_optimizer_steps)
            setattr(lr_scheduler, 'step_per_batch', True)
            logger.info("Using CosineAnnealingLR scheduler without warmup.")
    else:
        logger.warning("Total optimizer steps is zero. Skipping LR scheduler setup.")


    # --- 7. Loss Function ---
    loss_fn = torch.nn.CrossEntropyLoss()

    # --- 8. Trainer Initialization and Training Execution ---
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        global_config=global_config, # Pass the entire global config object
        lr_scheduler=lr_scheduler
    )

    # Start the training process
    trainer.train()

if __name__ == "__main__":
    main()
