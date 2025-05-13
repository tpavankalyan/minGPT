import os
import torch
from dataclasses import dataclass, field
from typing import List

@dataclass
class DataConfig:
    """Configuration for dataset loading and preparation."""
    dataset_name: str = "Pavankalyan/sorting-reasoning"
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    # filter_train_length: int = 4 # Uncomment and use if filtering is needed (e.g., filter out 'length'=4)
    block_size: int = 300 # The sequence length for the model's input

@dataclass
class ModelConfig:
    """Configuration for the GPT model architecture."""
    model_type: str = 'gpt-mini' # Choose from 'gpt-mini', 'gpt-micro', 'gpt-nano', 'gpt-base'
    # These will be overridden by actual tokenizer vocab size and block_size from DataConfig
    vocab_size: int = 50257
    block_size: int = 300
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 192

    def __post_init__(self):
        """Sets model-specific parameters based on model_type."""
        if self.model_type == 'gpt-mini':
            self.n_layer = 6
            self.n_head = 6
            self.n_embd = 192
        elif self.model_type == 'gpt-micro':
            self.n_layer = 4
            self.n_head = 4
            self.n_embd = 128
        elif self.model_type == 'gpt-nano':
            self.n_layer = 3
            self.n_head = 3
            self.n_embd = 48
        elif self.model_type == 'gpt-base':
            self.n_layer = 12
            self.n_head = 12
            self.n_embd = 768
        # Note: The GPT model in model.py has more types ('gpt2', 'openai-gpt', etc.)
        # You might want to expand this list in ModelConfig or rely solely on the model's internal mapping.
        # For now, we'll keep it simple and rely on the model's mapping in main.py.
        # The block_size and vocab_size will be set explicitly in main.py from data/tokenizer.
        pass # Keep the values from default or config file if model_type is not one of these


@dataclass
class TrainingConfig:
    """Configuration for the training process."""
    learning_rate: float = 3e-4
    batch_size: int = 256
    max_epochs: int = 50
    num_workers: int = 4 # Number of CPU workers for data loading
    gradient_accumulation_steps: int = 1 # Accumulate gradients over N batches
    weight_decay: float = 0.1
    beta1: float = 0.9 # AdamW beta1
    beta2: float = 0.95 # AdamW beta2
    warmup_steps: int = 0 # Number of warmup steps for LR scheduler
    log_interval: int = 10 # Log metrics to console/WandB every N batches
    eval_interval: int = 1 # Evaluate on validation set every N epochs
    save_interval: int = 10 # Save checkpoint every N epochs
    clip_grad_norm: float = 1.0 # Gradient clipping norm
    mixed_precision: bool = True # Use Automatic Mixed Precision (AMP)
    seed: int = 42 # Random seed for reproducibility

@dataclass
class WandbConfig:
    """Configuration for Weights & Biases logging."""
    project: str = 'reasoning-gpt'
    name: str = 'bpe-reasoning-packed-run'
    entity: str = None # Your WandB entity (username or team name), if any
    mode: str = 'online' # 'online', 'offline', 'disabled'
    group: str = None # Group runs together
    tags: List[str] = field(default_factory=lambda: []) # Tags for filtering runs

@dataclass
class GlobalConfig:
    """Global configuration combining all sub-configurations."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    output_dir: str = 'checkpoints' # Directory to save model checkpoints
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' # Device to run training on

    def __post_init__(self):
        """Ensures consistency and creates output directory."""
        # block_size will be set from data.block_size in main.py for the model config
        os.makedirs(self.output_dir, exist_ok=True) # Create checkpoint directory if it doesn't exist

