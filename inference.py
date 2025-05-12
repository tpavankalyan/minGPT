import torch
import sys
import os
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory of mingpt to the Python path
# This assumes your directory structure is like:
# project_root/
# ├── inference.py
# ├── main.py
# ├── trainer.py
# ├── dataset.py
# ├── config.py
# └── mingpt/
#     ├── __init__.py
#     ├── model.py
#     ├── utils.py # Assuming CfgNode is here
#     └── bpe_new.py # Assuming BPETokenizerReasoning is here

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the project root directory to the sys.path
# Adjust this path if your structure is different.
# Assuming inference.py is in project_root, and mingpt is a subdir
project_root_dir = current_dir
sys.path.insert(0, project_root_dir)


try:
    # Import the necessary classes from your project structure
    from mingpt.model import GPT
    # Assuming CfgNode is in mingpt.utils
    from mingpt.utils import CfgNode as CN
    # Assuming BPETokenizerReasoning is in mingpt.bpe_new
    from mingpt.bpe_new import BPETokenizerReasoning
except ImportError as e:
    logger.error(f"Error importing mingpt components: {e}")
    logger.error("Please ensure that the 'mingpt' directory is correctly placed and accessible in your Python path.")
    logger.error("Current sys.path:", sys.path)
    sys.exit(1)


def run_inference():
    """
    Sets up the model and tokenizer, and runs inference for single and batched inputs.
    """
    # --- 1. Configuration Loading ---
    # We'll use a simplified config for inference, matching the model's expected CN structure.
    # In a real scenario, you might load the same config used for training.
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-mini' # Or the type you trained
    # These will be set based on the tokenizer and desired inference block size
    model_config.vocab_size = None
    model_config.block_size = 300 # Must match or be smaller than the block_size used for training

    # If using a predefined type, merge it (this is handled by GPT.__init__ now, but good to be explicit)
    # The GPT.__init__ will merge based on model_type if provided.

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device for inference: {device}")

    # --- 2. Tokenizer Initialization ---
    logger.info("Initializing tokenizer...")
    try:
        tokenizer = BPETokenizerReasoning()
        # Set vocab_size on the config from the tokenizer
        model_config.vocab_size = tokenizer.vocab_size
        logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        logger.error(f"Failed to initialize tokenizer: {e}")
        return

    # --- 3. Model Initialization ---
    logger.info(f"Initializing GPT model: {model_config.model_type}...")
    model = GPT(model_config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    # --- 4. Load Trained Weights (IMPORTANT) ---
    # Replace 'path/to/your/trained_model.pth' with the actual path to your saved model checkpoint.
    # If you don't load trained weights, the model will generate random sequences.
    checkpoint_path = 'checkpoints/packed_non4_epoch_50_final.pth' # Example path from your training script
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading model weights from {checkpoint_path}")
        try:
            # Load the state dictionary
            checkpoint = torch.load(checkpoint_path, map_location=device)
            # Load the model state dict
            # Note: If you saved the entire state (model, optimizer, etc.),
            # you need to load the 'model_state_dict' key.
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Model weights loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model weights from {checkpoint_path}: {e}")
            logger.warning("Proceeding with randomly initialized model.")
    else:
        logger.warning(f"Model checkpoint not found at {checkpoint_path}. Proceeding with randomly initialized model.")

    # Move model to device and set to evaluation mode
    model.to(device)
    model.eval() # Set the model to evaluation mode for inference

    # --- 5. Prepare Input Data ---

    # Example 5a: Single-sequence inference input
    print("\n--- Single-Sequence Inference ---")
    single_input_text = "The quick brown fox jumps over the lazy"
    logger.info(f"Input text: '{single_input_text}'")

    # Tokenize the input text
    # The tokenizer should return a tensor (sequence_length,)
    single_input_ids = tokenizer(single_input_text)
    # Add a batch dimension of 1 for the model input (1, sequence_length)
    single_input_ids = single_input_ids.unsqueeze(0).to(device)

    logger.info(f"Single input IDs shape: {single_input_ids.shape}")
    logger.debug(f"Single input IDs:\n{single_input_ids}")

    # --- 6. Perform Single-Sequence Generation ---
    max_new_tokens_single = 50
    logger.info(f"Generating {max_new_tokens_single} new tokens...")
    with torch.no_grad(): # Ensure no gradients are calculated during inference
        generated_ids_single = model.generate(
            single_input_ids,
            max_new_tokens=max_new_tokens_single,
            temperature=0.8, # Controls randomness (higher = more random)
            do_sample=True, # Set to True for sampling, False for greedy
            top_k=None # Optional: restrict sampling to top K tokens
        )

    logger.info(f"\nGenerated IDs (Single) shape: {generated_ids_single.shape}")
    logger.debug(f"Generated IDs (Single):\n{generated_ids_single}")
    # Expected shape: (1, initial_sequence_length + max_new_tokens_single)

    # --- 7. Decode and Print Single-Sequence Output ---
    logger.info("\n--- Decoded Generated Text (Single) ---")
    # Remove the batch dimension before decoding (tokenizer expects (sequence_length,))
    decoded_text_single = tokenizer.decode(generated_ids_single[0].tolist()) # .tolist() converts tensor to list
    print(decoded_text_single)


    # Example 5b: Batched inference input
    print("\n--- Batched Inference ---")
    batch_input_texts = [
        "The quick brown fox jumps over the lazy",
        "Hello, my name is",
        "In the beginning was the",
        "Artificial intelligence is"
    ]
    logger.info(f"Input texts for batch ({len(batch_input_texts)} sequences):")
    for i, text in enumerate(batch_input_texts):
        logger.info(f"  Seq {i+1}: '{text}'")

    # Tokenize each text and pad/truncate to a common length for batching
    # A simple approach is to find the max length and pad, or pad/truncate to block_size
    # For this example, let's find the max length and pad with a padding token ID (e.g., 0)
    # In a real scenario, ensure your tokenizer handles padding correctly or use a dedicated padding function.
    # Assuming tokenizer returns a list of token IDs or a tensor
    tokenized_batch = [tokenizer(text) for text in batch_input_texts]

    # Find the maximum sequence length in the batch
    max_batch_length = max(len(ids) for ids in tokenized_batch)
    # Pad sequences to the max length
    # Assuming padding token ID is 0 (or use tokenizer.pad_token_id if available)
    pad_token_id = getattr(tokenizer, 'pad_token_id', 0)
    padded_batch_ids = [
        torch.cat([ids, torch.tensor([pad_token_id] * (max_batch_length - len(ids)), dtype=torch.long)])
        for ids in tokenized_batch
    ]

    # Stack the padded tensors to create a batch tensor (batch_size, max_batch_length)
    batch_input_ids = torch.stack(padded_batch_ids).to(device)

    logger.info(f"Batched input IDs shape: {batch_input_ids.shape}")
    logger.debug(f"Batched input IDs:\n{batch_input_ids}")


    # --- 6. Perform Batched Generation ---
    max_new_tokens_batch = 30
    logger.info(f"Generating {max_new_tokens_batch} new tokens for each sequence in the batch...")
    with torch.no_grad(): # Ensure no gradients are calculated during inference
        generated_ids_batch = model.generate(
            batch_input_ids,
            max_new_tokens=max_new_tokens_batch,
            temperature=0.8,
            do_sample=True,
            top_k=50 # Example using top_k
        )

    logger.info(f"\nGenerated IDs (Batched) shape: {generated_ids_batch.shape}")
    logger.debug(f"Generated IDs (Batched):\n{generated_ids_batch}")
    # Expected shape: (batch_size, initial_sequence_length_batch + max_new_tokens_batch)


    # --- 7. Decode and Print Batched Output ---
    logger.info("\n--- Decoded Generated Text (Batched) ---")
    # Decode each sequence in the batch
    decoded_texts_batch = [tokenizer.decode(ids.tolist()) for ids in generated_ids_batch]
    for i, text in enumerate(decoded_texts_batch):
        print(f"Sequence {i+1}:\n{text}\n---")

    logger.info("\nInference examples complete.")
    logger.info("Note: To get meaningful text output, ensure you load a trained model's weights.")


if __name__ == "__main__":
    run_inference()
