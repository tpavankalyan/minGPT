import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datasets import load_dataset
import logging

# Assuming mingpt.bpe_new.BPETokenizerReasoning exists and works as expected
# from mingpt.bpe_new import BPETokenizerReasoning
# from config import DataConfig # Import for type hinting if needed

logger = logging.getLogger(__name__)

class PackedReasoningDataset(Dataset):
    """
    A PyTorch Dataset for language modeling, packing multiple short texts
    into fixed-size chunks to avoid padding and maximize GPU utilization.
    """
    def __init__(self, texts: list, tokenizer, block_size: int):
        """
        Args:
            texts (list): A list of strings, where each string is a document or text.
            tokenizer: An initialized tokenizer with a `__call__` method that takes
                       a string and returns a `torch.Tensor` of token IDs.
                       It should also have an `eos_token_id` attribute.
            block_size (int): The desired fixed length of input sequences for the model.
                              The actual packed chunks (for `x` and `y`) will be `block_size - 1`.
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        # `pack_size` is the length of the actual input sequence `x` (and target `y`)
        self.pack_size = block_size - 1

        if not hasattr(tokenizer, 'eos_token_id'):
            logger.warning("Tokenizer does not have 'eos_token_id'. Using 0 as separator.")
            self.separator = 0 # Default separator if EOS token not available
        else:
            self.separator = tokenizer.eos_token_id

        logger.info(f"Tokenizing and packing data into chunks of size {self.pack_size}...")

        all_tokens = []
        for text in tqdm(texts, desc="Tokenizing texts"):
            tokens = tokenizer(text) # Assumes tokenizer returns a torch.Tensor
            if not isinstance(tokens, torch.Tensor):
                raise TypeError(f"Tokenizer output for text '{text[:50]}...' must be a torch.Tensor, but got {type(tokens)}")

            all_tokens.append(tokens.long()) # Ensure token IDs are of type torch.long
            # Add separator token after each text for demarcation
            all_tokens.append(torch.tensor([self.separator], dtype=torch.long))

        # Concatenate all tokens into a single long tensor
        try:
            all_tokens = torch.cat(all_tokens).long()
        except RuntimeError as e:
            logger.error(f"Error concatenating tokens: {e}")
            print("Check if all tokenizer outputs are torch.Tensor and have compatible dtypes.")
            raise

        # Pack the concatenated tokens into fixed-size chunks
        self.packed_data = []
        # The loop iterates such that each `chunk` has `self.pack_size + 1` elements.
        # `i` is the starting index of each chunk.
        # The upper bound `len(all_tokens) - self.pack_size` ensures there's enough
        # data for a full `self.pack_size + 1` chunk.
        for i in range(0, len(all_tokens) - self.pack_size, self.pack_size):
            chunk = all_tokens[i : i + self.pack_size + 1]
            # The condition `len(chunk) == self.pack_size + 1` should always be true
            # with the given range calculation if `all_tokens` is long enough.
            # It acts as a safeguard against malformed chunks.
            if len(chunk) == self.pack_size + 1:
                self.packed_data.append(chunk)
            else:
                # This scenario would indicate an issue with loop logic or insufficient remaining data.
                logger.warning(f"Skipped an incomplete chunk of size {len(chunk)} at index {i}. Expected {self.pack_size + 1}.")


        logger.info(f"Created {len(self.packed_data)} packed examples of size {self.pack_size} (input/target length).")
        if len(self.packed_data) == 0 and len(all_tokens) > 0:
             logger.warning(f"No packed examples created. Total tokens: {len(all_tokens)}, Block size: {block_size}. Ensure total tokens are sufficient to create at least one full chunk.")


    def __len__(self) -> int:
        """Returns the number of packed examples."""
        return len(self.packed_data)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves a packed example (input and target) at the given index.

        Args:
            idx (int): The index of the packed example.

        Returns:
            dict: A dictionary containing 'x' (input tensor) and 'y' (target tensor).
                  Both tensors have shape (pack_size,).
        """
        chunk = self.packed_data[idx]
        x = chunk[:-1]  # Input sequence (all but the last token), length = pack_size
        y = chunk[1:]   # Target sequence (all but the first token, shifted version of x), length = pack_size

        return {'x': x, 'y': y}

def collate_batch_packed(batch: list) -> dict:
    """
    Collates a list of packed examples into a single batch.
    Since all sequences are already of fixed size due to packing, this is a simple stack operation.

    Args:
        batch (list): A list of dictionaries, where each dictionary contains 'x' and 'y' tensors.

    Returns:
        dict: A dictionary with 'x' and 'y' batch tensors.
              'x' and 'y' will have shape (batch_size, sequence_length).
    """
    x = torch.stack([item['x'] for item in batch])
    y = torch.stack([item['y'] for item in batch])
    return {'x': x, 'y': y}

def apply_template(dataset) -> list:
    """
    Applies a template to the dataset to format the text for training.

    Args:
        dataset: The dataset to process.

    Returns:
        list: A list of formatted texts.
    """
    # Example template application (modify as needed)
    ques = [' '.join(map(str, q)) for q in dataset['ques']]
    ans = [' '.join(map(str, a)) for a in dataset['ans']]
    algos = ["bubble", "selection", "insertion", "quick", "heap", "shell", "gnome", "cycle"]
    texts = []
    for i in tqdm(range(len(ques))):
        for alg in algos:
            soln = '<|T|> '+' <|T|> '.join([' '.join(map(str, step)) for step in dataset[alg][i]])
            texts.append("<|Q|> "+ques[i]+" <|S|> "+alg+" "+soln+" <|A|> "+ans[i]+" <|E|>")
            if i%(len(ques)//5) == 0:
                #show sample texts
                logger.info(f"Sample text {i}: {texts[-1]}")
    return texts


def load_and_prepare_datasets(data_config, tokenizer) -> tuple: # DataConfig can be imported if type hinting `data_config: DataConfig`
    """
    Loads datasets from Hugging Face and prepares them into PackedReasoningDataset instances.

    Args:
        data_config: An object containing dataset configuration (e.g., DataConfig from config.py).
        tokenizer: The tokenizer instance to use for processing text.

    Returns:
        tuple: A tuple containing (train_dataset, val_dataset).
    """
    logger.info(f"Loading dataset: {data_config.dataset_name}")
    try:
        dataset = load_dataset(data_config.dataset_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset '{data_config.dataset_name}': {e}")

    train_texts = apply_template(dataset[data_config.train_split])
    val_texts = apply_template(dataset[data_config.val_split])

    # train_texts = dataset[data_config.train_split]['text']
    # val_texts = dataset[data_config.val_split]['text']
    # test_texts = dataset[data_config.test_split]['text'] # Uncomment if test set is needed

    # Example of applying a filter if needed (as per original code's commented line)
    # if hasattr(data_config, 'filter_train_length') and data_config.filter_train_length is not None:
    #     train_dataset_hf = dataset[data_config.train_split].filter(lambda x: x['length'] != data_config.filter_train_length)
    #     train_texts = train_dataset_hf['text']
    #     logger.info(f"Filtered training set: removed examples with length {data_config.filter_train_length}. New size: {len(train_texts)}")

    train_dataset = PackedReasoningDataset(train_texts, tokenizer, data_config.block_size)
    val_dataset = PackedReasoningDataset(val_texts, tokenizer, data_config.block_size)

    return train_dataset, val_dataset
