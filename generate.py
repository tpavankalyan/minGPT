import torch
from mingpt.model import GPT
from mingpt.bpe import BPETokenizerReasoning  # Import your tokenizer
import os

# --- Load Config and Model ---
def load_model_and_tokenizer(checkpoint_path, block_size):
    """Loads the model and tokenizer from a checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint.
        block_size (int): The block size used during training.

    Returns:
        tuple: (model, tokenizer)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = BPETokenizerReasoning()  # Initialize your tokenizer
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-mini'  # Make sure this matches your training config
    model_config.vocab_size = tokenizer.vocab_size
    model_config.block_size = block_size  # Set the block size
    model = GPT(model_config).to(device)

    # Load the model state dict
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print(f"Checkpoint path: {checkpoint_path}")
        raise  # Re-raise the exception to halt execution

    model.eval()  # Set to evaluation mode
    return model, tokenizer, device

# --- Generation Function ---
def generate_text(model, tokenizer, device, prompt, max_new_tokens=50):
    """Generates text using the loaded model.

    Args:
        model (GPT): The loaded GPT model.
        tokenizer (BPETokenizerReasoning): The tokenizer.
        device (str): The device to use ('cuda' or 'cpu').
        prompt (str): The initial text prompt.
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 50.

    Returns:
        str: The generated text.
    """
    model.eval()  # Ensure model is in eval mode
    with torch.no_grad():
        context = tokenizer(prompt).to(device).unsqueeze(0)  # Tokenize and add batch dimension
        generated_tokens = model.generate(context, max_new_tokens=max_new_tokens)[0]  # Generate tokens, take the first (and only) batch.  Important change.
        generated_text = tokenizer.decode(generated_tokens)
        return generated_text

if __name__ == "__main__":
    # --- Configuration ---
    checkpoint_path = '/scratch/azureml/cr/j/46c59005a69d40498c72602c63c83f9e/cap/data-capability/wd/INPUT_asdf/repos/syn-reasoning/synthetic_reasoning/minGPT/projects/sorting/checkpoints/packed_non4_epoch_15.pth'  # Replace with your checkpoint path
    block_size = 300 # this needs to be the same as the block size used during training.

    # --- Load Model and Tokenizer ---
    try:
        model, tokenizer, device = load_model_and_tokenizer(checkpoint_path, block_size)
    except Exception as e:
        print(f"Failed to load model: {e}")
        exit(1) # Exit if the model fails to load

    from datasets import load_dataset
    dataset = load_dataset("Pavankalyan/syn-reasoning")
    test_texts = dataset['test'].filter(lambda x: x['length'] == 6)['question']
    # test_texts = dataset['test'].filter(lambda x: x['length'] == 4)['question']
    q_list = list(set(test_texts))
    prompts = [f"<|Q|> {q} 9 <|S|>" for q in q_list]  # Create prompts for each unique question
    alg_dict = {}
    # prompt = "<|Q|> 0 9 1 4 2 <|S|>"  # Example prompt
    # generated_text = generate_text(model, tokenizer, device, prompt, max_new_tokens=200)
    # print(generated_text)
    # exit()
    # --- Generate Text ---
    from tqdm import tqdm
    # prompts = prompts[:1000]  # Limit to the first 100 prompts for demonstration
    total = 0
    correct = 0
    for prompt in tqdm(prompts):
        try: 
            generated_text = generate_text(model, tokenizer, device, prompt, max_new_tokens=100)
            a = generated_text.split('<|A|>')[1].split('<|E|>')[0].strip().split(' ')
            a = [int(x) for x in a]
            b = prompt.split('<|Q|>')[1].split('<|S|>')[0].strip().split(' ')
            b = [int(x) for x in b]
            b = sorted(b)

            algo = generated_text.replace(prompt, "").strip().split('<|T|>')[0].strip()
            
            if algo not in alg_dict:
                alg_dict[algo] = 1
            else:
                alg_dict[algo] += 1
            total += 1
            if a == b:
                correct += 1
            if total % 100 == 0:
                print(f"Total: {total}, Correct: {correct}, Accuracy: {correct / total:.2%}")
                print(f"Prompt: {prompt}")
                print(f"Generated Text: {generated_text}")
                print(f"Algorithm: {algo}")
                print(f"Expected: {b}")
                print(f"Generated: {a}")
                print("===" * 20)
        except Exception as e:
            print(f"Error generating text for prompt '{prompt}': {e}")
            continue  # Skip to the next prompt if there's an error

    # --- Print Results ---
    for algo, count in alg_dict.items():
        print(f"Algorithm: {algo}, Count: {count}")
        

    # --- Prompt and Generate ---
    # prompt = "<|Q|> 0 8 9 5 <|S|>"
    # generated_text = generate_text(model, tokenizer, device, prompt, max_new_tokens=100)  # Generate up to 100 new tokens
    # print(f"Prompt: {prompt}")
    # print(f"Generated Text: {generated_text}")
