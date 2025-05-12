"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

from mingpt.utils import CfgNode as CN # Assuming CfgNode is in mingpt.utils

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Helper Modules

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the GELU activation function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying GELU.
        """
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but an explicit
    implementation is included to show the mechanics.
    """

    def __init__(self, config: CN):
        """
        Args:
            config (CN): Configuration object containing model hyperparameters.
                         Must have n_embd, n_head, attn_pdrop, resid_pdrop, block_size.
        """
        super().__init__()
        # Ensure embedding dimension is divisible by the number of heads
        assert config.n_embd % config.n_head == 0

        # Key, query, value projections for all heads, concatenated
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # Regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # Causal mask to ensure that attention is only applied to the left in the input sequence
        # This mask is registered as a buffer so it's part of the model state but not a learnable parameter
        causal_mask = torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)
        self.register_buffer("bias", causal_mask)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head # Dimension of each attention head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the causal self-attention module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, n_embd).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, n_embd).
        """
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # c_attn output shape: (B, T, 3 * C)
        # After split: q, k, v are each (B, T, C)
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape and transpose for multi-head attention: (B, T, C) -> (B, T, n_head, head_size) -> (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # Causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))

        # Apply causal mask: set attention scores to -inf for future tokens
        # The mask `self.bias` is (1, 1, block_size, block_size), slice it to (1, 1, T, T)
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))

        # Apply softmax to get attention probabilities
        att = F.softmax(att, dim=-1)
        # Apply attention dropout
        att = self.attn_dropout(att)

        # Multiply attention probabilities with values: (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = att @ v

        # Re-assemble all head outputs side by side: (B, n_head, T, head_size) -> (B, T, n_head, head_size) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection and residual dropout
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    """ An unassuming Transformer block, comprising of self-attention and MLP. """

    def __init__(self, config: CN):
        """
        Args:
            config (CN): Configuration object containing model hyperparameters.
                         Must have n_embd, attn_pdrop, resid_pdrop, block_size.
        """
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd) # Layer norm before attention
        self.attn = CausalSelfAttention(config) # Causal self-attention module
        self.ln_2 = nn.LayerNorm(config.n_embd) # Layer norm before MLP

        # MLP module
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd), # Expand dimension
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd), # Project back
            act     = NewGELU(), # Activation function
            dropout = nn.Dropout(config.resid_pdrop), # Dropout
        ))
        # Define the forward pass for the MLP
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, n_embd).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, n_embd).
        """
        # Apply attention with pre-layer normalization and residual connection
        x = x + self.attn(self.ln_1(x))
        # Apply MLP with pre-layer normalization and residual connection
        x = x + self.mlpf(self.ln_2(x))
        return x

# -----------------------------------------------------------------------------
# The main GPT model

class GPT(nn.Module):
    """ GPT Language Model """

    @staticmethod
    def get_default_config() -> CN:
        """
        Returns a default configuration object for the GPT model.
        """
        C = CN()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd =  None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        return C

    def __init__(self, config: CN):
        """
        Args:
            config (CN): Configuration object containing model hyperparameters.
                         Must have vocab_size, block_size, and either model_type
                         or (n_layer, n_head, n_embd).
        """
        super().__init__()
        # Validate required configuration parameters
        assert config.vocab_size is not None, "vocab_size must be specified in the config."
        assert config.block_size is not None, "block_size must be specified in the config."
        self.block_size = config.block_size

        # Check if either model_type or specific parameters are given (exclusive OR)
        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        assert type_given ^ params_given, "Must specify either model_type OR (n_layer, n_head, n_embd), but not both or neither."

        if type_given:
            # Translate from model_type to detailed configuration if a predefined type is used
            # Use .get with an empty dictionary to avoid KeyError if model_type is not found
            model_params = {
                # names follow the huggingface naming conventions
                # GPT-1
                'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
                # GPT-2 configs
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
                # Gophers
                'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
                # (there are a number more...)
                # Custom tiny models
                'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
            }.get(config.model_type, {}) # Get parameters for the model type, or empty dict if not found

            if not model_params:
                 raise ValueError(f"Unknown model_type: {config.model_type}. Please provide valid model_type or specify n_layer, n_head, n_embd.")

            config.merge_from_dict(model_params) # Merge predefined parameters into the config

        # Validate that required parameters are now set after potentially merging
        assert config.n_layer is not None, "n_layer must be specified in the config or via model_type."
        assert config.n_head is not None, "n_head must be specified in the config or via model_type."
        assert config.n_embd is not None, "n_embd must be specified in the config or via model_type."

        # Define the transformer layers
        self.transformer = nn.ModuleDict(dict(
            # Token embeddings
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # Positional embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # Input dropout
            drop = nn.Dropout(config.embd_pdrop),
            # Stack of transformer blocks
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # Final layer normalization
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # Language modeling head (maps transformer output to vocabulary size)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                # Scale residual projections
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # Report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        logger.info(f"Number of parameters in transformer: {n_params/1e6:.2f}M")
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Total number of parameters: {total_params/1e6:.2f}M")


    def _init_weights(self, module: nn.Module):
        """
        Initializes the weights of the model modules.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    @classmethod
    def from_pretrained(cls, model_type: str):
        """
        Initialize a pretrained GPT model by copying over the weights
        from a huggingface/transformers checkpoint.

        Args:
            model_type (str): The name of the pretrained model (e.g., 'gpt2').

        Returns:
            GPT: An instance of the GPT model with loaded weights.
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        # Import transformers library here to avoid making it a hard dependency
        try:
            from transformers import GPT2LMHeadModel
        except ImportError:
            logger.error("Hugging Face transformers library not found. Please install it (`pip install transformers`).")
            raise

        logger.info(f"Loading pretrained weights from huggingface: {model_type}")

        # create a from-scratch initialized minGPT model
        config = cls.get_default_config()
        config.model_type = model_type
        # Set vocab_size and block_size based on the standard GPT-2 models
        config.vocab_size = 50257 # openai's model vocabulary
        config.block_size = 1024  # openai's model block_size
        model = GPT(config) # This will also set n_layer, n_head, n_embd from model_type
        sd = model.state_dict()

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        # Ignore attention mask bias as it's handled internally
        keys = [k for k in sd_hf if not k.endswith('attn.masked_bias')]
        # List of weights that need transposing due to Conv1D vs Linear difference
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # Ensure the number of keys match (ignoring the bias)
        if len(keys) != len(sd):
             logger.warning(f"Number of keys mismatch: HF ({len(keys)}) vs minGPT ({len(sd)}). This might indicate missing parameters.")

        for k in keys:
            if any(k.endswith(w) for w in transposed):
                # Special treatment for the Conv1D weights we need to transpose
                if sd_hf[k].shape[::-1] != sd[k].shape:
                     logger.warning(f"Shape mismatch for transposed weight {k}: HF {sd_hf[k].shape} vs minGPT {sd[k].shape}. Expected {sd[k].shape} for transposed.")
                     continue # Skip if shapes don't match even after transposing attempt
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # Vanilla copy over the other parameters
                if sd_hf[k].shape != sd[k].shape:
                    logger.warning(f"Shape mismatch for weight {k}: HF {sd_hf[k].shape} vs minGPT {sd[k].shape}.")
                    continue # Skip if shapes don't match
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        logger.info(f"Successfully loaded pretrained weights for {model_type}.")
        return model

    def configure_optimizers(self, train_config) -> torch.optim.AdamW:
        """
        Separates parameters into those that will and won't experience weight decay
        and returns the PyTorch AdamW optimizer object.

        Args:
            train_config: Configuration object containing training hyperparameters
                          (learning_rate, weight_decay, beta1, beta2).

        Returns:
            torch.optim.AdamW: The configured optimizer.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # parameters with 2 or more dimensions are usually weights, apply decay
                # parameters with 1 dimension are usually biases or layernorm/embedding weights, don't decay
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules (like nn.Linear) will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules (like nn.LayerNorm, nn.Embedding) will NOT be weight decayed
                    no_decay.add(fpn)
                # Parameters that don't fit the above criteria (e.g., custom parameters)
                # might not be included. Add a warning or handle them explicitly if needed.
                # For standard models, this separation covers most parameters.

        # Validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"parameters {inter_params} made it into both decay/no_decay sets!"
        # Check for parameters that were not separated
        not_separated = param_dict.keys() - union_params
        if len(not_separated) > 0:
             logger.warning(f"Parameters {not_separated} were not separated into either decay/no_decay set! They will not be included in the optimizer.")
             # Optionally assert if you want to be strict:
             # assert len(not_separated) == 0, f"parameters {not_separated} were not separated into either decay/no_decay set!"


        # Create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        # FIX: Pass betas as a tuple (beta1, beta2)
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=(train_config.beta1, train_config.beta2))
        logger.info("Optimizer configured with AdamW.")
        logger.info(f"Learning rate: {train_config.learning_rate}, Weight decay: {train_config.weight_decay}")
        logger.info(f"Betas: ({train_config.beta1}, {train_config.beta2})")

        return optimizer

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Performs the forward pass of the GPT model.

        Args:
            idx (torch.Tensor): Input tensor of token indices, shape (batch_size, sequence_length).
            targets (torch.Tensor, optional): Target tensor of token indices, shape (batch_size, sequence_length).
                                              Used for calculating the loss. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: A tuple containing:
                - logits (torch.Tensor): The output logits for each token, shape (batch_size, sequence_length, vocab_size).
                - loss (torch.Tensor | None): The cross-entropy loss if targets are provided, otherwise None.
        """
        device = idx.device
        b, t = idx.size() # batch size, sequence length

        # Ensure sequence length does not exceed model's block size
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"

        # Create positional indices (0, 1, ..., t-1)
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # Forward pass through the GPT model
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb) # Combine and apply dropout

        # Pass through transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final layer normalization
        x = self.transformer.ln_f(x)

        # Language modeling head to get logits
        logits = self.lm_head(x) # shape (b, t, vocab_size)

        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            # Reshape logits and targets for CrossEntropyLoss
            # logits: (b * t, vocab_size), targets: (b * t)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1) # ignore_index=-1 is common for padding

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, do_sample: bool = False, top_k: int | None = None) -> torch.Tensor:
        """
        Generates new tokens autoregressively, conditioned on the input sequence idx.
        Supports batched inference.

        Args:
            idx (torch.Tensor): Input tensor of token indices, shape (batch_size, initial_sequence_length).
                                This is the starting prompt(s).
            max_new_tokens (int): The maximum number of new tokens to generate for each sequence in the batch.
            temperature (float, optional): Controls the randomness of sampling. Higher values make output more random. Defaults to 1.0.
            do_sample (bool, optional): If True, sample from the probability distribution. If False, use greedy decoding (pick the token with the highest probability). Defaults to False.
            top_k (int | None, optional): If specified, sample from the top K most likely tokens. Defaults to None.

        Returns:
            torch.Tensor: The generated sequences, shape (batch_size, initial_sequence_length + max_new_tokens).
                          Includes the original input sequence followed by the generated tokens.
        """
        # Set model to evaluation mode (already done by @torch.no_grad(), but good practice)
        self.eval()

        # The loop generates one token at a time for the entire batch
        for _ in range(max_new_tokens):
            # If the sequence context is growing too long, crop it to the model's block size
            # This is important for models with fixed positional embeddings
            # idx_cond will be (batch_size, min(current_sequence_length, block_size))
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]

            # Forward the model to get the logits for the next token
            # The model processes the entire batch (idx_cond) in parallel
            # logits shape: (batch_size, sequence_length_cond, vocab_size)
            logits, _ = self(idx_cond)

            # Pluck the logits for the *last* token in the sequence for each item in the batch
            # logits[:, -1, :] shape: (batch_size, vocab_size)
            logits = logits[:, -1, :]

            # Apply temperature scaling to the logits
            logits = logits / temperature

            # Optionally crop the logits to only the top k options
            if top_k is not None:
                # torch.topk returns values and indices
                v, _ = torch.topk(logits, top_k)
                # Set logits of tokens not in the top k to -inf
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Apply softmax to convert logits to (normalized) probabilities
            # probs shape: (batch_size, vocab_size)
            probs = F.softmax(logits, dim=-1)

            # Either sample from the distribution or take the most likely element
            if do_sample:
                # Sample one token for each sequence in the batch
                # idx_next shape: (batch_size, 1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                # Get the index of the token with the highest probability for each sequence
                # idx_next shape: (batch_size, 1)
                _, idx_next = torch.topk(probs, k=1, dim=-1)

            # Append the sampled index to the running sequence for the entire batch
            # idx shape changes from (batch_size, T) to (batch_size, T + 1)
            idx = torch.cat((idx, idx_next), dim=1)

        # Return the complete generated sequences (initial prompt + generated tokens)
        return idx
