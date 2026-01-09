# %% [markdown]
# # LLM Inference from Scratch: Demystifying Language Models
#
# Welcome! In this notebook, you'll learn that **HuggingFace models are not magic** - they're just PyTorch models with convenient wrappers.
#
# We will:
# 1. Load GPT-2 (a small, CPU-friendly language model)
# 2. Explore the **actual PyTorch architecture**
# 3. Perform **manual tokenization** (without using pipelines)
# 4. Implement the **token generation loop from scratch**
# 5. Understand **logits**, **temperature**, and **sampling strategies**
#
# By the end, you'll understand exactly what happens inside an LLM during inference.

# %%
# Install dependencies (uncomment if needed)
# %pip install torch transformers -qqq

# %% [markdown]
# ## Part 1: The Imports - What Are We Actually Loading?
#
# Let's start with the imports and understand what each one does:
#
# - **`torch`**: The PyTorch library - the actual deep learning framework
# - **`torch.nn.functional`**: Contains functions like softmax that we'll use for sampling
# - **`GPT2LMHeadModel`**: This is just a PyTorch `nn.Module` subclass!
# - **`GPT2Tokenizer`**: Converts text to numbers and back

# %%
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Set seed for reproducibility
torch.manual_seed(42)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %% [markdown]
# ## Part 2: Loading the Model - It's Just PyTorch!
#
# When we load a HuggingFace model, we're downloading:
# 1. **Model weights**: The learned parameters (numbers)
# 2. **Model config**: Architecture details (how many layers, dimensions, etc.)
# 3. **Tokenizer files**: Vocabulary and encoding rules
#
# The model itself is a standard PyTorch `nn.Module`. Let's prove it!

# %%
# Load the model and tokenizer
model_name = "gpt2"  # This is the smallest GPT-2 variant (~124M parameters)

print("Loading tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

print("Loading model...")
model = GPT2LMHeadModel.from_pretrained(model_name)
model = model.to(device)
model.eval()  # Set to evaluation mode (disables dropout)

print("\n" + "="*60)
print("MODEL LOADED SUCCESSFULLY!")
print("="*60)

# %% [markdown]
# ### Proof: It's a PyTorch nn.Module
#
# Let's verify that this HuggingFace model is indeed a standard PyTorch module:

# %%
# Verify it's a PyTorch module
print(f"Is the model an nn.Module? {isinstance(model, torch.nn.Module)}")
print(f"Model class: {type(model)}")
print(f"Parent classes: {type(model).__bases__}")

# Count parameters (just like any PyTorch model)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (float32)")

# %% [markdown]
# ## Part 3: Exploring the Architecture
#
# GPT-2 is a **decoder-only transformer**. Let's look at its structure:
#
# ```
# GPT2LMHeadModel
# ├── transformer (GPT2Model)
# │   ├── wte: Word Token Embeddings (vocab_size x embed_dim)
# │   ├── wpe: Word Position Embeddings (max_positions x embed_dim)
# │   ├── h: ModuleList of transformer blocks
# │   │   └── [0-11]: GPT2Block (12 identical blocks)
# │   │       ├── ln_1: LayerNorm
# │   │       ├── attn: GPT2Attention (self-attention)
# │   │       ├── ln_2: LayerNorm
# │   │       └── mlp: GPT2MLP (feed-forward network)
# │   └── ln_f: Final LayerNorm
# └── lm_head: Linear layer (embed_dim -> vocab_size)
# ```

# %%
# Print the full architecture
print("FULL MODEL ARCHITECTURE:")
print("="*60)
print(model)

# %%
# Explore the model configuration
config = model.config

print("MODEL CONFIGURATION:")
print("="*60)
print(f"Vocabulary size: {config.vocab_size:,} tokens")
print(f"Maximum sequence length: {config.n_positions} tokens")
print(f"Embedding dimension: {config.n_embd}")
print(f"Number of attention heads: {config.n_head}")
print(f"Number of transformer layers: {config.n_layer}")
print(f"Feed-forward hidden size: {config.n_inner if config.n_inner else config.n_embd * 4}")

# %%
# Inspect individual layers
print("\nKEY LAYER DIMENSIONS:")
print("="*60)

# Token embeddings
wte = model.transformer.wte
print(f"Token Embedding (wte): {wte.weight.shape}")
print(f"  -> Maps {wte.weight.shape[0]:,} vocabulary tokens to {wte.weight.shape[1]}-dimensional vectors")

# Position embeddings
wpe = model.transformer.wpe
print(f"\nPosition Embedding (wpe): {wpe.weight.shape}")
print(f"  -> Maps {wpe.weight.shape[0]} positions to {wpe.weight.shape[1]}-dimensional vectors")

# First transformer block
block = model.transformer.h[0]
print(f"\nFirst Transformer Block:")
print(f"  Attention Q,K,V projection: {block.attn.c_attn.weight.shape}")
print(f"  Attention output projection: {block.attn.c_proj.weight.shape}")
print(f"  MLP first layer: {block.mlp.c_fc.weight.shape}")
print(f"  MLP second layer: {block.mlp.c_proj.weight.shape}")

# LM head
print(f"\nLanguage Model Head (lm_head): {model.lm_head.weight.shape}")
print(f"  -> Projects {model.lm_head.weight.shape[1]}-dimensional hidden states to {model.lm_head.weight.shape[0]:,} vocabulary logits")

# %% [markdown]
# ## Part 4: Tokenization - Converting Text to Numbers
#
# Before the model can process text, we need to convert it to numbers. This is called **tokenization**.
#
# GPT-2 uses **Byte Pair Encoding (BPE)**, which:
# - Breaks words into subword units
# - Can handle any text (no unknown tokens)
# - Balances vocabulary size with sequence length
#
# Let's see exactly how this works:

# %%
# Example text to tokenize
text = "Hello, I am a language model."

print("TOKENIZATION PROCESS:")
print("="*60)
print(f"Input text: '{text}'")
print()

# Step 1: Encode text to token IDs
token_ids = tokenizer.encode(text)
print(f"Step 1 - Token IDs: {token_ids}")
print(f"         Number of tokens: {len(token_ids)}")
print()

# Step 2: See what each token represents
print("Step 2 - Token breakdown:")
for i, token_id in enumerate(token_ids):
    token_text = tokenizer.decode([token_id])
    print(f"  Position {i}: ID={token_id:5d} -> '{token_text}'")
print()

# Step 3: Convert to PyTorch tensor
input_ids = torch.tensor([token_ids]).to(device)
print(f"Step 3 - PyTorch tensor shape: {input_ids.shape}")
print(f"         Tensor: {input_ids}")

# %%
# Let's see how different words get tokenized
examples = [
    "cat",
    "concatenate",
    "artificial intelligence",
    "GPT-2",
    "transformer",
    "antidisestablishmentarianism",
]

print("TOKENIZATION EXAMPLES:")
print("="*60)
for word in examples:
    tokens = tokenizer.encode(word)
    token_strs = [tokenizer.decode([t]) for t in tokens]
    print(f"'{word}'")
    print(f"  -> IDs: {tokens}")
    print(f"  -> Tokens: {token_strs}")
    print()

# %% [markdown]
# ## Part 5: The Forward Pass - Getting Logits
#
# Now let's run the model and see exactly what comes out. The output is **logits** - raw scores for each possible next token.
#
# **Key concept**: The model outputs a score for EVERY token in the vocabulary for EACH position in the sequence.

# %%
# Prepare input
prompt = "The capital of France is"
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

print("FORWARD PASS:")
print("="*60)
print(f"Input prompt: '{prompt}'")
print(f"Input shape: {input_ids.shape}  (batch_size=1, sequence_length={input_ids.shape[1]})")
print()

# Run the forward pass (no gradients needed for inference)
with torch.no_grad():
    outputs = model(input_ids)

# The output contains logits
logits = outputs.logits
print(f"Output logits shape: {logits.shape}")
print(f"  -> (batch_size={logits.shape[0]}, sequence_length={logits.shape[1]}, vocab_size={logits.shape[2]})")
print()
print("This means: for each of the {} input tokens, we get {} scores (one per vocabulary token)".format(
    logits.shape[1], logits.shape[2]))

# %%
# Let's look at the logits for the LAST position (predicting the next token)
last_token_logits = logits[0, -1, :]  # Shape: (vocab_size,)

print("EXAMINING LOGITS FOR NEXT TOKEN PREDICTION:")
print("="*60)
print(f"Logits for last position: shape {last_token_logits.shape}")
print(f"Min logit: {last_token_logits.min().item():.4f}")
print(f"Max logit: {last_token_logits.max().item():.4f}")
print(f"Mean logit: {last_token_logits.mean().item():.4f}")
print()

# Find the top predictions
top_k = 10
top_logits, top_indices = torch.topk(last_token_logits, top_k)

print(f"Top {top_k} predictions:")
print("-" * 40)
for i, (logit, idx) in enumerate(zip(top_logits, top_indices)):
    token = tokenizer.decode([idx.item()])
    print(f"  {i+1}. '{token}' (ID: {idx.item()}, logit: {logit.item():.4f})")

# %% [markdown]
# ## Part 6: From Logits to Probabilities - The Softmax Function
#
# Logits are raw scores - they can be any number. To convert them to probabilities (values between 0 and 1 that sum to 1), we use **softmax**:
#
# $$P(token_i) = \frac{e^{logit_i}}{\sum_j e^{logit_j}}$$

# %%
# Convert logits to probabilities using softmax
probabilities = F.softmax(last_token_logits, dim=-1)

print("SOFTMAX: CONVERTING LOGITS TO PROBABILITIES")
print("="*60)
print(f"Probabilities shape: {probabilities.shape}")
print(f"Sum of all probabilities: {probabilities.sum().item():.6f}")
print(f"Min probability: {probabilities.min().item():.2e}")
print(f"Max probability: {probabilities.max().item():.4f}")
print()

# Show top predictions with probabilities
top_probs, top_indices = torch.topk(probabilities, top_k)

print(f"Top {top_k} predictions with probabilities:")
print("-" * 50)
for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
    token = tokenizer.decode([idx.item()])
    logit = last_token_logits[idx].item()
    print(f"  {i+1}. '{token}' -> logit: {logit:7.3f} -> prob: {prob.item():.4f} ({prob.item()*100:.2f}%)")

# %% [markdown]
# ## Part 7: Temperature - Controlling Randomness
#
# **Temperature** is a parameter that controls how "confident" or "creative" the model is:
#
# $$P(token_i) = \frac{e^{logit_i / T}}{\sum_j e^{logit_j / T}}$$
#
# - **T = 1.0**: Normal behavior (default)
# - **T < 1.0**: More confident (sharper distribution, more deterministic)
# - **T > 1.0**: More creative (flatter distribution, more random)
# - **T → 0**: Always picks the highest probability token (greedy)
# - **T → ∞**: Uniform distribution (completely random)

# %%
import matplotlib.pyplot as plt

def apply_temperature(logits, temperature):
    """Apply temperature scaling to logits."""
    return F.softmax(logits / temperature, dim=-1)

# Test different temperatures
temperatures = [0.1, 0.5, 1.0, 1.5, 2.0]

print("TEMPERATURE EFFECT ON PROBABILITY DISTRIBUTION")
print("="*70)
print(f"{'Temperature':<12} | {'Top 1 prob':<12} | {'Top 5 sum':<12} | Top prediction")
print("-" * 70)

for temp in temperatures:
    probs = apply_temperature(last_token_logits, temp)
    top_prob = probs.max().item()
    top_5_sum = probs.topk(5).values.sum().item()
    top_token = tokenizer.decode([probs.argmax().item()])
    print(f"{temp:<12.1f} | {top_prob:<12.4f} | {top_5_sum:<12.4f} | '{top_token}'")

# %%
# Visualize temperature effects
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, temp in zip(axes, [0.3, 1.0, 2.0]):
    probs = apply_temperature(last_token_logits, temp)
    top_probs, top_indices = probs.topk(15)
    tokens = [tokenizer.decode([idx.item()]).strip() for idx in top_indices]

    ax.barh(range(len(tokens)), top_probs.cpu().numpy())
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens)
    ax.set_xlabel('Probability')
    ax.set_title(f'Temperature = {temp}')
    ax.invert_yaxis()

plt.tight_layout()
plt.suptitle(f'Prompt: "{prompt}" + ?', y=1.02, fontsize=12)
plt.show()

# %% [markdown]
# ## Part 8: Sampling Strategies
#
# Once we have probabilities, we need to select the next token. There are several strategies:
#
# 1. **Greedy**: Always pick the highest probability token
# 2. **Random Sampling**: Sample according to the probability distribution
# 3. **Top-K Sampling**: Only consider the top K tokens
# 4. **Top-P (Nucleus) Sampling**: Only consider tokens until cumulative probability reaches P

# %%
def greedy_sample(logits):
    """Select the token with highest probability."""
    return logits.argmax(dim=-1)

def random_sample(logits, temperature=1.0):
    """Sample from the probability distribution."""
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)

def top_k_sample(logits, k=50, temperature=1.0):
    """Sample from the top-k tokens only."""
    # Get top-k logits
    top_k_logits, top_k_indices = logits.topk(k)
    # Apply temperature and softmax
    probs = F.softmax(top_k_logits / temperature, dim=-1)
    # Sample from top-k
    sampled_idx = torch.multinomial(probs, num_samples=1)
    # Map back to original vocabulary index
    return top_k_indices.gather(-1, sampled_idx).squeeze(-1)

def top_p_sample(logits, p=0.9, temperature=1.0):
    """Sample using nucleus (top-p) sampling."""
    # Apply temperature
    probs = F.softmax(logits / temperature, dim=-1)
    # Sort probabilities in descending order
    sorted_probs, sorted_indices = probs.sort(descending=True)
    # Calculate cumulative probabilities
    cumulative_probs = sorted_probs.cumsum(dim=-1)
    # Find where cumulative probability exceeds p
    mask = cumulative_probs <= p
    # Always include at least one token
    mask[..., 0] = True
    # Zero out probabilities for tokens outside nucleus
    sorted_probs = sorted_probs * mask.float()
    # Renormalize
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    # Sample
    sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
    # Map back to original vocabulary index
    return sorted_indices.gather(-1, sampled_idx).squeeze(-1)

print("Sampling functions defined!")

# %%
# Demonstrate each sampling method
print("SAMPLING STRATEGIES DEMO")
print("="*60)
print(f"Prompt: '{prompt}'")
print()

# Greedy
greedy_token = greedy_sample(last_token_logits)
print(f"Greedy sampling: '{tokenizer.decode([greedy_token.item()])}'")

# Random sampling (run multiple times to see variation)
print("\nRandom sampling (10 samples):")
for i in range(10):
    token = random_sample(last_token_logits, temperature=1.0)
    print(f"  {i+1}. '{tokenizer.decode([token.item()])}'")

# Top-K sampling
print("\nTop-K sampling (k=5, 10 samples):")
for i in range(10):
    token = top_k_sample(last_token_logits, k=5, temperature=1.0)
    print(f"  {i+1}. '{tokenizer.decode([token.item()])}'")

# Top-P sampling
print("\nTop-P sampling (p=0.9, 10 samples):")
for i in range(10):
    token = top_p_sample(last_token_logits, p=0.9, temperature=1.0)
    print(f"  {i+1}. '{tokenizer.decode([token.item()])}'")

# %% [markdown]
# ## Part 9: The Token Generation Loop - Putting It All Together
#
# Now we'll implement the complete text generation loop from scratch. This is what happens inside `model.generate()` or HuggingFace pipelines:
#
# ```
# 1. Tokenize input prompt
# 2. LOOP:
#    a. Forward pass through model -> get logits
#    b. Get logits for last position
#    c. Apply temperature
#    d. Sample next token
#    e. Append to sequence
#    f. Check stopping conditions
# 3. Decode tokens back to text
# ```

# %%
def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens=50,
    temperature=1.0,
    top_k=None,
    top_p=None,
    verbose=False
):
    """
    Generate text using the model, token by token.

    Args:
        model: The GPT-2 model
        tokenizer: The tokenizer
        prompt: Starting text
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: If set, only sample from top-k tokens
        top_p: If set, use nucleus sampling with this threshold
        verbose: If True, print each token as it's generated

    Returns:
        Generated text string
    """
    # Step 1: Tokenize the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    if verbose:
        print(f"Starting generation...")
        print(f"Initial sequence length: {input_ids.shape[1]} tokens")
        print(f"Prompt: '{prompt}'")
        print("\nGenerating:", end=" ")

    # Step 2: Generation loop
    generated_tokens = []

    for step in range(max_new_tokens):
        # Step 2a: Forward pass
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        # Step 2b: Get logits for the last position
        next_token_logits = logits[0, -1, :]

        # Step 2c & 2d: Apply temperature and sample
        if top_k is not None:
            next_token = top_k_sample(next_token_logits, k=top_k, temperature=temperature)
        elif top_p is not None:
            next_token = top_p_sample(next_token_logits, p=top_p, temperature=temperature)
        elif temperature == 0:
            next_token = greedy_sample(next_token_logits)
        else:
            next_token = random_sample(next_token_logits, temperature=temperature)

        # Step 2e: Append to sequence
        next_token = next_token.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        generated_tokens.append(next_token.item())

        if verbose:
            token_text = tokenizer.decode([next_token.item()])
            print(token_text, end="", flush=True)

        # Step 2f: Check stopping condition (EOS token)
        if next_token.item() == tokenizer.eos_token_id:
            if verbose:
                print("\n[EOS token reached]")
            break

    if verbose:
        print(f"\n\nGenerated {len(generated_tokens)} new tokens")

    # Step 3: Decode full sequence
    full_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return full_text

print("Generation function defined!")

# %%
# Test the generation function
prompt = "Once upon a time, in a land far away,"

print("TEXT GENERATION DEMO")
print("="*60)
print()

# Generate with verbose output to see the process
generated = generate_text(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_new_tokens=50,
    temperature=0.8,
    top_k=50,
    verbose=True
)

print("\n" + "="*60)
print("FINAL OUTPUT:")
print("="*60)
print(generated)

# %% [markdown]
# ## Part 10: Comparing Different Generation Settings
#
# Let's see how different settings affect the generated text:

# %%
prompt = "The future of artificial intelligence is"

print("COMPARING TEMPERATURE SETTINGS")
print("="*70)
print(f"Prompt: '{prompt}'")
print()

for temp in [0.3, 0.7, 1.0, 1.5]:
    print(f"\n{'='*70}")
    print(f"Temperature = {temp}")
    print("-"*70)

    torch.manual_seed(42)  # Reset seed for fair comparison
    text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=40,
        temperature=temp,
        top_p=0.9,
        verbose=False
    )
    print(text)

# %%
prompt = "Scientists have discovered that"

print("COMPARING SAMPLING STRATEGIES")
print("="*70)
print(f"Prompt: '{prompt}'")

settings = [
    {"name": "Greedy (temp=0)", "temperature": 0, "top_k": None, "top_p": None},
    {"name": "Top-K (k=10)", "temperature": 0.8, "top_k": 10, "top_p": None},
    {"name": "Top-K (k=50)", "temperature": 0.8, "top_k": 50, "top_p": None},
    {"name": "Top-P (p=0.5)", "temperature": 0.8, "top_k": None, "top_p": 0.5},
    {"name": "Top-P (p=0.95)", "temperature": 0.8, "top_k": None, "top_p": 0.95},
]

for setting in settings:
    print(f"\n{'='*70}")
    print(f"{setting['name']}")
    print("-"*70)

    torch.manual_seed(42)
    text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=40,
        temperature=setting['temperature'],
        top_k=setting['top_k'],
        top_p=setting['top_p'],
        verbose=False
    )
    print(text)

# %% [markdown]
# ## Part 11: Detailed Step-by-Step Visualization
#
# Let's generate a few tokens and visualize exactly what happens at each step:

# %%
def generate_with_details(model, tokenizer, prompt, num_tokens=5, temperature=1.0):
    """Generate tokens and show detailed information at each step."""

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    print("STEP-BY-STEP GENERATION")
    print("="*70)
    print(f"Prompt: '{prompt}'")
    print(f"Tokenized: {input_ids[0].tolist()}")
    print()

    for step in range(num_tokens):
        print(f"\n{'='*70}")
        print(f"STEP {step + 1}")
        print(f"{'='*70}")

        # Current sequence
        current_text = tokenizer.decode(input_ids[0])
        print(f"Current sequence: '{current_text}'")
        print(f"Sequence length: {input_ids.shape[1]} tokens")

        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]

        # Apply temperature and get probabilities
        probs = F.softmax(logits / temperature, dim=-1)

        # Show top candidates
        print(f"\nTop 5 candidates (temperature={temperature}):")
        top_probs, top_indices = probs.topk(5)
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            token_text = tokenizer.decode([idx.item()])
            logit = logits[idx].item()
            print(f"  {i+1}. '{token_text}' | logit: {logit:7.2f} | prob: {prob.item():.4f} ({prob.item()*100:.1f}%)")

        # Sample next token
        next_token = torch.multinomial(probs, num_samples=1)
        next_token_text = tokenizer.decode([next_token.item()])
        print(f"\n>>> Sampled token: '{next_token_text}' (ID: {next_token.item()})")

        # Append
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    final_text = tokenizer.decode(input_ids[0])
    print(f"\n{'='*70}")
    print(f"FINAL RESULT: '{final_text}'")
    return final_text

# Run the detailed generation
torch.manual_seed(123)
result = generate_with_details(
    model=model,
    tokenizer=tokenizer,
    prompt="The secret to happiness is",
    num_tokens=5,
    temperature=0.8
)

# %% [markdown]
# ## Summary: Key Takeaways
#
# ### 1. HuggingFace Models Are Just PyTorch
# - `GPT2LMHeadModel` is a subclass of `torch.nn.Module`
# - You can inspect parameters, count them, and access layers like any PyTorch model
# - The "magic" is just well-organized code
#
# ### 2. The Architecture
# - **Token Embeddings**: Convert token IDs to vectors
# - **Position Embeddings**: Add position information
# - **Transformer Blocks**: Self-attention + feed-forward networks
# - **LM Head**: Project hidden states to vocabulary logits
#
# ### 3. Tokenization
# - Converts text to sequences of integers
# - GPT-2 uses Byte Pair Encoding (BPE)
# - Words may be split into multiple subword tokens
#
# ### 4. Logits and Probabilities
# - Model outputs raw logits (unnormalized scores) for each vocabulary token
# - Softmax converts logits to probabilities
# - Temperature scales logits before softmax to control randomness
#
# ### 5. Sampling Strategies
# - **Greedy**: Always pick highest probability (deterministic)
# - **Random**: Sample from full distribution
# - **Top-K**: Only consider K highest probability tokens
# - **Top-P (Nucleus)**: Only consider tokens until cumulative probability reaches P
#
# ### 6. The Generation Loop
# ```
# For each new token:
#   1. Forward pass → get logits
#   2. Apply temperature (optional)
#   3. Sample next token
#   4. Append to sequence
#   5. Repeat until stopping condition
# ```

# %% [markdown]
# ## Exercises
#
# Try these exercises to deepen your understanding:
#
# 1. **Modify the generation function** to implement a `repetition_penalty` that reduces the probability of tokens that have already appeared.
#
# 2. **Visualize attention patterns**: Access `outputs.attentions` (with `output_attentions=True`) and plot which tokens attend to which.
#
# 3. **Compare models**: Load `gpt2-medium` or `gpt2-large` and compare the quality of generation.
#
# 4. **Implement beam search**: Instead of sampling one token at a time, track multiple candidate sequences and choose the best overall.
#
# 5. **Add a stopping criteria**: Stop generation when the model produces a period or newline character.

# %%
# Space for your experiments!
# Try modifying the generation parameters or implementing the exercises above.

prompt = "In the year 2050,"

# Your code here...
