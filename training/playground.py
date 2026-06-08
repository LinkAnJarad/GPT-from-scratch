import torch
import torch.nn as nn
from model import LanguageModel
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Train a causal language model")

    parser.add_argument("--vocab_size", type=int, default=128256, help="Tokenizer vocabulary size (Llama-3)")
    parser.add_argument("--dim", type=int, default=1024, help="Model embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=2816, help="SwiGLU hidden dimension (~2.75x dim)")
    parser.add_argument("--n_heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--n_kv_heads", type=int, default=8, help="Number of key/value heads (GQA)")
    parser.add_argument("--n_layers", type=int, default=20, help="Number of transformer layers")
    parser.add_argument("--seq_len", type=int, default=4096, help="Sequence length")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    parser.add_argument("--prompt", type=str, default="Hello", help="Prompt")

    return parser.parse_args()

args = parse_args()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

from transformers import AutoTokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b")


print("Initializing model...")
model = LanguageModel(
    vocab_size=args.vocab_size,
    dim=args.dim,
    hidden_dim=args.hidden_dim,
    n_heads=args.n_heads,
    n_kv_heads=args.n_kv_heads,
    n_layers=args.n_layers,
    seq_len=args.seq_len,
    local_att_every=-1,  # Full attention only for initial training
    window=128,
    dropout=args.dropout,
)
model.to(device)

# Print parameter count
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {total_params / 1e6:.1f}M total, {trainable_params / 1e6:.1f}M trainable")


def load_checkpoint(filepath, model, optimizer, device):
    """Load model and optimizer state from a checkpoint."""
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_step = checkpoint['step']
    loss = checkpoint.get('loss', float('inf'))
    print(f"  Resumed from step {start_step} (loss={loss:.4f})")
    return start_step

print("Loading checkpoint...")

checkpoint = 'checkpoints/model-step-15000.pt'
load_checkpoint(checkpoint, model, optimizer=None, device=device)

print("Checkpoint loaded.")


@torch.no_grad()
def generate_sample_text(model, device, tokenizer, max_new_tokens=128, temperature=0.8, top_k=50):
    """Generate a short text sample from the model for qualitative monitoring."""
    model.eval()
    # Start from a BOS-like token (token id 1, or use a small prompt)
    input_ids = torch.tensor([[128000]], dtype=torch.long, device=device)  # Llama-3 <|begin_of_text|>

    for _ in tqdm(range(max_new_tokens)):
        # Crop to seq_len if necessary
        ctx = input_ids[:, -4096:]  # keep last seq_len tokens

        logits = model(ctx)
        logits = logits[:, -1, :] / temperature

        # Top-k filtering
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        # Stop on EOS (Llama-3 EOS token id = 128001)
        if next_token.item() == 128001:
            break

    model.train()
    token_ids = input_ids[0].tolist()
    try:
        decoded_text = tokenizer.decode(token_ids)
        return decoded_text
    except Exception as e:
        print(f"  Decoding failed: {e}")


@torch.no_grad()
def generate(
    model,
    device,
    tokenizer,
    prompt: str = "",
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.3,
):
    """
    Generate text with multinomial sampling, top-k/top-p filtering,
    and repetition penalty.

    Args:
        prompt:             Text to condition generation on (empty = BOS only).
        temperature:        Sampling temperature. Lower = sharper, higher = more random.
        top_k:              Keep only the k highest-probability tokens (0 = disabled).
        top_p:              Keep the smallest set of tokens whose cumulative
                            probability >= p (1.0 = disabled).
        repetition_penalty: Penalise previously seen tokens. 1.0 = no penalty,
                            >1.0 = increasingly penalise repeats.
    """
    model.eval()

    # --- Encode the prompt, or start from BOS ---
    if prompt:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor(
            [[128000] + prompt_ids], dtype=torch.long, device=device
        )
    else:
        input_ids = torch.tensor([[128000]], dtype=torch.long, device=device)

    generated_ids = input_ids.clone()

    for _ in tqdm(range(max_new_tokens)):
        ctx = generated_ids[:, -4096:]

        logits = model(ctx)
        logits = logits[:, -1, :]  # (1, vocab_size)

        # --- Repetition penalty ---
        # Divide logits of seen tokens by the penalty factor.
        # For positive logits this lowers the probability; for negative logits
        # we multiply instead (so they become more negative, i.e. less likely).
        if repetition_penalty != 1.0:
            seen = generated_ids[0].tolist()
            for token_id in set(seen):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

        # --- Temperature ---
        logits = logits / temperature

        # --- Top-k filtering ---
        if top_k > 0:
            k = min(top_k, logits.size(-1))
            top_k_vals, _ = torch.topk(logits, k)
            threshold = top_k_vals[:, -1].unsqueeze(-1)  # kth-largest value
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        # --- Top-p (nucleus) filtering ---
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            # Shift right so the token that pushes cumsum over p is kept
            remove_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
            # Scatter back to original token order
            logits = torch.zeros_like(logits).scatter(
                1, sorted_indices, sorted_logits
            )

        # --- Sample ---
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

        if next_token.item() == 128001:  # Llama-3 EOS
            break

    model.train()

    token_ids = generated_ids[0].tolist()
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    except Exception as e:
        print(f"  Decoding failed: {e}")
        return token_ids

generated_text = generate(model, device, tokenizer, args.prompt)
print(f"\nGenerated Text:\n\n{generated_text}")