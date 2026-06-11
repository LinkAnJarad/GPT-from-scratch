#!/usr/bin/env python3


import os
import sys
import math
import time
import argparse
import torch
import torch.nn as nn
from model import LanguageModel
from dataloader import get_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Train a causal language model")

    parser.add_argument("--vocab_size", type=int, default=32000, help="Tokenizer vocabulary size (Llama-2)")
    parser.add_argument("--dim", type=int, default=512, help="Model embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=1536, help="SwiGLU hidden dimension (~2.75x dim)")
    parser.add_argument("--n_heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--n_kv_heads", type=int, default=8, help="Number of key/value heads (GQA)")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--seq_len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=12, help="Micro-batch size per step")
    parser.add_argument("--grad_accum_steps", type=int, default=8, help="Gradient accumulation steps (effective batch = batch_size * grad_accum_steps)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=3e-5, help="Minimum learning rate (end of cosine decay)")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Number of linear warmup steps")
    parser.add_argument("--max_steps", type=int, default=100000, help="Maximum number of training steps")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="AdamW weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--beta2", type=float, default=0.95, help="AdamW beta2")

    # Data
    parser.add_argument("--data_path", type=str, default="data_memmap/tokens.memmap", help="Path to token memmap file")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader worker count")

    # Logging and checkpointing
    parser.add_argument("--log_interval", type=int, default=5, help="Log training metrics every N steps")
    parser.add_argument("--save_interval", type=int, default=2500, help="Save checkpoint every N steps")
    parser.add_argument("--sample_interval", type=int, default=500, help="Generate sample text every N steps")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory for saving checkpoints")
    parser.add_argument("--log_file", type=str, default="training/train.log", help="Training log file path")

    # Resume
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Learning rate schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------
def get_lr(step, warmup_steps, max_steps, lr, min_lr):
    """Linear warmup followed by cosine decay to min_lr."""
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    # Cosine decay phase
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


# ---------------------------------------------------------------------------
# Sample text generation (greedy, for monitoring training quality)
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_sample_text(model, device, seq_len, max_new_tokens=128, temperature=0.8, top_k=50):
    """Generate a short text sample from the model for qualitative monitoring."""
    model.eval()
    # Start from a BOS-like token (token id 1, or use a small prompt)
    input_ids = torch.tensor([[1]], dtype=torch.long, device=device)  # Llama-2 <s>

    for _ in range(max_new_tokens):
        # Crop to seq_len if necessary
        ctx = input_ids[:, -seq_len:]  # keep last seq_len tokens

        logits = model(ctx)
        logits = logits[:, -1, :] / temperature

        # Top-k filtering
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        # Stop on EOS (Llama-2 EOS token id = 2)
        if next_token.item() == 2:
            break

    model.train()
    return input_ids[0].tolist()


# ---------------------------------------------------------------------------
# Checkpoint saving and loading
# ---------------------------------------------------------------------------
def save_checkpoint(model, optimizer, step, loss, args, filepath):
    """Save model and optimizer state."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'args': vars(args),
    }, filepath)
    print(f"  Checkpoint saved to {filepath}")


def load_checkpoint(filepath, model, optimizer, device):
    """Load model and optimizer state from a checkpoint."""
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_step = checkpoint['step']
    loss = checkpoint.get('loss', float('inf'))
    print(f"  Resumed from step {start_step} (loss={loss:.4f})")
    return start_step


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    from transformers import AutoTokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-2-7b")

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # DataLoader
    # -----------------------------------------------------------------------
    print("Initializing DataLoader...")
    dataloader = get_dataloader(
        filepath=args.data_path,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )
    total_samples = len(dataloader.dataset)
    print(f"Dataset: {total_samples:,} samples, {len(dataloader):,} batches per epoch")

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
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
    print(f"Effective batch size: {args.batch_size * args.grad_accum_steps} (micro={args.batch_size} x accum={args.grad_accum_steps})")

    # -----------------------------------------------------------------------
    # Optimizer and loss
    # -----------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    loss_fn = nn.CrossEntropyLoss()

    # Create a GradScaler for mixed precision with float16
    scaler = torch.cuda.amp.GradScaler()

    # Resume from checkpoint if specified
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer, device)

    # -----------------------------------------------------------------------
    # Training loop (step-based, not epoch-based)
    # -----------------------------------------------------------------------
    print(f"\nStarting training from step {start_step}...")
    print(f"  Max steps: {args.max_steps:,}")
    print(f"  Warmup steps: {args.warmup_steps:,}")
    print(f"  Peak LR: {args.lr}, Min LR: {args.min_lr}")
    print(f"  Gradient clipping: {args.grad_clip}")
    print("=" * 70)

    model.train()
    data_iter = iter(dataloader)
    running_loss = 0.0
    tokens_processed = 0
    total_tokens_processed = 0
    train_start_time = time.time()
    step_start_time = time.time()

    # Open log file for writing metrics
    log_f = open(args.log_file, "a")
    if start_step == 0:
        log_f.write("step,loss,lr,tokens_per_sec,total_tokens,elapsed_min\n")

    for step in range(start_step, args.max_steps):
        # Update learning rate
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Gradient accumulation loop
        optimizer.zero_grad()
        accum_loss = 0.0

        for micro_step in range(args.grad_accum_steps):
            # Get next batch; restart iterator if exhausted (new epoch)
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                x, y = next(data_iter)

            x = x.to(device)
            y = y.to(device)

            # Forward pass with mixed precision
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
                # Reshape for cross-entropy: (batch * seq_len, vocab_size) vs (batch * seq_len,)
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                # Scale loss by accumulation steps to average gradients correctly
                scaled_loss = loss / args.grad_accum_steps
                
            scaler.scale(scaled_loss).backward()

            accum_loss += loss.item()
            tokens_processed += x.numel()
            total_tokens_processed += x.numel()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        # Track loss
        avg_loss = accum_loss / args.grad_accum_steps
        running_loss += avg_loss

        # ---------------------------------------------------------------
        # Logging
        # ---------------------------------------------------------------
        if (step + 1) % args.log_interval == 0:
            elapsed = time.time() - train_start_time
            step_elapsed = time.time() - step_start_time
            tok_per_sec = tokens_processed / step_elapsed if step_elapsed > 0 else 0
            avg_running_loss = running_loss / args.log_interval

            print(
                f"Step {step + 1:>7d}/{args.max_steps} | "
                f"Loss: {avg_running_loss:.4f} | "
                f"LR: {lr:.2e} | "
                f"Tok/s: {tok_per_sec:,.0f} | "
                f"Total Tok: {total_tokens_processed:,} | "
                f"Elapsed: {elapsed / 60:.1f}min"
            )

            log_f.write(f"{step + 1},{avg_running_loss:.6f},{lr:.8f},{tok_per_sec:.0f},{total_tokens_processed},{elapsed / 60:.2f}\n")
            log_f.flush()

            running_loss = 0.0
            tokens_processed = 0
            step_start_time = time.time()

        # ---------------------------------------------------------------
        # Checkpointing
        # ---------------------------------------------------------------
        if (step + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"model-step-{step + 1}.pt")
            save_checkpoint(model, optimizer, step + 1, avg_loss, args, ckpt_path)

        # ---------------------------------------------------------------
        # Sample generation
        # ---------------------------------------------------------------
        if (step + 1) % args.sample_interval == 0:
            print(f"\n--- Sample at step {step + 1} ---")
            token_ids = generate_sample_text(model, device, args.seq_len, max_new_tokens=128, temperature=0.8, top_k=50)
            print(f"  Token IDs (first 64): {token_ids[:64]}")
            print(f"  Total tokens generated: {len(token_ids)}")
            
            # Decode using the tokenizer we loaded
            try:
                decoded_text = tokenizer.decode(token_ids)
                print(f"  Decoded Text:\n{decoded_text}")
            except Exception as e:
                print(f"  Decoding failed: {e}")
            print("---\n")

    # -----------------------------------------------------------------------
    # Save final checkpoint
    # -----------------------------------------------------------------------
    final_path = os.path.join(args.checkpoint_dir, f"model-step-{args.max_steps}-final.pt")
    save_checkpoint(model, optimizer, args.max_steps, avg_loss, args, final_path)

    log_f.close()
    total_time = time.time() - train_start_time
    print(f"\nTraining complete. Total time: {total_time / 3600:.2f} hours")
    print(f"Final checkpoint saved to: {final_path}")


if __name__ == "__main__":
    main()
