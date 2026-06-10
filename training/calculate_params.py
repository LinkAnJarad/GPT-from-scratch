#!/usr/bin/env python3
import argparse
import torch
from model import LanguageModel

def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate the parameter count and structural breakdown of the LanguageModel."
    )
    parser.add_argument("--vocab_size", type=int, default=128256, help="Tokenizer vocabulary size")
    parser.add_argument("--dim", type=int, default=256, help="Model embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=768, help="SwiGLU hidden dimension")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n_kv_heads", type=int, default=4, help="Number of key/value heads (GQA)")
    parser.add_argument("--n_layers", type=int, default=10, help="Number of transformer layers")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    parser.add_argument("--local_att_every", type=int, default=-1, help="Local attention interval")
    parser.add_argument("--window", type=int, default=128, help="Local attention window size")
    
    return parser.parse_args()

def main():
    args = parse_args()

    print("=" * 60)
    print("           MODEL CONFIGURATION")
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"  {k:<20}: {v}")
    print("=" * 60)

    # Instantiate model on CPU
    print("[*] Instantiating model on CPU to calculate parameter counts...")
    model = LanguageModel(
        vocab_size=args.vocab_size,
        dim=args.dim,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        n_layers=args.n_layers,
        seq_len=args.seq_len,
        local_att_every=args.local_att_every,
        window=args.window,
        dropout=0.0
    )

    # Calculations
    # Since weight tying is used, the embedding and the output layer share weights.
    # We will compute both physical parameter count (unique memory) and virtual parameter count (including tied references).
    unique_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Let's count virtual params by traversing modules manually
    embedding_params = sum(p.numel() for p in model.embedding.parameters())
    transformer_params = sum(p.numel() for p in model.transformer.parameters())
    norm_params = sum(p.numel() for p in model.norm.parameters())
    # Note: model.out shares weights with model.embedding, so its weight count is the same.
    out_params = sum(p.numel() for p in model.out.parameters())
    
    # Virtual total (counting tied weights twice)
    virtual_total = embedding_params + transformer_params + norm_params + out_params

    # Single block breakdown
    single_block = model.transformer[0]
    attn_params = sum(p.numel() for p in single_block.attention.parameters())
    ffn_params = sum(p.numel() for p in single_block.ffn.parameters())
    block_norm_params = sum(p.numel() for p in single_block.norm1.parameters()) + sum(p.numel() for p in single_block.norm2.parameters())
    block_total = attn_params + ffn_params + block_norm_params

    print("\n" + "=" * 60)
    print("           PARAMETER COUNT BREAKDOWN")
    print("=" * 60)
    print(f"  Total Unique Parameters  : {unique_params:,} ({unique_params / 1e6:.2f}M)")
    print(f"  Trainable Parameters     : {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
    print(f"  Tied/Shared Parameters   : {embedding_params:,} ({embedding_params / 1e6:.2f}M) (Embedding <-> Out Head)")
    print(f"  Virtual Total Parameters : {virtual_total:,} ({virtual_total / 1e6:.2f}M) (if untied)")
    print("-" * 60)
    print("  Component Breakdown (Unique Weights):")
    print(f"    - Token Embeddings     : {embedding_params:,} ({embedding_params / unique_params * 100:.1f}%)")
    print(f"    - Transformer Blocks   : {transformer_params:,} ({transformer_params / unique_params * 100:.1f}%)")
    print(f"    - Final RMSNorm        : {norm_params:,} (<0.1%)")
    print(f"    - Output LM Head (Tied): 0 (Shared with Token Embeddings)")
    print("-" * 60)
    print("  Per Transformer Block Breakdown:")
    print(f"    - Attention Projections: {attn_params:,} ({attn_params / block_total * 100:.1f}%)")
    print(f"    - FeedForward (SwiGLU) : {ffn_params:,} ({ffn_params / block_total * 100:.1f}%)")
    print(f"    - RMSNorms             : {block_norm_params:,} ({block_norm_params / block_total * 100:.1f}%)")
    print(f"    - Total per Block      : {block_total:,} ({block_total / 1e6:.2f}M)")
    print("=" * 60)

if __name__ == "__main__":
    main()
