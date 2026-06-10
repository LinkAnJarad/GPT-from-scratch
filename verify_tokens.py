#!/usr/bin/env python3
"""
Verification script to check saved token chunks.
Reads the .npy files and validates their properties.
"""

import os
import sys
import argparse
import numpy as np
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Verify tokenized chunks")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/project/data",
        help="Directory containing tokenized chunks"
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="unsloth/llama-2-7b",
        help="HF tokenizer to use for decoding validation"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of samples to decode and print"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.output_dir):
        print(f"Error: Output directory {args.output_dir} does not exist.")
        sys.exit(1)
        
    print(f"Scanning directory {args.output_dir} for token files...")
    files = sorted([f for f in os.listdir(args.output_dir) if f.startswith("tokens_") and f.endswith(".npy")])
    
    if not files:
        print("No tokens_*.npy files found.")
        sys.exit(1)
        
    print(f"Found {len(files)} token chunk files.")
    
    # Load tokenizer for validation decode
    print(f"Loading tokenizer {args.tokenizer_name} for decoding check...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    total_tokens = 0
    for idx, filename in enumerate(files):
        filepath = os.path.join(args.output_dir, filename)
        try:
            arr = np.load(filepath)
            total_tokens += arr.size
            print(f"\n[{filename}]")
            print(f" - Path: {filepath}")
            print(f" - Shape: {arr.shape}")
            print(f" - Dtype: {arr.dtype}")
            print(f" - Token count: {arr.size:,}")
            print(f" - Min token ID: {arr.min()}")
            print(f" - Max token ID: {arr.max()}")
            
            # Basic validation checks
            if arr.min() < 0:
                print("   [Warning] Array contains negative token IDs!")
            if arr.max() >= len(tokenizer):
                print(f"   [Warning] Array contains token IDs >= tokenizer vocab size ({len(tokenizer)})!")
                
            # Decode sample text from first and last file
            if idx == 0 or idx == len(files) - 1:
                print(f" - Sample Decoded Tokens (first 100):")
                sample = arr[:100]
                decoded_text = tokenizer.decode(sample)
                print("-" * 60)
                print(decoded_text)
                print("-" * 60)
                
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            
    print("\n" + "="*50)
    print("VERIFICATION COMPLETED")
    print(f"Total token files verified: {len(files)}")
    print(f"Total tokens across all files: {total_tokens:,}")
    print("="*50)

if __name__ == "__main__":
    main()
