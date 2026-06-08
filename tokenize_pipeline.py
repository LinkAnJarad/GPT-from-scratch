#!/usr/bin/env python3
"""
Pipeline to tokenize and chunk text from multiple streaming datasets.
Saves tokenized chunks as numpy arrays under the target output directory.
"""

import os
import sys
import argparse
import time
import random
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Streaming Multi-Dataset Tokenizer Pipeline")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/project/data",
        help="Directory to save tokenized chunks (.npy files)"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10_000_000,
        help="Number of tokens per chunk"
    )
    parser.add_argument(
        "--target_tokens",
        type=int,
        default=20_000_000_000,
        help="Total target tokens to extract (approximate)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for parallel tokenizer encoding"
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="unsloth/llama-3-8b",
        help="HF model path or name for the tokenizer"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset interleaving"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set seed for reproducibility of interleaving choices
    random.seed(args.seed)
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading tokenizer: {args.tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    eos_token_id = tokenizer.eos_token_id
    vocab_size = len(tokenizer)
    
    # Determine appropriate numpy dtype to prevent overflow
    # Llama 3 has a vocabulary size of 128,256, which exceeds the max of uint16 (65,535).
    if vocab_size > 65535:
        dtype = np.uint32
        print(f"Vocab size is {vocab_size} (> 65,535). Using np.uint32 to prevent overflow.")
    else:
        dtype = np.uint16
        print(f"Vocab size is {vocab_size} (<= 65,535). Using np.uint16.")

    # Dataset configs
    # We define paths, configurations, splits, text columns, and weights.
    dataset_configs = [
        {
            "path": "openbmb/Ultra-FineWeb",
            "name": None,
            "split": "en",
            "text_column": "content",
            "weight": 0.6,
        },
        {
            "path": "Salesforce/wikitext",
            "name": "wikitext-103-raw-v1",
            "split": "train",
            "text_column": "text",
            "weight": 0.2,
        },
        {
            "path": "sedthh/gutenberg_english",
            "name": None,
            "split": "train",
            "text_column": "TEXT",
            "weight": 0.2,
        },
    ]

    print("\nInitializing streaming datasets:")
    iterators = {}
    weights = {}
    text_columns = {}
    dataset_names = {}
    
    for idx, cfg in enumerate(dataset_configs):
        path = cfg["path"]
        name = cfg["name"]
        split = cfg["split"]
        
        print(f" - Loading stream for {path} (split: {split}, name: {name})...")
        try:
            ds = load_dataset(path, name, split=split, streaming=True)
            iterators[idx] = iter(ds)
            weights[idx] = cfg["weight"]
            text_columns[idx] = cfg["text_column"]
            dataset_names[idx] = path
            print(f"   Successfully loaded {path}")
        except Exception as e:
            print(f"   Error loading dataset {path}: {e}")
            print("   Proceeding with remaining datasets...")

    # Text buffers for batch tokenization
    text_buffers = {idx: [] for idx in iterators}
    
    # Progress metrics
    total_tokens_saved = 0
    chunk_idx = 0
    current_chunk = []
    
    tokens_per_dataset = {idx: 0 for idx in iterators}
    docs_per_dataset = {idx: 0 for idx in iterators}
    
    start_time = time.time()
    last_report_time = start_time
    last_report_tokens = 0

    print(f"\nStarting tokenization pipeline. Target: {args.target_tokens:,} tokens. Chunk size: {args.chunk_size:,} tokens.")
    
    # Initialize tqdm progress bar
    pbar = tqdm(total=args.target_tokens, desc="Tokens processed", unit="tok")

    while iterators and total_tokens_saved < args.target_tokens:
        # Sample an active dataset based on normalized weights
        active_indices = list(iterators.keys())
        active_weights = [weights[idx] for idx in active_indices]
        
        # Draw a dataset choice
        sampled_idx = random.choices(active_indices, weights=active_weights, k=1)[0]
        
        # Retrieve the next document
        try:
            row = next(iterators[sampled_idx])
            text = row[text_columns[sampled_idx]]
            if text and text.strip():
                text_buffers[sampled_idx].append(text)
                docs_per_dataset[sampled_idx] += 1
        except StopIteration:
            pbar.write(f"\n[Info] Dataset {dataset_names[sampled_idx]} has been exhausted. Removing from active streams.")
            del iterators[sampled_idx]
            continue
        except Exception as e:
            pbar.write(f"\n[Warning] Error reading from {dataset_names[sampled_idx]}: {e}. Skipping row.")
            continue
            
        # If the batch is full, tokenize and append
        if len(text_buffers[sampled_idx]) >= args.batch_size:
            texts = text_buffers[sampled_idx]
            text_buffers[sampled_idx] = []
            
            # Batch tokenization
            batch_tokens = tokenizer(texts, add_special_tokens=False)["input_ids"]
            
            # Append tokens and EOS token id
            batch_total_tokens = 0
            for tokens in batch_tokens:
                current_chunk.extend(tokens)
                current_chunk.append(eos_token_id)
                num_tok = len(tokens) + 1
                tokens_per_dataset[sampled_idx] += num_tok
                batch_total_tokens += num_tok
                
            pbar.update(batch_total_tokens)
                
        # Save chunk(s) if size target is met
        while len(current_chunk) >= args.chunk_size:
            chunk_to_save = current_chunk[:args.chunk_size]
            current_chunk = current_chunk[args.chunk_size:]
            
            arr = np.array(chunk_to_save, dtype=dtype)
            chunk_path = os.path.join(args.output_dir, f"tokens_{chunk_idx}.npy")
            np.save(chunk_path, arr)
            
            total_tokens_saved += args.chunk_size
            chunk_idx += 1
            pbar.set_postfix({"chunks": chunk_idx}, refresh=True)
            
            # Print periodic report
            now = time.time()
            elapsed = now - start_time
            interval_elapsed = now - last_report_time
            interval_tokens = total_tokens_saved - last_report_tokens
            
            speed = interval_tokens / interval_elapsed if interval_elapsed > 0 else 0
            overall_speed = total_tokens_saved / elapsed if elapsed > 0 else 0
            
            pbar.write(
                f"Saved chunk {chunk_idx - 1} | Total tokens saved: {total_tokens_saved:,} | "
                f"Speed: {speed:,.1f} tok/sec (Overall: {overall_speed:,.1f} tok/sec) | "
                f"Elapsed: {elapsed/60:.2f} min"
            )
            
            last_report_time = now
            last_report_tokens = total_tokens_saved

    # Stream loop ended or target reached. Flush remaining buffers.
    pbar.write("\nFlushing remaining buffers...")
    for idx, texts in text_buffers.items():
        if texts:
            batch_tokens = tokenizer(texts, add_special_tokens=False)["input_ids"]
            batch_total_tokens = 0
            for tokens in batch_tokens:
                current_chunk.extend(tokens)
                current_chunk.append(eos_token_id)
                num_tok = len(tokens) + 1
                tokens_per_dataset[idx] += num_tok
                batch_total_tokens += num_tok
            pbar.update(batch_total_tokens)

    # Save remaining chunks of size CHUNK_SIZE
    while len(current_chunk) >= args.chunk_size:
        chunk_to_save = current_chunk[:args.chunk_size]
        current_chunk = current_chunk[args.chunk_size:]
        
        arr = np.array(chunk_to_save, dtype=dtype)
        chunk_path = os.path.join(args.output_dir, f"tokens_{chunk_idx}.npy")
        np.save(chunk_path, arr)
        
        total_tokens_saved += args.chunk_size
        chunk_idx += 1
        pbar.write(f"Saved chunk {chunk_idx - 1} | Total tokens saved: {total_tokens_saved:,}")
        pbar.set_postfix({"chunks": chunk_idx}, refresh=True)

    # Save final remainder chunk if any tokens are left
    if current_chunk:
        arr = np.array(current_chunk, dtype=dtype)
        chunk_path = os.path.join(args.output_dir, f"tokens_{chunk_idx}.npy")
        np.save(chunk_path, arr)
        total_tokens_saved += len(current_chunk)
        pbar.write(f"Saved remainder chunk {chunk_idx} with {len(current_chunk):,} tokens.")
        chunk_idx += 1
        
    pbar.close()

    # Print final summary stats
    elapsed = time.time() - start_time
    print("\n" + "="*50)
    print("TOKENIZATION PIPELINE COMPLETE")
    print("="*50)
    print(f"Total time elapsed: {elapsed/60:.2f} minutes")
    print(f"Total tokens saved: {total_tokens_saved:,}")
    print(f"Total chunks saved: {chunk_idx}")
    print("\nTokens processed by dataset:")
    for idx, name in dataset_names.items():
        share = (tokens_per_dataset[idx] / max(1, sum(tokens_per_dataset.values()))) * 100
        print(f" - {name}: {tokens_per_dataset[idx]:,} tokens ({share:.1f}%), {docs_per_dataset[idx]:,} documents")
    print("="*50)

if __name__ == "__main__":
    main()
