#!/usr/bin/env python3
"""
Utility script to merge tokenized chunks (.npy files) into a single 
NumPy memmap file for efficient memory-mapped streaming during training.
"""

import os
import re
import argparse
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Create a single memmap file from token chunks")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/project/data",
        help="Directory containing the tokens_*.npy files"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/project/data/tokens.memmap",
        help="Path to the output memmap file"
    )
    return parser.parse_args()

def extract_chunk_idx(filename):
    match = re.search(r'tokens_(\d+)\.npy', filename)
    return int(match.group(1)) if match else -1

def main():
    args = parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist.")
        return

    # 1. Find and sort token files numerically
    print(f"Scanning {args.input_dir} for token files...")
    all_files = os.listdir(args.input_dir)
    token_files = [f for f in all_files if f.startswith("tokens_") and f.endswith(".npy")]
    
    if not token_files:
        print("Error: No tokens_*.npy files found in the input directory.")
        return
        
    token_files = sorted(token_files, key=extract_chunk_idx)
    print(f"Found {len(token_files)} token files. Ordering from {token_files[0]} to {token_files[-1]}.")

    # 2. Inspect the first file to determine dtype and read shapes
    first_filepath = os.path.join(args.input_dir, token_files[0])
    first_arr = np.load(first_filepath, mmap_mode='r')
    dtype = first_arr.dtype
    print(f"Detected token dtype: {dtype}")

    # 3. Calculate total tokens across all chunks
    print("Calculating total tokens...")
    total_tokens = 0
    file_lengths = []
    
    for filename in tqdm(token_files, desc="Scanning chunk sizes"):
        filepath = os.path.join(args.input_dir, filename)
        # Using mmap_mode='r' reads only metadata/header without loading array into memory
        arr = np.load(filepath, mmap_mode='r')
        total_tokens += arr.size
        file_lengths.append(arr.size)
        
    expected_size_gb = (total_tokens * dtype.itemsize) / (1024 ** 3)
    print(f"Total tokens to write: {total_tokens:,}")
    print(f"Expected output file size: {expected_size_gb:.2f} GB")

    # 4. Initialize the memmap file
    print(f"Initializing memmap file at {args.output_file}...")
    # 'w+' creates or overwrites the file for reading and writing
    memmap_arr = np.memmap(args.output_file, dtype=dtype, mode='w+', shape=(total_tokens,))

    # 5. Copy data from npy chunks to the memmap
    offset = 0
    for idx, filename in enumerate(tqdm(token_files, desc="Writing to memmap")):
        filepath = os.path.join(args.input_dir, filename)
        arr = np.load(filepath)
        length = file_lengths[idx]
        memmap_arr[offset : offset + length] = arr
        offset += length

    # 6. Flush to disk
    print("Flushing data to disk...")
    memmap_arr.flush()
    print("Memmap successfully flushed.")

    # 7. Verification step
    print("Verifying the generated memmap file...")
    # Load in read-only mode
    verification_arr = np.memmap(args.output_file, dtype=dtype, mode='r', shape=(total_tokens,))
    
    # Check shape
    assert verification_arr.shape == (total_tokens,), "Verification shape mismatch!"
    
    # Verify first chunk
    first_chunk_len = file_lengths[0]
    first_chunk_orig = np.load(os.path.join(args.input_dir, token_files[0]))
    assert np.array_equal(verification_arr[:first_chunk_len], first_chunk_orig), "First chunk verification failed!"
    
    # Verify last chunk
    last_chunk_len = file_lengths[-1]
    last_chunk_orig = np.load(os.path.join(args.input_dir, token_files[-1]))
    assert np.array_equal(verification_arr[-last_chunk_len:], last_chunk_orig), "Last chunk verification failed!"
    
    print("Verification successful! All checks passed.")
    print(f"Merged memmap file is ready at: {args.output_file}")

if __name__ == "__main__":
    main()
