import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MemmapDataset(Dataset):
    """
    Memory-mapped Dataset for causal language modeling.
    Lazily loads the numpy memmap to prevent multiprocessing file handle conflicts
    when num_workers > 0.
    """
    def __init__(self, filepath, seq_len, dtype=np.uint32):
        self.filepath = filepath
        self.seq_len = seq_len
        self.dtype = np.dtype(dtype)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Memmap file not found at: {filepath}")
            
        # Get total number of tokens by dividing file size by size of the dtype
        self.total_tokens = os.path.getsize(filepath) // self.dtype.itemsize
        
        # Initialize tokens array placeholder for lazy loading in worker processes
        self.tokens = None

    def __len__(self):
        # We need seq_len + 1 tokens for each (x, y) sample pair
        return (self.total_tokens - 1) // self.seq_len

    def __getitem__(self, idx):
        if self.tokens is None:
            # Open the memory map lazily in the current process/worker
            self.tokens = np.memmap(self.filepath, dtype=self.dtype, mode='r')
            
        start_idx = idx * self.seq_len
        
        x_np = self.tokens[start_idx : start_idx + self.seq_len]
        y_np = self.tokens[start_idx + 1 : start_idx + 1 + self.seq_len]
        
        x = torch.from_numpy(x_np.astype(np.int64))
        y = torch.from_numpy(y_np.astype(np.int64))
        
        return x, y

def get_dataloader(
    filepath="/project/data_memmap/tokens.memmap",
    seq_len=2048,
    batch_size=8,
    dtype=np.uint32,
    num_workers=4,
    pin_memory=True,
    shuffle=False, # Dont shuffle: causes overhead
    drop_last=True
):

    dataset = MemmapDataset(filepath, seq_len, dtype=dtype)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0)
    )
    
    return dataloader

if __name__ == "__main__":
    filepath = "/project/data_memmap/tokens.memmap"
    if os.path.exists(filepath):
        print(f"File found. Instantiating DataLoader...")
        loader = get_dataloader(
            filepath=filepath,
            seq_len=16, # Small seq_len for testing
            batch_size=4,
            num_workers=2,
            shuffle=False
        )
        print(f"Total dataset length (samples): {len(loader.dataset):,}")
        
        # Fetch a single batch
        print("Fetching first batch...")
        x_batch, y_batch = next(iter(loader))
        
        print("\nBatch statistics:")
        print(f"Input shape: {x_batch.shape} (Expected: [batch_size, seq_len])")
        print(f"Target shape: {y_batch.shape} (Expected: [batch_size, seq_len])")
        print(f"Input dtype: {x_batch.dtype} (Expected: torch.int64)")
        print(f"Target dtype: {y_batch.dtype} (Expected: torch.int64)")
        
        # Print a sample pair to verify target shift
        print("\nSample token comparison (Target should be shifted right by 1):")
        print(f"X[0]: {x_batch[0].tolist()}")
        print(f"Y[0]: {y_batch[0].tolist()}")
        
        # Assert target shift matches input
        assert torch.equal(x_batch[0, 1:], y_batch[0, :-1]), "Target is not correctly shifted!"
        print("Verification successful: targets are correctly shifted by 1 token.")
    else:
        print(f"Could not perform sanity check: no file found at {filepath}")
