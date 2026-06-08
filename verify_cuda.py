import torch
import numpy as np
import matplotlib
import transformers
import datasets

print("Checking PyTorch and CUDA status:")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    # Run a simple tensor calculation on GPU
    device = torch.device("cuda")
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    z = torch.matmul(x, y)
    print("Tensor calculation on GPU succeeded!")
else:
    print("Warning: CUDA GPU is NOT available to PyTorch!")
