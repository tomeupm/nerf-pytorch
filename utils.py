"""
Utility functions for NeRF PyTorch implementation
"""

import torch


def get_device():
    """Get the best available device (CUDA > MPS > CPU)"""
    if torch.cuda.is_available():
        return "cuda"
    # elif torch.backends.mps.is_available():
    #   print("Using MPS backend for Apple Silicon devices.")
    #   return "mps"
    else:
        return "cpu"
