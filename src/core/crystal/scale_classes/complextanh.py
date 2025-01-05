from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Dict, Any, Union, Optional
from contextlib import contextmanager
import gc
import logging

import numpy as np
import torch
from torch import nn

# Import memory optimization utilities
from src.core.performance.cpu.memory_management import MemoryManager, MemoryMetrics
from src.utils.memory_management_util import optimize_memory, register_tensor
from src.core.crystal.scale_classes.rgflow import RGFlow







class ComplexTanh(nn.Module):
    """Complex-valued tanh activation function."""
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.is_complex():
            return torch.tanh(input.real) + 1j * torch.tanh(input.imag)
        else:
            # For real inputs, just apply regular tanh
            return torch.tanh(input)

