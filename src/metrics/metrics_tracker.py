"""Metrics tracking for attention benchmarks."""

from typing import Dict

import psutil
import torch
from torch import nn


class MetricsTracker:
    """Tracks and computes metrics for attention mechanisms."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.reset()

    def reset(self):
        """Reset all tracked metrics."""
        self.total_flops = 0
        self.total_memory = 0
        self.total_time = 0
        self.num_runs = 0

    def compute_metrics(
        self, model: nn.Module, input_tensor: torch.Tensor, output_tensor: torch.Tensor
    ) -> Dict[str, float]:
        """Compute metrics for a model run.

        Args:
            model: The model being benchmarked
            input_tensor: Input tensor to the model
            output_tensor: Output tensor from the model

        Returns:
            Dictionary of computed metrics
        """
        metrics = {}

        # Basic metrics
        metrics["input_size"] = input_tensor.numel()
        metrics["output_size"] = output_tensor.numel()

        # Memory metrics
        process = psutil.Process()
        metrics["peak_memory"] = process.memory_info().rss
        metrics["memory_allocated"] = process.memory_info().vms
        
        # PyTorch memory metrics - using device-agnostic tracking
        metrics["torch_allocated"] = 0  # Placeholder for device-specific memory tracking
        metrics["torch_reserved"] = 0   # Placeholder for device-specific memory tracking

        # Model metrics
        total_params = sum(p.numel() for p in model.parameters())
        metrics["model_size"] = total_params * 4  # Assuming float32

        # Convert any tensor values to float
        metrics = {
            k: float(v) if isinstance(v, torch.Tensor) else v
            for k, v in metrics.items()
        }

        return metrics
