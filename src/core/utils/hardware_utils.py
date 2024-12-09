"""Hardware detection and configuration utilities."""

from dataclasses import dataclass
from typing import Dict

import psutil
import torch


@dataclass
class HardwareProfile:
    """System hardware profile."""

    total_memory_gb: float
    available_memory_gb: float
    cpu_count: int
    has_vulkan: bool
    device_name: str
    device_capabilities: Dict[str, any]


def get_hardware_profile() -> HardwareProfile:
    """Get current system hardware profile.

    Returns:
        HardwareProfile: Current hardware configuration
    """
    memory = psutil.virtual_memory()
    total_memory_gb = memory.total / (1024**3)  # Convert to GB
    available_memory_gb = memory.available / (1024**3)

    # Check Vulkan availability
    has_vulkan = hasattr(torch, "vulkan") and torch.vulkan.is_available()

    # Determine device name and capabilities
    if has_vulkan:
        device_name = "vulkan"
        device_capabilities = {
            "memory": available_memory_gb
        }  # Use system memory as proxy until Vulkan API provides this
    else:
        device_name = "cpu"
        device_capabilities = {"memory": available_memory_gb}

    return HardwareProfile(
        total_memory_gb=total_memory_gb,
        available_memory_gb=available_memory_gb,
        cpu_count=psutil.cpu_count(),
        has_vulkan=has_vulkan,
        device_name=device_name,
        device_capabilities=device_capabilities,
    )


def get_safe_model_config(profile: HardwareProfile) -> Dict[str, int]:
    """Get safe model configuration based on hardware profile.

    Args:
        profile: Current hardware profile

    Returns:
        Dict containing safe model configuration parameters
    """
    # Base configuration for minimal resource usage
    min_config = {
        "batch_size": 1,
        "seq_length": 64,
        "hidden_dim": 128,
        "num_heads": 2,
        "num_runs": 3,
    }

    # Medium configuration for average systems
    medium_config = {
        "batch_size": 1,
        "seq_length": 128,
        "hidden_dim": 256,
        "num_heads": 4,
        "num_runs": 5,
    }

    # Full configuration for high-end systems
    full_config = {
        "batch_size": 2,
        "seq_length": 256,
        "hidden_dim": 512,
        "num_heads": 8,
        "num_runs": 10,
    }

    # Calculate approximate memory requirements (in GB)
    def estimate_memory_gb(config: Dict[str, int]) -> float:
        # Each tensor element is 4 bytes (float32)
        # Key tensors: input, key, query, value, attention weights, output
        # Plus overhead factor of 3.5 for intermediate computations, gradients, PyTorch overhead,
        # and additional memory spikes during computation
        bytes_per_element = 4
        overhead_factor = 3.5

        # Main tensor sizes
        input_size = config["batch_size"] * config["seq_length"] * config["hidden_dim"]
        key_query_value_size = 3 * input_size  # One each for key, query, value
        attention_size = (
            config["batch_size"]
            * config["num_heads"]
            * config["seq_length"]
            * config["seq_length"]
        )
        output_size = input_size

        # Add model parameters size
        num_params = (4 * config["hidden_dim"] * config["hidden_dim"]) + (
            2 * config["hidden_dim"]
        )  # Linear layers + biases

        total_elements = (
            input_size
            + key_query_value_size
            + attention_size
            + output_size
            + num_params
        ) * overhead_factor
        return (total_elements * bytes_per_element) / (1024**3)  # Convert to GB

    # Get device memory limit
    memory_limit = profile.available_memory_gb

    # Use more conservative memory limits:
    # 10% for minimal config
    # 20% for medium/full configs
    safe_memory_min = memory_limit * 0.10
    safe_memory = memory_limit * 0.20

    # Select configuration based on estimated memory usage
    estimated_min = estimate_memory_gb(min_config)
    estimated_medium = estimate_memory_gb(medium_config)
    estimated_full = estimate_memory_gb(full_config)

    # If we have less than 4GB available, always use minimal config with reduced sequence length
    if memory_limit < 4.0:
        min_config["seq_length"] = 32
        min_config["batch_size"] = 1
        min_config["hidden_dim"] = 64
        min_config["num_heads"] = 1
        return min_config

    if estimated_full < safe_memory:
        return full_config
    if estimated_medium < safe_memory:
        return medium_config
    if estimated_min < safe_memory_min:
        return min_config
    # If even minimal config is too big, reduce all parameters
    min_config["seq_length"] = 32
    min_config["batch_size"] = 1
    min_config["hidden_dim"] = 64
    min_config["num_heads"] = 1
    return min_config
