"""Tests for hardware utilities."""

from unittest.mock import MagicMock, patch

import pytest

from src.core.utils.hardware_utils import (
    HardwareProfile,
    get_hardware_profile,
    get_safe_model_config,
)


@pytest.fixture
def mock_hardware_low():
    """Mock low-end hardware profile."""
    with patch("psutil.virtual_memory") as mock_memory, patch(
        "psutil.cpu_count"
    ) as mock_cpu, patch("torch.vulkan", create=True) as vulkan_mock:

        vulkan_mock.is_available.return_value = False
        mock_memory.return_value = MagicMock(
            total=8 * (1024**3), available=2 * (1024**3)  # 8GB total  # 2GB available
        )
        mock_cpu.return_value = 2
        yield


@pytest.fixture
def mock_hardware_high():
    """Mock high-end hardware profile."""
    with patch("psutil.virtual_memory") as mock_memory, patch(
        "psutil.cpu_count"
    ) as mock_cpu, patch("torch.vulkan", create=True) as vulkan_mock:

        vulkan_mock.is_available.return_value = True
        mock_memory.return_value = MagicMock(
            total=32 * (1024**3),  # 32GB total
            available=24 * (1024**3),  # 24GB available
        )
        mock_cpu.return_value = 16
        yield


def test_get_hardware_profile_low_end(mock_hardware_low):
    """Test hardware profile detection on low-end system."""
    profile = get_hardware_profile()

    assert profile.total_memory_gb == pytest.approx(8.0)
    assert profile.available_memory_gb == pytest.approx(2.0)
    assert profile.cpu_count == 2
    assert not profile.has_vulkan
    assert profile.device_name == "cpu"


def test_get_hardware_profile_high_end(mock_hardware_high):
    """Test hardware profile detection on high-end system."""
    profile = get_hardware_profile()

    assert profile.total_memory_gb == pytest.approx(32.0)
    assert profile.available_memory_gb == pytest.approx(24.0)
    assert profile.cpu_count == 16
    assert profile.has_vulkan
    assert profile.device_name == "vulkan"


def test_get_safe_model_config_low_memory():
    """Test safe model config generation for low memory system."""
    profile = HardwareProfile(
        total_memory_gb=8.0,
        available_memory_gb=1.0,  # Even less memory to force minimal config
        cpu_count=2,
        has_vulkan=False,
        device_name="cpu",
        device_capabilities={"memory": 1.0},
    )

    config = get_safe_model_config(profile)
    assert config["batch_size"] == 1
    assert config["seq_length"] == 64  # Should get reduced sequence length
    assert config["hidden_dim"] == 256
    assert config["num_heads"] == 4


def test_get_safe_model_config_high_memory():
    """Test safe model config generation for high memory system."""
    profile = HardwareProfile(
        total_memory_gb=32.0,
        available_memory_gb=24.0,
        cpu_count=16,
        has_vulkan=True,
        device_name="vulkan",
        device_capabilities={"memory": 24.0},
    )

    config = get_safe_model_config(profile)
    assert config["batch_size"] == 4
    assert config["seq_length"] == 1024
    assert config["hidden_dim"] == 768
    assert config["num_heads"] == 12


def test_memory_estimation():
    """Test memory estimation is reasonable."""
    profile = get_hardware_profile()
    config = get_safe_model_config(profile)

    # Ensure we're not using more than 25% of available memory
    def estimate_memory_gb(config):
        bytes_per_element = 4
        overhead_factor = 2.5

        input_size = config["batch_size"] * config["seq_length"] * config["hidden_dim"]
        key_query_value_size = 3 * input_size
        attention_size = (
            config["batch_size"]
            * config["num_heads"]
            * config["seq_length"]
            * config["seq_length"]
        )
        output_size = input_size

        total_elements = (
            input_size + key_query_value_size + attention_size + output_size
        ) * overhead_factor
        return (total_elements * bytes_per_element) / (1024**3)

    estimated_memory = estimate_memory_gb(config)
    assert estimated_memory < profile.available_memory_gb * 0.25
