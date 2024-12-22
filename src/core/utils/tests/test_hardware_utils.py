"""Test hardware utilities."""

import pytest
from unittest.mock import patch

import psutil
import torch

from ..hardware_utils import get_hardware_info, get_memory_info


def test_get_hardware_info():
    """Test hardware info retrieval."""
    with patch("psutil.virtual_memory") as mock_vm, patch("psutil.cpu_count") as mock_cpu:
        mock_vm.return_value.total = 16 * 1024**3  # 16GB
        mock_vm.return_value.available = 8 * 1024**3  # 8GB
        mock_cpu.return_value = 8

        info = get_hardware_info()
        assert info.device_name == "cpu"
        assert info.total_memory == 16 * 1024**3
        assert info.available_memory == 8 * 1024**3
        assert info.cpu_count == 8


def test_get_memory_info():
    """Test memory info retrieval."""
    with patch("psutil.virtual_memory") as mock_vm:
        mock_vm.return_value.total = 16 * 1024**3  # 16GB
        mock_vm.return_value.available = 8 * 1024**3  # 8GB
        mock_vm.return_value.used = 8 * 1024**3  # 8GB

        info = get_memory_info()
        assert info["total"] == 16 * 1024**3
        assert info["available"] == 8 * 1024**3
        assert info["used"] == 8 * 1024**3
