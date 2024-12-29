"""Hardware utilities for system configuration."""

from dataclasses import dataclass
from typing import Dict

import psutil
import torch


@dataclass
class HardwareInfo:
    """Hardware information."""
    
    device_name: str
    total_memory: int
    available_memory: int
    cpu_count: int


def get_hardware_info() -> HardwareInfo:
    """Get hardware information.
    
    Returns:
        HardwareInfo: Hardware information
    """
    device_name = "cpu"
    vm = psutil.virtual_memory()
    
    return HardwareInfo(
        device_name=device_name,
        total_memory=vm.total,
        available_memory=vm.available,
        cpu_count=psutil.cpu_count() or 1
    )


def get_memory_info() -> Dict[str, int]:
    """Get memory information.
    
    Returns:
        Dict[str, int]: Memory information in bytes
    """
    vm = psutil.virtual_memory()
    return {
        'total': vm.total,
        'available': vm.available,
        'used': vm.used
    }
