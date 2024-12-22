"""Base infrastructure components.

This module provides base classes and utilities for:
1. Resource management
2. Performance monitoring
3. System integration
"""

from dataclasses import dataclass
from typing import Dict, Optional, Any

import torch
import psutil


@dataclass
class DeviceInfo:
    """Device information."""
    
    name: str
    total_memory: int
    available_memory: int
    
    def __str__(self) -> str:
        return (
            f"Device: {self.name}\n"
            f"Total Memory: {self.total_memory / 1024**2:.1f}MB\n"
            f"Available Memory: {self.available_memory / 1024**2:.1f}MB"
        )


class SystemMonitor:
    """System resource monitoring."""
    
    def __init__(self):
        """Initialize system monitor."""
        self.device = torch.device('cpu')
        
    def get_device_info(self) -> DeviceInfo:
        """Get device information.
        
        Returns:
            Device information
        """
        vm = psutil.virtual_memory()
        return DeviceInfo(
            name="cpu",
            total_memory=vm.total,
            available_memory=vm.available
        )
        
    def get_memory_info(self) -> Dict[str, int]:
        """Get memory information.
        
        Returns:
            Dictionary with memory information
        """
        vm = psutil.virtual_memory()
        return {
            'total': vm.total,
            'available': vm.available,
            'used': vm.used
        }
        
    def get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information.
        
        Returns:
            Dictionary with CPU information
        """
        return {
            'count': psutil.cpu_count(),
            'percent': psutil.cpu_percent(interval=1),
            'freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {}
        }