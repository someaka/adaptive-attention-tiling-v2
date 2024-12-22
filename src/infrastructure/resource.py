"""Resource allocation infrastructure."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class ResourceAllocator:
    """Allocator for compute and memory resources."""

    def __init__(
        self,
        device: str = "cpu",
        memory_limit: Optional[int] = None,
        compute_limit: Optional[int] = None
    ):
        """Initialize resource allocator.
        
        Args:
            device: Device to allocate resources on
            memory_limit: Maximum memory allocation in bytes
            compute_limit: Maximum compute units to use
        """
        self.device = device
        self.memory_limit = memory_limit
        self.compute_limit = compute_limit
        
        # Track allocations
        self.memory_allocated = 0
        self.compute_allocated = 0
        self.allocations: Dict[str, Dict] = {}
        
    def allocate_memory(
        self,
        size: int,
        name: str,
        priority: int = 0
    ) -> bool:
        """Allocate memory.
        
        Args:
            size: Size in bytes to allocate
            name: Name for allocation tracking
            priority: Priority level (higher = more important)
            
        Returns:
            True if allocation successful
        """
        # Check if allocation possible
        if (
            self.memory_limit is not None and
            self.memory_allocated + size > self.memory_limit
        ):
            return False
            
        # Track allocation
        self.memory_allocated += size
        self.allocations[name] = {
            "type": "memory",
            "size": size,
            "priority": priority
        }
        
        return True
        
    def allocate_compute(
        self,
        units: int,
        name: str,
        priority: int = 0
    ) -> bool:
        """Allocate compute units.
        
        Args:
            units: Number of compute units
            name: Name for allocation tracking
            priority: Priority level
            
        Returns:
            True if allocation successful
        """
        # Check if allocation possible
        if (
            self.compute_limit is not None and
            self.compute_allocated + units > self.compute_limit
        ):
            return False
            
        # Track allocation
        self.compute_allocated += units
        self.allocations[name] = {
            "type": "compute",
            "units": units,
            "priority": priority
        }
        
        return True
        
    def free(self, name: str):
        """Free allocation.
        
        Args:
            name: Name of allocation to free
        """
        if name not in self.allocations:
            return
            
        allocation = self.allocations[name]
        if allocation["type"] == "memory":
            self.memory_allocated -= allocation["size"]
        else:
            self.compute_allocated -= allocation["units"]
            
        del self.allocations[name]
        
    def get_available_memory(self) -> float:
        """Get available memory in bytes.
        
        Returns:
            Available memory in bytes
        """
        if self.memory_limit is None:
            return float("inf")
        return float(self.memory_limit - self.memory_allocated)
        
    def get_available_compute(self) -> float:
        """Get available compute units.
        
        Returns:
            Available compute units
        """
        if self.compute_limit is None:
            return float("inf")
        return float(self.compute_limit - self.compute_allocated)
        
    def cleanup(self):
        """Clean up all allocations."""
        self.memory_allocated = 0
        self.compute_allocated = 0
        self.allocations.clear()
