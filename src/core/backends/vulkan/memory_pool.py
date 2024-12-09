from typing import Dict, List, Optional, Tuple
import vulkan as vk
from dataclasses import dataclass
import numpy as np

@dataclass
class MemoryBlock:
    """Represents a block of GPU memory."""
    memory: vk.DeviceMemory
    size: int
    offset: int
    is_free: bool
    alignment: int

class MemoryPool:
    """Manages a pool of Vulkan memory blocks for efficient reuse."""
    
    def __init__(self, device: vk.Device, memory_type_index: int, initial_size: int = 1024 * 1024):
        self.device = device
        self.memory_type_index = memory_type_index
        self.initial_size = initial_size
        
        # Memory pools for different usage patterns
        self.pools: Dict[vk.MemoryPropertyFlags, List[MemoryBlock]] = {
            vk.MEMORY_PROPERTY_DEVICE_LOCAL_BIT: [],
            vk.MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.MEMORY_PROPERTY_HOST_COHERENT_BIT: []
        }
        
        # Statistics for adaptive sizing
        self.allocation_stats = {
            'hits': 0,
            'misses': 0,
            'fragmentation': 0.0
        }
    
    def allocate(self, 
                 size: int, 
                 alignment: int,
                 memory_properties: vk.MemoryPropertyFlags) -> Tuple[vk.DeviceMemory, int]:
        """Allocate memory from the pool."""
        
        # Adjust size for alignment
        aligned_size = (size + alignment - 1) & ~(alignment - 1)
        
        # Try to find existing block
        block = self._find_free_block(aligned_size, alignment, memory_properties)
        if block is not None:
            self.allocation_stats['hits'] += 1
            block.is_free = False
            return block.memory, block.offset
        
        self.allocation_stats['misses'] += 1
        
        # Create new block if none found
        allocation_size = max(aligned_size, self.initial_size)
        memory = self._allocate_memory(allocation_size, memory_properties)
        
        block = MemoryBlock(
            memory=memory,
            size=allocation_size,
            offset=0,
            is_free=False,
            alignment=alignment
        )
        
        self.pools[memory_properties].append(block)
        return memory, 0
    
    def free(self, memory: vk.DeviceMemory, offset: int):
        """Return memory block to the pool."""
        for blocks in self.pools.values():
            for block in blocks:
                if block.memory == memory and block.offset == offset:
                    block.is_free = True
                    self._maybe_merge_blocks(blocks)
                    self._update_fragmentation()
                    return
    
    def _find_free_block(self,
                        size: int,
                        alignment: int,
                        memory_properties: vk.MemoryPropertyFlags) -> Optional[MemoryBlock]:
        """Find a suitable free block."""
        if memory_properties not in self.pools:
            return None
            
        blocks = self.pools[memory_properties]
        
        # Best-fit strategy
        best_fit = None
        min_waste = float('inf')
        
        for block in blocks:
            if not block.is_free:
                continue
                
            # Check alignment
            aligned_offset = (block.offset + alignment - 1) & ~(alignment - 1)
            waste = aligned_offset - block.offset
            
            if block.size - waste >= size and waste < min_waste:
                best_fit = block
                min_waste = waste
        
        return best_fit
    
    def _allocate_memory(self, size: int, memory_properties: vk.MemoryPropertyFlags) -> vk.DeviceMemory:
        """Allocate new Vulkan memory."""
        alloc_info = vk.MemoryAllocateInfo(
            allocation_size=size,
            memory_type_index=self.memory_type_index
        )
        
        return vk.allocate_memory(self.device, alloc_info, None)
    
    def _maybe_merge_blocks(self, blocks: List[MemoryBlock]):
        """Merge adjacent free blocks."""
        i = 0
        while i < len(blocks) - 1:
            curr = blocks[i]
            next_block = blocks[i + 1]
            
            if (curr.is_free and next_block.is_free and
                curr.memory == next_block.memory and
                curr.offset + curr.size == next_block.offset):
                
                # Merge blocks
                curr.size += next_block.size
                blocks.pop(i + 1)
                continue
            
            i += 1
    
    def _update_fragmentation(self):
        """Update fragmentation statistics."""
        total_size = 0
        free_size = 0
        
        for blocks in self.pools.values():
            for block in blocks:
                total_size += block.size
                if block.is_free:
                    free_size += block.size
        
        if total_size > 0:
            self.allocation_stats['fragmentation'] = 1.0 - (free_size / total_size)
    
    def get_stats(self) -> Dict:
        """Get pool statistics."""
        return {
            'hits': self.allocation_stats['hits'],
            'misses': self.allocation_stats['misses'],
            'fragmentation': self.allocation_stats['fragmentation'],
            'total_pools': sum(len(blocks) for blocks in self.pools.values())
        }
    
    def cleanup(self):
        """Free all allocated memory."""
        for blocks in self.pools.values():
            for block in blocks:
                vk.free_memory(self.device, block.memory, None)
        self.pools.clear()
