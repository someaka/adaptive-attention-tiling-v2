"""Vulkan memory management system for efficient tensor operations."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import vulkan as vk

@dataclass
class VulkanMemoryMetrics:
    """Metrics for Vulkan memory usage tracking."""
    allocated_memory: int  # Current allocated memory in bytes
    peak_memory: int  # Peak memory usage in bytes
    fragmentation_ratio: float  # Memory fragmentation ratio
    buffer_memory: int  # Memory in Vulkan buffers
    operation_type: str  # Type of operation (allocate, free, transfer)
    device_index: int  # Physical device index

class VulkanMemoryError(Exception):
    """Exception raised for Vulkan memory management errors."""
    pass

class VulkanMemoryManager:
    """Manages Vulkan memory allocation and tracking for tensors."""
    
    def __init__(self, device_index: int = 0):
        """Initialize Vulkan memory manager.
        
        Args:
            device_index: Physical device index to manage
        """
        self._device_index = device_index
        self._allocated_memory = 0
        self._peak_memory = 0
        self._buffer_allocations: Dict[int, int] = {}  # buffer_id -> size
        self._buffer_memory: Dict[int, Any] = {}  # buffer_id -> memory handle
        self._metrics: List[VulkanMemoryMetrics] = []
        self._buffer_pool_size = 1024 * 1024 * 128  # 128MB default buffer pool
        
        try:
            # Initialize Vulkan instance and device
            self._init_vulkan()
        except Exception as e:
            raise VulkanMemoryError(f"Failed to initialize Vulkan: {e}")
        
    def _init_vulkan(self) -> None:
        """Initialize Vulkan instance and select physical device."""
        # Create Vulkan instance
        app_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="AAT",
            applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            pEngineName="No Engine",
            engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            apiVersion=vk.VK_API_VERSION_1_0
        )
        
        create_info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info
        )
        
        self._instance = vk.vkCreateInstance(create_info, None)
        
        # Get physical device
        physical_devices = vk.vkEnumeratePhysicalDevices(self._instance)
        if not physical_devices:
            raise VulkanMemoryError("No Vulkan devices found")
            
        self._physical_device = physical_devices[self._device_index]
        
        # Create logical device
        queue_create_info = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=0,
            queueCount=1,
            pQueuePriorities=[1.0]
        )
        
        device_create_info = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            pQueueCreateInfos=[queue_create_info],
            queueCreateInfoCount=1
        )
        
        self._device = vk.vkCreateDevice(self._physical_device, device_create_info, None)
        
    def allocate_buffer(self, size: Union[int, Tuple[int, ...]], dtype=np.float32) -> Any:
        """Allocate a new buffer on Vulkan device with given size.
        
        Args:
            size: Buffer size in bytes or dimensions
            dtype: Buffer data type if size is dimensions
            
        Returns:
            Vulkan buffer
            
        Raises:
            VulkanMemoryError: If buffer allocation fails or size is invalid
        """
        try:
            # Validate size
            if isinstance(size, tuple):
                if any(s <= 0 for s in size):
                    raise VulkanMemoryError("Buffer dimensions must be positive")
                element_size = np.dtype(dtype).itemsize
                memory_size = element_size * np.prod(size)
            else:
                if size <= 0:
                    raise VulkanMemoryError("Buffer size must be positive")
                memory_size = size
            
            # Check if we need to clear buffer pool
            if self._allocated_memory + memory_size > self._peak_memory + self._buffer_pool_size:
                self._clear_buffer_pool()
            
            # Create buffer
            buffer_create_info = vk.VkBufferCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                size=memory_size,
                usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE
            )
            
            try:
                buffer = vk.vkCreateBuffer(self._device, buffer_create_info, None)
            except Exception as e:
                raise VulkanMemoryError(f"Failed to create buffer: {e}")
            
            try:
                # Allocate memory
                mem_requirements = vk.vkGetBufferMemoryRequirements(self._device, buffer)
                
                alloc_info = vk.VkMemoryAllocateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                    allocationSize=mem_requirements.size,
                    memoryTypeIndex=self._find_memory_type(
                        mem_requirements.memoryTypeBits,
                        vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                    )
                )
                
                memory = vk.vkAllocateMemory(self._device, alloc_info, None)
                
                # Bind memory to buffer
                vk.vkBindBufferMemory(self._device, buffer, memory, 0)
                
            except Exception as e:
                # Clean up buffer if memory allocation fails
                vk.vkDestroyBuffer(self._device, buffer, None)
                raise VulkanMemoryError(f"Failed to allocate memory: {e}")
            
            # Update tracking
            buffer_id = id(buffer)
            self._allocated_memory += memory_size
            self._peak_memory = max(self._peak_memory, self._allocated_memory)
            self._buffer_allocations[buffer_id] = memory_size
            self._buffer_memory[buffer_id] = memory
            
            # Record metrics
            self._metrics.append(VulkanMemoryMetrics(
                allocated_memory=self._allocated_memory,
                peak_memory=self._peak_memory,
                fragmentation_ratio=self.get_fragmentation_ratio(),
                buffer_memory=self.get_buffer_memory(),
                operation_type="allocate",
                device_index=self._device_index
            ))
            
            return buffer
            
        except VulkanMemoryError:
            raise
        except Exception as e:
            raise VulkanMemoryError(f"Failed to allocate buffer: {e}")
    
    def transfer_to_device(self, host_array: np.ndarray) -> Any:
        """Transfer a host array to Vulkan device with memory tracking.
        
        Args:
            host_array: Array on host
            
        Returns:
            Vulkan buffer containing the data
            
        Raises:
            VulkanMemoryError: If transfer fails
        """
        try:
            # Create and allocate buffer
            buffer = self.allocate_buffer(host_array.shape, host_array.dtype)
            
            # Get memory handle
            memory = self._buffer_memory[id(buffer)]
            
            # Map memory
            data_ptr = vk.vkMapMemory(self._device, memory, 0, host_array.nbytes, 0)
            
            # Copy data
            data_ptr[:host_array.nbytes] = host_array.tobytes()
            
            # Unmap memory
            vk.vkUnmapMemory(self._device, memory)
            
            return buffer
            
        except Exception as e:
            # Cleanup on error
            if 'buffer' in locals():
                self.free_buffer(buffer)
            raise VulkanMemoryError(f"Failed to transfer data to device: {e}")
    
    def transfer_to_host(self, device_buffer: Any, shape: Optional[Tuple[int, ...]] = None, dtype=np.float32) -> np.ndarray:
        """Transfer data from device buffer to host array.
        
        Args:
            device_buffer: Vulkan buffer to read from
            shape: Optional shape of output array
            dtype: Data type of output array
            
        Returns:
            NumPy array containing the data
            
        Raises:
            VulkanMemoryError: If transfer fails
        """
        try:
            # Get memory handle
            memory = self._buffer_memory.get(id(device_buffer))
            if memory is None:
                raise VulkanMemoryError("Invalid buffer")
            
            # Get memory requirements
            mem_requirements = vk.vkGetBufferMemoryRequirements(self._device, device_buffer)
            
            # Map memory
            data_ptr = vk.vkMapMemory(self._device, memory, 0, mem_requirements.size, 0)
            
            # Create a copy of the data
            data_size = mem_requirements.size
            if shape is not None:
                data_size = np.prod(shape) * np.dtype(dtype).itemsize
            data_bytes = bytes(data_ptr[:data_size])
            
            # Unmap memory
            vk.vkUnmapMemory(self._device, memory)
            
            # Create numpy array from the copy
            array_data = np.frombuffer(data_bytes, dtype=dtype)
            if shape is not None:
                array_data = array_data.reshape(shape).copy()  # Create a copy to ensure memory safety
            else:
                array_data = array_data.copy()  # Create a copy to ensure memory safety
            
            return array_data
            
        except Exception as e:
            raise VulkanMemoryError(f"Failed to transfer data from device: {e}")
    
    def free_buffer(self, buffer: Any) -> None:
        """Free a Vulkan buffer and its memory.
        
        Args:
            buffer: Buffer to free
            
        Raises:
            VulkanMemoryError: If buffer free fails or buffer is invalid
        """
        try:
            if buffer is None:
                raise VulkanMemoryError("Cannot free None buffer")
                
            # Extract buffer handle if we got a tuple
            if isinstance(buffer, tuple):
                buffer = buffer[0]
                
            buffer_id = id(buffer)
            
            # Get memory handle
            memory = self._buffer_memory.get(buffer_id)
            if memory is None:
                raise VulkanMemoryError("Invalid buffer: no associated memory found")
            
            try:
                # Free memory first
                vk.vkFreeMemory(self._device, memory, None)
                del self._buffer_memory[buffer_id]
                
                # Update tracking
                if buffer_id in self._buffer_allocations:
                    self._allocated_memory -= self._buffer_allocations[buffer_id]
                    del self._buffer_allocations[buffer_id]
                
                # Free buffer
                vk.vkDestroyBuffer(self._device, buffer, None)
                
            except Exception as e:
                raise VulkanMemoryError(f"Failed to free buffer resources: {e}")
            
        except VulkanMemoryError:
            raise
        except Exception as e:
            raise VulkanMemoryError(f"Failed to free buffer: {e}")
    
    def _find_memory_type(self, type_filter: int, properties: int) -> int:
        """Find suitable memory type index.
        
        Args:
            type_filter: Memory type bits
            properties: Required memory properties
            
        Returns:
            Memory type index
            
        Raises:
            VulkanMemoryError: If no suitable memory type found
        """
        try:
            mem_properties = vk.vkGetPhysicalDeviceMemoryProperties(self._physical_device)
            
            for i in range(mem_properties.memoryTypeCount):
                if (type_filter & (1 << i)) and (mem_properties.memoryTypes[i].propertyFlags & properties) == properties:
                    return i
                    
            raise VulkanMemoryError("Failed to find suitable memory type")
            
        except Exception as e:
            raise VulkanMemoryError(f"Failed to find memory type: {e}")
    
    def get_allocated_memory(self) -> int:
        """Get current allocated memory in bytes."""
        return self._allocated_memory
    
    def get_peak_memory(self) -> int:
        """Get peak memory usage in bytes."""
        return self._peak_memory
    
    def get_buffer_memory(self) -> int:
        """Get current Vulkan buffer memory in bytes."""
        return sum(self._buffer_allocations.values())
    
    def get_fragmentation_ratio(self) -> float:
        """Calculate memory fragmentation ratio."""
        if not self._allocated_memory:
            return 0.0
        
        # Calculate fragmentation based on buffer allocation patterns
        total_gaps = 0
        sorted_allocations = sorted(self._buffer_allocations.values())
        
        for i in range(len(sorted_allocations) - 1):
            gap = sorted_allocations[i+1] - sorted_allocations[i]
            if gap > 0:
                total_gaps += gap
        
        return total_gaps / self._allocated_memory if self._allocated_memory else 0.0
    
    def optimize_memory_layout(self) -> None:
        """Optimize Vulkan memory layout to reduce fragmentation."""
        if not self._buffer_allocations:
            return
            
        # Get all buffers and their sizes
        buffers = []
        for buffer_id, size in self._buffer_allocations.items():
            buffer = None
            try:
                buffer = vk.vkGetBufferDeviceAddress(self._device, buffer_id)
                if buffer:
                    buffers.append((buffer, size))
            except:
                pass
        
        if not buffers:
            return
            
        # Sort buffers by size for better memory locality
        buffers.sort(key=lambda x: x[1], reverse=True)
        
        # TODO: Implement actual Vulkan buffer defragmentation
        # This would involve:
        # 1. Creating new buffers
        # 2. Copying data from old to new buffers
        # 3. Destroying old buffers
        # 4. Updating buffer references
    
    def _clear_buffer_pool(self) -> None:
        """Clear Vulkan buffer pool."""
        try:
            # Reset tracking first
            self._allocated_memory = 0
            self._buffer_allocations.clear()
            self._buffer_memory.clear()
            
        except Exception as e:
            raise VulkanMemoryError(f"Failed to clear buffer pool: {e}")
    
    def cleanup(self) -> None:
        """Clean up Vulkan resources."""
        try:
            # Clear buffer pool first
            try:
                self._clear_buffer_pool()
            except:
                pass
                
            # Then destroy device and instance
            if hasattr(self, '_device'):
                try:
                    vk.vkDeviceWaitIdle(self._device)
                    vk.vkDestroyDevice(self._device, None)
                    delattr(self, '_device')
                except:
                    pass
                    
            if hasattr(self, '_instance'):
                try:
                    vk.vkDestroyInstance(self._instance, None)
                    delattr(self, '_instance')
                except:
                    pass
                    
        except Exception as e:
            raise VulkanMemoryError(f"Failed to cleanup: {e}")
    
    def __del__(self):
        """Cleanup when manager is destroyed."""
        try:
            self.cleanup()
        except:
            pass
