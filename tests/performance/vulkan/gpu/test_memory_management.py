"""Tests for Vulkan memory management functionality."""

import logging
import pytest
import numpy as np
from pathlib import Path
from ctypes import c_void_p, cast, POINTER, c_size_t

from src.core.performance.vulkan_memory import (
    VulkanMemoryManager,
    VulkanBuffer,
    MemoryError as VulkanMemoryError
)

# Configure logging
@pytest.fixture(scope="session", autouse=True)
def configure_logging():
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "vulkan_memory_tests.log"
    
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)

@pytest.fixture
def memory_manager(request):
    """Fixture to create and cleanup VulkanMemoryManager."""
    manager = None
    try:
        # Initialize Vulkan instance and device first
        import vulkan as vk
        app_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="AAT Tests",
            applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            pEngineName="No Engine",
            engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            apiVersion=vk.VK_API_VERSION_1_0
        )
        
        create_info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info
        )
        
        instance = vk.vkCreateInstance(create_info, None)
        physical_devices = vk.vkEnumeratePhysicalDevices(instance)
        if not physical_devices:
            pytest.skip("No Vulkan devices found")
            
        physical_device = physical_devices[0]
        
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
        
        device = vk.vkCreateDevice(physical_device, device_create_info, None)
        
        # Convert device handles to c_void_p
        device_handle = c_void_p(int(device))
        physical_device_handle = c_void_p(int(physical_device))
        
        manager = VulkanMemoryManager(device_handle, physical_device_handle)
        yield manager
    finally:
        if manager:
            manager.cleanup()

@pytest.mark.gpu
def test_tensor_allocation(memory_manager, configure_logging):
    """Test basic tensor allocation."""
    logger = configure_logging
    
    # Test allocation with tuple size
    buffer1 = memory_manager.allocate_tensor((100, 100), np.float32)
    assert isinstance(buffer1, VulkanBuffer)
    assert buffer1.size > 0
    logger.info("Successfully allocated buffer with tuple size")

    # Test allocation with different shape
    buffer2 = memory_manager.allocate_tensor((64, 32, 16), np.float32)
    assert isinstance(buffer2, VulkanBuffer)
    assert buffer2.size > 0
    logger.info("Successfully allocated buffer with 3D shape")

@pytest.mark.gpu
def test_data_transfer(memory_manager, configure_logging):
    """Test data transfer between host and device."""
    logger = configure_logging
    
    # Create test data
    test_data = np.random.rand(100, 100).astype(np.float32)
    
    # Allocate buffer and transfer data
    buffer = memory_manager.allocate_tensor(test_data.shape, test_data.dtype)
    memory_manager.copy_to_device(test_data, buffer)
    
    # Read back data
    result = np.zeros_like(test_data)
    memory_manager.copy_from_device(buffer, result)
    assert np.allclose(test_data, result)
    logger.info("Successfully completed data transfer test")

@pytest.mark.gpu
def test_memory_tracking(memory_manager, configure_logging):
    """Test memory tracking and metrics."""
    logger = configure_logging
    
    # Record initial memory usage
    initial_usage = memory_manager._allocated_memory
    
    # Allocate some buffers
    buffers = []
    for _ in range(5):
        buf = memory_manager.allocate_tensor((50, 50), np.float32)
        buffers.append(buf)
    
    # Check memory increased
    current_usage = memory_manager._allocated_memory
    assert current_usage > initial_usage
    
    # Free buffers
    for buf in buffers:
        memory_manager.free_tensor(buf)
    
    # Verify memory returned to initial state
    final_usage = memory_manager._allocated_memory
    assert final_usage == initial_usage
    logger.info("Successfully completed memory tracking test")

@pytest.mark.gpu
def test_error_handling(memory_manager, configure_logging):
    """Test error handling scenarios."""
    logger = configure_logging
    
    # Test invalid allocation
    with pytest.raises(VulkanMemoryError):
        memory_manager.allocate_tensor((-1, -1), np.float32)
    
    # Test invalid data transfer
    buffer = memory_manager.allocate_tensor((10, 10), np.float32)
    with pytest.raises(VulkanMemoryError):
        memory_manager.copy_to_device(np.zeros((20, 20)), buffer)  # Wrong size
    
    logger.info("Successfully completed error handling test")

@pytest.mark.gpu
def test_buffer_pool_cleanup(memory_manager, configure_logging):
    """Test buffer pool cleanup mechanism."""
    logger = configure_logging
    
    # Allocate multiple buffers
    buffers = []
    for _ in range(10):
        buf = memory_manager.allocate_tensor((20, 20), np.float32)
        buffers.append(buf)
    
    # Free half of them
    for buf in buffers[:5]:
        memory_manager.free_tensor(buf)
    
    # Force cleanup
    memory_manager._clear_buffer_pool()
    
    # Verify remaining buffers are still valid
    for buf in buffers[5:]:
        assert buf.buffer is not None
        assert buf.memory is not None
    
    logger.info("Successfully completed buffer pool cleanup test")
