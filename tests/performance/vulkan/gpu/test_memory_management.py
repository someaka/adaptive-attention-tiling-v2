"""Tests for Vulkan memory management functionality."""

import logging
import pytest
import numpy as np
from pathlib import Path

from src.core.performance.vulkan.memory_management import (
    VulkanMemoryError,
    VulkanMemoryManager,
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
        manager = VulkanMemoryManager()
        yield manager
    finally:
        if manager:
            manager.cleanup()

@pytest.mark.gpu
def test_buffer_allocation(memory_manager, configure_logging):
    """Test basic buffer allocation."""
    logger = configure_logging
    
    # Test allocation with tuple size
    buffer1 = memory_manager.allocate_buffer((100, 100), np.float32)
    assert buffer1 is not None
    logger.info("Successfully allocated buffer with tuple size")

    # Test allocation with byte size
    buffer2 = memory_manager.allocate_buffer(1024)
    assert buffer2 is not None
    logger.info("Successfully allocated buffer with byte size")

@pytest.mark.gpu
def test_data_transfer(memory_manager, configure_logging):
    """Test data transfer between host and device."""
    logger = configure_logging
    
    # Create test data
    test_data = np.random.rand(100, 100).astype(np.float32)
    
    # Allocate buffer and transfer data
    buffer = memory_manager.allocate_buffer(test_data.shape, test_data.dtype)
    memory_manager.copy_to_device(buffer, test_data)
    
    # Read back data
    result = memory_manager.copy_to_host(buffer)
    assert np.allclose(test_data, result)
    logger.info("Successfully completed data transfer test")

@pytest.mark.gpu
def test_memory_tracking(memory_manager, configure_logging):
    """Test memory tracking and metrics."""
    logger = configure_logging
    
    # Record initial memory usage
    initial_usage = memory_manager.get_memory_usage()
    
    # Allocate some buffers
    buffers = []
    for _ in range(5):
        buf = memory_manager.allocate_buffer((50, 50), np.float32)
        buffers.append(buf)
    
    # Check memory increased
    current_usage = memory_manager.get_memory_usage()
    assert current_usage > initial_usage
    
    # Cleanup buffers
    for buf in buffers:
        memory_manager.free_buffer(buf)
    
    # Verify memory returned to initial state
    final_usage = memory_manager.get_memory_usage()
    assert final_usage == initial_usage
    logger.info("Successfully completed memory tracking test")

@pytest.mark.gpu
def test_error_handling(memory_manager, configure_logging):
    """Test error handling scenarios."""
    logger = configure_logging
    
    # Test invalid allocation
    with pytest.raises(VulkanMemoryError):
        memory_manager.allocate_buffer((-1, -1), np.float32)
    
    # Test double free
    buffer = memory_manager.allocate_buffer((10, 10), np.float32)
    memory_manager.free_buffer(buffer)
    with pytest.raises(VulkanMemoryError):
        memory_manager.free_buffer(buffer)
    
    logger.info("Successfully completed error handling test")

@pytest.mark.gpu
def test_buffer_pool_cleanup(memory_manager, configure_logging):
    """Test buffer pool cleanup mechanism."""
    logger = configure_logging
    
    # Allocate multiple buffers
    buffers = []
    for _ in range(10):
        buf = memory_manager.allocate_buffer((20, 20), np.float32)
        buffers.append(buf)
    
    # Free half of them
    for buf in buffers[:5]:
        memory_manager.free_buffer(buf)
    
    # Force cleanup
    memory_manager.cleanup_pool()
    
    # Verify remaining buffers are still valid
    for buf in buffers[5:]:
        assert memory_manager.is_buffer_valid(buf)
    
    logger.info("Successfully completed buffer pool cleanup test")
