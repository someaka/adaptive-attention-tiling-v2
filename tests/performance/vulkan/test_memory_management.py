"""Tests for Vulkan memory management functionality."""

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from src.core.performance.vulkan.memory_management import VulkanMemoryError, VulkanMemoryManager

# Configure logging
log_dir = Path(__file__).parent.parent.parent / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

log_file = log_dir / "vulkan_memory_tests.log"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


@pytest.fixture
def memory_manager():
    """Fixture to create and cleanup VulkanMemoryManager."""
    manager = None
    try:
        manager = VulkanMemoryManager()
        yield manager
    finally:
        if manager:
            manager.cleanup()


def test_buffer_allocation(memory_manager):
    """Test basic buffer allocation."""
    try:
        # Test allocation with tuple size
        buffer1 = memory_manager.allocate_buffer((100, 100), np.float32)
        assert buffer1 is not None
        logger.info("Successfully allocated buffer with tuple size")

        # Test allocation with byte size
        buffer2 = memory_manager.allocate_buffer(1024)
        assert buffer2 is not None
        logger.info("Successfully allocated buffer with byte size")

        # Verify memory tracking
        assert memory_manager.get_allocated_memory() > 0
        assert memory_manager.get_peak_memory() >= memory_manager.get_allocated_memory()
        logger.info(
            "Memory stats - Allocated: %d, Peak: %d",
            memory_manager.get_allocated_memory(),
            memory_manager.get_peak_memory(),
        )

    except VulkanMemoryError as e:
        logger.exception("Buffer allocation failed: %s", e)
        raise


def test_data_transfer(memory_manager):
    """Test data transfer between host and device."""
    try:
        # Create test data
        test_data = np.random.default_rng(42).random((10, 10))

        # Transfer to device
        device_buffer = memory_manager.transfer_to_device(test_data)
        assert device_buffer is not None
        logger.info("Successfully transferred data to device")

        # Transfer back to host
        host_data = memory_manager.transfer_to_host(device_buffer, test_data.shape, test_data.dtype)
        assert np.allclose(test_data, host_data)
        logger.info("Successfully transferred data back to host")
        logger.debug(
            "Data verification passed. Original sum: %f, Transferred sum: %f",
            test_data.sum(),
            host_data.sum(),
        )

    except VulkanMemoryError as e:
        logger.exception("Data transfer failed: %s", e)
        raise


def test_memory_tracking(memory_manager):
    """Test memory tracking and metrics."""
    try:
        initial_memory = memory_manager.get_allocated_memory()
        logger.info("Initial allocated memory: %d", initial_memory)

        # Allocate multiple buffers
        buffers = []
        sizes = [(100, 100), (200, 200), (300, 300)]

        for size in sizes:
            buffer = memory_manager.allocate_buffer(size, np.float32)
            buffers.append(buffer)
            logger.info("Allocated buffer of size %s", size)

        # Verify memory increases
        assert memory_manager.get_allocated_memory() > initial_memory
        logger.info("Final allocated memory: %d", memory_manager.get_allocated_memory())
        logger.info("Peak memory usage: %d", memory_manager.get_peak_memory())
        logger.info("Fragmentation ratio: %f", memory_manager.get_fragmentation_ratio())

        # Free buffers
        for buffer in buffers:
            memory_manager.free_buffer(buffer)

        logger.info("Successfully freed all buffers")
        logger.info("Memory after cleanup: %d", memory_manager.get_allocated_memory())

    except VulkanMemoryError as e:
        logger.exception("Memory tracking test failed: %s", e)
        raise


def test_error_handling(memory_manager):
    """Test error handling scenarios."""
    try:
        # Test invalid buffer free
        with pytest.raises(VulkanMemoryError):
            memory_manager.free_buffer(None)

        # Test invalid memory map
        with pytest.raises(VulkanMemoryError):
            memory_manager.transfer_to_host(None, (10, 10), np.float32)

        # Test invalid buffer allocation
        with pytest.raises(VulkanMemoryError):
            memory_manager.allocate_buffer((-1, -1))

    except Exception as e:
        logger.exception("Error handling test failed: %s", e)
        raise


def test_buffer_pool_cleanup(memory_manager):
    """Test buffer pool cleanup mechanism."""
    try:
        # Allocate enough buffers to trigger cleanup
        large_size = (1000, 1000)  # 4MB for float32
        buffers = []

        for _ in range(5):  # Should exceed default pool size
            buffer = memory_manager.allocate_buffer(large_size, np.float32)
            buffers.append(buffer)
            logger.info(
                "Allocated large buffer, current memory: %d", memory_manager.get_allocated_memory()
            )

        # Verify pool cleanup occurred
        assert memory_manager.get_allocated_memory() > 0
        logger.info("Final memory after pool test: %d", memory_manager.get_allocated_memory())

    except VulkanMemoryError as e:
        logger.exception("Buffer pool cleanup test failed: %s", e)
        raise


@pytest.fixture(scope="session", autouse=True)
def log_test_session():
    """Log test session start and end."""
    session_start = datetime.now(timezone.utc)
    logger.info("Starting Vulkan memory management test session at %s", session_start)
    yield
    session_end = datetime.now(timezone.utc)
    duration = session_end - session_start
    logger.info("Completed test session. Duration: %s", duration)
    logger.info("=" * 80)
