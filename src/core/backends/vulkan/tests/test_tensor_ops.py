"""Tests for Vulkan tensor operations."""

import pytest
import torch
import vulkan as vk

from core.backends.vulkan.tensor_ops import TensorDescriptor, VulkanTensorOps
from core.backends.vulkan.pipeline import PipelineType


@pytest.fixture
def vulkan_device():
    """Create a Vulkan device for testing."""
    # Create instance
    app_info = vk.VkApplicationInfo(
        pApplicationName="Test",
        applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
        pEngineName="No Engine",
        engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
        apiVersion=vk.VK_API_VERSION_1_0
    )
    
    create_info = vk.VkInstanceCreateInfo(
        pApplicationInfo=app_info,
        enabledLayerCount=0,
        ppEnabledLayerNames=None,
        enabledExtensionCount=0,
        ppEnabledExtensionNames=None
    )
    instance = vk.vkCreateInstance(create_info, None)
    
    # Get physical device
    physical_devices = vk.vkEnumeratePhysicalDevices(instance)
    if not physical_devices:
        pytest.skip("No Vulkan devices available")
    physical_device = physical_devices[0]
    
    # Find compute queue family
    queue_families = vk.vkGetPhysicalDeviceQueueFamilyProperties(physical_device)
    compute_queue_family = next(
        i for i, props in enumerate(queue_families)
        if props.queueFlags & vk.VK_QUEUE_COMPUTE_BIT
    )
    
    # Create device
    queue_create_info = vk.VkDeviceQueueCreateInfo(
        queueFamilyIndex=compute_queue_family,
        queueCount=1,
        pQueuePriorities=[1.0]
    )
    
    device_create_info = vk.VkDeviceCreateInfo(
        queueCreateInfoCount=1,
        pQueueCreateInfos=[queue_create_info],
        enabledLayerCount=0,
        ppEnabledLayerNames=None,
        enabledExtensionCount=0,
        ppEnabledExtensionNames=None
    )
    
    device = vk.vkCreateDevice(physical_device, device_create_info, None)
    queue = vk.vkGetDeviceQueue(device, compute_queue_family, 0)
    
    yield device, queue, compute_queue_family
    
    # Cleanup
    vk.vkDestroyDevice(device, None)
    vk.vkDestroyInstance(instance, None)


@pytest.fixture
def tensor_ops(vulkan_device):
    """Create VulkanTensorOps instance."""
    device, queue, queue_family = vulkan_device
    ops = VulkanTensorOps(device, queue, queue_family)
    yield ops
    ops.cleanup()


def test_tensor_descriptor_creation(tensor_ops):
    """Test creating a tensor descriptor."""
    tensor = torch.randn(2, 3)
    descriptor = tensor_ops.register_tensor(tensor)
    
    assert isinstance(descriptor, TensorDescriptor)
    assert descriptor.shape == tensor.shape
    assert descriptor.dtype == tensor.dtype
    assert descriptor.is_contiguous == tensor.is_contiguous()
    assert descriptor.size == tensor.nelement() * tensor.element_size()


def test_matmul(tensor_ops):
    """Test matrix multiplication."""
    # Create test matrices
    a = torch.randn(32, 16)
    b = torch.randn(16, 32)
    
    # Compute expected result using PyTorch
    expected = torch.matmul(a, b)
    
    # Compute result using Vulkan
    result = tensor_ops.matmul(a, b)
    
    # Compare results
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_attention(tensor_ops):
    """Test adaptive attention computation."""
    batch, num_heads, seq_len, head_dim = 2, 4, 128, 64
    queries = torch.randn(batch, num_heads, seq_len, head_dim)
    keys = torch.randn(batch, num_heads, seq_len, head_dim)
    values = torch.randn(batch, num_heads, seq_len, head_dim)
    
    # Run adaptive attention
    output, metrics = tensor_ops.adaptive_attention(
        queries=queries,
        keys=keys,
        values=values,
        resolution=0.8,
        density_threshold=0.5,
    )
    
    # Basic shape checks
    assert output.shape == (batch, num_heads, seq_len, head_dim)
    assert metrics.shape == (batch, num_heads, seq_len // 32)  # Using tile_size=32
    
    # Check output has valid values
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    
    # Check metrics are in valid range [0, 1]
    assert (metrics >= 0).all() and (metrics <= 1).all()


def test_cleanup(tensor_ops):
    """Test cleanup releases resources properly."""
    tensor = torch.randn(2, 3)
    descriptor = tensor_ops.register_tensor(tensor)
    
    # Cache should contain the tensor
    assert tensor in tensor_ops._tensor_cache
    
    # Cleanup
    tensor_ops.cleanup()
    
    # Cache should be empty
    assert len(tensor_ops._tensor_cache) == 0
