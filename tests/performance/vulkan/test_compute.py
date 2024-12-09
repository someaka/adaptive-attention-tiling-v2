"""Performance tests for Vulkan compute operations.

This module tests the performance characteristics of Vulkan compute operations
in the Adaptive Attention Tiling system, focusing on:
1. Shader compilation and execution
2. Memory transfer efficiency
3. Workgroup optimization
4. Resource management
"""

import os
import subprocess
import pytest
import torch

from src.core.performance.vulkan.compute import VulkanCompute

# Test configurations
WORKGROUP_SIZES = [(8, 8), (16, 16), (32, 32)]
MATRIX_SIZES = [(256, 256), (1024, 1024), (4096, 4096)]
BATCH_SIZES = [1, 8, 32]
SHADER_TYPES = ["pattern", "flow", "attention"]

# Shader paths
SHADER_DIR = os.path.join("src", "core", "performance", "vulkan", "shaders")
SHADER_PATHS = {
    "pattern": os.path.join(SHADER_DIR, "pattern_compute.comp"),
    "flow": os.path.join(SHADER_DIR, "flow_compute.comp"),
    "attention": os.path.join(SHADER_DIR, "attention_compute.comp")
}

def compile_shader(shader_path: str) -> str:
    """Compile GLSL shader to SPIR-V."""
    spv_path = shader_path + ".spv"
    
    # Only recompile if source is newer than SPIR-V
    if os.path.exists(spv_path):
        src_time = os.path.getmtime(shader_path)
        spv_time = os.path.getmtime(spv_path)
        if src_time <= spv_time:
            return spv_path

    result = subprocess.run(
        ["glslc", shader_path, "-o", spv_path],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Shader compilation failed: {result.stderr}")
    
    return spv_path

@pytest.fixture
def vulkan_compute():
    """Create a VulkanCompute instance for testing."""
    # Ensure all shaders are compiled
    for shader_path in SHADER_PATHS.values():
        compile_shader(shader_path)
    return VulkanCompute(enable_profiling=True)

def generate_test_data(size: tuple[int, int], batch_size: int) -> torch.Tensor:
    """Generate test data for compute operations."""
    return torch.randn(batch_size, *size)

@pytest.mark.parametrize("shader_type", SHADER_TYPES)
@pytest.mark.parametrize("matrix_size", MATRIX_SIZES)
def test_shader_compilation(
    vulkan_compute: VulkanCompute, shader_type: str, matrix_size: tuple[int, int]
):
    """Test shader compilation performance and efficiency."""
    # Ensure shader is compiled
    shader_path = SHADER_PATHS[shader_type]
    spv_path = compile_shader(shader_path)
    assert os.path.exists(spv_path), f"SPIR-V shader {spv_path} not found"

    # Test shader loading and compilation
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    shader = vulkan_compute.compile_shader(shader_type, matrix_size)
    end_time.record()

    torch.cuda.synchronize()
    compilation_time = start_time.elapsed_time(end_time)

    metrics = vulkan_compute.get_metrics()

    # Performance assertions
    assert compilation_time < 1000  # Less than 1 second
    assert metrics.shader_size < 1024 * 1024  # Less than 1MB
    assert metrics.compilation_success

    # Verify shader properties
    assert shader.specialization_constants["LOCAL_SIZE_X"] > 0
    assert shader.specialization_constants["LOCAL_SIZE_Y"] > 0


@pytest.mark.parametrize("workgroup_size", WORKGROUP_SIZES)
@pytest.mark.parametrize("matrix_size", MATRIX_SIZES)
def test_workgroup_optimization(
    vulkan_compute: VulkanCompute,
    workgroup_size: tuple[int, int],
    matrix_size: tuple[int, int],
):
    """Test impact of different workgroup sizes on performance."""
    data = generate_test_data(matrix_size, batch_size=1)

    # Configure workgroup size
    vulkan_compute.set_workgroup_size(*workgroup_size)

    # Run compute operation
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    vulkan_compute.execute_compute(data)
    end_time.record()

    torch.cuda.synchronize()
    execution_time = start_time.elapsed_time(end_time)

    metrics = vulkan_compute.get_metrics()

    # Performance analysis
    occupancy = metrics.active_warps / metrics.max_warps
    assert occupancy > 0.6  # At least 60% occupancy

    # Record performance for different sizes
    return {
        "workgroup_size": workgroup_size,
        "execution_time": execution_time,
        "occupancy": occupancy,
        "memory_transfers": metrics.memory_transfers,
    }


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("matrix_size", MATRIX_SIZES)
def test_memory_transfer(
    vulkan_compute: VulkanCompute, batch_size: int, matrix_size: tuple[int, int]
):
    """Test memory transfer performance between host and device."""
    data = generate_test_data(matrix_size, batch_size)

    # Test host to device transfer
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    device_data = vulkan_compute.transfer_to_device(data)
    end_time.record()

    torch.cuda.synchronize()
    h2d_time = start_time.elapsed_time(end_time)

    # Test device to host transfer
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    host_data = vulkan_compute.transfer_to_host(device_data)
    end_time.record()

    torch.cuda.synchronize()
    d2h_time = start_time.elapsed_time(end_time)

    vulkan_compute.get_metrics()

    # Performance assertions
    data_size = data.numel() * data.element_size()
    h2d_bandwidth = data_size / (h2d_time / 1000) / 1e9  # GB/s
    d2h_bandwidth = data_size / (d2h_time / 1000) / 1e9  # GB/s

    assert h2d_bandwidth > 1.0  # At least 1 GB/s
    assert d2h_bandwidth > 1.0  # At least 1 GB/s
    assert torch.allclose(data, host_data, rtol=1e-5)


@pytest.mark.parametrize("shader_type", SHADER_TYPES)
def test_resource_management(vulkan_compute: VulkanCompute, shader_type: str):
    """Test Vulkan resource management efficiency."""
    size = (1024, 1024)
    data = generate_test_data(size, batch_size=1)

    # Warm-up
    _ = vulkan_compute.execute_compute(data, shader_type)

    # Test resource creation and deletion
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    for _ in range(100):
        buffer = vulkan_compute.create_buffer(data.size())
        vulkan_compute.delete_buffer(buffer)
    end_time.record()

    torch.cuda.synchronize()
    resource_time = start_time.elapsed_time(end_time)

    metrics = vulkan_compute.get_metrics()

    # Resource management assertions
    assert metrics.active_buffers == 0  # All buffers cleaned up
    assert metrics.memory_leaks == 0  # No memory leaks
    assert resource_time < 1000  # Less than 1 second for 100 operations


def test_descriptor_set_optimization(vulkan_compute: VulkanCompute):
    """Test descriptor set allocation and binding optimization."""
    size = (512, 512)
    num_sets = 100

    # Create multiple descriptor sets
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    descriptor_sets = []
    for _ in range(num_sets):
        data = generate_test_data(size, batch_size=1)
        desc_set = vulkan_compute.create_descriptor_set(data)
        descriptor_sets.append(desc_set)
    end_time.record()

    torch.cuda.synchronize()
    creation_time = start_time.elapsed_time(end_time)

    metrics = vulkan_compute.get_metrics()

    # Performance assertions
    assert creation_time / num_sets < 1.0  # Less than 1ms per set
    assert metrics.descriptor_pool_fragmentation < 0.1  # Less than 10% fragmentation

    # Cleanup
    for desc_set in descriptor_sets:
        vulkan_compute.delete_descriptor_set(desc_set)

    assert metrics.active_descriptor_sets == 0  # All sets cleaned up
