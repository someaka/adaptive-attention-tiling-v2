"""Tests for Vulkan shader performance."""

import time
import vulkan as vk
import numpy as np
import pytest
from ctypes import c_void_p, cast, POINTER, c_size_t, c_uint32
from typing import Tuple, Optional
import struct

from src.core.performance.vulkan_memory import VulkanMemoryManager


def create_vulkan_device() -> Tuple[c_void_p, c_void_p]:
    """Create Vulkan instance and device for testing."""
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
    
    return device_handle, physical_device_handle


def setup_pattern_data(
    batch_size: int, pattern_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create pattern and flow test data.

    Args:
        batch_size: Number of patterns in batch
        pattern_size: Size of each pattern

    Returns:
        Tuple of pattern and flow arrays
    """
    rng = np.random.default_rng()
    pattern = rng.random((batch_size, pattern_size, pattern_size), dtype=np.float32)
    flow = rng.random((batch_size, pattern_size, pattern_size), dtype=np.float32)
    return pattern, flow


def setup_flow_data(
    batch_size: int, manifold_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create metric and connection test data.

    Args:
        batch_size: Number of samples in batch
        manifold_dim: Dimension of manifold

    Returns:
        Tuple of metric and connection arrays
    """
    rng = np.random.default_rng()
    metric = rng.random((batch_size, manifold_dim, manifold_dim), dtype=np.float32)
    connection = rng.random(
        (batch_size, manifold_dim, manifold_dim, manifold_dim), dtype=np.float32
    )
    return metric, connection


PATTERN_EVOLUTION_SHADER = """
#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0) buffer PatternBuffer {
    float pattern[];
};
layout(binding = 1) buffer FlowBuffer {
    float flow[];
};
layout(push_constant) uniform PushConstants {
    float rate;
    uint width;
    uint height;
} constants;

void main() {
    uvec2 pos = gl_GlobalInvocationID.xy;
    if (pos.x >= constants.width || pos.y >= constants.height) {
        return;
    }
    
    uint idx = pos.y * constants.width + pos.x;
    float current = pattern[idx];
    float flow_val = flow[idx];
    
    // Evolution step
    pattern[idx] = current + constants.rate * flow_val;
}
"""

FLOW_COMPUTATION_SHADER = """
#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0) buffer MetricBuffer {
    float metric[];  // [batch][i][j]
};
layout(binding = 1) buffer ConnectionBuffer {
    float connection[];  // [batch][i][j][k]
};
layout(binding = 2) buffer FlowBuffer {
    float flow[];  // [batch][i][j]
};

layout(push_constant) uniform PushConstants {
    uint batch_size;
    uint dim;
} constants;

void main() {
    uint batch_idx = gl_GlobalInvocationID.z;
    uint i = gl_GlobalInvocationID.x;
    uint j = gl_GlobalInvocationID.y;
    
    if (batch_idx >= constants.batch_size || i >= constants.dim || j >= constants.dim) {
        return;
    }
    
    // Compute flow from metric and connection
    float flow_val = 0.0;
    for (uint k = 0; k < constants.dim; k++) {
        uint metric_idx = (batch_idx * constants.dim * constants.dim) + (i * constants.dim) + j;
        uint conn_idx = (batch_idx * constants.dim * constants.dim * constants.dim) + 
                       (i * constants.dim * constants.dim) + 
                       (j * constants.dim) + k;
        
        flow_val += metric[metric_idx] * connection[conn_idx];
    }
    
    uint flow_idx = (batch_idx * constants.dim * constants.dim) + (i * constants.dim) + j;
    flow[flow_idx] = flow_val;
}
"""

CONFIGURABLE_WORKGROUP_SHADER = """
#version 450

layout(local_size_x_id = 0, local_size_y_id = 1) in;

layout(binding = 0) buffer PatternBuffer {
    float pattern[];
};
layout(binding = 1) buffer FlowBuffer {
    float flow[];
};

layout(push_constant) uniform PushConstants {
    float rate;
    uint width;
    uint height;
} constants;

void main() {
    uvec2 pos = gl_GlobalInvocationID.xy;
    if (pos.x >= constants.width || pos.y >= constants.height) {
        return;
    }
    
    uint idx = pos.y * constants.width + pos.x;
    float current = pattern[idx];
    float flow_val = flow[idx];
    
    // Evolution step
    pattern[idx] = current + constants.rate * flow_val;
}
"""

PUSH_CONSTANT_SHADER = """
#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0) buffer PatternBuffer {
    float pattern[];
};
layout(binding = 1) buffer FlowBuffer {
    float flow[];
};

layout(push_constant) uniform PushConstants {
    float rate;
    uint width;
    uint height;
    float params[13];  // Additional parameters to test push constant size
} constants;

void main() {
    uvec2 pos = gl_GlobalInvocationID.xy;
    if (pos.x >= constants.width || pos.y >= constants.height) {
        return;
    }
    
    uint idx = pos.y * constants.width + pos.x;
    float current = pattern[idx];
    float flow_val = flow[idx];
    
    // Apply parameters
    float result = current;
    for (int i = 0; i < 13; i++) {
        result = result + constants.params[i] * flow_val;
    }
    pattern[idx] = result;
}
"""

DESCRIPTOR_SET_SHADER = """
#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0) buffer PatternBuffer {
    float pattern[];
};
layout(binding = 1) buffer FlowBuffer {
    float flow[];
};
layout(binding = 2) uniform ParamBuffer {
    float rate;
    uint width;
    uint height;
    float params[13];
} constants;

void main() {
    uvec2 pos = gl_GlobalInvocationID.xy;
    if (pos.x >= constants.width || pos.y >= constants.height) {
        return;
    }
    
    uint idx = pos.y * constants.width + pos.x;
    float current = pattern[idx];
    float flow_val = flow[idx];
    
    // Apply parameters
    float result = current;
    for (int i = 0; i < 13; i++) {
        result = result + constants.params[i] * flow_val;
    }
    pattern[idx] = result;
}
"""

SINGLE_ITEM_SHADER = """
#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0) buffer PatternBuffer {
    float pattern[];
};
layout(binding = 1) buffer FlowBuffer {
    float flow[];
};

layout(push_constant) uniform PushConstants {
    uint width;
    uint height;
} constants;

void main() {
    uvec2 pos = gl_GlobalInvocationID.xy;
    if (pos.x >= constants.width || pos.y >= constants.height) {
        return;
    }
    
    uint idx = pos.y * constants.width + pos.x;
    float current = pattern[idx];
    float flow_val = flow[idx];
    
    // Simple evolution step
    pattern[idx] = current + 0.1 * flow_val;
}
"""

BATCH_SHADER = """
#version 450
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(binding = 0) buffer PatternBuffer {
    float pattern[];  // [batch][height][width]
};
layout(binding = 1) buffer FlowBuffer {
    float flow[];  // [batch][height][width]
};

layout(push_constant) uniform PushConstants {
    uint batch_size;
    uint width;
    uint height;
} constants;

void main() {
    uvec3 pos = gl_GlobalInvocationID;
    if (pos.x >= constants.width || pos.y >= constants.height || pos.z >= constants.batch_size) {
        return;
    }
    
    uint idx = pos.z * constants.width * constants.height + pos.y * constants.width + pos.x;
    float current = pattern[idx];
    float flow_val = flow[idx];
    
    // Simple evolution step
    pattern[idx] = current + 0.1 * flow_val;
}
"""


class TestVulkanShaders:
    """Test suite for Vulkan shader performance."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up test parameters."""
        self.batch_sizes = [1, 4, 16]
        self.pattern_sizes = [128, 256, 512]
        self.manifold_dims = [16, 32, 64]
        self.iterations = 3
        self.device, self.physical_device = create_vulkan_device()
        
        # Compile shaders
        self.pattern_evolution_spv = self._compile_shader(PATTERN_EVOLUTION_SHADER)
        self.flow_computation_spv = self._compile_shader(FLOW_COMPUTATION_SHADER)
        self.configurable_workgroup_spv = self._compile_shader(CONFIGURABLE_WORKGROUP_SHADER)
        self.push_constant_spv = self._compile_shader(PUSH_CONSTANT_SHADER)
        self.descriptor_set_spv = self._compile_shader(DESCRIPTOR_SET_SHADER)
        self.single_item_spv = self._compile_shader(SINGLE_ITEM_SHADER)
        self.batch_spv = self._compile_shader(BATCH_SHADER)
        
    def _compile_shader(self, source: str) -> bytes:
        """Compile GLSL shader to SPIR-V.
        
        Args:
            source: GLSL shader source code
            
        Returns:
            SPIR-V bytecode
            
        Raises:
            subprocess.CalledProcessError: If shader compilation fails
        """
        import tempfile
        import subprocess
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.comp') as f:
            f.write(source)
            f.flush()
            
            output_file = f.name + '.spv'
            try:
                subprocess.run(
                    ['glslc', f.name, '-o', output_file],
                    check=True,
                    capture_output=True
                )
                
                with open(output_file, 'rb') as spv:
                    return spv.read()
            finally:
                if os.path.exists(output_file):
                    os.remove(output_file)
                    
    def _create_shader_module(self, code: bytes) -> c_void_p:
        """Create Vulkan shader module from SPIR-V code.
        
        Args:
            code: SPIR-V bytecode
            
        Returns:
            Vulkan shader module handle as a void pointer
            
        Raises:
            VulkanError: If shader module creation fails
        """
        # Convert bytes to uint32 array for Vulkan
        code_arr = np.frombuffer(code, dtype=np.uint32)
        code_ptr = code_arr.ctypes.data_as(POINTER(c_uint32))
        
        create_info = vk.VkShaderModuleCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(code),
            pCode=code_ptr
        )
        
        shader_module = c_void_p(0)
        result = vk.vkCreateShaderModule(
            self.device,
            create_info,
            None,
            shader_module
        )
        
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"Failed to create shader module: {result}")
            
        return shader_module

    def test_pattern_evolution_performance(self) -> None:
        """Test pattern evolution shader performance."""
        for batch_size in self.batch_sizes:
            for pattern_size in self.pattern_sizes:
                pattern, flow = setup_pattern_data(batch_size, pattern_size)

                # Create Vulkan buffers
                memory_manager = VulkanMemoryManager(self.device, self.physical_device)
                pattern_buffer = memory_manager.allocate_tensor(pattern.shape, pattern.dtype)
                flow_buffer = memory_manager.allocate_tensor(flow.shape, flow.dtype)
                memory_manager.copy_to_device(pattern, pattern_buffer)
                memory_manager.copy_to_device(flow, flow_buffer)

                # Test with different evolution rates
                rates = [0.01, 0.1, 0.5]
                times = []

                for rate in rates:
                    start = time.time()
                    for _ in range(self.iterations):
                        # Create shader module
                        shader_module = self._create_shader_module(self.pattern_evolution_spv)
                        
                        # Create pipeline layout
                        push_constant_range = vk.VkPushConstantRange(
                            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                            offset=0,
                            size=16  # float + 2 * uint
                        )
                        
                        layout_info = vk.VkPipelineLayoutCreateInfo(
                            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                            pushConstantRangeCount=1,
                            pPushConstantRanges=[push_constant_range]
                        )
                        pipeline_layout = vk.vkCreatePipelineLayout(self.device, layout_info, None)
                        
                        # Create compute pipeline
                        stage_info = vk.VkPipelineShaderStageCreateInfo(
                            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                            module=shader_module,
                            pName="main"
                        )
                        
                        pipeline_info = vk.VkComputePipelineCreateInfo(
                            sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                            stage=stage_info,
                            layout=pipeline_layout
                        )
                        
                        pipeline = vk.vkCreateComputePipelines(
                            self.device, None, 1, [pipeline_info], None
                        )[0]
                        
                        # Create command buffer
                        cmd_pool_info = vk.VkCommandPoolCreateInfo(
                            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                            queueFamilyIndex=0
                        )
                        cmd_pool = vk.vkCreateCommandPool(self.device, cmd_pool_info, None)
                        
                        cmd_alloc_info = vk.VkCommandBufferAllocateInfo(
                            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                            commandPool=cmd_pool,
                            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                            commandBufferCount=1
                        )
                        cmd_buffer = vk.vkAllocateCommandBuffers(self.device, cmd_alloc_info)[0]
                        
                        # Record commands
                        begin_info = vk.VkCommandBufferBeginInfo(
                            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
                        )
                        vk.vkBeginCommandBuffer(cmd_buffer, begin_info)
                        
                        vk.vkCmdBindPipeline(
                            cmd_buffer,
                            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline
                        )
                        
                        # Push constants
                        push_data = struct.pack("=fII", rate, pattern_size, pattern_size)
                        vk.vkCmdPushConstants(
                            cmd_buffer,
                            pipeline_layout,
                            vk.VK_SHADER_STAGE_COMPUTE_BIT,
                            0,
                            len(push_data),
                            push_data
                        )
                        
                        # Dispatch
                        group_size = 16
                        group_count_x = (pattern_size + group_size - 1) // group_size
                        group_count_y = (pattern_size + group_size - 1) // group_size
                        vk.vkCmdDispatch(cmd_buffer, group_count_x, group_count_y, 1)
                        
                        vk.vkEndCommandBuffer(cmd_buffer)
                        
                        # Submit and wait
                        submit_info = vk.VkSubmitInfo(
                            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                            commandBufferCount=1,
                            pCommandBuffers=[cmd_buffer]
                        )
                        
                        queue = vk.vkGetDeviceQueue(self.device, 0, 0)
                        vk.vkQueueSubmit(queue, 1, [submit_info], None)
                        vk.vkQueueWaitIdle(queue)
                        
                        # Cleanup
                        vk.vkDestroyCommandPool(self.device, cmd_pool, None)
                        vk.vkDestroyPipeline(self.device, pipeline, None)
                        vk.vkDestroyPipelineLayout(self.device, pipeline_layout, None)
                        vk.vkDestroyShaderModule(self.device, shader_module, None)
                        
                    times.append((time.time() - start) / self.iterations)

                print(f"\nPattern Evolution {batch_size}x{pattern_size}x{pattern_size}")
                for rate, t in zip(rates, times):
                    print(f"Rate {rate}: {t:.4f}s")
                print(f"Batch processing efficiency: {times[0]/times[-1]:.2f}x")

                # Cleanup
                memory_manager.cleanup()

    def test_flow_computation_performance(self) -> None:
        """Test flow computation shader performance."""
        for batch_size in self.batch_sizes:
            for dim in self.manifold_dims:
                metric, connection = setup_flow_data(batch_size, dim)

                # Create Vulkan buffers
                memory_manager = VulkanMemoryManager(self.device, self.physical_device)
                metric_buffer = memory_manager.allocate_tensor(metric.shape, metric.dtype)
                connection_buffer = memory_manager.allocate_tensor(connection.shape, connection.dtype)
                flow_buffer = memory_manager.allocate_tensor((batch_size, dim, dim), np.float32)
                memory_manager.copy_to_device(metric, metric_buffer)
                memory_manager.copy_to_device(connection, connection_buffer)

                # Test computation
                start = time.time()
                for _ in range(self.iterations):
                    # Create shader module
                    shader_module = self._create_shader_module(self.flow_computation_spv)
                    
                    # Create pipeline layout
                    push_constant_range = vk.VkPushConstantRange(
                        stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                        offset=0,
                        size=8  # 2 * uint
                    )
                    
                    layout_info = vk.VkPipelineLayoutCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                        pushConstantRangeCount=1,
                        pPushConstantRanges=[push_constant_range]
                    )
                    pipeline_layout = vk.vkCreatePipelineLayout(self.device, layout_info, None)
                    
                    # Create compute pipeline
                    stage_info = vk.VkPipelineShaderStageCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                        stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                        module=shader_module,
                        pName="main"
                    )
                    
                    pipeline_info = vk.VkComputePipelineCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                        stage=stage_info,
                        layout=pipeline_layout
                    )
                    
                    pipeline = vk.vkCreateComputePipelines(
                        self.device, None, 1, [pipeline_info], None
                    )[0]
                    
                    # Create command buffer
                    cmd_pool_info = vk.VkCommandPoolCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                        queueFamilyIndex=0
                    )
                    cmd_pool = vk.vkCreateCommandPool(self.device, cmd_pool_info, None)
                    
                    cmd_alloc_info = vk.VkCommandBufferAllocateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                        commandPool=cmd_pool,
                        level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                        commandBufferCount=1
                    )
                    cmd_buffer = vk.vkAllocateCommandBuffers(self.device, cmd_alloc_info)[0]
                    
                    # Record commands
                    begin_info = vk.VkCommandBufferBeginInfo(
                        sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
                    )
                    vk.vkBeginCommandBuffer(cmd_buffer, begin_info)
                    
                    vk.vkCmdBindPipeline(
                        cmd_buffer,
                        vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                        pipeline
                    )
                    
                    # Push constants
                    push_data = struct.pack("=II", batch_size, dim)
                    vk.vkCmdPushConstants(
                        cmd_buffer,
                        pipeline_layout,
                        vk.VK_SHADER_STAGE_COMPUTE_BIT,
                        0,
                        len(push_data),
                        push_data
                    )
                    
                    # Dispatch
                    group_size = 16
                    group_count_x = (dim + group_size - 1) // group_size
                    group_count_y = (dim + group_size - 1) // group_size
                    vk.vkCmdDispatch(cmd_buffer, group_count_x, group_count_y, batch_size)
                    
                    vk.vkEndCommandBuffer(cmd_buffer)
                    
                    # Submit and wait
                    submit_info = vk.VkSubmitInfo(
                        sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                        commandBufferCount=1,
                        pCommandBuffers=[cmd_buffer]
                    )
                    
                    queue = vk.vkGetDeviceQueue(self.device, 0, 0)
                    vk.vkQueueSubmit(queue, 1, [submit_info], None)
                    vk.vkQueueWaitIdle(queue)
                    
                    # Cleanup
                    vk.vkDestroyCommandPool(self.device, cmd_pool, None)
                    vk.vkDestroyPipeline(self.device, pipeline, None)
                    vk.vkDestroyPipelineLayout(self.device, pipeline_layout, None)
                    vk.vkDestroyShaderModule(self.device, shader_module, None)
                    
                avg_time = (time.time() - start) / self.iterations

                print(f"\nFlow Computation {batch_size}x{dim}x{dim}")
                print(f"Average time: {avg_time:.4f}s")
                print(f"GFLOPS: {2*batch_size*dim*dim*dim/avg_time/1e9:.2f}")

                # Cleanup
                memory_manager.cleanup()

    def test_workgroup_impact(self) -> None:
        """Test impact of different workgroup sizes."""
        pattern_size = 256
        batch_size = 4
        pattern, flow = setup_pattern_data(batch_size, pattern_size)

        # Create Vulkan buffers
        memory_manager = VulkanMemoryManager(self.device, self.physical_device)
        pattern_buffer = memory_manager.allocate_tensor(pattern.shape, pattern.dtype)
        flow_buffer = memory_manager.allocate_tensor(flow.shape, flow.dtype)
        memory_manager.copy_to_device(pattern, pattern_buffer)
        memory_manager.copy_to_device(flow, flow_buffer)

        # Test different workgroup sizes
        workgroup_sizes = [8, 16, 32]
        times = []

        for size in workgroup_sizes:
            start = time.time()
            for _ in range(self.iterations):
                # Create shader module with specialization constants
                shader_module = self._create_shader_module(self.configurable_workgroup_spv)
                
                # Create specialization info
                spec_data = struct.pack("=II", size, size)
                spec_map_entries = [
                    vk.VkSpecializationMapEntry(
                        constantID=0,
                        offset=0,
                        size=4
                    ),
                    vk.VkSpecializationMapEntry(
                        constantID=1,
                        offset=4,
                        size=4
                    )
                ]
                spec_info = vk.VkSpecializationInfo(
                    mapEntryCount=2,
                    pMapEntries=spec_map_entries,
                    dataSize=len(spec_data),
                    pData=spec_data
                )
                
                # Create pipeline layout
                push_constant_range = vk.VkPushConstantRange(
                    stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                    offset=0,
                    size=16  # float + 2 * uint
                )
                
                layout_info = vk.VkPipelineLayoutCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                    pushConstantRangeCount=1,
                    pPushConstantRanges=[push_constant_range]
                )
                pipeline_layout = vk.vkCreatePipelineLayout(self.device, layout_info, None)
                
                # Create compute pipeline
                stage_info = vk.VkPipelineShaderStageCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                    module=shader_module,
                    pName="main",
                    pSpecializationInfo=spec_info
                )
                
                pipeline_info = vk.VkComputePipelineCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                    stage=stage_info,
                    layout=pipeline_layout
                )
                
                pipeline = vk.vkCreateComputePipelines(
                    self.device, None, 1, [pipeline_info], None
                )[0]
                
                # Create command buffer
                cmd_pool_info = vk.VkCommandPoolCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                    queueFamilyIndex=0
                )
                cmd_pool = vk.vkCreateCommandPool(self.device, cmd_pool_info, None)
                
                cmd_alloc_info = vk.VkCommandBufferAllocateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                    commandPool=cmd_pool,
                    level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                    commandBufferCount=1
                )
                cmd_buffer = vk.vkAllocateCommandBuffers(self.device, cmd_alloc_info)[0]
                
                # Record commands
                begin_info = vk.VkCommandBufferBeginInfo(
                    sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
                )
                vk.vkBeginCommandBuffer(cmd_buffer, begin_info)
                
                vk.vkCmdBindPipeline(
                    cmd_buffer,
                    vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                    pipeline
                )
                
                # Push constants
                push_data = struct.pack("=fII", 0.1, pattern_size, pattern_size)
                vk.vkCmdPushConstants(
                    cmd_buffer,
                    pipeline_layout,
                    vk.VK_SHADER_STAGE_COMPUTE_BIT,
                    0,
                    len(push_data),
                    push_data
                )
                
                # Dispatch
                group_count_x = (pattern_size + size - 1) // size
                group_count_y = (pattern_size + size - 1) // size
                vk.vkCmdDispatch(cmd_buffer, group_count_x, group_count_y, 1)
                
                vk.vkEndCommandBuffer(cmd_buffer)
                
                # Submit and wait
                submit_info = vk.VkSubmitInfo(
                    sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    commandBufferCount=1,
                    pCommandBuffers=[cmd_buffer]
                )
                
                queue = vk.vkGetDeviceQueue(self.device, 0, 0)
                vk.vkQueueSubmit(queue, 1, [submit_info], None)
                vk.vkQueueWaitIdle(queue)
                
                # Cleanup
                vk.vkDestroyCommandPool(self.device, cmd_pool, None)
                vk.vkDestroyPipeline(self.device, pipeline, None)
                vk.vkDestroyPipelineLayout(self.device, pipeline_layout, None)
                vk.vkDestroyShaderModule(self.device, shader_module, None)
                
            times.append((time.time() - start) / self.iterations)

        print("\nWorkgroup Size Impact")
        for size, t in zip(workgroup_sizes, times):
            print(f"Size {size}: {t:.4f}s")
        print(f"Optimal size impact: {(min(times)-max(times))/max(times)*100:.1f}%")

        # Cleanup
        memory_manager.cleanup()

    def test_push_constant_performance(self) -> None:
        """Test push constant vs descriptor performance."""
        pattern_size = 256
        batch_size = 4
        pattern, flow = setup_pattern_data(batch_size, pattern_size)

        # Create Vulkan buffers
        memory_manager = VulkanMemoryManager(self.device, self.physical_device)
        pattern_buffer = memory_manager.allocate_tensor(pattern.shape, pattern.dtype)
        flow_buffer = memory_manager.allocate_tensor(flow.shape, flow.dtype)
        memory_manager.copy_to_device(pattern, pattern_buffer)
        memory_manager.copy_to_device(flow, flow_buffer)

        # Create parameter data
        params = np.random.random(13).astype(np.float32)
        param_data = struct.pack("=fII13f", 0.1, pattern_size, pattern_size, *params)
        param_buffer = memory_manager.allocate_tensor((16 + 13*4,), np.uint8)
        memory_manager.copy_to_device(np.frombuffer(param_data, dtype=np.uint8), param_buffer)

        # Test with push constants
        start = time.time()
        for _ in range(self.iterations):
            # Create shader module
            shader_module = self._create_shader_module(self.push_constant_spv)
            
            # Create pipeline layout
            push_constant_range = vk.VkPushConstantRange(
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                offset=0,
                size=len(param_data)
            )
            
            layout_info = vk.VkPipelineLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                pushConstantRangeCount=1,
                pPushConstantRanges=[push_constant_range]
            )
            pipeline_layout = vk.vkCreatePipelineLayout(self.device, layout_info, None)
            
            # Create compute pipeline
            stage_info = vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                module=shader_module,
                pName="main"
            )
            
            pipeline_info = vk.VkComputePipelineCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                stage=stage_info,
                layout=pipeline_layout
            )
            
            pipeline = vk.vkCreateComputePipelines(
                self.device, None, 1, [pipeline_info], None
            )[0]
            
            # Create command buffer
            cmd_pool_info = vk.VkCommandPoolCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                queueFamilyIndex=0
            )
            cmd_pool = vk.vkCreateCommandPool(self.device, cmd_pool_info, None)
            
            cmd_alloc_info = vk.VkCommandBufferAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                commandPool=cmd_pool,
                level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount=1
            )
            cmd_buffer = vk.vkAllocateCommandBuffers(self.device, cmd_alloc_info)[0]
            
            # Record commands
            begin_info = vk.VkCommandBufferBeginInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
            )
            vk.vkBeginCommandBuffer(cmd_buffer, begin_info)
            
            vk.vkCmdBindPipeline(
                cmd_buffer,
                vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                pipeline
            )
            
            # Push constants
            vk.vkCmdPushConstants(
                cmd_buffer,
                pipeline_layout,
                vk.VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                len(param_data),
                param_data
            )
            
            # Dispatch
            group_size = 16
            group_count_x = (pattern_size + group_size - 1) // group_size
            group_count_y = (pattern_size + group_size - 1) // group_size
            vk.vkCmdDispatch(cmd_buffer, group_count_x, group_count_y, 1)
            
            vk.vkEndCommandBuffer(cmd_buffer)
            
            # Submit and wait
            submit_info = vk.VkSubmitInfo(
                sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                commandBufferCount=1,
                pCommandBuffers=[cmd_buffer]
            )
            
            queue = vk.vkGetDeviceQueue(self.device, 0, 0)
            vk.vkQueueSubmit(queue, 1, [submit_info], None)
            vk.vkQueueWaitIdle(queue)
            
            # Cleanup
            vk.vkDestroyCommandPool(self.device, cmd_pool, None)
            vk.vkDestroyPipeline(self.device, pipeline, None)
            vk.vkDestroyPipelineLayout(self.device, pipeline_layout, None)
            vk.vkDestroyShaderModule(self.device, shader_module, None)
            
        push_time = (time.time() - start) / self.iterations

        # Test with descriptor set
        start = time.time()
        for _ in range(self.iterations):
            # Create shader module
            shader_module = self._create_shader_module(self.descriptor_set_spv)
            
            # Create descriptor set layout
            binding = vk.VkDescriptorSetLayoutBinding(
                binding=2,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT
            )
            
            layout_info = vk.VkDescriptorSetLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                bindingCount=1,
                pBindings=[binding]
            )
            desc_layout = vk.vkCreateDescriptorSetLayout(self.device, layout_info, None)
            
            # Create pipeline layout
            pipeline_layout_info = vk.VkPipelineLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                setLayoutCount=1,
                pSetLayouts=[desc_layout]
            )
            pipeline_layout = vk.vkCreatePipelineLayout(self.device, pipeline_layout_info, None)
            
            # Create compute pipeline
            stage_info = vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                module=shader_module,
                pName="main"
            )
            
            pipeline_info = vk.VkComputePipelineCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                stage=stage_info,
                layout=pipeline_layout
            )
            
            pipeline = vk.vkCreateComputePipelines(
                self.device, None, 1, [pipeline_info], None
            )[0]
            
            # Create descriptor pool and set
            pool_size = vk.VkDescriptorPoolSize(
                type=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                descriptorCount=1
            )
            
            pool_info = vk.VkDescriptorPoolCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                maxSets=1,
                poolSizeCount=1,
                pPoolSizes=[pool_size]
            )
            desc_pool = vk.vkCreateDescriptorPool(self.device, pool_info, None)
            
            alloc_info = vk.VkDescriptorSetAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                descriptorPool=desc_pool,
                descriptorSetCount=1,
                pSetLayouts=[desc_layout]
            )
            desc_set = vk.vkAllocateDescriptorSets(self.device, alloc_info)[0]
            
            # Update descriptor set
            buffer_info = vk.VkDescriptorBufferInfo(
                buffer=param_buffer,
                offset=0,
                range=len(param_data)
            )
            
            write = vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=desc_set,
                dstBinding=2,
                dstArrayElement=0,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                pBufferInfo=[buffer_info]
            )
            vk.vkUpdateDescriptorSets(self.device, 1, [write], 0, None)
            
            # Create command buffer
            cmd_pool_info = vk.VkCommandPoolCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                queueFamilyIndex=0
            )
            cmd_pool = vk.vkCreateCommandPool(self.device, cmd_pool_info, None)
            
            cmd_alloc_info = vk.VkCommandBufferAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                commandPool=cmd_pool,
                level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount=1
            )
            cmd_buffer = vk.vkAllocateCommandBuffers(self.device, cmd_alloc_info)[0]
            
            # Record commands
            begin_info = vk.VkCommandBufferBeginInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
            )
            vk.vkBeginCommandBuffer(cmd_buffer, begin_info)
            
            vk.vkCmdBindPipeline(
                cmd_buffer,
                vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                pipeline
            )
            
            vk.vkCmdBindDescriptorSets(
                cmd_buffer,
                vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                pipeline_layout,
                0,
                1,
                [desc_set],
                0,
                None
            )
            
            # Dispatch
            group_size = 16
            group_count_x = (pattern_size + group_size - 1) // group_size
            group_count_y = (pattern_size + group_size - 1) // group_size
            vk.vkCmdDispatch(cmd_buffer, group_count_x, group_count_y, 1)
            
            vk.vkEndCommandBuffer(cmd_buffer)
            
            # Submit and wait
            submit_info = vk.VkSubmitInfo(
                sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                commandBufferCount=1,
                pCommandBuffers=[cmd_buffer]
            )
            
            queue = vk.vkGetDeviceQueue(self.device, 0, 0)
            vk.vkQueueSubmit(queue, 1, [submit_info], None)
            vk.vkQueueWaitIdle(queue)
            
            # Cleanup
            vk.vkDestroyDescriptorPool(self.device, desc_pool, None)
            vk.vkDestroyDescriptorSetLayout(self.device, desc_layout, None)
            vk.vkDestroyCommandPool(self.device, cmd_pool, None)
            vk.vkDestroyPipeline(self.device, pipeline, None)
            vk.vkDestroyPipelineLayout(self.device, pipeline_layout, None)
            vk.vkDestroyShaderModule(self.device, shader_module, None)
            
        desc_time = (time.time() - start) / self.iterations
        
        print("\nParameter Passing Performance")
        print(f"Push constant time: {push_time:.4f}s")
        print(f"Descriptor set time: {desc_time:.4f}s")
        print(f"Overhead: {(desc_time-push_time)/push_time*100:.1f}%")

        # Cleanup
        memory_manager.cleanup()

    def test_batch_processing_efficiency(self) -> None:
        """Test batch processing efficiency."""
        pattern_size = 256
        pattern, flow = setup_pattern_data(1, pattern_size)

        times_single = []
        times_batch = []

        for batch_size in self.batch_sizes:
            # Single processing
            memory_manager = VulkanMemoryManager(self.device, self.physical_device)
            start = time.time()
            for _ in range(batch_size):
                pattern_buffer = memory_manager.allocate_tensor(pattern.shape, pattern.dtype)
                flow_buffer = memory_manager.allocate_tensor(flow.shape, flow.dtype)
                memory_manager.copy_to_device(pattern, pattern_buffer)
                memory_manager.copy_to_device(flow, flow_buffer)
                
                # Create shader module
                shader_module = self._create_shader_module(self.single_item_spv)
                
                # Create pipeline layout
                push_constant_range = vk.VkPushConstantRange(
                    stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                    offset=0,
                    size=8  # 2 * uint
                )
                
                layout_info = vk.VkPipelineLayoutCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                    pushConstantRangeCount=1,
                    pPushConstantRanges=[push_constant_range]
                )
                pipeline_layout = vk.vkCreatePipelineLayout(self.device, layout_info, None)
                
                # Create compute pipeline
                stage_info = vk.VkPipelineShaderStageCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                    module=shader_module,
                    pName="main"
                )
                
                pipeline_info = vk.VkComputePipelineCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                    stage=stage_info,
                    layout=pipeline_layout
                )
                
                pipeline = vk.vkCreateComputePipelines(
                    self.device, None, 1, [pipeline_info], None
                )[0]
                
                # Create command buffer
                cmd_pool_info = vk.VkCommandPoolCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                    queueFamilyIndex=0
                )
                cmd_pool = vk.vkCreateCommandPool(self.device, cmd_pool_info, None)
                
                cmd_alloc_info = vk.VkCommandBufferAllocateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                    commandPool=cmd_pool,
                    level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                    commandBufferCount=1
                )
                cmd_buffer = vk.vkAllocateCommandBuffers(self.device, cmd_alloc_info)[0]
                
                # Record commands
                begin_info = vk.VkCommandBufferBeginInfo(
                    sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
                )
                vk.vkBeginCommandBuffer(cmd_buffer, begin_info)
                
                vk.vkCmdBindPipeline(
                    cmd_buffer,
                    vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                    pipeline
                )
                
                # Push constants
                push_data = struct.pack("=II", pattern_size, pattern_size)
                vk.vkCmdPushConstants(
                    cmd_buffer,
                    pipeline_layout,
                    vk.VK_SHADER_STAGE_COMPUTE_BIT,
                    0,
                    len(push_data),
                    push_data
                )
                
                # Dispatch
                group_size = 16
                group_count_x = (pattern_size + group_size - 1) // group_size
                group_count_y = (pattern_size + group_size - 1) // group_size
                vk.vkCmdDispatch(cmd_buffer, group_count_x, group_count_y, 1)
                
                vk.vkEndCommandBuffer(cmd_buffer)
                
                # Submit and wait
                submit_info = vk.VkSubmitInfo(
                    sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    commandBufferCount=1,
                    pCommandBuffers=[cmd_buffer]
                )
                
                queue = vk.vkGetDeviceQueue(self.device, 0, 0)
                vk.vkQueueSubmit(queue, 1, [submit_info], None)
                vk.vkQueueWaitIdle(queue)
                
                # Cleanup
                vk.vkDestroyCommandPool(self.device, cmd_pool, None)
                vk.vkDestroyPipeline(self.device, pipeline, None)
                vk.vkDestroyPipelineLayout(self.device, pipeline_layout, None)
                vk.vkDestroyShaderModule(self.device, shader_module, None)
                
            times_single.append(time.time() - start)

            # Batch processing
            pattern_batch = np.tile(pattern, (batch_size, 1, 1))
            flow_batch = np.tile(flow, (batch_size, 1, 1))
            pattern_buffer = memory_manager.allocate_tensor(pattern_batch.shape, pattern_batch.dtype)
            flow_buffer = memory_manager.allocate_tensor(flow_batch.shape, flow_batch.dtype)
            memory_manager.copy_to_device(pattern_batch, pattern_buffer)
            memory_manager.copy_to_device(flow_batch, flow_buffer)

            start = time.time()
            # Create shader module
            shader_module = self._create_shader_module(self.batch_spv)
            
            # Create pipeline layout
            push_constant_range = vk.VkPushConstantRange(
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                offset=0,
                size=12  # 3 * uint
            )
            
            layout_info = vk.VkPipelineLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                pushConstantRangeCount=1,
                pPushConstantRanges=[push_constant_range]
            )
            pipeline_layout = vk.vkCreatePipelineLayout(self.device, layout_info, None)
            
            # Create compute pipeline
            stage_info = vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                module=shader_module,
                pName="main"
            )
            
            pipeline_info = vk.VkComputePipelineCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                stage=stage_info,
                layout=pipeline_layout
            )
            
            pipeline = vk.vkCreateComputePipelines(
                self.device, None, 1, [pipeline_info], None
            )[0]
            
            # Create command buffer
            cmd_pool_info = vk.VkCommandPoolCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                queueFamilyIndex=0
            )
            cmd_pool = vk.vkCreateCommandPool(self.device, cmd_pool_info, None)
            
            cmd_alloc_info = vk.VkCommandBufferAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                commandPool=cmd_pool,
                level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount=1
            )
            cmd_buffer = vk.vkAllocateCommandBuffers(self.device, cmd_alloc_info)[0]
            
            # Record commands
            begin_info = vk.VkCommandBufferBeginInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
            )
            vk.vkBeginCommandBuffer(cmd_buffer, begin_info)
            
            vk.vkCmdBindPipeline(
                cmd_buffer,
                vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                pipeline
            )
            
            # Push constants
            push_data = struct.pack("=III", batch_size, pattern_size, pattern_size)
            vk.vkCmdPushConstants(
                cmd_buffer,
                pipeline_layout,
                vk.VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                len(push_data),
                push_data
            )
            
            # Dispatch
            group_size = 16
            group_count_x = (pattern_size + group_size - 1) // group_size
            group_count_y = (pattern_size + group_size - 1) // group_size
            vk.vkCmdDispatch(cmd_buffer, group_count_x, group_count_y, batch_size)
            
            vk.vkEndCommandBuffer(cmd_buffer)
            
            # Submit and wait
            submit_info = vk.VkSubmitInfo(
                sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                commandBufferCount=1,
                pCommandBuffers=[cmd_buffer]
            )
            
            queue = vk.vkGetDeviceQueue(self.device, 0, 0)
            vk.vkQueueSubmit(queue, 1, [submit_info], None)
            vk.vkQueueWaitIdle(queue)
            
            # Cleanup
            vk.vkDestroyCommandPool(self.device, cmd_pool, None)
            vk.vkDestroyPipeline(self.device, pipeline, None)
            vk.vkDestroyPipelineLayout(self.device, pipeline_layout, None)
            vk.vkDestroyShaderModule(self.device, shader_module, None)
            
            times_batch.append(time.time() - start)

            # Cleanup
            memory_manager.cleanup()

        print("\nBatch Processing Efficiency")
        for batch_size, t_single, t_batch in zip(
            self.batch_sizes, times_single, times_batch
        ):
            print(f"Batch size {batch_size}:")
            print(f"  Single: {t_single:.4f}s")
            print(f"  Batch:  {t_batch:.4f}s")
            print(f"  Speedup: {t_single/t_batch:.2f}x")
