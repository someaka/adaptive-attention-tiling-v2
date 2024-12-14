"""Vulkan Shader Management.

This module provides tools for managing Vulkan compute shaders including:
- Shader compilation
- Pipeline creation
- Resource binding
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, List
from ctypes import c_void_p, c_uint32, c_int, Structure, POINTER, byref, c_char_p, c_size_t
import vulkan as vk

# Vulkan type definitions
VkDevice = c_void_p
VkQueue = c_void_p
VkCommandPool = c_void_p
VkShaderModule = c_void_p
VkPipeline = c_void_p
VkPipelineLayout = c_void_p
VkDescriptorSet = c_void_p
VkDescriptorSetLayout = c_void_p
VkPipelineCache = c_void_p
VkFence = c_void_p
VkCommandBuffer = c_void_p

# Vulkan constants
VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO = 15
VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32
VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO = 30
VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO = 29
VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO = 18
VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = 40
VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 42
VK_STRUCTURE_TYPE_SUBMIT_INFO = 4

VK_SHADER_STAGE_COMPUTE_BIT = 0x00000020
VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7
VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0
VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT = 0x00000001
VK_PIPELINE_BIND_POINT_COMPUTE = 1

class VkShaderModuleCreateInfo(Structure):
    _fields_ = [
        ("sType", c_int),
        ("pNext", c_void_p),
        ("flags", c_uint32),
        ("codeSize", c_size_t),
        ("pCode", POINTER(c_uint32))
    ]

class VkDescriptorSetLayoutBinding(Structure):
    _fields_ = [
        ("binding", c_uint32),
        ("descriptorType", c_uint32),
        ("descriptorCount", c_uint32),
        ("stageFlags", c_uint32),
        ("pImmutableSamplers", c_void_p)
    ]

class VkDescriptorSetLayoutCreateInfo(Structure):
    _fields_ = [
        ("sType", c_int),
        ("pNext", c_void_p),
        ("flags", c_uint32),
        ("bindingCount", c_uint32),
        ("pBindings", POINTER(VkDescriptorSetLayoutBinding))
    ]

class VkPushConstantRange(Structure):
    _fields_ = [
        ("stageFlags", c_uint32),
        ("offset", c_uint32),
        ("size", c_uint32)
    ]

class VkPipelineLayoutCreateInfo(Structure):
    _fields_ = [
        ("sType", c_int),
        ("pNext", c_void_p),
        ("flags", c_uint32),
        ("setLayoutCount", c_uint32),
        ("pSetLayouts", POINTER(c_void_p)),
        ("pushConstantRangeCount", c_uint32),
        ("pPushConstantRanges", POINTER(VkPushConstantRange))
    ]

class VkPipelineShaderStageCreateInfo(Structure):
    _fields_ = [
        ("sType", c_int),
        ("pNext", c_void_p),
        ("flags", c_uint32),
        ("stage", c_uint32),
        ("module", c_void_p),
        ("pName", c_char_p),
        ("pSpecializationInfo", c_void_p)
    ]

class VkComputePipelineCreateInfo(Structure):
    _fields_ = [
        ("sType", c_int),
        ("pNext", c_void_p),
        ("flags", c_uint32),
        ("stage", VkPipelineShaderStageCreateInfo),
        ("layout", c_void_p),
        ("basePipelineHandle", c_void_p),
        ("basePipelineIndex", c_int)
    ]

class VkCommandBufferAllocateInfo(Structure):
    _fields_ = [
        ("sType", c_int),
        ("pNext", c_void_p),
        ("commandPool", c_void_p),
        ("level", c_uint32),
        ("commandBufferCount", c_uint32)
    ]

class VkCommandBufferBeginInfo(Structure):
    _fields_ = [
        ("sType", c_int),
        ("pNext", c_void_p),
        ("flags", c_uint32),
        ("pInheritanceInfo", c_void_p)
    ]

class VkSubmitInfo(Structure):
    _fields_ = [
        ("sType", c_int),
        ("pNext", c_void_p),
        ("waitSemaphoreCount", c_uint32),
        ("pWaitSemaphores", c_void_p),
        ("pWaitDstStageMask", POINTER(c_uint32)),
        ("commandBufferCount", c_uint32),
        ("pCommandBuffers", POINTER(c_void_p)),
        ("signalSemaphoreCount", c_uint32),
        ("pSignalSemaphores", c_void_p)
    ]

class ShaderManager:
    """Manages Vulkan compute shaders."""

    def __init__(self, device: VkDevice, shader_dir: str = "shaders"):
        self.device = device
        self.shader_dir = Path(shader_dir)
        self.shader_modules: Dict[str, VkShaderModule] = {}
        self.pipelines: Dict[str, VkPipeline] = {}
        self.pipeline_layouts: Dict[str, VkPipelineLayout] = {}

    def create_shader_module(self, shader_name: str) -> VkShaderModule:
        """Create a shader module from SPIR-V code."""
        shader_path = self.shader_dir / f"{shader_name}.spv"
        with open(shader_path, "rb") as f:
            code = f.read()

        code_size = len(code)
        code_array = (c_uint32 * (code_size // 4)).from_buffer_copy(code)
        
        create_info = VkShaderModuleCreateInfo(
            sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            pNext=None,
            flags=0,
            codeSize=code_size,
            pCode=code_array
        )

        shader_module = c_void_p()
        result = vk.vkCreateShaderModule(self.device, byref(create_info), None, byref(shader_module))
        if result != 0:
            raise RuntimeError(f"Failed to create shader module: {result}")

        self.shader_modules[shader_name] = shader_module
        return shader_module

    def create_compute_pipeline(
        self, shader_name: str, push_constant_size: int = 0
    ) -> Tuple[VkPipeline, VkPipelineLayout]:
        """Create a compute pipeline with the specified shader."""
        # Create descriptor set layout
        bindings = []
        for i in range(4):  # Assuming max 4 bindings per shader
            binding = VkDescriptorSetLayoutBinding(
                binding=i,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
                pImmutableSamplers=None
            )
            bindings.append(binding)

        bindings_array = (VkDescriptorSetLayoutBinding * len(bindings))(*bindings)
        descriptor_set_layout_info = VkDescriptorSetLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            pNext=None,
            flags=0,
            bindingCount=len(bindings),
            pBindings=bindings_array
        )

        descriptor_set_layout = c_void_p()
        result = vk.vkCreateDescriptorSetLayout(
            self.device, byref(descriptor_set_layout_info), None, byref(descriptor_set_layout)
        )
        if result != 0:
            raise RuntimeError(f"Failed to create descriptor set layout: {result}")

        # Create pipeline layout
        push_constant_range = None
        if push_constant_size > 0:
            push_constant_range = VkPushConstantRange(
                stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
                offset=0,
                size=push_constant_size
            )

        set_layouts = (c_void_p * 1)(descriptor_set_layout)
        push_constant_ranges = None
        push_constant_range_count = 0
        if push_constant_range:
            push_constant_ranges = (VkPushConstantRange * 1)(push_constant_range)
            push_constant_range_count = 1

        pipeline_layout_info = VkPipelineLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            pNext=None,
            flags=0,
            setLayoutCount=1,
            pSetLayouts=set_layouts,
            pushConstantRangeCount=push_constant_range_count,
            pPushConstantRanges=push_constant_ranges
        )

        pipeline_layout = c_void_p()
        result = vk.vkCreatePipelineLayout(
            self.device, byref(pipeline_layout_info), None, byref(pipeline_layout)
        )
        if result != 0:
            raise RuntimeError(f"Failed to create pipeline layout: {result}")

        # Create compute pipeline
        shader_stage = VkPipelineShaderStageCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            pNext=None,
            flags=0,
            stage=VK_SHADER_STAGE_COMPUTE_BIT,
            module=self.shader_modules[shader_name],
            pName=b"main",
            pSpecializationInfo=None
        )

        compute_pipeline_info = VkComputePipelineCreateInfo(
            sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            pNext=None,
            flags=0,
            stage=shader_stage,
            layout=pipeline_layout,
            basePipelineHandle=None,
            basePipelineIndex=-1
        )

        pipeline = c_void_p()
        result = vk.vkCreateComputePipelines(
            self.device, None, 1, byref(compute_pipeline_info), None, byref(pipeline)
        )
        if result != 0:
            raise RuntimeError(f"Failed to create compute pipeline: {result}")

        self.pipelines[shader_name] = pipeline
        self.pipeline_layouts[shader_name] = pipeline_layout

        return pipeline, pipeline_layout

    def cleanup(self) -> None:
        """Clean up Vulkan resources."""
        for shader_module in self.shader_modules.values():
            vk.vkDestroyShaderModule(self.device, shader_module, None)

        for pipeline in self.pipelines.values():
            vk.vkDestroyPipeline(self.device, pipeline, None)

        for layout in self.pipeline_layouts.values():
            vk.vkDestroyPipelineLayout(self.device, layout, None)


class ComputeShaderDispatcher:
    """Dispatches compute shader workloads."""

    def __init__(
        self, device: VkDevice, queue: VkQueue, command_pool: VkCommandPool
    ):
        self.device = device
        self.queue = queue
        self.command_pool = command_pool

    def dispatch(
        self,
        pipeline: VkPipeline,
        pipeline_layout: VkPipelineLayout,
        descriptor_set: VkDescriptorSet,
        push_constants: Optional[bytes] = None,
        group_count: Tuple[int, int, int] = (1, 1, 1),
    ) -> None:
        """Dispatch a compute shader workload."""
        # Create command buffer
        command_buffer_info = VkCommandBufferAllocateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            pNext=None,
            commandPool=self.command_pool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1
        )

        command_buffer = c_void_p()
        result = vk.vkAllocateCommandBuffers(self.device, byref(command_buffer_info), byref(command_buffer))
        if result != 0:
            raise RuntimeError(f"Failed to allocate command buffer: {result}")

        # Begin command buffer
        begin_info = VkCommandBufferBeginInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext=None,
            flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            pInheritanceInfo=None
        )
        
        result = vk.vkBeginCommandBuffer(command_buffer, byref(begin_info))
        if result != 0:
            raise RuntimeError(f"Failed to begin command buffer: {result}")

        # Bind pipeline
        vk.vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)

        # Bind descriptor set
        descriptor_sets = (c_void_p * 1)(descriptor_set)
        vk.vkCmdBindDescriptorSets(
            command_buffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout,
            0,
            1,
            descriptor_sets,
            0,
            None
        )

        # Push constants if provided
        if push_constants:
            vk.vkCmdPushConstants(
                command_buffer,
                pipeline_layout,
                VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                len(push_constants),
                push_constants
            )

        # Dispatch compute work
        vk.vkCmdDispatch(command_buffer, *group_count)

        # End command buffer
        result = vk.vkEndCommandBuffer(command_buffer)
        if result != 0:
            raise RuntimeError(f"Failed to end command buffer: {result}")

        # Submit command buffer
        command_buffers = (c_void_p * 1)(command_buffer)
        submit_info = VkSubmitInfo(
            sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
            pNext=None,
            waitSemaphoreCount=0,
            pWaitSemaphores=None,
            pWaitDstStageMask=None,
            commandBufferCount=1,
            pCommandBuffers=command_buffers,
            signalSemaphoreCount=0,
            pSignalSemaphores=None
        )

        result = vk.vkQueueSubmit(self.queue, 1, byref(submit_info), None)
        if result != 0:
            raise RuntimeError(f"Failed to submit queue: {result}")

        vk.vkQueueWaitIdle(self.queue)

        # Cleanup
        vk.vkFreeCommandBuffers(self.device, self.command_pool, 1, command_buffers)


class ShaderCompiler:
    """Compiles GLSL shaders to SPIR-V."""

    def __init__(self, shader_dir: str = "shaders"):
        self.shader_dir = Path(shader_dir)
        self.shader_dir.mkdir(parents=True, exist_ok=True)

    def compile_shader(self, shader_name: str) -> None:
        """Compile a GLSL shader to SPIR-V."""
        import subprocess

        input_file = self.shader_dir / f"{shader_name}.comp"
        output_file = self.shader_dir / f"{shader_name}.spv"

        result = subprocess.run(
            ["glslangValidator", "-V", str(input_file), "-o", str(output_file)], check=False
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to compile shader: {shader_name}")

    def compile_all_shaders(self) -> None:
        """Compile all GLSL shaders in the shader directory."""
        for shader_file in self.shader_dir.glob("*.comp"):
            self.compile_shader(shader_file.stem)
