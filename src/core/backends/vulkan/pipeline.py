"""Vulkan pipeline management for adaptive attention tiling."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Optional
from ctypes import c_void_p, cast, POINTER, c_uint32, byref

import vulkan as vk


class PipelineType(Enum):
    """Types of compute pipelines."""

    TILE_PROCESSOR = auto()
    CROSS_TILE_ROUTER = auto()
    METRICS_COLLECTOR = auto()
    MATMUL = auto()
    ADAPTIVE_ATTENTION = auto()
    ADAPTIVE_ATTENTION_BACKWARD = auto()


@dataclass
class ShaderStage:
    """Shader stage configuration."""

    module: int  # vk.VkShaderModule
    entry_point: str
    stage: int  # vk.VkShaderStageFlagBits


def handle_to_int(handle: c_void_p) -> int:
    """Convert a Vulkan handle (CData) to integer."""
    return cast(handle, POINTER(c_uint32)).contents.value


class VulkanPipeline:
    """Manages Vulkan compute pipelines."""

    def __init__(self, device: int):  # vk.VkDevice
        self.device = device
        self.pipeline_cache = handle_to_int(self._create_pipeline_cache())
        self.descriptor_pool = handle_to_int(self._create_descriptor_pool())
        self.pipelines: Dict[PipelineType, int] = {}  # vk.VkPipeline
        self.pipeline_layouts: Dict[PipelineType, int] = {}  # vk.VkPipelineLayout
        self.descriptor_set_layouts: Dict[PipelineType, int] = {}  # vk.VkDescriptorSetLayout

    def create_pipeline(
        self, type: PipelineType, shader_code: bytes, push_constant_size: int = 0
    ) -> int:  # vk.VkPipeline
        """Create a compute pipeline."""
        # Create shader module
        shader_module = handle_to_int(self._create_shader_module(shader_code))
        shader_stage = vk.VkPipelineShaderStageCreateInfo(
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT, module=shader_module, pName="main"
        )

        # Create descriptor set layout
        descriptor_layout = handle_to_int(self._create_descriptor_set_layout(type))
        self.descriptor_set_layouts[type] = descriptor_layout

        # Create pipeline layout
        push_constant_range = None
        if push_constant_size > 0:
            push_constant_range = vk.VkPushConstantRange(
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                offset=0,
                size=push_constant_size,
            )

        layout_info = vk.VkPipelineLayoutCreateInfo(
            setLayoutCount=1,
            pSetLayouts=[descriptor_layout],
            pushConstantRangeCount=1 if push_constant_range else 0,
            pPushConstantRanges=[push_constant_range] if push_constant_range else None,
        )
        
        pipeline_layout_handle = c_void_p()
        result = vk.vkCreatePipelineLayout(self.device, byref(layout_info), None, byref(pipeline_layout_handle))
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"Failed to create pipeline layout: {result}")
        pipeline_layout = handle_to_int(pipeline_layout_handle)
        self.pipeline_layouts[type] = pipeline_layout

        # Create compute pipeline
        pipeline_info = vk.VkComputePipelineCreateInfo(
            stage=shader_stage, layout=pipeline_layout
        )
        
        pipeline_handle = c_void_p()
        result = vk.vkCreateComputePipelines(
            self.device, self.pipeline_cache, 1, [pipeline_info], None, byref(pipeline_handle)
        )
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"Failed to create compute pipeline: {result}")
        pipeline = handle_to_int(pipeline_handle)

        self.pipelines[type] = pipeline
        vk.vkDestroyShaderModule(self.device, shader_module, None)

        return pipeline

    def get_pipeline(self, type: PipelineType) -> Optional[int]:  # vk.VkPipeline
        """Get an existing pipeline."""
        return self.pipelines.get(type)

    def get_layout(self, type: PipelineType) -> Optional[int]:  # vk.VkPipelineLayout
        """Get pipeline layout."""
        return self.pipeline_layouts.get(type)

    def get_descriptor_layout(
        self, type: PipelineType
    ) -> Optional[int]:  # vk.VkDescriptorSetLayout
        """Get descriptor set layout."""
        return self.descriptor_set_layouts.get(type)

    def _create_pipeline_cache(self) -> c_void_p:  # vk.VkPipelineCache
        """Create pipeline cache."""
        cache_info = vk.VkPipelineCacheCreateInfo()
        pipeline_cache = c_void_p()
        result = vk.vkCreatePipelineCache(self.device, byref(cache_info), None, byref(pipeline_cache))
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"Failed to create pipeline cache: {result}")
        return pipeline_cache

    def _create_descriptor_pool(self) -> c_void_p:  # vk.VkDescriptorPool
        """Create descriptor pool."""
        pool_sizes = [
            vk.VkDescriptorPoolSize(
                type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=100,  # Adjust based on needs
            ),
            vk.VkDescriptorPoolSize(
                type=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                descriptorCount=100,  # Adjust based on needs
            ),
        ]

        pool_info = vk.VkDescriptorPoolCreateInfo(
            maxSets=100, poolSizeCount=len(pool_sizes), pPoolSizes=pool_sizes
        )

        descriptor_pool = c_void_p()
        result = vk.vkCreateDescriptorPool(self.device, byref(pool_info), None, byref(descriptor_pool))
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"Failed to create descriptor pool: {result}")
        return descriptor_pool

    def _create_shader_module(self, code: bytes) -> c_void_p:  # vk.VkShaderModule
        """Create shader module from SPIR-V code."""
        create_info = vk.VkShaderModuleCreateInfo(codeSize=len(code), pCode=code)
        shader_module = c_void_p()
        result = vk.vkCreateShaderModule(self.device, byref(create_info), None, byref(shader_module))
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"Failed to create shader module: {result}")
        return shader_module

    def _create_descriptor_set_layout(
        self, type: PipelineType
    ) -> c_void_p:  # vk.VkDescriptorSetLayout
        """Create descriptor set layout based on pipeline type."""
        bindings = []

        if type == PipelineType.TILE_PROCESSOR:
            bindings = [
                # Input state buffer
                vk.VkDescriptorSetLayoutBinding(
                    binding=0,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount=1,
                    stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                ),
                # Output state buffer
                vk.VkDescriptorSetLayoutBinding(
                    binding=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount=1,
                    stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                ),
                # Metrics buffer
                vk.VkDescriptorSetLayoutBinding(
                    binding=2,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount=1,
                    stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                ),
            ]
        elif type == PipelineType.CROSS_TILE_ROUTER:
            bindings = [
                # Source tile buffers
                vk.VkDescriptorSetLayoutBinding(
                    binding=0,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount=1,
                    stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                ),
                # Target tile buffers
                vk.VkDescriptorSetLayoutBinding(
                    binding=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount=1,
                    stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                ),
            ]

        layout_info = vk.VkDescriptorSetLayoutCreateInfo(
            bindingCount=len(bindings), pBindings=bindings
        )
        descriptor_layout = c_void_p()
        result = vk.vkCreateDescriptorSetLayout(self.device, byref(layout_info), None, byref(descriptor_layout))
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"Failed to create descriptor set layout: {result}")
        return descriptor_layout

    def cleanup(self) -> None:
        """Clean up Vulkan resources."""
        for pipeline in self.pipelines.values():
            vk.vkDestroyPipeline(self.device, pipeline, None)

        for layout in self.pipeline_layouts.values():
            vk.vkDestroyPipelineLayout(self.device, layout, None)

        for descriptor_layout in self.descriptor_set_layouts.values():
            vk.vkDestroyDescriptorSetLayout(self.device, descriptor_layout, None)

        vk.vkDestroyDescriptorPool(self.device, self.descriptor_pool, None)
        vk.vkDestroyPipelineCache(self.device, self.pipeline_cache, None)
