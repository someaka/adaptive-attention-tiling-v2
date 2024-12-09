"""Vulkan pipeline management for adaptive attention tiling."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Optional

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


class VulkanPipeline:
    """Manages Vulkan compute pipelines."""

    def __init__(self, device: int):  # vk.VkDevice
        self.device = device
        self.pipeline_cache: int = self._create_pipeline_cache()
        self.descriptor_pool: int = self._create_descriptor_pool()
        self.pipelines: Dict[PipelineType, int] = {}  # vk.VkPipeline
        self.pipeline_layouts: Dict[PipelineType, int] = {}  # vk.VkPipelineLayout
        self.descriptor_set_layouts: Dict[PipelineType, int] = {}  # vk.VkDescriptorSetLayout

    def create_pipeline(
        self, type: PipelineType, shader_code: bytes, push_constant_size: int = 0
    ) -> int:  # vk.VkPipeline
        """Create a compute pipeline."""
        # Create shader module
        shader_module = self._create_shader_module(shader_code)
        shader_stage = vk.VkPipelineShaderStageCreateInfo(
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT, module=shader_module, pName="main"
        )

        # Create descriptor set layout
        descriptor_layout = self._create_descriptor_set_layout(type)
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
        pipeline_layout = vk.vkCreatePipelineLayout(self.device, layout_info, None)
        self.pipeline_layouts[type] = pipeline_layout

        # Create compute pipeline
        pipeline_info = vk.VkComputePipelineCreateInfo(
            stage=shader_stage, layout=pipeline_layout
        )

        pipeline = vk.vkCreateComputePipelines(
            self.device, self.pipeline_cache, 1, [pipeline_info], None
        )[0]

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

    def _create_pipeline_cache(self) -> int:  # vk.VkPipelineCache
        """Create pipeline cache."""
        cache_info = vk.VkPipelineCacheCreateInfo()
        return vk.vkCreatePipelineCache(self.device, cache_info, None)

    def _create_descriptor_pool(self) -> int:  # vk.VkDescriptorPool
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

        return vk.vkCreateDescriptorPool(self.device, pool_info, None)

    def _create_shader_module(self, code: bytes) -> int:  # vk.VkShaderModule
        """Create shader module from SPIR-V code."""
        create_info = vk.VkShaderModuleCreateInfo(codeSize=len(code), pCode=code)
        return vk.vkCreateShaderModule(self.device, create_info, None)

    def _create_descriptor_set_layout(
        self, type: PipelineType
    ) -> int:  # vk.VkDescriptorSetLayout
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
        return vk.vkCreateDescriptorSetLayout(self.device, layout_info, None)

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
