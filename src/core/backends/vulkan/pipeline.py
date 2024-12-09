"""Vulkan pipeline management for adaptive attention tiling."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import vulkan as vk

from src.core.common.constants import MAX_COMPUTE_WORK_GROUP_SIZE


class PipelineType(Enum):
    """Types of compute pipelines."""
    TILE_PROCESSOR = auto()
    CROSS_TILE_ROUTER = auto()
    METRICS_COLLECTOR = auto()


@dataclass
class ShaderStage:
    """Shader stage configuration."""
    module: vk.ShaderModule
    entry_point: str
    specialization_info: Optional[vk.SpecializationInfo] = None


class VulkanPipeline:
    """Manages Vulkan compute pipelines."""
    
    def __init__(self, device: vk.Device):
        self.device = device
        self.pipeline_cache = self._create_pipeline_cache()
        self.descriptor_pool = self._create_descriptor_pool()
        self.pipelines: Dict[PipelineType, vk.Pipeline] = {}
        self.pipeline_layouts: Dict[PipelineType, vk.PipelineLayout] = {}
        self.descriptor_set_layouts: Dict[PipelineType, vk.DescriptorSetLayout] = {}
        
    def create_pipeline(self, 
                       type: PipelineType, 
                       shader_code: bytes,
                       push_constant_size: int = 0) -> vk.Pipeline:
        """Create a compute pipeline."""
        # Create shader module
        shader = self._create_shader_module(shader_code)
        shader_stage = vk.PipelineShaderStageCreateInfo(
            stage=vk.ShaderStageFlagBits.COMPUTE,
            module=shader,
            name="main"
        )
        
        # Create descriptor set layout
        descriptor_layout = self._create_descriptor_set_layout(type)
        self.descriptor_set_layouts[type] = descriptor_layout
        
        # Create pipeline layout
        push_constant_range = None
        if push_constant_size > 0:
            push_constant_range = vk.PushConstantRange(
                stage_flags=vk.ShaderStageFlagBits.COMPUTE,
                offset=0,
                size=push_constant_size
            )
        
        layout_info = vk.PipelineLayoutCreateInfo(
            set_layouts=[descriptor_layout],
            push_constant_ranges=[push_constant_range] if push_constant_range else []
        )
        pipeline_layout = vk.create_pipeline_layout(self.device, layout_info, None)
        self.pipeline_layouts[type] = pipeline_layout
        
        # Create compute pipeline
        pipeline_info = vk.ComputePipelineCreateInfo(
            stage=shader_stage,
            layout=pipeline_layout
        )
        
        pipeline = vk.create_compute_pipelines(
            self.device, self.pipeline_cache, [pipeline_info], None
        )[0]
        
        self.pipelines[type] = pipeline
        vk.destroy_shader_module(self.device, shader, None)
        
        return pipeline
    
    def get_pipeline(self, type: PipelineType) -> Optional[vk.Pipeline]:
        """Get an existing pipeline."""
        return self.pipelines.get(type)
    
    def get_layout(self, type: PipelineType) -> Optional[vk.PipelineLayout]:
        """Get pipeline layout."""
        return self.pipeline_layouts.get(type)
    
    def get_descriptor_layout(self, type: PipelineType) -> Optional[vk.DescriptorSetLayout]:
        """Get descriptor set layout."""
        return self.descriptor_set_layouts.get(type)
    
    def _create_pipeline_cache(self) -> vk.PipelineCache:
        """Create pipeline cache."""
        cache_info = vk.PipelineCacheCreateInfo()
        return vk.create_pipeline_cache(self.device, cache_info, None)
    
    def _create_descriptor_pool(self) -> vk.DescriptorPool:
        """Create descriptor pool."""
        pool_sizes = [
            vk.DescriptorPoolSize(
                type=vk.DescriptorType.STORAGE_BUFFER,
                descriptor_count=100  # Adjust based on needs
            ),
            vk.DescriptorPoolSize(
                type=vk.DescriptorType.UNIFORM_BUFFER,
                descriptor_count=100  # Adjust based on needs
            )
        ]
        
        pool_info = vk.DescriptorPoolCreateInfo(
            max_sets=100,  # Adjust based on needs
            pool_sizes=pool_sizes
        )
        
        return vk.create_descriptor_pool(self.device, pool_info, None)
    
    def _create_shader_module(self, code: bytes) -> vk.ShaderModule:
        """Create shader module from SPIR-V code."""
        create_info = vk.ShaderModuleCreateInfo(code=code)
        return vk.create_shader_module(self.device, create_info, None)
    
    def _create_descriptor_set_layout(self, type: PipelineType) -> vk.DescriptorSetLayout:
        """Create descriptor set layout based on pipeline type."""
        bindings = []
        
        if type == PipelineType.TILE_PROCESSOR:
            bindings = [
                # Input state buffer
                vk.DescriptorSetLayoutBinding(
                    binding=0,
                    descriptor_type=vk.DescriptorType.STORAGE_BUFFER,
                    descriptor_count=1,
                    stage_flags=vk.ShaderStageFlagBits.COMPUTE
                ),
                # Output state buffer
                vk.DescriptorSetLayoutBinding(
                    binding=1,
                    descriptor_type=vk.DescriptorType.STORAGE_BUFFER,
                    descriptor_count=1,
                    stage_flags=vk.ShaderStageFlagBits.COMPUTE
                ),
                # Metrics buffer
                vk.DescriptorSetLayoutBinding(
                    binding=2,
                    descriptor_type=vk.DescriptorType.STORAGE_BUFFER,
                    descriptor_count=1,
                    stage_flags=vk.ShaderStageFlagBits.COMPUTE
                )
            ]
        elif type == PipelineType.CROSS_TILE_ROUTER:
            bindings = [
                # Source tile buffers
                vk.DescriptorSetLayoutBinding(
                    binding=0,
                    descriptor_type=vk.DescriptorType.STORAGE_BUFFER,
                    descriptor_count=1,
                    stage_flags=vk.ShaderStageFlagBits.COMPUTE
                ),
                # Target tile buffers
                vk.DescriptorSetLayoutBinding(
                    binding=1,
                    descriptor_type=vk.DescriptorType.STORAGE_BUFFER,
                    descriptor_count=1,
                    stage_flags=vk.ShaderStageFlagBits.COMPUTE
                )
            ]
        
        layout_info = vk.DescriptorSetLayoutCreateInfo(bindings=bindings)
        return vk.create_descriptor_set_layout(self.device, layout_info, None)
    
    def cleanup(self):
        """Clean up Vulkan resources."""
        for pipeline in self.pipelines.values():
            vk.destroy_pipeline(self.device, pipeline, None)
            
        for layout in self.pipeline_layouts.values():
            vk.destroy_pipeline_layout(self.device, layout, None)
            
        for descriptor_layout in self.descriptor_set_layouts.values():
            vk.destroy_descriptor_set_layout(self.device, descriptor_layout, None)
            
        vk.destroy_descriptor_pool(self.device, self.descriptor_pool, None)
        vk.destroy_pipeline_cache(self.device, self.pipeline_cache, None)
