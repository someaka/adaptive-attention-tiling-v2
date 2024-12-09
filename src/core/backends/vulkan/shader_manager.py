"""Shader management system for Vulkan compute pipelines."""

import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Optional

import vulkan as vk

from src.core.common.constants import SHADER_DIR


class ShaderType(Enum):
    """Types of compute shaders."""
    TILE_PROCESSOR = auto()
    CROSS_TILE_ROUTER = auto()


@dataclass
class ShaderConfig:
    """Configuration for shader compilation."""
    local_size_x: int = 256
    use_fp16: bool = True
    enable_validation: bool = True


class ShaderManager:
    """Manages shader modules and compilation."""
    
    def __init__(self, device: vk.Device):
        self.device = device
        self._shader_modules: Dict[ShaderType, vk.ShaderModule] = {}
        self._shader_paths = {
            ShaderType.TILE_PROCESSOR: os.path.join(SHADER_DIR, "tile_processor.comp"),
            ShaderType.CROSS_TILE_ROUTER: os.path.join(SHADER_DIR, "cross_tile_router.comp")
        }
        
    def load_shader(self, 
                   type: ShaderType, 
                   config: Optional[ShaderConfig] = None) -> vk.ShaderModule:
        """Load and compile a shader module."""
        if type in self._shader_modules:
            return self._shader_modules[type]
            
        config = config or ShaderConfig()
        
        # Read shader file
        with open(self._shader_paths[type], 'rb') as f:
            spv_code = f.read()
            
        # Create shader module
        create_info = vk.ShaderModuleCreateInfo(
            code=spv_code
        )
        shader_module = vk.create_shader_module(
            self.device, create_info, None
        )
        
        self._shader_modules[type] = shader_module
        return shader_module
    
    def get_push_constant_size(self, type: ShaderType) -> int:
        """Get push constant buffer size for shader type."""
        if type == ShaderType.TILE_PROCESSOR:
            return 5 * 4  # sequence_length, d_model, d_state, min_resolution, density_threshold
        elif type == ShaderType.CROSS_TILE_ROUTER:
            return 4 * 4  # num_tiles, tile_size, d_model, flow_threshold
        return 0
    
    def get_descriptor_layout_bindings(self, type: ShaderType) -> list:
        """Get descriptor layout bindings for shader type."""
        if type == ShaderType.TILE_PROCESSOR:
            return [
                # Input buffer
                vk.DescriptorSetLayoutBinding(
                    binding=0,
                    descriptor_type=vk.DescriptorType.STORAGE_BUFFER,
                    descriptor_count=1,
                    stage_flags=vk.ShaderStageFlagBits.COMPUTE
                ),
                # State buffer
                vk.DescriptorSetLayoutBinding(
                    binding=1,
                    descriptor_type=vk.DescriptorType.STORAGE_BUFFER,
                    descriptor_count=1,
                    stage_flags=vk.ShaderStageFlagBits.COMPUTE
                ),
                # Density buffer
                vk.DescriptorSetLayoutBinding(
                    binding=2,
                    descriptor_type=vk.DescriptorType.STORAGE_BUFFER,
                    descriptor_count=1,
                    stage_flags=vk.ShaderStageFlagBits.COMPUTE
                ),
                # Output buffer
                vk.DescriptorSetLayoutBinding(
                    binding=3,
                    descriptor_type=vk.DescriptorType.STORAGE_BUFFER,
                    descriptor_count=1,
                    stage_flags=vk.ShaderStageFlagBits.COMPUTE
                )
            ]
        elif type == ShaderType.CROSS_TILE_ROUTER:
            return [
                # Tile states
                vk.DescriptorSetLayoutBinding(
                    binding=0,
                    descriptor_type=vk.DescriptorType.STORAGE_BUFFER,
                    descriptor_count=1,
                    stage_flags=vk.ShaderStageFlagBits.COMPUTE
                ),
                # Tile resolutions
                vk.DescriptorSetLayoutBinding(
                    binding=1,
                    descriptor_type=vk.DescriptorType.STORAGE_BUFFER,
                    descriptor_count=1,
                    stage_flags=vk.ShaderStageFlagBits.COMPUTE
                ),
                # Cross-tile flow
                vk.DescriptorSetLayoutBinding(
                    binding=2,
                    descriptor_type=vk.DescriptorType.STORAGE_BUFFER,
                    descriptor_count=1,
                    stage_flags=vk.ShaderStageFlagBits.COMPUTE
                )
            ]
        return []
    
    def get_workgroup_size(self, type: ShaderType) -> int:
        """Get workgroup size for shader type."""
        return 256  # Both shaders use 256 threads per workgroup
        
    def cleanup(self):
        """Clean up shader modules."""
        for shader_module in self._shader_modules.values():
            vk.destroy_shader_module(self.device, shader_module, None)
        self._shader_modules.clear()
