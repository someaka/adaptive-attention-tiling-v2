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

    def __init__(self, device: int):  # VkDevice
        self.device = device
        self._shader_modules: Dict[ShaderType, int] = {}  # Dict[ShaderType, VkShaderModule]
        self._shader_paths = {
            ShaderType.TILE_PROCESSOR: os.path.join(SHADER_DIR, "tile_processor.comp"),
            ShaderType.CROSS_TILE_ROUTER: os.path.join(
                SHADER_DIR, "cross_tile_router.comp"
            ),
        }

    def load_shader(
        self, type: ShaderType, config: Optional[ShaderConfig] = None
    ) -> int:  # VkShaderModule
        """Load and compile a shader module."""
        if type in self._shader_modules:
            return self._shader_modules[type]

        config = config or ShaderConfig()

        # Read shader file
        with open(self._shader_paths[type], "rb") as f:
            shader_code = f.read()

        # Create shader module
        create_info = vk.VkShaderModuleCreateInfo(
            codeSize=len(shader_code),
            pCode=shader_code,
        )

        shader_module = vk.vkCreateShaderModule(self.device, create_info, None)
        self._shader_modules[type] = shader_module

        return shader_module

    def get_shader(self, type: ShaderType) -> Optional[int]:  # Optional[VkShaderModule]
        """Get an existing shader module."""
        return self._shader_modules.get(type)

    def get_push_constant_size(self, type: ShaderType) -> int:
        """Get push constant buffer size for shader type."""
        if type == ShaderType.TILE_PROCESSOR:
            return (
                5 * 4
            )  # sequence_length, d_model, d_state, min_resolution, density_threshold
        if type == ShaderType.CROSS_TILE_ROUTER:
            return 4 * 4  # num_tiles, tile_size, d_model, flow_threshold
        return 0

    def get_descriptor_layout_bindings(self, type: ShaderType) -> list:
        """Get descriptor layout bindings for shader type."""
        if type == ShaderType.TILE_PROCESSOR:
            return [
                # Input buffer
                vk.VkDescriptorSetLayoutBinding(
                    binding=0,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount=1,
                    stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                ),
                # State buffer
                vk.VkDescriptorSetLayoutBinding(
                    binding=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount=1,
                    stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                ),
                # Density buffer
                vk.VkDescriptorSetLayoutBinding(
                    binding=2,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount=1,
                    stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                ),
                # Output buffer
                vk.VkDescriptorSetLayoutBinding(
                    binding=3,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount=1,
                    stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                ),
            ]
        if type == ShaderType.CROSS_TILE_ROUTER:
            return [
                # Tile states
                vk.VkDescriptorSetLayoutBinding(
                    binding=0,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount=1,
                    stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                ),
                # Tile resolutions
                vk.VkDescriptorSetLayoutBinding(
                    binding=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount=1,
                    stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                ),
                # Cross-tile flow
                vk.VkDescriptorSetLayoutBinding(
                    binding=2,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount=1,
                    stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                ),
            ]
        return []

    def get_workgroup_size(self, type: ShaderType) -> int:
        """Get workgroup size for shader type."""
        return 256  # Both shaders use 256 threads per workgroup

    def cleanup(self) -> None:
        """Clean up shader modules."""
        for shader_module in self._shader_modules.values():
            vk.vkDestroyShaderModule(self.device, shader_module, None)
        self._shader_modules.clear()
