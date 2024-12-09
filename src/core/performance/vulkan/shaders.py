"""Vulkan Shader Management.

This module provides tools for managing Vulkan compute shaders including:
- Shader compilation
- Pipeline creation
- Resource binding
"""

import vulkan as vk
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from pathlib import Path
import ctypes

class ShaderManager:
    """Manages Vulkan compute shaders."""
    
    def __init__(self, device: vk.Device, shader_dir: str = 'shaders'):
        self.device = device
        self.shader_dir = Path(shader_dir)
        self.shader_modules: Dict[str, vk.ShaderModule] = {}
        self.pipelines: Dict[str, vk.Pipeline] = {}
        self.pipeline_layouts: Dict[str, vk.PipelineLayout] = {}
        
    def create_shader_module(self, shader_name: str) -> vk.ShaderModule:
        """Create a shader module from SPIR-V code."""
        shader_path = self.shader_dir / f"{shader_name}.spv"
        with open(shader_path, 'rb') as f:
            code = f.read()
            
        create_info = vk.ShaderModuleCreateInfo(
            code=code
        )
        
        shader_module = vk.CreateShaderModule(self.device, create_info, None)
        self.shader_modules[shader_name] = shader_module
        return shader_module
    
    def create_compute_pipeline(self,
                              shader_name: str,
                              push_constant_size: int = 0) -> Tuple[vk.Pipeline, vk.PipelineLayout]:
        """Create a compute pipeline with the specified shader."""
        # Create descriptor set layout
        bindings = []
        for i in range(4):  # Assuming max 4 bindings per shader
            binding = vk.DescriptorSetLayoutBinding(
                binding=i,
                descriptorType=vk.DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.SHADER_STAGE_COMPUTE_BIT
            )
            bindings.append(binding)
            
        descriptor_set_layout_info = vk.DescriptorSetLayoutCreateInfo(
            bindings=bindings
        )
        descriptor_set_layout = vk.CreateDescriptorSetLayout(
            self.device, descriptor_set_layout_info, None
        )
        
        # Create pipeline layout
        push_constant_range = vk.PushConstantRange(
            stageFlags=vk.SHADER_STAGE_COMPUTE_BIT,
            offset=0,
            size=push_constant_size
        ) if push_constant_size > 0 else None
        
        pipeline_layout_info = vk.PipelineLayoutCreateInfo(
            setLayouts=[descriptor_set_layout],
            pushConstantRanges=[push_constant_range] if push_constant_range else []
        )
        pipeline_layout = vk.CreatePipelineLayout(
            self.device, pipeline_layout_info, None
        )
        
        # Create compute pipeline
        shader_stage = vk.PipelineShaderStageCreateInfo(
            stage=vk.SHADER_STAGE_COMPUTE_BIT,
            module=self.shader_modules[shader_name],
            pName="main"
        )
        
        compute_pipeline_info = vk.ComputePipelineCreateInfo(
            layout=pipeline_layout,
            stage=shader_stage
        )
        
        pipeline = vk.CreateComputePipelines(
            self.device, vk.PipelineCache(0), 1, compute_pipeline_info, None
        )[0]
        
        self.pipelines[shader_name] = pipeline
        self.pipeline_layouts[shader_name] = pipeline_layout
        
        return pipeline, pipeline_layout
    
    def cleanup(self) -> None:
        """Clean up Vulkan resources."""
        for shader_module in self.shader_modules.values():
            vk.DestroyShaderModule(self.device, shader_module, None)
        
        for pipeline in self.pipelines.values():
            vk.DestroyPipeline(self.device, pipeline, None)
        
        for layout in self.pipeline_layouts.values():
            vk.DestroyPipelineLayout(self.device, layout, None)

class ComputeShaderDispatcher:
    """Dispatches compute shader workloads."""
    
    def __init__(self, 
                 device: vk.Device,
                 queue: vk.Queue,
                 command_pool: vk.CommandPool):
        self.device = device
        self.queue = queue
        self.command_pool = command_pool
        
    def dispatch(self,
                pipeline: vk.Pipeline,
                pipeline_layout: vk.PipelineLayout,
                descriptor_set: vk.DescriptorSet,
                push_constants: Optional[bytes] = None,
                group_count: Tuple[int, int, int] = (1, 1, 1)) -> None:
        """Dispatch a compute shader workload."""
        # Create command buffer
        command_buffer_info = vk.CommandBufferAllocateInfo(
            commandPool=self.command_pool,
            level=vk.COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1
        )
        command_buffer = vk.AllocateCommandBuffers(self.device, command_buffer_info)[0]
        
        # Begin command buffer
        begin_info = vk.CommandBufferBeginInfo(
            flags=vk.COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        )
        vk.BeginCommandBuffer(command_buffer, begin_info)
        
        # Bind pipeline
        vk.CmdBindPipeline(command_buffer, vk.PIPELINE_BIND_POINT_COMPUTE, pipeline)
        
        # Bind descriptor set
        vk.CmdBindDescriptorSets(
            command_buffer,
            vk.PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout,
            0, 1, [descriptor_set],
            0, None
        )
        
        # Push constants if provided
        if push_constants:
            vk.CmdPushConstants(
                command_buffer,
                pipeline_layout,
                vk.SHADER_STAGE_COMPUTE_BIT,
                0,
                len(push_constants),
                push_constants
            )
        
        # Dispatch compute work
        vk.CmdDispatch(command_buffer, *group_count)
        
        # End command buffer
        vk.EndCommandBuffer(command_buffer)
        
        # Submit command buffer
        submit_info = vk.SubmitInfo(
            commandBuffers=[command_buffer]
        )
        vk.QueueSubmit(self.queue, 1, submit_info, vk.Fence(0))
        vk.QueueWaitIdle(self.queue)
        
        # Cleanup
        vk.FreeCommandBuffers(self.device, self.command_pool, 1, [command_buffer])

class ShaderCompiler:
    """Compiles GLSL shaders to SPIR-V."""
    
    def __init__(self, shader_dir: str = 'shaders'):
        self.shader_dir = Path(shader_dir)
        self.shader_dir.mkdir(parents=True, exist_ok=True)
    
    def compile_shader(self, shader_name: str) -> None:
        """Compile a GLSL shader to SPIR-V."""
        import subprocess
        
        input_file = self.shader_dir / f"{shader_name}.comp"
        output_file = self.shader_dir / f"{shader_name}.spv"
        
        result = subprocess.run([
            'glslangValidator',
            '-V',
            str(input_file),
            '-o',
            str(output_file)
        ])
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to compile shader: {shader_name}")
    
    def compile_all_shaders(self) -> None:
        """Compile all GLSL shaders in the shader directory."""
        for shader_file in self.shader_dir.glob('*.comp'):
            self.compile_shader(shader_file.stem)
