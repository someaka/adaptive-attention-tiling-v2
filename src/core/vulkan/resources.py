"""Vulkan resource management."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import vulkan as vk

from .memory import MemoryBlock, VulkanMemory


@dataclass
class BufferResource:
    """Buffer resource."""
    
    buffer: int  # VkBuffer
    memory: MemoryBlock
    size: int
    usage: int


@dataclass
class ImageResource:
    """Image resource."""
    
    image: int  # VkImage
    memory: MemoryBlock
    view: int  # VkImageView
    format: int
    extent: vk.VkExtent3D


class VulkanResources:
    """Manages Vulkan resources."""
    
    def __init__(self, device: int, memory_manager: VulkanMemory):
        """Initialize resource manager.
        
        Args:
            device: Vulkan device handle
            memory_manager: Memory manager
        """
        self.device = device
        self.memory = memory_manager
        
        # Track resources
        self.buffers: Dict[int, BufferResource] = {}
        self.images: Dict[int, ImageResource] = {}
        
    def create_buffer(
        self,
        size: int,
        usage: int,
        sharing_mode: int = vk.VK_SHARING_MODE_EXCLUSIVE,
        memory_properties: int = vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    ) -> BufferResource:
        """Create buffer resource.
        
        Args:
            size: Buffer size in bytes
            usage: Buffer usage flags
            sharing_mode: Buffer sharing mode
            memory_properties: Memory property flags
            
        Returns:
            Buffer resource
        """
        # Create buffer
        create_info = vk.VkBufferCreateInfo(
            size=size,
            usage=usage,
            sharingMode=sharing_mode,
        )
        buffer = vk.vkCreateBuffer(self.device, create_info, None)
        
        # Get memory requirements
        requirements = vk.vkGetBufferMemoryRequirements(self.device, buffer)
        
        # Allocate memory
        block = self.memory.allocate(
            requirements.size,
            requirements.memoryTypeBits,
            memory_properties,
        )
        
        # Bind memory
        vk.vkBindBufferMemory(self.device, buffer, block.memory, 0)
        
        # Create resource
        resource = BufferResource(
            buffer=buffer,
            memory=block,
            size=size,
            usage=usage,
        )
        
        # Track resource
        self.buffers[id(buffer)] = resource
        
        return resource
        
    def create_image(
        self,
        format: int,
        extent: vk.VkExtent3D,
        usage: int,
        sharing_mode: int = vk.VK_SHARING_MODE_EXCLUSIVE,
        memory_properties: int = vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    ) -> ImageResource:
        """Create image resource.
        
        Args:
            format: Image format
            extent: Image extent
            usage: Image usage flags
            sharing_mode: Image sharing mode
            memory_properties: Memory property flags
            
        Returns:
            Image resource
        """
        # Create image
        create_info = vk.VkImageCreateInfo(
            imageType=vk.VK_IMAGE_TYPE_2D,
            format=format,
            extent=extent,
            mipLevels=1,
            arrayLayers=1,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            tiling=vk.VK_IMAGE_TILING_OPTIMAL,
            usage=usage,
            sharingMode=sharing_mode,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
        )
        image = vk.vkCreateImage(self.device, create_info, None)
        
        # Get memory requirements
        requirements = vk.vkGetImageMemoryRequirements(self.device, image)
        
        # Allocate memory
        block = self.memory.allocate(
            requirements.size,
            requirements.memoryTypeBits,
            memory_properties,
        )
        
        # Bind memory
        vk.vkBindImageMemory(self.device, image, block.memory, 0)
        
        # Create image view
        view_create_info = vk.VkImageViewCreateInfo(
            image=image,
            viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
            format=format,
            components=vk.VkComponentMapping(
                r=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                g=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                b=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                a=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
            ),
            subresourceRange=vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0,
                levelCount=1,
                baseArrayLayer=0,
                layerCount=1,
            ),
        )
        view = vk.vkCreateImageView(self.device, view_create_info, None)
        
        # Create resource
        resource = ImageResource(
            image=image,
            memory=block,
            view=view,
            format=format,
            extent=extent,
        )
        
        # Track resource
        self.images[id(image)] = resource
        
        return resource
        
    def destroy_buffer(self, resource: BufferResource):
        """Destroy buffer resource.
        
        Args:
            resource: Buffer resource to destroy
        """
        vk.vkDestroyBuffer(self.device, resource.buffer, None)
        self.memory.free(resource.memory)
        del self.buffers[id(resource.buffer)]
        
    def destroy_image(self, resource: ImageResource):
        """Destroy image resource.
        
        Args:
            resource: Image resource to destroy
        """
        vk.vkDestroyImageView(self.device, resource.view, None)
        vk.vkDestroyImage(self.device, resource.image, None)
        self.memory.free(resource.memory)
        del self.images[id(resource.image)]
        
    def cleanup(self):
        """Cleanup all resources."""
        for resource in list(self.buffers.values()):
            self.destroy_buffer(resource)
        for resource in list(self.images.values()):
            self.destroy_image(resource)
