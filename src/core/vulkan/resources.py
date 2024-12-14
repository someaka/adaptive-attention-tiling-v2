"""Vulkan resource management."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any, cast as type_cast, Type, TypeVar

from ctypes import c_void_p, cast, POINTER, Structure, c_uint32, c_uint64, _CData, _CArgObject

import vulkan as vk

from .memory import MemoryBlock, VulkanMemory


# Type aliases for Vulkan handles
VkHandle = Any  # Opaque handle type for Vulkan objects

# Type variable for structure types
T = TypeVar('T', bound=Structure)


class VkMemoryRequirements(Structure):
    """Memory requirements structure."""
    _fields_ = [
        ("size", c_uint64),
        ("alignment", c_uint64),
        ("memoryTypeBits", c_uint32)
    ]


@dataclass
class MemoryRequirements:
    """Memory requirements wrapper."""
    size: int
    alignment: int
    memory_type_bits: int

    @classmethod
    def from_vulkan(cls, vulkan_reqs: VkMemoryRequirements) -> 'MemoryRequirements':
        """Create from Vulkan structure."""
        return cls(
            size=vulkan_reqs.size,
            alignment=vulkan_reqs.alignment,
            memory_type_bits=vulkan_reqs.memoryTypeBits
        )


@dataclass
class BufferResource:
    """Buffer resource."""
    
    buffer: VkHandle  # VkBuffer handle
    memory: MemoryBlock
    size: int
    usage: int


@dataclass
class ImageResource:
    """Image resource."""
    
    image: VkHandle  # VkImage handle
    memory: MemoryBlock
    view: VkHandle  # VkImageView handle
    format: int
    extent: VkHandle  # VkExtent3D structure


def _cast_to_struct(obj: Any, struct_type: Type[Structure]) -> Any:
    """Safely cast a CData object to a structure pointer.
    
    Args:
        obj: CData object to cast
        struct_type: Target structure type
        
    Returns:
        Cast pointer to structure
    """
    # First cast to void pointer to ensure compatibility
    void_ptr = cast(obj, c_void_p)
    # Then cast to target structure pointer
    return cast(void_ptr, POINTER(struct_type))


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
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=size,
            usage=usage,
            sharingMode=sharing_mode,
        )
        buffer = vk.vkCreateBuffer(self.device, create_info, None)
        
        # Get memory requirements
        raw_reqs = vk.vkGetBufferMemoryRequirements(self.device, buffer)
        reqs_ptr = _cast_to_struct(raw_reqs, VkMemoryRequirements)
        requirements = MemoryRequirements.from_vulkan(reqs_ptr.contents)
        
        # Allocate memory
        block = self.memory.allocate(
            requirements.size,
            requirements.memory_type_bits,
            memory_properties,
        )
        
        # Bind memory
        vk.vkBindBufferMemory(self.device, buffer, block.memory, 0)
        
        # Create resource
        resource = BufferResource(
            buffer=buffer,  # Already a Vulkan handle
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
        width: int,
        height: int,
        depth: int = 1,
        usage: int = vk.VK_IMAGE_USAGE_SAMPLED_BIT | vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        sharing_mode: int = vk.VK_SHARING_MODE_EXCLUSIVE,
        memory_properties: int = vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    ) -> ImageResource:
        """Create image resource.
        
        Args:
            format: Image format
            width: Image width
            height: Image height
            depth: Image depth (default: 1 for 2D images)
            usage: Image usage flags
            sharing_mode: Image sharing mode
            memory_properties: Memory property flags
            
        Returns:
            Image resource
        """
        # Create extent structure
        extent = vk.VkExtent3D(width=width, height=height, depth=depth)
        
        # Create image
        create_info = vk.VkImageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
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
        raw_reqs = vk.vkGetImageMemoryRequirements(self.device, image)
        reqs_ptr = _cast_to_struct(raw_reqs, VkMemoryRequirements)
        requirements = MemoryRequirements.from_vulkan(reqs_ptr.contents)
        
        # Allocate memory
        block = self.memory.allocate(
            requirements.size,
            requirements.memory_type_bits,
            memory_properties,
        )
        
        # Bind memory
        vk.vkBindImageMemory(self.device, image, block.memory, 0)
        
        # Create image view
        view_create_info = vk.VkImageViewCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
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
            image=image,  # Already a Vulkan handle
            memory=block,
            view=view,  # Already a Vulkan handle
            format=format,
            extent=extent,  # Already a Vulkan structure
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
