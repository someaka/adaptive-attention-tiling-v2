"""Common constants."""

import os

# Memory constants
MEMORY_ALIGNMENT = 256  # bytes
PAGE_SIZE = 4096  # bytes
DEFAULT_BUFFER_BLOCK_SIZE = 1024 * 1024  # 1 MB

# Compute constants
WARP_SIZE = 32
BLOCK_SIZE = 256

# Attention constants
MAX_SEQ_LENGTH = 4096
MAX_BATCH_SIZE = 32
MAX_HEAD_DIM = 128

# Performance thresholds
MIN_OCCUPANCY = 0.3
MAX_OCCUPANCY = 0.8
MIN_EFFICIENCY = 0.5
MAX_EFFICIENCY = 0.9

# Tiling constants
MIN_TILE_SIZE = 32
MAX_TILE_SIZE = 256
TILE_SIZE_INCREMENT = 32

# Resource limits
MAX_SHARED_MEMORY = 48 * 1024  # 48 KB
MAX_REGISTERS_PER_THREAD = 255
MAX_THREADS_PER_BLOCK = 1024

# Shader paths
SHADER_DIR = os.path.join("src", "core", "performance", "vulkan", "shaders")
