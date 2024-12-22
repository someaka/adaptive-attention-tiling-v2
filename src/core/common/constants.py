"""Common constants used throughout the project."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
SRC_DIR = PROJECT_ROOT / "src"
CORE_DIR = SRC_DIR / "core"
TESTS_DIR = PROJECT_ROOT / "tests"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# File extensions
PY_EXT = ".py"
YAML_EXT = ".yaml"
JSON_EXT = ".json"
LOG_EXT = ".log"

# Default values
DEFAULT_BATCH_SIZE = 32
DEFAULT_SEQ_LEN = 128
DEFAULT_HIDDEN_DIM = 256
DEFAULT_NUM_HEADS = 8
DEFAULT_DROPOUT = 0.1
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_STEPS = 1000
DEFAULT_MAX_STEPS = 100000
DEFAULT_SAVE_STEPS = 1000
DEFAULT_LOG_STEPS = 100

# Memory constants
KB = 1024
MB = KB * 1024
GB = MB * 1024

# Performance constants
MAX_THREADS = os.cpu_count() or 1
DEFAULT_NUM_WORKERS = min(4, MAX_THREADS)
MEMORY_LIMIT = 0.9  # Use up to 90% of available memory
