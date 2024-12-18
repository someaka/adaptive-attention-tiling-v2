"""DEPRECATED: Pattern Dynamics Implementation for Neural Attention.

This module has been deprecated. All functionality has been moved to pattern/dynamics.py.
This file now serves only as a compatibility layer and will be removed in a future version.

Please update your imports to use:
from .pattern.dynamics import PatternDynamics
"""

import warnings
from typing import List, Optional, Tuple, Callable, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pattern.models import (
    ReactionDiffusionState,
    StabilityInfo,
    StabilityMetrics,
    ControlSignal,
    BifurcationPoint,
    BifurcationDiagram
)

from .pattern.dynamics import PatternDynamics as _PatternDynamics

class PatternDynamics(_PatternDynamics):
    """DEPRECATED: Pattern dynamics system with attention-specific features.
    
    This class has been deprecated. Please use pattern.dynamics.PatternDynamics instead.
    All functionality has been moved to the new implementation.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize pattern dynamics system."""
        warnings.warn(
            "This class is deprecated. Please use pattern.dynamics.PatternDynamics instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)

# Re-export all the public classes and functions
__all__ = [
    'ReactionDiffusionState',
    'StabilityInfo',
    'StabilityMetrics',
    'ControlSignal',
    'BifurcationPoint',
    'BifurcationDiagram',
    'PatternDynamics'
]
