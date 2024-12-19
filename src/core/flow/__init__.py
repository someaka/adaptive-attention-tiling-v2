"""Geometric Flow Package.

This package provides implementations of geometric flows for various applications:
1. Base geometric flow with common functionality
2. Quantum geometric flow with uncertainty principles
3. Neural geometric flow with learned dynamics
4. Pattern formation flow with reaction-diffusion

The flows share a common interface defined by GeometricFlowProtocol.
"""

from .base import BaseGeometricFlow
from .neural import NeuralGeometricFlow
from .pattern import PatternFormationFlow
from .protocol import FlowMetrics, GeometricFlowProtocol, SingularityInfo
from .quantum import QuantumGeometricFlow

__all__ = [
    'BaseGeometricFlow',
    'FlowMetrics',
    'GeometricFlowProtocol',
    'NeuralGeometricFlow',
    'PatternFormationFlow',
    'QuantumGeometricFlow',
    'SingularityInfo',
]
