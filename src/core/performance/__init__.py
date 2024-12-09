"""Performance optimization module for Adaptive Attention Tiling.

This module provides tools and utilities for optimizing performance:
1. CPU optimization (vectorization, memory management)
2. Profiling tools
3. Performance metrics
"""

from .cpu_optimizer import CPUOptimizer, PerformanceMetrics

__all__ = ['CPUOptimizer', 'PerformanceMetrics']
