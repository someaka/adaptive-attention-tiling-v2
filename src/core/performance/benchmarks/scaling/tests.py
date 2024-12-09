"""Scaling Benchmarks.

This module provides benchmarks for scaling behavior including:
- Strong scaling (fixed problem size, varying resources)
- Weak scaling (fixed problem size per resource)
- Memory scaling (memory usage patterns with scale)
"""

import torch
import time
import psutil
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np
from ...cpu.vectorization import VectorizationOptimizer
from ...cpu.memory import MemoryManager
from ..core.operations import CoreBenchmarks

@dataclass
class ScalingResult:
    """Results from a scaling benchmark."""
    test_type: str
    scale_factor: int
    execution_time: float
    speedup: float
    efficiency: float
    memory_per_unit: float
    total_memory: float

class ScalingBenchmarks:
    """Benchmarks for scaling behavior."""
    
    def __init__(self,
                 base_size: int = 1024,
                 max_scale: int = 8,
                 device: str = 'cpu'):
        self.base_size = base_size
        self.max_scale = max_scale
        self.device = device
        self.results: List[ScalingResult] = []
        
        # Initialize benchmarking components
        self.core_benchmarks = CoreBenchmarks(device=device)
        self.memory_manager = MemoryManager()
    
    def benchmark_strong_scaling(self,
                               computation: Callable[[torch.Tensor], torch.Tensor],
                               input_size: Tuple[int, ...]) -> List[ScalingResult]:
        """Benchmark strong scaling behavior."""
        results = []
        base_tensor = torch.randn(*input_size, device=self.device)
        base_time = None
        
        for scale in range(1, self.max_scale + 1):
            # Split computation across processes
            chunk_size = input_size[0] // scale
            if chunk_size == 0:
                break
                
            times = []
            memory_usage = []
            
            for _ in range(3):  # Multiple runs for stability
                start = time.perf_counter()
                
                if scale == 1:
                    _ = computation(base_tensor)
                else:
                    chunks = base_tensor.split(chunk_size)
                    with mp.Pool(scale) as pool:
                        _ = pool.map(computation, chunks)
                
                end = time.perf_counter()
                times.append(end - start)
                
                memory_info = psutil.Process().memory_info()
                memory_usage.append(memory_info.rss)
            
            avg_time = sum(times) / len(times)
            avg_memory = sum(memory_usage) / len(memory_usage)
            
            if base_time is None:
                base_time = avg_time
            
            speedup = base_time / avg_time
            efficiency = speedup / scale
            
            result = ScalingResult(
                test_type='strong',
                scale_factor=scale,
                execution_time=avg_time,
                speedup=speedup,
                efficiency=efficiency,
                memory_per_unit=avg_memory / scale,
                total_memory=avg_memory
            )
            
            results.append(result)
            self.results.append(result)
        
        return results
    
    def benchmark_weak_scaling(self,
                             computation: Callable[[torch.Tensor], torch.Tensor],
                             base_size: Optional[int] = None) -> List[ScalingResult]:
        """Benchmark weak scaling behavior."""
        if base_size is None:
            base_size = self.base_size
            
        results = []
        base_time = None
        
        for scale in range(1, self.max_scale + 1):
            # Increase problem size with scale
            size = base_size * scale
            tensor = torch.randn(size, size, device=self.device)
            
            times = []
            memory_usage = []
            
            for _ in range(3):  # Multiple runs for stability
                start = time.perf_counter()
                _ = computation(tensor)
                end = time.perf_counter()
                times.append(end - start)
                
                memory_info = psutil.Process().memory_info()
                memory_usage.append(memory_info.rss)
            
            avg_time = sum(times) / len(times)
            avg_memory = sum(memory_usage) / len(memory_usage)
            
            if base_time is None:
                base_time = avg_time
            
            efficiency = base_time / avg_time
            
            result = ScalingResult(
                test_type='weak',
                scale_factor=scale,
                execution_time=avg_time,
                speedup=1.0,  # Not applicable for weak scaling
                efficiency=efficiency,
                memory_per_unit=avg_memory / scale,
                total_memory=avg_memory
            )
            
            results.append(result)
            self.results.append(result)
        
        return results
    
    def benchmark_memory_scaling(self,
                               base_size: Optional[int] = None) -> List[ScalingResult]:
        """Benchmark memory usage scaling."""
        if base_size is None:
            base_size = self.base_size
            
        results = []
        base_memory = None
        
        for scale in range(1, self.max_scale + 1):
            size = base_size * scale
            tensors = []
            
            start = time.perf_counter()
            
            # Allocate tensors with increasing sizes
            for _ in range(scale):
                if self.memory_manager:
                    tensor = self.memory_manager.pool.acquire((size, size), 
                                                           torch.float32)
                else:
                    tensor = torch.randn(size, size, device=self.device)
                tensors.append(tensor)
            
            end = time.perf_counter()
            
            memory_info = psutil.Process().memory_info()
            total_memory = memory_info.rss
            
            if base_memory is None:
                base_memory = total_memory
            
            result = ScalingResult(
                test_type='memory',
                scale_factor=scale,
                execution_time=end - start,
                speedup=1.0,  # Not applicable for memory scaling
                efficiency=base_memory / (total_memory / scale),
                memory_per_unit=total_memory / scale,
                total_memory=total_memory
            )
            
            results.append(result)
            self.results.append(result)
            
            # Cleanup
            for tensor in tensors:
                if self.memory_manager:
                    self.memory_manager.release_tensor(tensor)
                del tensor
            
            torch.cuda.empty_cache() if self.device == 'cuda' else None
        
        return results
    
    def run_all_benchmarks(self) -> Dict[str, List[ScalingResult]]:
        """Run all scaling benchmarks."""
        # Define test computation
        def test_computation(x: torch.Tensor) -> torch.Tensor:
            return torch.matmul(x, x.t())
        
        return {
            'strong': self.benchmark_strong_scaling(test_computation, 
                                                 (self.base_size, self.base_size)),
            'weak': self.benchmark_weak_scaling(test_computation),
            'memory': self.benchmark_memory_scaling()
        }
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all benchmarks."""
        summary = {}
        for result in self.results:
            if result.test_type not in summary:
                summary[result.test_type] = {
                    'avg_time': result.execution_time,
                    'avg_speedup': result.speedup,
                    'avg_efficiency': result.efficiency,
                    'avg_memory_per_unit': result.memory_per_unit,
                    'max_scale': result.scale_factor,
                    'count': 1
                }
            else:
                stats = summary[result.test_type]
                stats['avg_time'] = (stats['avg_time'] * stats['count'] + 
                                   result.execution_time) / (stats['count'] + 1)
                stats['avg_speedup'] = (stats['avg_speedup'] * stats['count'] + 
                                      result.speedup) / (stats['count'] + 1)
                stats['avg_efficiency'] = (stats['avg_efficiency'] * stats['count'] + 
                                         result.efficiency) / (stats['count'] + 1)
                stats['avg_memory_per_unit'] = (stats['avg_memory_per_unit'] * 
                                              stats['count'] + 
                                              result.memory_per_unit) / (stats['count'] + 1)
                stats['max_scale'] = max(stats['max_scale'], result.scale_factor)
                stats['count'] += 1
        return summary
