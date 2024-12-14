"""Profiling Tools.

This module provides profiling tools for:
- CPU profiling
- Memory profiling
- Cache profiling
- System profiling
"""

import time
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Dict, List, Optional, Any, TypeVar, cast

import line_profiler
import memory_profiler
import numpy as np
import psutil


T = TypeVar('T')

@dataclass
class ProfileResult:
    """Results from profiling."""

    function_name: str
    execution_time: float
    memory_usage: float
    cpu_percent: float
    line_stats: Optional[Dict[int, Dict[str, float]]] = None
    cache_stats: Optional[Dict[str, int]] = None


class CPUProfiler:
    """Profiles CPU usage and execution time."""

    def __init__(self, enable_line_profiling: bool = True):
        self.enable_line_profiling = enable_line_profiling
        self.line_profiler: Optional[line_profiler.LineProfiler] = None
        if enable_line_profiling:
            try:
                self.line_profiler = line_profiler.LineProfiler()
            except Exception as e:
                print(f"Warning: Failed to initialize line profiler: {e}")
                self.enable_line_profiling = False
        self.results: List[ProfileResult] = []

    def profile(self, func: Callable[..., T]) -> Callable[..., T]:
        """Profile a function's CPU usage."""
        profiled_func = func
        if self.enable_line_profiling and self.line_profiler is not None:
            profiled_func = self.line_profiler(func)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            start_time = time.perf_counter()
            process = psutil.Process()
            start_cpu = process.cpu_percent()

            # Run the function
            result = profiled_func(*args, **kwargs)

            end_time = time.perf_counter()
            end_cpu = process.cpu_percent()

            # Get line profiling stats if enabled
            line_stats = None
            if self.enable_line_profiling and self.line_profiler is not None:
                try:
                    self.line_profiler.enable_by_count()
                    stats = self.line_profiler.get_stats()
                    if hasattr(stats, 'timings'):
                        for key, timings in stats.timings.items():
                            if key[2] == func.__name__:
                                line_stats = {
                                    line: {
                                        "hits": hits,
                                        "time": timing / 1e6,  # Convert to seconds
                                    }
                                    for line, hits, timing in timings
                                }
                except Exception as e:
                    print(f"Warning: Failed to collect line profiling stats: {e}")

            self.results.append(
                ProfileResult(
                    function_name=func.__name__,
                    execution_time=end_time - start_time,
                    memory_usage=process.memory_info().rss / 1024**2,  # MB
                    cpu_percent=(end_cpu + start_cpu) / 2,
                    line_stats=line_stats,
                )
            )

            return result

        return wrapper


class MemoryProfiler:
    """Profiles memory usage and patterns."""

    def __init__(self):
        self.results: List[ProfileResult] = []

    def profile(self, func: Callable) -> Callable:
        """Profile a function's memory usage."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Setup memory profiling
            process = psutil.Process()
            memory_usage = []

            def memory_callback():
                memory_usage.append(process.memory_info().rss / 1024**2)

            # Profile the function
            profiler = memory_profiler.profile(func, precision=4)
            start_time = time.perf_counter()
            result = profiler(*args, **kwargs)
            end_time = time.perf_counter()

            self.results.append(
                ProfileResult(
                    function_name=func.__name__,
                    execution_time=end_time - start_time,
                    memory_usage=max(memory_usage) if memory_usage else 0,
                    cpu_percent=0.0,  # Not tracked in memory profiling
                    line_stats=None,
                )
            )

            return result

        return wrapper


class CacheProfiler:
    """Profiles cache behavior."""

    def __init__(self):
        self.results: List[ProfileResult] = []
        self.cache_stats = {"hits": 0, "misses": 0}

    def profile(self, func: Callable) -> Callable:
        """Profile a function's cache behavior."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()

            # Reset cache stats
            initial_hits = self.cache_stats["hits"]
            initial_misses = self.cache_stats["misses"]

            result = func(*args, **kwargs)

            end_time = time.perf_counter()

            # Calculate cache statistics
            cache_stats = {
                "hits": self.cache_stats["hits"] - initial_hits,
                "misses": self.cache_stats["misses"] - initial_misses,
            }

            self.results.append(
                ProfileResult(
                    function_name=func.__name__,
                    execution_time=end_time - start_time,
                    memory_usage=0.0,  # Not tracked in cache profiling
                    cpu_percent=0.0,  # Not tracked in cache profiling
                    cache_stats=cache_stats,
                )
            )

            return result

        return wrapper

    def record_cache_access(self, hit: bool) -> None:
        """Record a cache hit or miss."""
        if hit:
            self.cache_stats["hits"] += 1
        else:
            self.cache_stats["misses"] += 1


class SystemProfiler:
    """Profiles system-wide metrics."""

    def __init__(self, sampling_interval: float = 0.1):
        self.sampling_interval = sampling_interval
        self.results: List[Dict[str, float]] = []

    def start_monitoring(self) -> None:
        """Start system monitoring."""
        self.monitoring = True
        self.results.clear()

        while self.monitoring:
            stats = self._collect_system_stats()
            self.results.append(stats)
            time.sleep(self.sampling_interval)

    def stop_monitoring(self) -> None:
        """Stop system monitoring."""
        self.monitoring = False

    def _collect_system_stats(self) -> Dict[str, float]:
        """Collect system statistics."""
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        disk = psutil.disk_io_counters()

        return {
            "timestamp": time.time(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "swap_percent": psutil.swap_memory().percent,
            "disk_read_bytes": disk.read_bytes if disk else 0,
            "disk_write_bytes": disk.write_bytes if disk else 0,
        }

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of system metrics."""
        if not self.results:
            return {}

        summary = {}
        metrics = list(self.results[0].keys())

        for metric in metrics:
            if metric == "timestamp":
                continue

            values = [r[metric] for r in self.results]
            summary[metric] = {
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "std": np.std(values),
            }

        return summary


class ProfilingManager:
    """Manages all profiling tools."""

    def __init__(
        self,
        enable_cpu: bool = True,
        enable_memory: bool = True,
        enable_cache: bool = True,
        enable_system: bool = True,
    ):
        self.cpu_profiler = CPUProfiler() if enable_cpu else None
        self.memory_profiler = MemoryProfiler() if enable_memory else None
        self.cache_profiler = CacheProfiler() if enable_cache else None
        self.system_profiler = SystemProfiler() if enable_system else None

    def profile(self, func: Callable) -> Callable:
        """Apply all enabled profilers to a function."""
        profiled_func = func

        if self.cpu_profiler:
            profiled_func = self.cpu_profiler.profile(profiled_func)
        if self.memory_profiler:
            profiled_func = self.memory_profiler.profile(profiled_func)
        if self.cache_profiler:
            profiled_func = self.cache_profiler.profile(profiled_func)

        @wraps(profiled_func)
        def wrapper(*args, **kwargs):
            if self.system_profiler:
                self.system_profiler.start_monitoring()

            result = profiled_func(*args, **kwargs)

            if self.system_profiler:
                self.system_profiler.stop_monitoring()

            return result

        return wrapper

    def get_results(self) -> Dict[str, List[ProfileResult]]:
        """Get results from all profilers."""
        results = {}

        if self.cpu_profiler:
            results["cpu"] = self.cpu_profiler.results
        if self.memory_profiler:
            results["memory"] = self.memory_profiler.results
        if self.cache_profiler:
            results["cache"] = self.cache_profiler.results
        if self.system_profiler:
            results["system"] = self.system_profiler.results

        return results

    def clear_results(self) -> None:
        """Clear results from all profilers."""
        if self.cpu_profiler:
            self.cpu_profiler.results.clear()
        if self.memory_profiler:
            self.memory_profiler.results.clear()
        if self.cache_profiler:
            self.cache_profiler.results.clear()
        if self.system_profiler:
            self.system_profiler.results.clear()
