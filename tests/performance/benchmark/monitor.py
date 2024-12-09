"""Continuous benchmark monitoring system for Adaptive Attention Tiling.

This module provides a framework for continuous performance monitoring,
focusing on:
1. Automated benchmark execution
2. Performance regression detection
3. Resource utilization tracking
4. Historical trend analysis
"""

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import psutil
import torch


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    test_name: str
    execution_time: float
    memory_usage: float
    cpu_utilization: float
    gpu_utilization: Optional[float]
    timestamp: datetime
    metrics: dict[str, float]


class BenchmarkMonitor:
    """Continuous benchmark monitoring system."""

    def __init__(
        self,
        results_dir: Union[str, Path],
        history_length: int = 100,
        alert_threshold: float = 0.2,
    ):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.history_length = history_length
        self.alert_threshold = alert_threshold
        self.current_results: list[BenchmarkResult] = []

        # Set up logging
        logging.basicConfig(
            filename=self.results_dir / "benchmark.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def start_benchmark(self, test_name: str) -> None:
        """Start timing a benchmark."""
        self.start_time = time.perf_counter()
        self.start_memory = psutil.Process().memory_info().rss
        self.start_cpu = psutil.cpu_percent()
        self.current_test = test_name

        self.logger.info(f"Starting benchmark: {test_name}")

    def end_benchmark(
        self, additional_metrics: Optional[dict[str, float]] = None
    ) -> BenchmarkResult:
        """End timing and record benchmark results."""
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss
        end_cpu = psutil.cpu_percent()

        # Calculate metrics
        execution_time = end_time - self.start_time
        memory_usage = (end_memory - self.start_memory) / 1024 / 1024  # MB
        cpu_utilization = (self.start_cpu + end_cpu) / 2

        # Get GPU utilization if available
        try:
            gpu_util = torch.cuda.utilization()
        except (AttributeError, RuntimeError):
            gpu_util = None

        result = BenchmarkResult(
            test_name=self.current_test,
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_utilization=cpu_utilization,
            gpu_utilization=gpu_util,
            timestamp=datetime.now(),
            metrics=additional_metrics or {},
        )

        self.current_results.append(result)
        self._save_result(result)
        self._check_regression(result)

        return result

    def _save_result(self, result: BenchmarkResult) -> None:
        """Save benchmark result to disk."""
        result_dict = {
            "test_name": result.test_name,
            "execution_time": result.execution_time,
            "memory_usage": result.memory_usage,
            "cpu_utilization": result.cpu_utilization,
            "gpu_utilization": result.gpu_utilization,
            "timestamp": result.timestamp.isoformat(),
            "metrics": result.metrics,
        }

        result_file = (
            self.results_dir
            / f"{result.test_name}_{result.timestamp:%Y%m%d_%H%M%S}.json"
        )
        with open(result_file, "w") as f:
            json.dump(result_dict, f, indent=2)

    def _check_regression(self, result: BenchmarkResult) -> None:
        """Check for performance regression."""
        # Load historical results
        historical_results = self._load_historical_results(result.test_name)

        if not historical_results:
            return

        # Calculate baseline statistics
        baseline_times = [r.execution_time for r in historical_results]
        baseline_mean = np.mean(baseline_times)
        baseline_std = np.std(baseline_times)

        # Check for regression
        z_score = (result.execution_time - baseline_mean) / (baseline_std + 1e-10)
        if z_score > self.alert_threshold:
            self.logger.warning(
                f"Performance regression detected in {result.test_name}:\n"
                f"Current time: {result.execution_time:.4f}s\n"
                f"Historical mean: {baseline_mean:.4f}s\n"
                f"Z-score: {z_score:.2f}"
            )

    def _load_historical_results(self, test_name: str) -> list[BenchmarkResult]:
        """Load historical benchmark results."""
        results = []
        result_files = sorted(
            self.results_dir.glob(f"{test_name}_*.json"), reverse=True
        )[: self.history_length]

        for file in result_files:
            with open(file) as f:
                data = json.load(f)
                result = BenchmarkResult(
                    test_name=data["test_name"],
                    execution_time=data["execution_time"],
                    memory_usage=data["memory_usage"],
                    cpu_utilization=data["cpu_utilization"],
                    gpu_utilization=data["gpu_utilization"],
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    metrics=data["metrics"],
                )
                results.append(result)

        return results

    def generate_report(self) -> dict[str, dict[str, float]]:
        """Generate a summary report of all benchmarks."""
        report = {}

        for result in self.current_results:
            if result.test_name not in report:
                report[result.test_name] = {
                    "mean_time": result.execution_time,
                    "mean_memory": result.memory_usage,
                    "mean_cpu": result.cpu_utilization,
                    "count": 1,
                }
            else:
                stats = report[result.test_name]
                stats["mean_time"] = (
                    stats["mean_time"] * stats["count"] + result.execution_time
                ) / (stats["count"] + 1)
                stats["mean_memory"] = (
                    stats["mean_memory"] * stats["count"] + result.memory_usage
                ) / (stats["count"] + 1)
                stats["mean_cpu"] = (
                    stats["mean_cpu"] * stats["count"] + result.cpu_utilization
                ) / (stats["count"] + 1)
                stats["count"] += 1

        return report

    def plot_trends(self, test_name: str) -> None:
        """Plot performance trends for a specific test."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.warning("matplotlib not installed. Cannot generate plots.")
            return

        results = self._load_historical_results(test_name)
        if not results:
            return

        timestamps = [r.timestamp for r in results]
        times = [r.execution_time for r in results]
        memories = [r.memory_usage for r in results]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Execution time trend
        ax1.plot(timestamps, times, "b-")
        ax1.set_title(f"{test_name} - Execution Time Trend")
        ax1.set_xlabel("Timestamp")
        ax1.set_ylabel("Time (s)")
        ax1.grid(True)

        # Memory usage trend
        ax2.plot(timestamps, memories, "r-")
        ax2.set_title(f"{test_name} - Memory Usage Trend")
        ax2.set_xlabel("Timestamp")
        ax2.set_ylabel("Memory (MB)")
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(self.results_dir / f"{test_name}_trends.png")
        plt.close()
