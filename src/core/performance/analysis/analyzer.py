"""Performance Analysis Tools.

This module provides tools for analyzing performance data including:
- Bottleneck detection
- Optimization suggestions
- Regression analysis
- Performance visualization
- Report generation
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


@dataclass
class PerformanceBottleneck:
    """Detected performance bottleneck."""

    component: str
    severity: float  # 0.0 to 1.0
    description: str
    suggestions: List[str]


@dataclass
class RegressionResult:
    """Performance regression analysis result."""

    metric: str
    baseline: float
    current: float
    change_percent: float
    is_regression: bool
    severity: float


class PerformanceAnalyzer:
    """Analyzes performance data to detect issues and suggest optimizations."""

    def __init__(
        self,
        threshold_cpu: float = 0.8,
        threshold_memory: float = 0.8,
        threshold_regression: float = 0.1,
    ):
        self.threshold_cpu = threshold_cpu
        self.threshold_memory = threshold_memory
        self.threshold_regression = threshold_regression
        self.bottlenecks: List[PerformanceBottleneck] = []
        self.regressions: List[RegressionResult] = []

    def analyze_metrics(self, metrics: Dict[str, Any]) -> List[PerformanceBottleneck]:
        """Analyze performance metrics to detect bottlenecks."""
        bottlenecks = []

        # CPU Analysis
        if "cpu_usage" in metrics:
            cpu_usage = metrics["cpu_usage"]
            if cpu_usage > self.threshold_cpu:
                bottlenecks.append(
                    PerformanceBottleneck(
                        component="CPU",
                        severity=(cpu_usage - self.threshold_cpu)
                        / (1 - self.threshold_cpu),
                        description=f"High CPU usage: {cpu_usage:.1%}",
                        suggestions=[
                            "Consider vectorizing operations",
                            "Optimize memory access patterns",
                            "Use batch processing where possible",
                        ],
                    )
                )

        # Memory Analysis
        if "memory_usage" in metrics:
            memory_usage = metrics["memory_usage"]
            total_memory = (
                torch.cuda.get_device_properties(0).total_memory
                if torch.cuda.is_available()
                else None
            )
            if total_memory and memory_usage / total_memory > self.threshold_memory:
                bottlenecks.append(
                    PerformanceBottleneck(
                        component="Memory",
                        severity=(memory_usage / total_memory - self.threshold_memory)
                        / (1 - self.threshold_memory),
                        description=f"High memory usage: {memory_usage/total_memory:.1%}",
                        suggestions=[
                            "Implement memory pooling",
                            "Use gradient checkpointing",
                            "Optimize tensor allocations",
                        ],
                    )
                )

        # Cache Analysis
        if "cache_hits" in metrics and "cache_misses" in metrics:
            hits = metrics["cache_hits"]
            misses = metrics["cache_misses"]
            if misses > hits:
                bottlenecks.append(
                    PerformanceBottleneck(
                        component="Cache",
                        severity=misses / (hits + misses),
                        description=f"Poor cache utilization: {hits/(hits+misses):.1%} hit rate",
                        suggestions=[
                            "Optimize data layout",
                            "Implement cache prefetching",
                            "Reduce working set size",
                        ],
                    )
                )

        self.bottlenecks.extend(bottlenecks)
        return bottlenecks

    def analyze_regression(
        self, baseline: Dict[str, float], current: Dict[str, float]
    ) -> List[RegressionResult]:
        """Analyze performance regression between baseline and current metrics."""
        regressions = []

        for metric in baseline:
            if metric not in current:
                continue

            change = (current[metric] - baseline[metric]) / baseline[metric]
            is_regression = change > self.threshold_regression

            if is_regression:
                regressions.append(
                    RegressionResult(
                        metric=metric,
                        baseline=baseline[metric],
                        current=current[metric],
                        change_percent=change * 100,
                        is_regression=True,
                        severity=min(1.0, change / self.threshold_regression),
                    )
                )

        self.regressions.extend(regressions)
        return regressions


class PerformanceVisualizer:
    """Visualizes performance data through various plots and graphs."""

    def __init__(self, output_dir: str = "performance_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_timeline(
        self, metrics: Dict[str, List[float]], metric_name: str, save: bool = True
    ) -> None:
        """Plot metric timeline."""
        plt.figure(figsize=(12, 6))
        times = range(len(metrics[metric_name]))
        plt.plot(times, metrics[metric_name])
        plt.title(f"{metric_name} Timeline")
        plt.xlabel("Time")
        plt.ylabel(metric_name)

        if save:
            plt.savefig(self.output_dir / f"{metric_name}_timeline.png")
            plt.close()

    def plot_memory_graph(
        self, allocations: List[float], deallocations: List[float], save: bool = True
    ) -> None:
        """Plot memory allocation graph."""
        plt.figure(figsize=(12, 6))
        times = range(len(allocations))
        plt.plot(times, allocations, label="Allocations")
        plt.plot(times, deallocations, label="Deallocations")
        plt.title("Memory Usage Over Time")
        plt.xlabel("Time")
        plt.ylabel("Bytes")
        plt.legend()

        if save:
            plt.savefig(self.output_dir / "memory_graph.png")
            plt.close()

    def plot_heatmap(
        self, data: np.ndarray, labels: List[str], title: str, save: bool = True
    ) -> None:
        """Plot performance heatmap."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(data, annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels)
        plt.title(title)

        if save:
            plt.savefig(self.output_dir / f"{title.lower()}_heatmap.png")
            plt.close()


class ReportGenerator:
    """Generates performance analysis reports."""

    def __init__(self, output_dir: str = "performance_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_performance_report(
        self,
        metrics: Dict[str, Any],
        bottlenecks: List[PerformanceBottleneck],
        include_plots: bool = True,
    ) -> str:
        """Generate comprehensive performance report."""
        report = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Header
        report.append("# Performance Analysis Report")
        report.append(f"Generated: {timestamp}\n")

        # Summary
        report.append("## Performance Summary")
        for metric, value in metrics.items():
            report.append(f"- {metric}: {value}")
        report.append("")

        # Bottlenecks
        report.append("## Detected Bottlenecks")
        for bottleneck in bottlenecks:
            report.append(
                f"### {bottleneck.component} (Severity: {bottleneck.severity:.1%})"
            )
            report.append(f"Description: {bottleneck.description}")
            report.append("Suggestions:")
            for suggestion in bottleneck.suggestions:
                report.append(f"- {suggestion}")
            report.append("")

        # Save report
        report_path = self.output_dir / f"performance_report_{timestamp}.md"
        report_text = "\n".join(report)
        report_path.write_text(report_text)

        return report_text

    def generate_regression_report(self, regressions: List[RegressionResult]) -> str:
        """Generate regression analysis report."""
        report = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Header
        report.append("# Performance Regression Report")
        report.append(f"Generated: {timestamp}\n")

        # Regressions
        report.append("## Detected Regressions")
        for regression in regressions:
            report.append(f"### {regression.metric}")
            report.append(f"- Baseline: {regression.baseline:.2f}")
            report.append(f"- Current: {regression.current:.2f}")
            report.append(f"- Change: {regression.change_percent:+.1f}%")
            report.append(f"- Severity: {regression.severity:.1%}\n")

        # Save report
        report_path = self.output_dir / f"regression_report_{timestamp}.md"
        report_text = "\n".join(report)
        report_path.write_text(report_text)

        return report_text

    def generate_optimization_report(
        self, metrics: Dict[str, Any], optimizations: List[str]
    ) -> str:
        """Generate optimization suggestions report."""
        report = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Header
        report.append("# Performance Optimization Report")
        report.append(f"Generated: {timestamp}\n")

        # Current Performance
        report.append("## Current Performance Metrics")
        for metric, value in metrics.items():
            report.append(f"- {metric}: {value}")
        report.append("")

        # Optimization Suggestions
        report.append("## Recommended Optimizations")
        for i, opt in enumerate(optimizations, 1):
            report.append(f"{i}. {opt}")

        # Save report
        report_path = self.output_dir / f"optimization_report_{timestamp}.md"
        report_text = "\n".join(report)
        report_path.write_text(report_text)

        return report_text
