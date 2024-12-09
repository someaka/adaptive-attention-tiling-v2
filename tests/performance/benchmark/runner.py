"""Test runner for continuous performance monitoring.

This module provides utilities to run performance tests and collect metrics
using the benchmark monitoring system.
"""

import argparse
import os
import sys
from typing import Optional

import pytest
from monitor import BenchmarkMonitor


def run_performance_tests(
    test_paths: list[str],
    results_dir: str,
    pattern: Optional[str] = None,
    history_length: int = 100,
    alert_threshold: float = 0.2,
) -> None:
    """Run performance tests with monitoring."""
    # Initialize benchmark monitor
    monitor = BenchmarkMonitor(
        results_dir=results_dir,
        history_length=history_length,
        alert_threshold=alert_threshold,
    )

    # Prepare pytest arguments
    pytest_args = ["-v"]  # Verbose output

    # Add test paths
    pytest_args.extend(test_paths)

    # Add pattern if specified
    if pattern:
        pytest_args.extend(["-k", pattern])

    # Add custom pytest plugin for monitoring
    class BenchmarkPlugin:
        """Pytest plugin for benchmark monitoring."""

        @pytest.hookimpl(tryfirst=True)
        def pytest_runtest_setup(self, item) -> None:
            """Setup benchmark monitoring before each test."""
            monitor.start_benchmark(item.name)

        @pytest.hookimpl(trylast=True)
        def pytest_runtest_teardown(self, item) -> None:
            """Record benchmark results after each test."""
            monitor.end_benchmark()

    # Run tests with monitoring
    pytest.main(pytest_args, plugins=[BenchmarkPlugin()])

    # Generate and save report
    report = monitor.generate_report()

    # Generate plots for each test
    for test_name in report:
        monitor.plot_trends(test_name)


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Run performance tests with monitoring"
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        default=["tests/performance/cpu", "tests/performance/vulkan"],
        help="Paths to test directories or files",
    )
    parser.add_argument(
        "--results-dir",
        default="benchmark_results",
        help="Directory to store benchmark results",
    )
    parser.add_argument("--pattern", help="Pattern to filter test names")
    parser.add_argument(
        "--history-length",
        type=int,
        default=100,
        help="Number of historical results to keep",
    )
    parser.add_argument(
        "--alert-threshold",
        type=float,
        default=0.2,
        help="Z-score threshold for regression alerts",
    )

    args = parser.parse_args()

    # Ensure all test paths exist
    for test_path in args.tests:
        if not os.path.exists(test_path):
            print(f"Error: Test path does not exist: {test_path}")
            sys.exit(1)

    # Run tests
    run_performance_tests(
        test_paths=args.tests,
        results_dir=args.results_dir,
        pattern=args.pattern,
        history_length=args.history_length,
        alert_threshold=args.alert_threshold,
    )


if __name__ == "__main__":
    main()
