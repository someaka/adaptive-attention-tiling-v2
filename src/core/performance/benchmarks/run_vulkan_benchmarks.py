"""Run Vulkan Memory Benchmarks.

This script runs the Vulkan memory benchmarks and updates the performance metrics.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

from src.core.performance.benchmarks.vulkan.benchmark_runner import run_benchmarks


def main():
    """Run benchmarks and process results."""
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "benchmark_results" / f"vulkan_memory_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging
    log_file = output_dir / "benchmark_run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    try:
        logging.info("Starting Vulkan memory benchmarks...")

        # Run benchmarks
        run_benchmarks(str(output_dir))

        # Process results
        results_file = output_dir / "metadata.json"
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)

            logging.info("\nBenchmark Results Summary:")
            logging.info(f"Timestamp: {results['timestamp']}")
            logging.info(f"Device: {results['device_name']}")
            logging.info(f"Vulkan Version: {results['vulkan_version']}")

        logging.info(f"\nDetailed results and visualizations saved to: {output_dir}")

    except Exception as e:
        logging.error(f"Benchmark run failed: {e!s}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
