"""Global pytest configuration and fixtures for testing."""

import gc
import logging
import os
import random
import resource
import signal
import subprocess
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NoReturn, Optional

import black
import numpy as np
import pytest
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("test.log", mode="w")],
)

logger = logging.getLogger(__name__)

# Resource limits
MAX_MEMORY_GB = 4  # Maximum memory limit in GB
MAX_TIME_SECONDS = 30  # Maximum time limit per test in seconds

# Test configurations - reduced sizes for safety
BATCH_SIZES = [32, 128]  # Removed larger sizes
SEQUENCE_LENGTHS = [64, 256]  # Removed larger sizes
FEATURE_DIMS = [32, 128]  # Removed larger sizes
CHUNK_SIZES = [64, 256]  # Removed larger sizes
MATRIX_SIZES = [(64, 64), (256, 256)]  # Removed larger sizes
POOL_SIZES = [1024, 4096]  # KB, removed larger sizes
BLOCK_SIZES = [32, 128]  # KB, removed larger sizes
CACHE_SIZES = [32, 256]  # KB, removed larger sizes


@contextmanager
def resource_guard() -> Generator[None, None, None]:
    """Set up resource limits for memory and time."""
    # Set memory limit
    memory_limit = MAX_MEMORY_GB * 1024 * 1024 * 1024  # Convert to bytes
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit, hard))

    # Set up timeout
    def timeout_handler(_signum: int, _frame: Any) -> NoReturn:
        msg = f"Test exceeded {MAX_TIME_SECONDS} seconds time limit"
        raise TimeoutError(msg)

    # Set signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(MAX_TIME_SECONDS)

    try:
        yield
    finally:
        # Reset signal handler and alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        # Reset memory limit
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


def _set_random_seeds() -> None:
    """Set random seeds for reproducibility."""
    seed = 42
    random.seed(seed)
    np.random.default_rng(seed)
    torch.manual_seed(seed)
    # Note: Vulkan support will be added when PyTorch's Python API supports it


@pytest.fixture(scope="session", autouse=True)
def _set_random_seeds_fixture() -> None:
    """Set random seeds for reproducibility across all tests."""
    _set_random_seeds()


@pytest.fixture(autouse=True)
def _resource_guard_fixture() -> Generator[None, None, None]:
    """Apply resource guards to all tests automatically."""
    with resource_guard():
        yield


def _find_ruff_executable() -> Path:
    """Find the ruff executable in common locations."""
    ruff_paths = [
        Path(".venv/bin/ruff").resolve(),
        Path(".venv/Scripts/ruff.exe").resolve(),
    ]
    try:
        # Try to find ruff in PATH, but don't fail if not found
        # Use full path to which to avoid S607
        which_path = "/usr/bin/which"
        if Path(which_path).exists():
            which_output = subprocess.run(
                [which_path, "ruff"], capture_output=True, text=True, check=False
            )
            if which_output.returncode == 0:
                ruff_paths.append(Path(which_output.stdout.strip()))
    except (subprocess.SubprocessError, OSError):
        pass

    ruff_path = next((p for p in ruff_paths if p.exists()), None)
    if not ruff_path:
        msg = "Could not find ruff executable. Please ensure ruff is installed in your virtual environment."
        raise FileNotFoundError(msg)
    return ruff_path


def _run_black_format(file_path: Path, content: str) -> str:
    """Run black formatting on file content."""
    mode = black.FileMode(line_length=100)
    try:
        formatted = black.format_str(content, mode=mode)
    except black.NothingChanged:
        return content  # File already formatted correctly
    except (black.InvalidInput, ValueError) as e:
        pytest.fail(f"Black formatting failed for {file_path}: {e}")

    # Only write if we got here (successful formatting)
    if formatted != content:
        file_path.write_text(formatted)
    return formatted


def _run_ruff_commands(ruff_path: Path, file_path: Path) -> None:
    """Run ruff commands on a file.

    Args:
        ruff_path: Path to ruff executable
        file_path: File to run ruff on
    """
    logger.info("Running ruff commands on %s", file_path)
    file_str = str(file_path)
    commands = [["check", "--fix", "--unsafe-fixes"], ["format"], ["check"]]

    try:
        for command in commands:
            result = subprocess.run(
                [str(ruff_path), *command, file_str], capture_output=True, text=True, check=True
            )
            if result.stderr:
                pytest.fail(f"Ruff {command[0]} failed for {file_path}:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        logger.exception("Failed to run ruff command %s: %s", command, str(e))
        raise


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Modify test items before test run.

    Args:
        items: List of test items to modify
    """
    logger.info("Collected %d test items", len(items))
    for item in items:
        logger.debug("Test item: %s", item.name)


def pytest_configure(config: Any) -> None:
    """Configure pytest.

    Args:
        config: Pytest config object
    """
    # Configure logging based on verbosity
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_level = logging.DEBUG if config.option.verbose > 0 else logging.INFO

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))

    # Add file handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "test.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def pytest_unconfigure(config: Any) -> None:
    """Clean up after pytest run.

    Args:
        config: Pytest config object
    """
    # Log test session summary
    if config.option.verbose > 0:
        passed = len(config.pluginmanager.get_plugin("terminalreporter").stats.get("passed", []))
        failed = len(config.pluginmanager.get_plugin("terminalreporter").stats.get("failed", []))
        logger.info(
            "Test session finished with %d passed, %d failed",
            passed,
            failed,
        )
    else:
        logger.info("Test session finished")


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Set up before each test.

    Args:
        item: Test item being run
    """
    logger.info("Setting up test: %s", item.name)


def pytest_runtest_teardown(item: pytest.Item) -> None:
    """Clean up after each test.

    Args:
        item: Test item that was run
    """
    logger.info("Tearing down test: %s", item.name)


@pytest.fixture(scope="session")
def root_dir() -> Path:
    """Get root directory of project.

    Returns:
        Path to project root directory
    """
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def ruff_path(root_dir: Path) -> Path:
    """Get path to ruff executable.

    Args:
        root_dir: Project root directory

    Returns:
        Path to ruff executable

    Raises:
        FileNotFoundError: If ruff executable not found
    """
    venv_dir = root_dir / ".venv"
    if not venv_dir.exists():
        venv_dir = root_dir / "venv"

    ruff = venv_dir / "Scripts" / "ruff.exe" if os.name == "nt" else venv_dir / "bin" / "ruff"

    if not ruff.exists():
        msg = f"Could not find ruff at {ruff}"
        raise FileNotFoundError(msg)

    return ruff


def pytest_collect_file(file_path: Path, parent: Any) -> Optional[pytest.Item]:
    """Process file before collection.

    Args:
        file_path: Path to file being collected
        parent: Parent collector

    Returns:
        None if file should be skipped
    """
    if file_path.suffix != ".py":
        return None

    try:
        # Get project root directory
        root_dir = Path(__file__).parent.parent

        # Try both .venv and venv directories
        ruff_path = root_dir / ".venv" / "bin" / "ruff"
        if not ruff_path.exists():
            ruff_path = root_dir / "venv" / "bin" / "ruff"

        if not ruff_path.exists():
            logger.error("Could not find ruff executable in %s", root_dir)
            return None

        # Run ruff commands and collect file if parent is a test module
        _run_ruff_commands(ruff_path, file_path)
        if parent.name.startswith("test_"):
            return parent.collect_file(file_path)

    except Exception as e:
        logger.exception("Error processing %s: %s", file_path, str(e))
        raise

    return None


@pytest.fixture(autouse=True)
def setup_logging(request: pytest.FixtureRequest) -> None:
    """Set up logging for each test.

    Args:
        request: Pytest request object
    """
    logger = logging.getLogger(request.node.name)
    logger.setLevel(logging.INFO)

    # Add test-specific log file
    log_file = Path("logs") / f"{request.node.name}.log"
    handler = logging.FileHandler(log_file, mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

    yield logger

    # Clean up
    handler.close()
    logger.removeHandler(handler)


@pytest.fixture(scope="session")
def device() -> str:
    """Return the device to use for tests."""
    # Note: Using CPU for now, will switch to Vulkan when PyTorch support is ready
    return "cpu"


@pytest.fixture(scope="session")
def dtype() -> torch.dtype:
    """Return the default dtype to use for tests."""
    return torch.float32


@pytest.fixture
def benchmark_config() -> dict[str, int | float]:
    """Return default benchmark configuration."""
    return {
        "warmup_iterations": 10,
        "num_iterations": 100,
        "min_time": 0.000005,  # 5 microseconds minimum time
        "max_time": 1.0,  # 1 second maximum time
        "min_rounds": 5,  # Minimum number of rounds
        "calibration_precision": 10,
        "disable_gc": False,  # Keep garbage collection enabled
    }
