"""Global pytest configuration and fixtures for testing."""

import gc
import random
import resource
import signal
import subprocess
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NoReturn

import black
import numpy as np
import pytest
import torch

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
    """Run ruff commands in sequence."""
    file_str = str(file_path.resolve())
    commands = [
        ["check", "--fix", "--unsafe-fixes"],
        ["format"],
        ["check"],
    ]

    for args in commands:
        result = subprocess.run(
            [str(ruff_path), *args, file_str],
            capture_output=True,
            text=True,
            check=False,
        )
        # Skip "no files to check" case only for the fix command
        if result.returncode != 0 and (
            args != ["check", "--fix", "--unsafe-fixes"]
            or "no files to check" not in result.stderr.lower()
        ):
            command_name = args[0] if args else "unknown"
            pytest.fail(f"Ruff {command_name} failed for {file_path}:\n{result.stderr}")


def pytest_collect_file(parent: pytest.File, file_path: Path) -> pytest.Item | None:  # noqa: ARG001
    """Format Python files and check style with black and ruff during test collection."""
    if not str(file_path).endswith(".py"):
        return None

    # Skip files in .venv directory
    if ".venv" in str(file_path):
        return None

    # Get project root directory
    root_dir = Path(__file__).parent.parent

    # Only process files under the project root
    try:
        file_path.relative_to(root_dir)
    except ValueError:
        return None

    try:
        ruff_path = _find_ruff_executable()
        content = file_path.read_text()
        content = _run_black_format(file_path, content)
        _run_ruff_commands(ruff_path, file_path)
    except (subprocess.CalledProcessError, OSError) as e:
        pytest.fail(f"Command failed for {file_path}: {e}")

    return None


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
