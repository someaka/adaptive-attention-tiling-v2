"""Global pytest configuration and fixtures for testing."""

import random
import subprocess
from pathlib import Path

import black
import numpy as np
import pytest
import torch


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
def benchmark_config() -> dict[str, int]:
    """Return default benchmark configuration."""
    return {
        "warmup_iterations": 10,
        "num_iterations": 100,
    }
