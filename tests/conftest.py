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
from src.validation.geometric.flow import TilingFlowValidator as FlowValidator
from src.validation.patterns.stability import PatternValidator
from src.validation.quantum.state import QuantumStateValidator
from src.validation.framework import ValidationFramework
from src.core.tiling.geometric_flow import GeometricFlow
from src.neural.attention.pattern.dynamics import PatternDynamics

# Configure logging
root_logger = logging.getLogger()
root_logger.handlers = []  # Remove any existing handlers

def tensor_repr(tensor: torch.Tensor) -> str:
    """Format tensor representation."""
    if tensor.numel() == 0:
        return "tensor([])"
    return f"tensor(shape={list(tensor.shape)}, dtype={tensor.dtype})"

def _tensor_str(self: torch.Tensor) -> str:
    """Format tensor string representation."""
    return tensor_repr(self)

def _tensor_repr(self: torch.Tensor, *, tensor_contents: Optional[Any] = None) -> str:
    """Format tensor repr.
    
    Args:
        self: The tensor to format
        tensor_contents: Optional contents to use for representation
    
    Returns:
        Formatted string representation
    """
    return tensor_repr(self)

# Patch tensor representations
torch.Tensor.__str__ = _tensor_str
torch.Tensor.__repr__ = _tensor_repr

class TensorReprPlugin:
    """Pytest plugin to format tensor representations."""
    
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item: pytest.Item, call):
        outcome = yield
        report = outcome.get_result()
        
        if report.longrepr:
            # Convert tensor representations in the output
            report.longrepr = str(report.longrepr).replace(
                str(torch.Tensor([])), tensor_repr(torch.Tensor([]))
            )

def pytest_configure(config: Any) -> None:
    """Configure pytest."""
    config.pluginmanager.register(TensorReprPlugin())

    # Register markers
    config.addinivalue_line("markers", "level0: base level tests with no dependencies")
    config.addinivalue_line("markers", "level1: tests depending on level0 components")
    config.addinivalue_line("markers", "level2: tests depending on level1 components")
    config.addinivalue_line("markers", "level3: tests depending on level2 components")
    config.addinivalue_line("markers", "level4: high-level integration tests")
    config.addinivalue_line("markers", "validation: marks validation framework tests")
    
    # Set up logging
    log_level = logging.WARNING
    
    # Configure console output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    # Configure file output
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "test.log"
    
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    # Remove any existing handlers and add new ones
    root_logger = logging.getLogger()
    root_logger.handlers = []  # Remove any existing handlers
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],  # Only use StreamHandler for console output
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
        return formatted  # File formatted successfully
    except (ValueError, Exception) as e:
        pytest.fail(f"Black formatting failed for {file_path}: {e}")
    return content  # Return original content if formatting fails


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
                [str(ruff_path), *command, file_str],
                capture_output=True,
                text=True,
                check=True,
            )
            if result.stderr:
                pytest.fail(f"Ruff {command[0]} failed for {file_path}:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        logger.exception("Failed to run ruff command %s: %s", command, str(e))
        raise


def pytest_collection_modifyitems(items):
    """Modify test items in place to ensure proper test execution order based on dependency analysis."""
    import json
    import os
    from pathlib import Path

    # Load dependency analysis results
    project_root = str(Path(__file__).parent.parent)
    dep_file = os.path.join(project_root, 'dependency_analysis.json')
    
    try:
        with open(dep_file, 'r') as f:
            dep_data = json.load(f)
            dependency_levels = dep_data['dependency_levels']
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        # Fallback to default levels if analysis file not found or invalid
        logger.warning("Dependency analysis file not found or invalid. Using default test ordering.")
        return

    def get_test_level(item):
        """Determine the dependency level for a test item based on analysis results."""
        # Convert test path to module path format
        test_path = str(item.fspath)
        if test_path.startswith(project_root):
            test_path = test_path[len(project_root):].lstrip(os.sep)
        test_path = os.path.splitext(test_path)[0].replace(os.sep, '.')
        
        # Check each level's modules to find where this test belongs
        for level in sorted(dependency_levels.keys(), key=int):
            level_modules = dependency_levels[level]
            # Check if test path matches any module in this level
            if any(test_path.startswith(mod.replace('src.', 'tests.test_')) or 
                  test_path.startswith(mod.replace('src.', 'tests.')) 
                  for mod in level_modules):
                return int(level)
        
        # Default to level 0 if no match found
        return 0

    # Sort items by dependency level
    items.sort(key=get_test_level)

    # Add markers based on levels
    for item in items:
        level = get_test_level(item)
        marker = getattr(pytest.mark, f'level{level}')
        item.add_marker(marker)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add summary info after tests finish."""
    session = terminalreporter._session
    if exitstatus == 0:
        logger.warning("All tests passed successfully")
    else:
        failed = len([i for i in session.items if i.get_closest_marker('failed')])
        logger.warning(f"Test session finished with {session.testscollected - failed} passed, {failed} failed")


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


@pytest.hookimpl
def pytest_collect_file(file_path: Path, parent: Any) -> Optional[pytest.Item]:
    """Process file before collection."""
    if file_path.suffix != ".py":
        return None

    # Only process test files
    if not str(file_path).endswith("_test.py") and not str(file_path).startswith("test_"):
        return None

    # Let pytest handle the actual collection
    return parent.session.collect_file(file_path, parent)


@pytest.fixture(autouse=True)
def setup_logging(request: pytest.FixtureRequest) -> logging.Logger:
    """Set up logging for each test."""
    # Get the test name
    test_name = request.node.name
    
    # Create a test-specific logger
    logger = logging.getLogger(test_name)
    logger.setLevel(logging.DEBUG)
    
    # Add a handler that uses tensor_repr for tensor formatting
    class TensorFormattingHandler(logging.StreamHandler):
        def format(self, record):
            if isinstance(record.msg, torch.Tensor):
                record.msg = tensor_repr(record.msg)
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    tensor_repr(arg) if isinstance(arg, torch.Tensor) else arg
                    for arg in record.args
                )
            return super().format(record)
    
    handler = TensorFormattingHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


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


@pytest.fixture
def test_params() -> dict[str, Any]:
    """Common test parameters."""
    return {
        'diffusion_coefficient': 0.2,  # Increased for faster convergence
        'dt': 0.2,  # Increased for faster convergence while maintaining stability
        'grid_size': 32,
        'batch_size': 1,
        'channels': 1,
        'device': 'cpu',
        'dtype': torch.float32,
        'rtol': 1e-5,
        'atol': 1e-5
    }


@pytest.fixture
def hidden_dim():
    """Hidden dimension for pattern dynamics tests"""
    return 32


@pytest.fixture
def num_patterns():
    """Number of patterns for pattern dynamics tests"""
    return 4


@pytest.fixture
def seq_length():
    """Sequence length for pattern dynamics tests"""
    return 16


@pytest.fixture(scope="session")
def pattern_dynamics(hidden_dim):
    """Create pattern dynamics system for testing."""
    return PatternDynamics(
        grid_size=32,  # Match test_params grid_size
        space_dim=2,
        boundary='periodic',
        dt=0.01,
        num_modes=8,
        hidden_dim=hidden_dim
    )


# Validation fixtures
@pytest.fixture(scope="session")
def flow_validator(setup_test_parameters):
    """Get flow validator for testing."""
    # Create a mock geometric flow for testing
    flow = GeometricFlow(
        hidden_dim=32,  # Match hidden_dim fixture
        manifold_dim=2,
        motive_rank=4,
        num_charts=1,
        integration_steps=10,
        dt=0.1,
        stability_threshold=1e-6
    )
    
    return FlowValidator(
        flow=flow,
        stability_threshold=1e-6,
        curvature_bounds=(-1.0, 1.0),
        max_energy=1e3
    )

@pytest.fixture(scope="session")
def stability_validator(flow_validator):
    """Get flow stability validator for testing."""
    return flow_validator

@pytest.fixture(scope="session")
def energy_validator(flow_validator):
    """Get energy validator for testing."""
    return flow_validator

@pytest.fixture(scope="session")
def convergence_validator(flow_validator):
    """Get convergence validator for testing."""
    return flow_validator

@pytest.fixture(scope="session")
def geometric_flow_validator(flow_validator):
    """Get complete geometric flow validator for testing."""
    return flow_validator

# Pattern validation fixtures
@pytest.fixture(scope="session")
def pattern_validator(pattern_dynamics, flow_validator):
    """Get pattern validator for testing."""
    return PatternValidator(
        linear_validator=flow_validator.linear_validator,
        nonlinear_validator=flow_validator.nonlinear_validator,
        structural_validator=flow_validator.structural_validator,
        lyapunov_threshold=0.1,
        perturbation_threshold=0.1
    )

# Quantum validation fixtures
@pytest.fixture(scope="session")
def quantum_validator():
    """Get quantum validator for testing."""
    return QuantumStateValidator()

# Combined validation framework
@pytest.fixture(scope="session")
def validation_framework(
    geometric_flow_validator,
    pattern_validator,
    quantum_validator
):
    """Get complete validation framework for testing."""
    return ValidationFramework(
        geometric_validator=geometric_flow_validator,
        pattern_validator=pattern_validator,
        quantum_validator=quantum_validator
    )
