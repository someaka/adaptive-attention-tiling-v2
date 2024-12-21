"""Tests for the PatternProcessor class."""

import pytest
import torch
import yaml
from pathlib import Path

# Explicitly disable CUDA and other non-CPU/Vulkan backends
if hasattr(torch, '_C'):
    # Disable CUDA
    torch.backends.cuda.is_built = lambda: False
    # Disable other backends
    if hasattr(torch.backends, 'mps'):
        torch.backends.mps.is_built = lambda: False

try:
    import torch_vulkan  # type: ignore
    HAS_VULKAN = torch_vulkan.is_available()
except ImportError:
    HAS_VULKAN = False

# Only allow CPU and Vulkan devices
ALLOWED_DEVICES = ["cpu"]
if HAS_VULKAN:
    ALLOWED_DEVICES.append("vulkan")

from src.core.patterns.pattern_processor import PatternProcessor
from src.core.tiling.patterns.pattern_fiber_bundle import PatternFiberBundle
from src.core.patterns.motivic_integration import MotivicRiemannianStructureImpl
from src.core.quantum.neural_quantum_bridge import NeuralQuantumBridge
from src.core.flow.pattern_heat import PatternHeatFlow


def load_test_config(regimen="debug"):
    """Load test configuration."""
    config_path = Path("configs/test_regimens") / f"{regimen}.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def processor():
    """Create a pattern processor instance."""
    print("Creating pattern processor...")  # Debug output
    device = torch.device("cpu")  # Always start with CPU for consistent testing
    
    # Load debug configuration
    config = load_test_config("debug")
    dims = config["geometric_tests"]["dimensions"]
    num_heads = config["geometric_tests"]["num_heads"]
    
    # Create processor with debug settings
    processor = PatternProcessor(
        manifold_dim=dims,      # From debug config
        hidden_dim=dims * 2,    # Common pattern is 2x manifold dim
        motive_rank=1,          # Minimal rank
        num_primes=1,           # Minimal primes
        num_heads=num_heads,    # From debug config
        dropout=0.0,            # No dropout for deterministic testing
        device=device
    )
    print("Pattern processor created successfully")  # Debug output
    return processor


@pytest.fixture
def test_pattern(processor):
    """Create a test pattern."""
    return torch.randn(10, processor.manifold_dim, device=processor.device)


def test_initialization(processor):
    """Test pattern processor initialization."""
    print("Starting initialization test...")  # Debug output
    config = load_test_config("debug")
    dims = config["geometric_tests"]["dimensions"]
    
    # Basic attribute checks
    assert processor.hidden_dim == dims * 2
    print("Hidden dim check passed")
    assert processor.manifold_dim == dims
    print("Manifold dim check passed")
    assert processor.motive_rank == 1
    print("Motive rank check passed")
    assert processor.num_primes == 1
    print("Num primes check passed")
    assert processor.device == torch.device("cpu")
    print("Device check passed")
    
    # Component checks
    assert isinstance(processor.pattern_bundle, PatternFiberBundle)
    print("Pattern bundle check passed")
    assert isinstance(processor.riemannian, MotivicRiemannianStructureImpl)
    print("Riemannian structure check passed")
    assert isinstance(processor.quantum_bridge, NeuralQuantumBridge)
    print("Quantum bridge check passed")
    assert isinstance(processor.flow, PatternHeatFlow)
    print("Flow check passed")
    
    print("All initialization checks passed successfully")


def test_process_pattern(processor, test_pattern):
    """Test pattern processing."""
    # Test with quantum processing
    result = processor.process_pattern(test_pattern, with_quantum=True, return_intermediates=True)
    assert isinstance(result, tuple)
    evolved, metrics = result
    assert evolved.shape == test_pattern.shape
    assert 'bundle_point' in metrics
    assert 'metric' in metrics
    assert 'quantum_corrections' in metrics
    assert 'evolved' in metrics
    assert 'dynamics' in metrics
    assert metrics['quantum_corrections'] is not None

    # Test without quantum processing
    evolved = processor.process_pattern(test_pattern, with_quantum=False, return_intermediates=False)
    assert evolved.shape == test_pattern.shape


def test_forward(processor, test_pattern):
    """Test forward pass."""
    # Test with intermediates
    result = processor.forward(test_pattern, return_intermediates=True)
    assert isinstance(result, tuple)
    output, intermediates = result
    assert output.shape == test_pattern.shape
    assert isinstance(intermediates, dict)

    # Test without intermediates
    output = processor.forward(test_pattern, return_intermediates=False)
    assert output.shape == test_pattern.shape


def test_device_handling(processor):
    """Test device placement and movement."""
    # Create processor on GPU
    gpu_processor = PatternProcessor(
        manifold_dim=16,
        hidden_dim=32,
        device=torch.device("vulkan")
    )
    
    # Create data on CPU
    data = torch.randn(10, processor.manifold_dim)
    
    # Process should automatically move to GPU
    result = gpu_processor.process_pattern(data, return_intermediates=True)
    evolved, _ = result
    assert evolved.device.type == "vulkan"


def test_error_handling(processor):
    """Test error handling."""
    # Test invalid pattern dimension
    with pytest.raises(ValueError):
        invalid_pattern = torch.randn(10, processor.manifold_dim + 1)
        processor.process_pattern(invalid_pattern)
    
    # Test invalid batch dimension
    with pytest.raises(ValueError):
        invalid_batch = torch.randn(10, 5, processor.manifold_dim)
        processor.process_pattern(invalid_batch)
  