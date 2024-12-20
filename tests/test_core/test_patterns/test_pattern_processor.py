"""Tests for the PatternProcessor class."""

import pytest
import torch

from src.core.patterns.pattern_processor import PatternProcessor
from src.core.tiling.patterns.pattern_fiber_bundle import PatternFiberBundle
from src.core.patterns.motivic_integration import MotivicRiemannianStructureImpl
from src.core.quantum.neural_quantum_bridge import NeuralQuantumBridge
from src.core.flow.pattern_heat import PatternHeatFlow


@pytest.fixture
def processor():
    """Create a pattern processor instance."""
    return PatternProcessor(
        manifold_dim=16,
        hidden_dim=32,
        motive_rank=4,
        num_primes=8,
        device=torch.device("cpu")
    )


@pytest.fixture
def test_pattern(processor):
    """Create a test pattern."""
    return torch.randn(10, processor.manifold_dim, device=processor.device)


def test_initialization(processor):
    """Test pattern processor initialization."""
    assert processor.hidden_dim == 32
    assert processor.manifold_dim == 16
    assert processor.motive_rank == 4
    assert processor.num_primes == 8
    assert processor.device == torch.device("cpu")
    assert isinstance(processor.pattern_bundle, PatternFiberBundle)
    assert isinstance(processor.riemannian, MotivicRiemannianStructureImpl)
    assert isinstance(processor.quantum_bridge, NeuralQuantumBridge)
    assert isinstance(processor.flow, PatternHeatFlow)


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
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    # Create processor on GPU
    gpu_processor = PatternProcessor(
        manifold_dim=16,
        hidden_dim=32,
        device=torch.device("cuda")
    )
    
    # Create data on CPU
    data = torch.randn(10, processor.manifold_dim)
    
    # Process should automatically move to GPU
    result = gpu_processor.process_pattern(data, return_intermediates=True)
    evolved, _ = result
    assert evolved.device.type == "cuda"


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
  