import pytest
import torch
from src.core.patterns.operadic_structure import EnrichedAttention, OperadicOperation

@pytest.fixture
def enriched_attention():
    """Create an EnrichedAttention instance for testing."""
    return EnrichedAttention(
        base_category="SymplecticVect",
        wave_enabled=True,
        _k=2.0,
        _omega=1.0,
        dtype=torch.float32
    )

@pytest.fixture
def tensor():
    """Create a test tensor."""
    return torch.randn(4, 8, 16)

@pytest.fixture
def operadic_operation():
    """Create a test OperadicOperation."""
    return OperadicOperation(
        source_dim=16,
        target_dim=16,
        composition_law=torch.eye(16),
        enrichment={'preserve_symplectic': True, 'preserve_metric': True},
        natural_transformation=None
    )

def test_wave_operator(enriched_attention, tensor):
    """Test wave operator functionality."""
    # Test with wave enabled
    result = enriched_attention.wave_operator(tensor)
    assert result.shape == tensor.shape
    assert torch.is_complex(result)

    # Test with wave disabled
    enriched_attention.wave_enabled = False
    result = enriched_attention.wave_operator(tensor)
    assert result is tensor

def test_create_wave_packet(enriched_attention):
    """Test wave packet creation."""
    position = torch.randn(4, 8, 16)
    momentum = torch.randn(4, 8, 16)

    # Test with wave enabled
    result = enriched_attention.create_wave_packet(position, momentum)
    assert result.shape == position.shape
    assert torch.is_complex(result)

    # Test with wave disabled
    enriched_attention.wave_enabled = False
    result = enriched_attention.create_wave_packet(position, momentum)
    assert result is position

def test_get_position(enriched_attention):
    """Test position extraction from wave packet."""
    wave = torch.randn(4, 8, 16, dtype=torch.complex64)
    
    # Test with wave enabled
    result = enriched_attention.get_position(wave)
    assert result.shape == wave.shape  # Position should maintain same shape for complex wave packets
    assert not torch.is_complex(result)
    assert result.dtype == enriched_attention.dtype

    # Test with wave disabled
    enriched_attention.wave_enabled = False
    real_tensor = torch.randn(4, 8, 16)
    result = enriched_attention.get_position(real_tensor)
    assert result is real_tensor

def test_get_momentum(enriched_attention):
    """Test momentum extraction from wave packet."""
    wave = torch.randn(4, 8, 16, dtype=torch.complex64)
    
    # Test with wave enabled
    result = enriched_attention.get_momentum(wave)
    assert result.shape == wave.shape[:-1] + (16,)
    assert not torch.is_complex(result)
    assert result.dtype == enriched_attention.dtype

    # Test with wave disabled
    enriched_attention.wave_enabled = False
    real_tensor = torch.randn(4, 8, 16)
    result = enriched_attention.get_momentum(real_tensor)
    assert result is real_tensor

def test_create_morphism(enriched_attention, tensor, operadic_operation):
    """Test enriched morphism creation."""
    # Test with wave enabled
    result = enriched_attention.create_morphism(tensor, operadic_operation)
    assert result.shape == tensor.shape
    assert torch.is_complex(result)
    assert result.dtype == torch.complex64

    # Test without wave
    result = enriched_attention.create_morphism(tensor, operadic_operation, include_wave=False)
    assert result.shape == tensor.shape
    assert not torch.is_complex(result)
    assert result.dtype == enriched_attention.dtype

    # Test with wave disabled
    enriched_attention.wave_enabled = False
    result = enriched_attention.create_morphism(tensor, operadic_operation)
    assert result.shape == tensor.shape
    assert not torch.is_complex(result)
    assert result.dtype == enriched_attention.dtype

def test_base_category(enriched_attention):
    """Test base category attribute."""
    assert enriched_attention.base_category == "SymplecticVect"
    
    # Test category change
    enriched_attention.base_category = "RiemannianVect"
    assert enriched_attention.base_category == "RiemannianVect"

def test_wave_parameters(enriched_attention):
    """Test wave parameter attributes."""
    assert enriched_attention._k == 2.0
    assert enriched_attention._omega == 1.0
    assert enriched_attention.dtype == torch.float32
    
    # Test parameter changes
    enriched_attention._k = 3.0
    enriched_attention._omega = 2.0
    enriched_attention.dtype = torch.float64
    assert enriched_attention._k == 3.0
    assert enriched_attention._omega == 2.0
    assert enriched_attention.dtype == torch.float64