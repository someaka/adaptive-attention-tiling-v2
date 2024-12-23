import torch
import pytest
from src.core.patterns.symplectic import SymplecticStructure, SymplecticForm
from src.core.patterns.operadic_structure import AttentionOperad, EnrichedAttention

def test_enriched_attention_initialization():
    """Test initialization of enriched attention structure."""
    enriched = EnrichedAttention(
        base_category="SymplecticVect",
        wave_enabled=True,
        _k=2.0,
        _omega=1.0
    )
    
    assert enriched.base_category == "SymplecticVect"
    assert enriched.wave_enabled is True
    assert enriched._k == 2.0
    assert enriched._omega == 1.0

def test_wave_operator_functionality():
    """Test wave operator methods."""
    enriched = EnrichedAttention(wave_enabled=True)
    
    # Test wave operator
    tensor = torch.randn(4)
    wave = enriched.wave_operator(tensor)
    assert wave.is_complex()
    
    # Test wave packet creation
    position = torch.randn(4)
    momentum = torch.randn(4)
    packet = enriched.create_wave_packet(position, momentum)
    assert packet.is_complex()
    
    # Test position and momentum extraction
    pos = enriched.get_position(packet)
    mom = enriched.get_momentum(packet)
    assert pos.shape == position.shape
    assert mom.shape == momentum.shape

def test_dimension_handling_with_wave():
    """Test dimension handling with wave emergence enabled."""
    # Initialize structure with wave emergence
    structure = SymplecticStructure(
        dim=3,  # Odd dimension to test transition
        preserve_structure=True,
        wave_enabled=True
    )
    
    # Create test tensor
    tensor = torch.randn(10, 3)  # Batch of 10 vectors in R^3
    
    # Transform to even dimension
    result = structure._handle_dimension(tensor)
    
    # Check dimension is even
    assert result.shape[-1] % 2 == 0
    
    # Check wave properties are preserved
    wave_op = structure.enriched.wave_operator(tensor)
    wave_transformed = structure._handle_dimension(wave_op)
    
    # Wave properties should be preserved
    assert torch.allclose(
        structure.enriched.wave_operator(result),
        wave_transformed,
        rtol=1e-5
    )

def test_quantum_geometric_tensor():
    """Test quantum geometric tensor computation."""
    structure = SymplecticStructure(
        dim=4,  # Even dimension
        preserve_structure=True,
        wave_enabled=True
    )
    
    # Create test point
    point = torch.randn(4)
    
    # Compute metric and symplectic components
    metric = structure.compute_metric(point)
    symplectic = structure.compute_form(point).matrix
    
    # Check properties of quantum geometric tensor
    quantum_tensor = metric + 1j * symplectic
    
    # Verify hermiticity
    assert torch.allclose(
        quantum_tensor,
        quantum_tensor.conj().transpose(-2, -1),
        rtol=1e-5
    )
    
    # Verify metric is positive definite
    eigenvals = torch.linalg.eigvalsh(metric)
    assert (eigenvals > 0).all()
    
    # Verify symplectic form is non-degenerate
    pfaffian = torch.linalg.det(symplectic) ** 0.5
    assert abs(pfaffian) > 1e-5

def test_dimension_transition_consistency():
    """Test consistency of dimension transitions."""
    structure = SymplecticStructure(
        dim=3,
        preserve_structure=True,
        wave_enabled=True
    )
    
    # Create test tensors of different dimensions
    tensors = [
        torch.randn(10, d) for d in [2, 3, 4, 5]
    ]
    
    for tensor in tensors:
        result = structure._handle_dimension(tensor)
        
        # Check dimension is even
        assert result.shape[-1] % 2 == 0
        
        # Check structure preservation
        form_before = structure.compute_form(tensor)
        form_after = structure.compute_form(result)
        
        # Verify symplectic properties are preserved
        assert torch.allclose(
            form_before.evaluate(tensor[0], tensor[0]),
            form_after.evaluate(result[0], result[0]),
            rtol=1e-5
        )

def test_wave_symplectic_interaction():
    """Test interaction between wave emergence and symplectic structure."""
    structure = SymplecticStructure(
        dim=4,
        preserve_structure=True,
        wave_enabled=True
    )
    
    # Create test wave packet
    position = torch.randn(4)
    momentum = torch.randn(4)
    wave = structure.enriched.create_wave_packet(position, momentum)
    
    # Transform through dimension change
    transformed = structure._handle_dimension(wave)
    
    # Verify wave packet properties preserved
    assert torch.allclose(
        structure.enriched.get_position(transformed),
        structure._handle_dimension(position),
        rtol=1e-5
    )
    assert torch.allclose(
        structure.enriched.get_momentum(transformed),
        structure._handle_dimension(momentum),
        rtol=1e-5
    )

def test_morphism_creation():
    """Test creation of enriched morphisms."""
    enriched = EnrichedAttention(wave_enabled=True)
    operad = AttentionOperad(base_dim=4)
    
    # Create test pattern and operation
    pattern = torch.randn(4)
    operation = operad.create_operation(4, 6, preserve_structure='symplectic')
    
    # Create morphism with wave structure
    result = enriched.create_morphism(pattern, operation, include_wave=True)
    
    # Verify shape
    assert result.shape[-1] == 6
    
    # Verify wave structure is included
    assert result.is_complex()

# ... existing code ... 