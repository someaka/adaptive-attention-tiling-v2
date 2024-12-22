import pytest
import torch
from src.core.patterns.symplectic import SymplecticStructure

def test_dimension_transition_consistency():
    """Test consistency of dimension transitions."""
    structure = SymplecticStructure(
        dim=3,
        preserve_structure=True,
        wave_enabled=True,
        dtype=torch.float32
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
        wave_enabled=True,
        dtype=torch.float32
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

def test_quantum_geometric_tensor():
    """Test quantum geometric tensor computation."""
    structure = SymplecticStructure(
        dim=4,
        preserve_structure=True,
        wave_enabled=True,
        dtype=torch.float32
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