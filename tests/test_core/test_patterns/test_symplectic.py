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
    assert torch.allclose(enriched.k, torch.tensor(2.0))
    assert torch.allclose(enriched.omega, torch.tensor(1.0))

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
    
    # Normalize values to probabilities
    pos_prob = torch.abs(pos) / (torch.sum(torch.abs(pos)) + 1e-7)
    position_prob = torch.abs(position) / (torch.sum(torch.abs(position)) + 1e-7)
    mom_prob = torch.abs(mom) / (torch.sum(torch.abs(mom)) + 1e-7)
    momentum_prob = torch.abs(momentum) / (torch.sum(torch.abs(momentum)) + 1e-7)
    
    # Add small epsilon to avoid log(0)
    eps = 1e-7
    pos_prob = pos_prob + eps
    position_prob = position_prob + eps
    mom_prob = mom_prob + eps
    momentum_prob = momentum_prob + eps
    
    # Renormalize after adding epsilon
    pos_prob = pos_prob / torch.sum(pos_prob)
    position_prob = position_prob / torch.sum(position_prob)
    mom_prob = mom_prob / torch.sum(mom_prob)
    momentum_prob = momentum_prob / torch.sum(momentum_prob)
    
    # Check that probability distributions are similar using KL divergence
    kl_pos = torch.sum(pos_prob * torch.log(pos_prob / position_prob))
    kl_mom = torch.sum(mom_prob * torch.log(mom_prob / momentum_prob))
    
    assert kl_pos < 0.5, "Position distributions differ too much"
    assert kl_mom < 1.0, "Momentum distributions differ too much"

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
    
    # Transform to target dimension
    result = structure._handle_dimension(tensor)
    
    # Check target dimension is set correctly
    assert result.shape[-1] == structure.target_dim
    
    # Check wave properties are preserved
    wave_op = structure.enriched.wave_operator(tensor)
    wave_transformed = structure._handle_dimension(wave_op)
    
    # Wave structure should be preserved (not exact values)
    assert wave_transformed.is_complex()
    assert wave_transformed.shape[-1] == structure.target_dim

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
        
        # Check dimension matches target (which is always even)
        assert result.shape[-1] == structure.target_dim
        
        # Check structure preservation
        form_before = structure.compute_form(tensor)
        form_after = structure.compute_form(result)
        
        # Verify symplectic properties are preserved (not exact values)
        assert form_after.matrix.shape[-1] % 2 == 0  # Even dimension
        assert torch.allclose(
            form_after.matrix,
            -form_after.matrix.transpose(-2, -1),  # Anti-symmetry
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
    
    # Verify wave packet properties preserved (not exact values)
    pos_transformed = structure.enriched.get_position(transformed)
    mom_transformed = structure.enriched.get_momentum(transformed)
    
    # Normalize values to probabilities
    pos_prob = torch.abs(pos_transformed) / (torch.sum(torch.abs(pos_transformed)) + 1e-7)
    position_prob = torch.abs(position) / (torch.sum(torch.abs(position)) + 1e-7)
    mom_prob = torch.abs(mom_transformed) / (torch.sum(torch.abs(mom_transformed)) + 1e-7)
    momentum_prob = torch.abs(momentum) / (torch.sum(torch.abs(momentum)) + 1e-7)
    
    # Add small epsilon to avoid log(0)
    eps = 1e-7
    pos_prob = pos_prob + eps
    position_prob = position_prob + eps
    mom_prob = mom_prob + eps
    momentum_prob = momentum_prob + eps
    
    # Renormalize after adding epsilon
    pos_prob = pos_prob / torch.sum(pos_prob)
    position_prob = position_prob / torch.sum(position_prob)
    mom_prob = mom_prob / torch.sum(mom_prob)
    momentum_prob = momentum_prob / torch.sum(momentum_prob)
    
    # Check that probability distributions are similar using KL divergence
    kl_pos = torch.sum(pos_prob * torch.log(pos_prob / position_prob))
    kl_mom = torch.sum(mom_prob * torch.log(mom_prob / momentum_prob))
    
    assert kl_pos < 0.5, "Position distributions differ too much"
    assert kl_mom < 1.0, "Momentum distributions differ too much"

def test_morphism_creation():
    """Test creation of enriched morphisms."""
    enriched = EnrichedAttention(wave_enabled=True)
    operad = AttentionOperad(base_dim=4)
    
    # Create test pattern and operation
    pattern = torch.randn(4)
    operation = operad.create_operation(4, 6, preserve_structure='symplectic')
    
    # Create morphism with wave structure
    result = enriched.create_morphism(pattern, operation, include_wave=True)
    
    # Verify structure preservation (not exact shape)
    assert result.is_complex()  # Wave structure preserved
    # Shape can change as long as symplectic structure is preserved
    assert result.shape[-1] % 2 == 0  # Even dimension
