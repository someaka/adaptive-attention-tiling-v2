"""Tests for tensor shape consistency in geometric flow."""

import pytest
import torch

from src.neural.flow.geometric_flow import GeometricFlow, RicciTensorNetwork
from src.neural.flow.hamiltonian import HamiltonianSystem
from src.validation.geometric.flow import FlowValidator, FlowValidationResult

class TestTensorShapes:
    """Test tensor shapes in geometric flow."""
    
    @pytest.fixture
    def batch_size(self):
        """Batch size for tests."""
        return 8
        
    @pytest.fixture
    def phase_dim(self):
        """Phase space dimension."""
        return 4  # 4D phase space (2D position + 2D momentum)
        
    @pytest.fixture
    def flow(self, phase_dim):
        """Create geometric flow instance."""
        return GeometricFlow(phase_dim=phase_dim)
        
    @pytest.fixture
    def ricci_network(self, phase_dim):
        """Create Ricci tensor network."""
        manifold_dim = phase_dim // 2
        return RicciTensorNetwork(manifold_dim=manifold_dim)
        
    @pytest.fixture
    def hamiltonian(self, phase_dim):
        """Create Hamiltonian system."""
        return HamiltonianSystem(manifold_dim=phase_dim)
        
    @pytest.fixture
    def phase_points(self, batch_size, phase_dim):
        """Create test points in phase space."""
        # Create points with gradients enabled
        return torch.randn(batch_size, phase_dim, requires_grad=True)
        
    def test_geometric_flow_shapes(self, flow, phase_points, ricci_network):
        """Test shapes in geometric flow computations."""
        batch_size = phase_points.shape[0]
        manifold_dim = flow.manifold_dim  # This is the position space dimension (2)
        
        # Extract position components from phase space points
        position = phase_points[..., :manifold_dim].clone()  # Clone to avoid in-place modifications
        position.requires_grad_(True)
        
        # Set points on flow before computing metrics
        flow.points = position
        
        # Test metric tensor computation
        metric = flow.compute_metric(position)  # Should use position components only
        assert metric.shape == (batch_size, manifold_dim, manifold_dim), \
            f"Metric tensor shape mismatch: expected {(batch_size, manifold_dim, manifold_dim)}, got {metric.shape}"
        
        # Test Ricci tensor computation using the network
        ricci_tensor = ricci_network(position)
        assert isinstance(ricci_tensor, torch.Tensor), "Ricci tensor should be a torch.Tensor"
        assert ricci_tensor.shape == (batch_size, manifold_dim, manifold_dim), \
            f"Ricci tensor shape mismatch: expected {(batch_size, manifold_dim, manifold_dim)}, got {ricci_tensor.shape}"
        
        # Test flow computation directly with tensor
        flow_vector = flow.compute_flow(position, ricci_tensor)  # Should use position components
        assert flow_vector.shape == (batch_size, manifold_dim), \
            f"Flow vector shape mismatch: expected {(batch_size, manifold_dim)}, got {flow_vector.shape}"
        
    def test_hamiltonian_shapes(self, hamiltonian, phase_points):
        """Test shapes in Hamiltonian system computations."""
        batch_size = phase_points.shape[0]
        phase_dim = phase_points.shape[1]
        manifold_dim = phase_dim // 2
        
        # Ensure points require grad
        if not phase_points.requires_grad:
            phase_points.requires_grad_(True)
        
        # Test phase space conversion
        phase_point = hamiltonian._to_phase_space(phase_points)
        assert phase_point.position.shape == (batch_size, manifold_dim), \
            f"Position shape mismatch: expected {(batch_size, manifold_dim)}, got {phase_point.position.shape}"
        assert phase_point.momentum.shape == (batch_size, manifold_dim), \
            f"Momentum shape mismatch: expected {(batch_size, manifold_dim)}, got {phase_point.momentum.shape}"
            
        # Test energy computation
        energy = hamiltonian.compute_energy(phase_points)
        assert energy.shape == (batch_size,), \
            f"Energy shape mismatch: expected {(batch_size,)}, got {energy.shape}"
            
        # Test evolution
        evolved = hamiltonian.evolve(phase_points)
        assert evolved.shape == phase_points.shape, \
            f"Evolved states shape mismatch: expected {phase_points.shape}, got {evolved.shape}"
            
    def test_hamiltonian_evolution_shapes(self, hamiltonian, phase_points):
        """Test shapes in Hamiltonian evolution computations."""
        batch_size = phase_points.shape[0]
        phase_dim = phase_points.shape[1]
        
        # Evolve points
        evolved_points = hamiltonian(phase_points)
        assert evolved_points.shape == (batch_size, phase_dim), \
            f"Evolved points shape mismatch: expected {(batch_size, phase_dim)}, got {evolved_points.shape}"
        
        # Check gradients are preserved
        assert evolved_points.requires_grad
        
        # Split evolved states
        manifold_dim = phase_points.shape[-1] // 2
        evolved_position = evolved_points[..., :manifold_dim]
        evolved_momentum = evolved_points[..., manifold_dim:]
        
        # Check component shapes
        assert evolved_position.shape == phase_points[..., :manifold_dim].shape
        assert evolved_momentum.shape == phase_points[..., manifold_dim:].shape
        
        # Verify momentum normalization
        momentum_norms = torch.norm(evolved_momentum, dim=-1)
        assert torch.allclose(momentum_norms, torch.ones_like(momentum_norms), atol=1e-6)
            
    def test_validation_shapes(self, flow, phase_points):
        """Test shapes in validation computations."""
        batch_size = phase_points.shape[0]
        manifold_dim = flow.manifold_dim  # This is the position space dimension (2)
        
        # Test energy validation (uses full phase space)
        validator = FlowValidator(
            energy_threshold=1e-6,
            monotonicity_threshold=1e-4,
            singularity_threshold=1.0,
            max_iterations=1000,
            tolerance=1e-6
        )
        result = validator.validate(phase_points)
        
        assert isinstance(result, FlowValidationResult), \
            "Validation result should be a FlowValidationResult"
        assert result.data is not None, \
            "Validation result should have data"
        assert 'energy_metrics' in result.data, \
            "Validation result should have energy metrics"
        
        energy_metrics = result.data['energy_metrics']
        assert isinstance(energy_metrics.initial_energy, float), \
            "Initial energy should be a float"
        assert isinstance(energy_metrics.final_energy, float), \
            "Final energy should be a float"
            
    def test_convergence_shapes(self, flow, phase_points):
        """Test shapes in convergence computations."""
        batch_size = phase_points.shape[0]
        manifold_dim = flow.manifold_dim  # This is the position space dimension (2)
        
        # Extract position components for convergence validation
        position = phase_points[..., :manifold_dim].clone()  # Clone to avoid in-place modifications
        position.requires_grad_(True)
        
        # Test convergence validation (uses position components only)
        validator = FlowValidator(
            energy_threshold=1e-6,
            monotonicity_threshold=1e-4,
            singularity_threshold=1.0,
            max_iterations=1000,
            tolerance=1e-6
        )
        result = validator.validate(position)
        
        assert isinstance(result, FlowValidationResult), \
            "Validation result should be a FlowValidationResult"
        assert result.data is not None, \
            "Validation result should have data"
        assert 'convergence_metrics' in result.data, \
            "Validation result should have convergence metrics"

    def test_energy_validation_shapes(self):
        """Test energy validation tensor shapes."""
        validator = FlowValidator(
            energy_threshold=1e-6,
            monotonicity_threshold=1e-4,
            singularity_threshold=1.0,
            max_iterations=1000,
            tolerance=1e-6
        )
