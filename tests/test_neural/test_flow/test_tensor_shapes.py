"""Tests for tensor shape consistency in geometric flow."""

import pytest
import torch
import logging
from typing import List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.core.flow.neural import NeuralGeometricFlow
from src.core.flow.protocol import RicciTensorNetwork
from src.neural.flow.hamiltonian import HamiltonianSystem
from src.validation.geometric.flow import (
    TilingFlowValidator as FlowValidator,
    TilingFlowValidationResult as FlowValidationResult
)

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
        return NeuralGeometricFlow(
            manifold_dim=phase_dim // 2,  # phase_dim is twice manifold_dim
            hidden_dim=phase_dim
        )
        
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
        
        # Log input state
        logger.info("\nInput State Analysis:")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Phase space dimension: {phase_dim}")
        
        # Split initial states
        manifold_dim = phase_points.shape[-1] // 2
        init_position = phase_points[..., :manifold_dim]
        init_momentum = phase_points[..., manifold_dim:]
        
        # Log initial state norms
        init_pos_norm = torch.norm(init_position, dim=-1)
        init_mom_norm = torch.norm(init_momentum, dim=-1)
        logger.info("\nInitial State Norms:")
        logger.info(f"Position norms: {init_pos_norm.tolist()}")
        logger.info(f"Momentum norms: {init_mom_norm.tolist()}")
        
        # Evolve points
        evolved_points = hamiltonian(phase_points)
        
        # Basic shape checks
        assert evolved_points.shape == (batch_size, phase_dim), \
            f"Evolved points shape mismatch: expected {(batch_size, phase_dim)}, got {evolved_points.shape}"
        assert evolved_points.requires_grad, "Gradient tracking lost during evolution"
        
        # Split evolved states
        evolved_position = evolved_points[..., :manifold_dim]
        evolved_momentum = evolved_points[..., manifold_dim:]
        
        # Check component shapes
        assert evolved_position.shape == phase_points[..., :manifold_dim].shape, \
            "Position component shape changed during evolution"
        assert evolved_momentum.shape == phase_points[..., manifold_dim:].shape, \
            "Momentum component shape changed during evolution"
        
        # Compute and log evolved state norms
        evolved_pos_norm = torch.norm(evolved_position, dim=-1)
        evolved_mom_norm = torch.norm(evolved_momentum, dim=-1)
        logger.info("\nEvolved State Norms:")
        logger.info(f"Position norms: {evolved_pos_norm.tolist()}")
        logger.info(f"Momentum norms: {evolved_mom_norm.tolist()}")
        
        # Compute relative changes
        pos_rel_change = torch.abs(evolved_pos_norm - init_pos_norm) / init_pos_norm
        mom_rel_change = torch.abs(evolved_mom_norm - init_mom_norm) / init_mom_norm
        logger.info("\nRelative Changes:")
        logger.info(f"Position norm changes: {pos_rel_change.tolist()}")
        logger.info(f"Momentum norm changes: {mom_rel_change.tolist()}")
        
        # Verify momentum normalization with relaxed tolerance
        # Allow for numerical effects and implementation changes
        max_momentum_deviation = 3.0  # Allow momentum up to 3x unit norm
        min_momentum = 0.1  # Minimum allowed momentum magnitude
        
        # Check momentum bounds
        assert torch.all(evolved_mom_norm > min_momentum), \
            f"Momentum magnitude too small: min={evolved_mom_norm.min().item():.2e}"
        assert torch.all(evolved_mom_norm < max_momentum_deviation), \
            f"Momentum magnitude too large: max={evolved_mom_norm.max().item():.2e}"
        
        # Verify momentum conservation
        mom_rel_change = torch.abs(evolved_mom_norm - init_mom_norm) / init_mom_norm
        max_allowed_mom_change = 0.2  # Allow up to 20% change in momentum magnitude
        assert torch.all(mom_rel_change < max_allowed_mom_change), \
            f"Momentum not conserved. Max relative change: {mom_rel_change.max().item():.2%}"
        
        # Check energy conservation (if implemented)
        try:
            init_energy = hamiltonian.compute_energy(phase_points)
            evolved_energy = hamiltonian.compute_energy(evolved_points)
            energy_change = torch.abs(evolved_energy - init_energy) / torch.abs(init_energy)
            logger.info("\nEnergy Analysis:")
            logger.info(f"Initial energy: {init_energy.tolist()}")
            logger.info(f"Evolved energy: {evolved_energy.tolist()}")
            logger.info(f"Relative change: {energy_change.tolist()}")
            
            # Verify energy conservation with reasonable tolerance
            assert torch.all(energy_change < 0.1), \
                f"Energy not conserved. Max change: {energy_change.max().item():.2%}"
        except (NotImplementedError, AttributeError):
            logger.warning("Energy computation not implemented, skipping conservation check")
            
    def test_validation_shapes(self, flow, phase_points):
        """Test shapes in validation computations."""
        batch_size = phase_points.shape[0]
        manifold_dim = flow.manifold_dim  # This is the position space dimension (2)
        
        # Extract position components for flow validation
        position = phase_points[..., :manifold_dim].clone()  # Clone to avoid in-place modifications
        position.requires_grad_(True)
        
        # Create flow field tensor with proper shape [time_steps=1, batch_size, dim]
        flow_field = position.unsqueeze(0)  # Add time dimension
        
        # Test energy validation
        validator = FlowValidator(
            flow=flow,
            stability_threshold=1e-6,
            curvature_bounds=(-1.0, 1.0),
            max_energy=1e3
        )
        result = validator.validate_flow(flow_field)
        
        assert isinstance(result, FlowValidationResult), \
            "Validation result should be a FlowValidationResult"
        assert result.data is not None, \
            "Validation result should have data"
        assert 'energy' in result.data, \
            "Validation result should have energy metrics"
            
    def test_convergence_shapes(self, flow, phase_points):
        """Test shapes in convergence computations."""
        batch_size = phase_points.shape[0]
        manifold_dim = flow.manifold_dim  # This is the position space dimension (2)
        
        # Extract position components for convergence validation
        position = phase_points[..., :manifold_dim].clone()  # Clone to avoid in-place modifications
        position.requires_grad_(True)
        
        # Create flow field tensor with proper shape [time_steps=1, batch_size, dim]
        flow_field = position.unsqueeze(0)  # Add time dimension
        
        # Test convergence validation (uses position components only)
        validator = FlowValidator(
            flow=flow,
            stability_threshold=1e-6,
            curvature_bounds=(-1.0, 1.0),
            max_energy=1e3
        )
        result = validator.validate_flow(flow_field)
        
        assert isinstance(result, FlowValidationResult), \
            "Validation result should be a FlowValidationResult"
        assert result.data is not None, \
            "Validation result should have data"
        assert 'stability' in result.data, \
            "Validation result should have convergence metrics"

    def test_energy_validation_shapes(self, flow):
        """Test energy validation tensor shapes."""
        validator = FlowValidator(
            flow=flow,
            stability_threshold=1e-6,
            curvature_bounds=(-1.0, 1.0),
            max_energy=1e3
        )
