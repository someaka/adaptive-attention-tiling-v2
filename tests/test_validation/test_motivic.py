"""Tests for motivic structure validation."""

import pytest
import torch
from torch import Tensor, nn
from typing import Tuple, Callable, Optional

from src.core.patterns.motivic_riemannian import (
    MotivicRiemannianStructure,
    MotivicMetricTensor,
    MotivicChristoffelSymbols,
    MotivicCurvatureTensor,
    MotivicCohomology
)
from src.validation.geometric.motivic import (
    MotivicValidator,
    MotivicRiemannianValidator,
    HeightValidation,
    MotivicValidation
)
from src.core.tiling.patterns.cohomology import HeightStructure, ArithmeticDynamics, ArithmeticForm
from src.core.patterns.riemannian_base import MetricTensor, VectorField
from src.validation.framework import ValidationFramework
from src.validation.geometric.model import ModelGeometricValidator
from src.validation.quantum.state import QuantumStateValidator
from src.validation.patterns.stability import PatternValidator as StabilityValidator
from src.core.models.base import ModelGeometry


class MockModelGeometry(ModelGeometry):
    """Mock model geometry for testing."""
    def __init__(self):
        super().__init__(
            manifold_dim=3,
            query_dim=3,
            key_dim=3,
            layers={},
            attention_heads=[]
        )

def safe_int_cast(value):
    """Safely cast value to int."""
    if isinstance(value, (int, float)):
        return int(value)
    elif isinstance(value, torch.Tensor):
        return int(value.item())
    elif isinstance(value, nn.Module):
        if hasattr(value, 'out_features'):
            out_feat = value.out_features
            if isinstance(out_feat, (int, float)):
                return int(out_feat)
            elif isinstance(out_feat, torch.Tensor):
                return int(out_feat.item())
    raise ValueError(f"Cannot safely cast {type(value)} to int")


class MockMotivicRiemannianStructure(MotivicRiemannianStructure):
    """Mock motivic Riemannian structure for testing."""
    
    def __init__(
        self,
        manifold_dim: int = 3,
        hidden_dim: int = 4,
        motive_rank: int = 2,
        num_primes: int = 4
    ):
        super().__init__(
            manifold_dim=manifold_dim,
            hidden_dim=hidden_dim
        )
        self.motive_rank = motive_rank
        self.num_primes = num_primes
        
        # Create fiber map
        self.fiber_map = nn.Linear(manifold_dim, hidden_dim)
        
        # Create connection map
        self.connection_map = nn.Linear(manifold_dim, manifold_dim * manifold_dim)
        
    def compute_metric(self, points: Tensor) -> MotivicMetricTensor:
        """Compute metric tensor with height structure."""
        # Ensure points have batch dimension
        if points.dim() == 1:
            points = points.unsqueeze(0)
            
        batch_size = points.shape[0]
        
        # Compute point norms for metric scaling
        point_norms = torch.norm(points, dim=1, keepdim=True)
        point_norms = torch.clamp(point_norms, min=1e-6)
        
        # Create base metric values that scale with point norms
        identity = torch.eye(self.manifold_dim)
        values = torch.zeros(batch_size, self.manifold_dim, self.manifold_dim)
        
        # Scale metric by point norms to ensure proper behavior
        for i in range(batch_size):
            values[i] = identity * point_norms[i].item()
        
        # Add small constant to ensure positive definiteness
        values = values + 0.1 * identity.unsqueeze(0)
        
        # Create metric tensor with height structure
        height_structure = HeightStructure(num_primes=self.num_primes)
        metric = MotivicMetricTensor(
            values=values,
            dimension=self.manifold_dim,
            height_structure=height_structure
        )
        
        # Special case for single point
        if batch_size == 1:
            metric.height_data = torch.tensor([0.1], device=points.device)  # Start with smallest value
            return metric
            
        # Compute heights directly from point norms
        norms = point_norms.squeeze()
        
        # Sort norms to ensure proper ordering
        sorted_norms, indices = torch.sort(norms)
        sorted_heights = torch.zeros_like(sorted_norms)
        
        # Create strictly increasing sequence from 0.1 to 0.9
        # Use exponential growth to ensure strict monotonicity
        positions = torch.arange(len(sorted_norms), dtype=torch.float32, device=points.device)
        base = 1.5  # Growth factor > 1 ensures strict increase
        exponents = -base * positions / (len(sorted_norms) - 1)
        sorted_heights = 0.1 + 0.8 * (1 - torch.exp(exponents))
        
        # Ensure strict monotonicity with minimum gap
        for i in range(1, len(sorted_heights)):
            min_gap = 0.05  # Minimum gap between consecutive heights
            min_height = sorted_heights[i-1] + min_gap
            sorted_heights[i] = torch.max(sorted_heights[i], min_height)
        
        # Map back to original order
        _, inverse_indices = torch.sort(indices)
        metric.height_data = sorted_heights[inverse_indices]
        
        return metric
    
    def geodesic_flow(
        self,
        initial_point: Tensor,
        initial_velocity: Tensor,
        steps: int = 100,
        step_size: float = 0.01
    ) -> Tuple[Tensor, Tensor]:
        """Mock implementation of geodesic flow."""
        # Simple straight line flow for testing
        t = torch.linspace(0, step_size * steps, steps, device=self.device)
        points = initial_point.unsqueeze(0) + t.unsqueeze(-1) * initial_velocity.unsqueeze(0)
        velocities = initial_velocity.unsqueeze(0).expand(steps, -1)
        return points, velocities
        
    def lie_derivative_metric(
        self,
        point: Tensor,
        vector_field: VectorField
    ) -> MetricTensor[Tensor]:
        """Mock implementation of Lie derivative."""
        # Zero derivative for testing
        metric = self.compute_metric(point.unsqueeze(0))
        return metric
        
    def sectional_curvature(
        self,
        point: Tensor,
        v1: Tensor,
        v2: Tensor
    ) -> float:
        """Mock implementation of sectional curvature."""
        # Constant curvature for testing
        return 0.0
    
    def compute_christoffel(self, points: Tensor) -> MotivicChristoffelSymbols:
        """Mock implementation of Christoffel symbol computation."""
        batch_size = points.shape[0]
        metric = self.compute_metric(points)
        
        # Create Christoffel symbols for constant positive curvature
        values = torch.zeros(
            (batch_size, self.manifold_dim, self.manifold_dim, self.manifold_dim),
            device=self.device
        )
        
        # Compute inverse metric
        metric_inv = torch.inverse(metric.values)
        
        # For a space of constant positive curvature K = 1
        K = 1.0  # Constant positive curvature
        
        # Christoffel symbols for constant positive curvature:
        # Γⁱⱼₖ = K/2 * (δⁱₖgⱼₘ + δⁱⱼgₖₘ - gⱼₖδⁱₘ)gᵐⁱ
        for i in range(self.manifold_dim):
            for j in range(self.manifold_dim):
                for k in range(self.manifold_dim):
                    for m in range(self.manifold_dim):
                        # First term: δⁱₖgⱼₘ
                        if i == k:
                            values[:, i, j, k] += 0.5 * K * metric.values[:, j, m] * metric_inv[:, m, i]
                        # Second term: δⁱⱼgₘ
                        if i == j:
                            values[:, i, j, k] += 0.5 * K * metric.values[:, k, m] * metric_inv[:, m, i]
                        # Third term: -gⱼₖδⁱₘ
                        if i == m:
                            values[:, i, j, k] -= 0.5 * K * metric.values[:, j, k] * metric_inv[:, m, i]
        
        # Create proper ArithmeticDynamics instance
        dynamics = ArithmeticDynamics(
            hidden_dim=self.hidden_dim,
            motive_rank=self.motive_rank,
            num_primes=self.num_primes
        )
        
        return MotivicChristoffelSymbols(
            values=values,
            metric=metric,
            dynamics=dynamics,
            is_symmetric=True
        )

    def compute_curvature(self, points: Tensor, connection: MotivicChristoffelSymbols) -> MotivicCurvatureTensor:
        """Mock implementation of curvature computation."""
        batch_size = points.shape[0]
        
        # For a space of constant positive curvature K = 0.1,
        # the Riemann tensor is:
        # R^i_jkl = K * (δⁱₖg_jl - δⁱₗg_jk)
        riemann = torch.zeros(
            (batch_size, self.manifold_dim, self.manifold_dim, self.manifold_dim, self.manifold_dim),
            device=self.device
        )
        
        K = 0.1  # Reduced constant positive curvature
        metric = connection.metric.values
        
        # Compute Riemann tensor for constant positive curvature
        for i in range(self.manifold_dim):
            for j in range(self.manifold_dim):
                for k in range(self.manifold_dim):
                    for l in range(self.manifold_dim):
                        if i == k:
                            riemann[:, i, j, k, l] += K * metric[:, j, l]
                        if i == l:
                            riemann[:, i, j, k, l] -= K * metric[:, j, k]
        
        # Compute Ricci tensor: Rⱼₖ = Rⁱⱼᵢₖ
        ricci = torch.einsum('bijil->bjl', riemann)
        
        # Compute scalar curvature: R = gⁱʲRᵢⱼ
        metric_inv = torch.inverse(metric)
        scalar = torch.einsum('bij,bij->b', metric_inv, ricci)
        
        # Ensure positive scalar curvature
        scalar = K * self.manifold_dim * (self.manifold_dim - 1) * torch.ones_like(scalar)
        
        # Create proper MotivicCohomology instance
        motive = MotivicCohomology(
            base_space=self,
            hidden_dim=self.hidden_dim,
            motive_rank=self.motive_rank,
            num_primes=self.num_primes
        )
        
        # Create curvature tensor
        curvature = MotivicCurvatureTensor(
            riemann=riemann,
            ricci=ricci,
            scalar_curvatures=scalar,
            motive=motive
        )
        
        # Compute cohomology class from curvature data
        # Project Riemann tensor to match cohomology dimension
        riemann_flat = riemann.reshape(batch_size, -1)  # [batch_size, *]
        
        # Normalize the flattened tensor
        riemann_norm = torch.norm(riemann_flat, dim=1, keepdim=True)
        riemann_normalized = riemann_flat / (riemann_norm + 1e-8)
        
        # Create arithmetic form with normalized coefficients
        form = ArithmeticForm(
            degree=2,  # Curvature is a 2-form
            coefficients=riemann_normalized,
            num_primes=self.num_primes
        )
        
        # Compute cohomology class with proper dimension
        cohomology_class = motive.compute_motive(form)
        
        # Ensure cohomology class has proper shape and matches batch size
        if cohomology_class.dim() == 1:
            cohomology_class = cohomology_class.unsqueeze(0)
        if cohomology_class.shape[0] != batch_size:
            cohomology_class = cohomology_class.expand(batch_size, -1)
            
        # Set cohomology class with proper shape
        curvature.cohomology_class = cohomology_class
        
        return curvature


@pytest.fixture
def motivic_structure():
    """Create a motivic Riemannian structure for testing."""
    return MockMotivicRiemannianStructure(
        manifold_dim=3,
        hidden_dim=4,
        motive_rank=2,
        num_primes=4
    )


@pytest.fixture
def points():
    """Create test points."""
    points = torch.randn(10, 3)
    points.requires_grad_(True)  # Enable gradients
    return points


@pytest.fixture
def validator(motivic_structure):
    """Create a motivic Riemannian validator."""
    return MotivicRiemannianValidator(motivic_structure)


@pytest.fixture
def mock_model_geometry():
    """Create mock model geometry for testing."""
    return MockModelGeometry()

@pytest.fixture
def geometric_validator(mock_model_geometry):
    """Create geometric validator."""
    return ModelGeometricValidator(
        model_geometry=mock_model_geometry,
        tolerance=1e-6,
        curvature_bounds=(-1.0, 1.0)
    )

@pytest.fixture
def quantum_validator():
    """Create quantum validator."""
    return QuantumStateValidator()

@pytest.fixture
def pattern_validator():
    """Create pattern validator."""
    return StabilityValidator(
        linear_validator=None,  # Mock implementation doesn't need these
        nonlinear_validator=None,
        lyapunov_threshold=1e-6,
        perturbation_threshold=1e-6
    )

@pytest.fixture
def validation_framework(geometric_validator, quantum_validator, pattern_validator):
    """Create validation framework."""
    return ValidationFramework(
        geometric_validator=geometric_validator,
        quantum_validator=quantum_validator,
        pattern_validator=pattern_validator
    )


class TestHeightValidation:
    """Test suite specifically for height validation."""
    
    def test_basic_height_properties(self, motivic_structure, points):
        """Test basic height function properties."""
        validator = MotivicValidator()
        metric = motivic_structure.compute_metric(points)
        
        result = validator.validate_height(metric)
        assert isinstance(result, HeightValidation)
        assert result.is_valid
        assert result.local_heights_valid
        assert result.global_height_valid
        assert result.northcott_property
        assert result.height_data is not None, "Height data should not be None"
        assert torch.is_tensor(result.height_data)
        
        # Check specific height properties
        height_data = result.height_data
        assert height_data is not None, "Height data should not be None"
        assert torch.all(height_data >= 0), "Heights must be non-negative"
        assert torch.all(torch.isfinite(height_data)), "Heights must be finite"
    
    def test_strictly_increasing_heights(self, motivic_structure):
        """Test that heights are strictly increasing."""
        validator = MotivicValidator()
        
        # Create sequence of points with increasing norms
        points = torch.randn(5, 3)
        points = points * torch.arange(1, 6).view(-1, 1)
        points.requires_grad_(True)
        
        metric = motivic_structure.compute_metric(points)
        result = validator.validate_height(metric)
        
        assert result.is_valid
        assert result.height_data is not None, "Height data should not be None"
        height_data = result.height_data  # Store in local variable after null check
        height_diffs = height_data[1:] - height_data[:-1]  # Use local variable
        assert torch.all(height_diffs > 0), "Heights must be strictly increasing"
    
    def test_height_edge_cases(self, motivic_structure):
        """Test height validation with edge cases."""
        validator = MotivicValidator()
        
        # Test single point
        single_point = torch.randn(1, 3)
        single_point.requires_grad_(True)
        metric = motivic_structure.compute_metric(single_point)
        result = validator.validate_height(metric)
        assert result.is_valid
        
        # Test zero point
        zero_point = torch.zeros(1, 3)
        zero_point.requires_grad_(True)
        metric = motivic_structure.compute_metric(zero_point)
        result = validator.validate_height(metric)
        assert result.is_valid
        assert result.height_data is not None, "Height data should not be None"
        height_data = result.height_data  # Store in local variable after null check
        assert height_data[0] > 0, "Height of zero point should be positive"  # Use local variable
    
    def test_invalid_height_cases(self):
        """Test cases where height validation should fail."""
        validator = MotivicValidator()
        
        # Test negative height
        invalid_height = torch.tensor([-1.0, 0.0, 1.0])
        height_structure = HeightStructure(num_primes=4)
        metric = MotivicMetricTensor(
            values=torch.eye(3),
            dimension=3,
            height_structure=height_structure
        )
        metric.height_data = invalid_height
        result = validator.validate_height(metric)
        assert not result.is_valid
        assert not result.local_heights_valid
        
        # Test infinite height
        inf_height = torch.tensor([float('inf'), 1.0, 2.0])
        metric.height_data = inf_height
        result = validator.validate_height(metric)
        assert not result.is_valid
        assert not result.local_heights_valid

class TestDynamicsValidation:
    """Test suite specifically for dynamics validation."""
    
    def test_basic_dynamics_properties(self, motivic_structure, points):
        """Test basic arithmetic dynamics properties."""
        validator = MotivicValidator()
        connection = motivic_structure.compute_christoffel(points)
        
        result = validator.validate_dynamics(connection)
        assert result.is_valid
        assert "passed" in result.message
        assert connection.dynamics_state is not None
    
    def test_dynamics_consistency(self, motivic_structure, points):
        """Test consistency of arithmetic dynamics."""
        validator = MotivicValidator()
        connection = motivic_structure.compute_christoffel(points)
        
        # Check that dynamics preserves structure
        dynamics = connection.dynamics
        assert isinstance(dynamics, ArithmeticDynamics)
        assert dynamics.hidden_dim == motivic_structure.hidden_dim
        assert dynamics.motive_rank == motivic_structure.motive_rank
        assert dynamics.num_primes == motivic_structure.num_primes
    
    def test_dynamics_edge_cases(self, motivic_structure):
        """Test dynamics validation with edge cases."""
        validator = MotivicValidator()
        
        # Test single point
        single_point = torch.randn(1, 3)
        single_point.requires_grad_(True)
        connection = motivic_structure.compute_christoffel(single_point)
        result = validator.validate_dynamics(connection)
        assert result.is_valid
        
        # Test zero connection
        zero_point = torch.zeros(1, 3)
        zero_point.requires_grad_(True)
        connection = motivic_structure.compute_christoffel(zero_point)
        result = validator.validate_dynamics(connection)
        assert result.is_valid
    
    def test_invalid_dynamics_cases(self, motivic_structure, points):
        """Test cases where dynamics validation should fail."""
        validator = MotivicValidator()
        connection = motivic_structure.compute_christoffel(points)
        
        # Test with no dynamics state
        connection.dynamics_state = None
        result = validator.validate_dynamics(connection)
        assert not result.is_valid
        assert "No dynamics state" in result.message
        
        # Test with invalid dynamics parameters
        connection.dynamics = ArithmeticDynamics(
            hidden_dim=5,  # Mismatched dimension
            motive_rank=2,
            num_primes=4
        )
        result = validator.validate_dynamics(connection)
        assert not result.is_valid

class TestCohomologyValidation:
    """Test suite specifically for cohomology validation."""
    
    def test_basic_cohomology_properties(self, motivic_structure, points):
        """Test basic cohomology properties."""
        validator = MotivicValidator()
        metric = motivic_structure.compute_metric(points)
        connection = motivic_structure.compute_christoffel(points)
        curvature = motivic_structure.compute_curvature(points, connection)
        
        result = validator.validate_cohomology(curvature)
        assert result.bounds_satisfied
        assert torch.is_tensor(result.error_bounds)
        assert torch.is_tensor(result.scalar_curvatures)
        
        # Check specific curvature properties
        assert torch.all(torch.isfinite(curvature.riemann))
        assert torch.all(torch.isfinite(curvature.ricci))
        assert torch.all(torch.isfinite(curvature.scalar_curvatures))
    
    def test_curvature_bounds(self, motivic_structure, points):
        """Test that curvature satisfies required bounds."""
        validator = MotivicValidator()
        metric = motivic_structure.compute_metric(points)
        connection = motivic_structure.compute_christoffel(points)
        curvature = motivic_structure.compute_curvature(points, connection)
        
        result = validator.validate_cohomology(curvature)
        assert result.bounds_satisfied
        
        # Check positive scalar curvature
        assert torch.all(curvature.scalar_curvatures > 0), "Scalar curvature should be positive"
        
        # Check error bounds are finite
        assert torch.all(torch.isfinite(result.error_bounds))
        assert torch.all(result.error_bounds >= 0)
    
    def test_cohomology_edge_cases(self, motivic_structure):
        """Test cohomology validation with edge cases."""
        validator = MotivicValidator()
        
        # Test single point
        single_point = torch.randn(1, 3)
        single_point.requires_grad_(True)
        metric = motivic_structure.compute_metric(single_point)
        connection = motivic_structure.compute_christoffel(single_point)
        curvature = motivic_structure.compute_curvature(single_point, connection)
        result = validator.validate_cohomology(curvature)
        assert result.bounds_satisfied
        
        # Test flat space (zero curvature)
        flat_point = torch.zeros(1, 3)
        flat_point.requires_grad_(True)
        metric = motivic_structure.compute_metric(flat_point)
        connection = motivic_structure.compute_christoffel(flat_point)
        curvature = motivic_structure.compute_curvature(flat_point, connection)
        result = validator.validate_cohomology(curvature)
        assert result.bounds_satisfied
    
    def test_invalid_cohomology_cases(self, motivic_structure, points):
        """Test cases where cohomology validation should fail."""
        validator = MotivicValidator()
        metric = motivic_structure.compute_metric(points)
        connection = motivic_structure.compute_christoffel(points)
        curvature = motivic_structure.compute_curvature(points, connection)
        
        # Test with missing cohomology class
        if hasattr(curvature, 'cohomology_class'):
            delattr(curvature, 'cohomology_class')
        result = validator.validate_cohomology(curvature)
        assert not result.bounds_satisfied
        
        # Test with infinite curvature
        curvature.riemann = torch.full_like(curvature.riemann, float('inf'))
        result = validator.validate_cohomology(curvature)
        assert not result.bounds_satisfied


class TestMotivicRiemannianValidator:
    """Test suite for MotivicRiemannianValidator."""

    def test_full_validation(self, validator, points):
        """Test complete validation of motivic Riemannian structure."""
        result = validator.validate(points)
        
        assert isinstance(result, MotivicValidation)
        assert result.is_valid
        assert result.height_valid
        assert result.dynamics_valid
        assert result.cohomology_valid
        assert "passed" in result.message
        
        # Check data contents
        assert "height" in result.data
        assert "dynamics" in result.data
        assert "cohomology" in result.data

    def test_validation_with_perturbation(self, validator, points):
        """Test validation with perturbed metric."""
        # Add noise to points
        noisy_points = points + torch.randn_like(points) * 0.1
        
        result = validator.validate(noisy_points)
        assert isinstance(result, MotivicValidation)
        # Even with noise, basic properties should hold
        assert result.height_valid
        assert result.dynamics_valid

    def test_boundary_cases(self, validator):
        """Test validation with boundary cases."""
        # Test with single point
        single_point = torch.randn(1, 3)
        single_point.requires_grad_(True)
        result = validator.validate(single_point)
        assert result.is_valid
        
        # Test with zero points
        zero_points = torch.zeros(10, 3)
        zero_points.requires_grad_(True)
        result = validator.validate(zero_points)
        assert result.is_valid
        
        # Test with large values
        large_points = torch.randn(5, 3) * 1e3
        large_points.requires_grad_(True)
        result = validator.validate(large_points)
        assert result.is_valid

    @pytest.mark.parametrize("manifold_dim,hidden_dim", [
        (2, 3),
        (4, 6),
        (5, 8)
    ])
    def test_different_dimensions(self, manifold_dim, hidden_dim):
        """Test validation with different manifold dimensions."""
        structure = MockMotivicRiemannianStructure(
            manifold_dim=manifold_dim,
            hidden_dim=hidden_dim,
            motive_rank=2,
            num_primes=4
        )
        validator = MotivicRiemannianValidator(structure)
        
        points = torch.randn(5, manifold_dim)
        points.requires_grad_(True)
        result = validator.validate(points)
        
        assert result.is_valid
        assert result.height_valid
        assert result.dynamics_valid
        assert result.cohomology_valid
  