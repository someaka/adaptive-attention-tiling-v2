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
from src.core.patterns.cohomology import (
    HeightStructure,
    ArithmeticDynamics,
    ArithmeticForm
)
from src.core.patterns.riemannian_base import MetricTensor, VectorField
from src.validation.framework import ValidationFramework
from src.validation.geometric.model import ModelGeometricValidator
from src.validation.quantum.state import QuantumStateValidator
from src.validation.patterns.stability import PatternValidator as StabilityValidator
from src.core.models.base import ModelGeometry
from tests.utils.config_loader import load_test_config


@pytest.fixture(scope="session")
def test_config():
    """Load test configuration."""
    return load_test_config()


@pytest.fixture
def motivic_structure(test_config):
    """Create a motivic Riemannian structure for testing."""
    return MockMotivicRiemannianStructure(
        manifold_dim=test_config["geometric"]["manifold_dim"],
        hidden_dim=test_config["performance"]["dimensions"]["hidden"],
        motive_rank=test_config["geometric"]["motive_rank"],
        num_primes=test_config["geometric"]["num_primes"]
    )


@pytest.fixture
def points(test_config):
    """Create test points."""
    manifold_dim = test_config["geometric"]["manifold_dim"]
    batch_size = test_config["performance"]["batch_sizes"]["geometric"]
    points = torch.randn(batch_size, manifold_dim)
    points.requires_grad_(True)  # Enable gradients
    return points


@pytest.fixture
def validator(motivic_structure, test_config):
    """Create a motivic Riemannian validator."""
    return MotivicRiemannianValidator(
        motivic_structure,
        tolerance=test_config["validation"]["tolerances"]["base"]
    )


@pytest.fixture
def mock_model_geometry(test_config):
    """Create mock model geometry for testing."""
    manifold_dim = test_config["geometric"]["manifold_dim"]
    return MockModelGeometry(manifold_dim=manifold_dim)


@pytest.fixture
def geometric_validator(mock_model_geometry, test_config):
    """Create geometric validator."""
    return ModelGeometricValidator(
        model_geometry=mock_model_geometry,
        tolerance=test_config["validation"]["tolerances"]["base"],
        curvature_bounds=(-1.0, 1.0)
    )


@pytest.fixture
def quantum_validator(test_config):
    """Create quantum validator."""
    return QuantumStateValidator()


@pytest.fixture
def pattern_validator(test_config):
    """Create pattern validator."""
    tolerance = test_config["validation"]["tolerances"]["base"]
    return StabilityValidator(
        linear_validator=None,  # Mock implementation doesn't need these
        nonlinear_validator=None,
        lyapunov_threshold=tolerance,
        perturbation_threshold=tolerance
    )


@pytest.fixture
def validation_framework(geometric_validator, quantum_validator, pattern_validator):
    """Create validation framework."""
    return ValidationFramework(
        geometric_validator=geometric_validator,
        quantum_validator=quantum_validator,
        pattern_validator=pattern_validator
    )


class MockModelGeometry(ModelGeometry):
    """Mock model geometry for testing."""
    def __init__(self, manifold_dim: int = 3):
        super().__init__(
            manifold_dim=manifold_dim,
            query_dim=manifold_dim,
            key_dim=manifold_dim,
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
        
        # Create fiber map with orthogonal initialization
        self.fiber_map = nn.Linear(manifold_dim, hidden_dim)
        nn.init.orthogonal_(self.fiber_map.weight)
        self.fiber_map.weight.data *= 0.1  # Scale for stability
        
        # Create connection map with stable initialization
        # Output size should be manifold_dim^3 for proper reshaping
        self.connection_map = nn.Linear(manifold_dim, manifold_dim * manifold_dim * manifold_dim)
        nn.init.xavier_uniform_(self.connection_map.weight)
        self.connection_map.weight.data *= 0.1  # Scale for stability
        
        # Create input projection for dynamics
        self.input_proj = nn.Linear(manifold_dim, hidden_dim)
        nn.init.xavier_uniform_(self.input_proj.weight)
        self.input_proj.weight.data *= 0.1  # Scale for stability
        
    def compute_metric(self, points: Tensor) -> MotivicMetricTensor:
        """Compute metric tensor with height structure."""
        # Ensure points have batch dimension
        if points.dim() == 1:
            points = points.unsqueeze(0)
            
        batch_size = points.shape[0]
        eps = 1e-8  # Numerical stability
        
        # Compute point norms for metric scaling with stability
        point_norms = torch.norm(points, dim=1, keepdim=True)
        point_norms = torch.clamp(point_norms, min=1e-6)
        
        # Create base metric values that scale with point norms
        identity = torch.eye(self.manifold_dim, device=points.device)
        values = torch.zeros(batch_size, self.manifold_dim, self.manifold_dim, device=points.device)
        
        # Scale metric by point norms to ensure proper behavior
        for i in range(batch_size):
            values[i] = identity * point_norms[i].item()
        
        # Add scaled identity for better conditioning
        values = values + 0.1 * identity.unsqueeze(0)
        
        # Ensure symmetry and positive definiteness
        values = 0.5 * (values + values.transpose(-2, -1))
        values = values + eps * identity.unsqueeze(0)
        
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
        
        # Normalize norms to [0, 1] range for consistent spacing
        norm_min = norms.min()
        norm_max = norms.max()
        if norm_min == norm_max:
            metric.height_data = torch.full_like(norms, 0.5)
            return metric
            
        # Scale norms to [0, 1] while preserving order with stability
        normalized_norms = (norms - norm_min) / (norm_max - norm_min + eps)
        
        # Use exponential function to ensure strict monotonicity
        base = 2.0  # Growth factor > 1 ensures strict increase
        heights = 0.1 + 0.8 * (1 - torch.exp(-base * normalized_norms))
        
        # Ensure minimum gap between consecutive heights
        min_gap = 0.05
        for i in range(1, len(heights)):
            min_height = heights[i-1] + min_gap
            heights[i] = torch.max(heights[i], min_height)
        
        metric.height_data = heights
        
        return metric
    
    def geodesic_flow(
        self,
        initial_point: Tensor,
        initial_velocity: Tensor,
        steps: int = 100,
        step_size: float = 0.01
    ) -> Tuple[Tensor, Tensor]:
        """Mock implementation of geodesic flow with stability."""
        # Ensure inputs have proper shape
        if initial_point.dim() == 1:
            initial_point = initial_point.unsqueeze(0)
        if initial_velocity.dim() == 1:
            initial_velocity = initial_velocity.unsqueeze(0)
            
        # Normalize velocity for stability
        velocity_norm = torch.norm(initial_velocity)
        if velocity_norm > 0:
            initial_velocity = initial_velocity / velocity_norm
            
        # Generate time steps with proper spacing
        t = torch.linspace(0, step_size * steps, steps, device=initial_point.device)
        
        # Compute flow with stability
        points = initial_point.unsqueeze(0) + t.unsqueeze(-1) * initial_velocity.unsqueeze(0)
        velocities = initial_velocity.unsqueeze(0).expand(steps, -1)
        
        # Add small perturbation to avoid exact zeros
        eps = 1e-8
        points = points + eps * torch.randn_like(points)
        velocities = velocities + eps * torch.randn_like(velocities)
        
        return points, velocities
        
    def lie_derivative_metric(
        self,
        point: Tensor,
        vector_field: VectorField
    ) -> MetricTensor[Tensor]:
        """Mock implementation of Lie derivative with stability."""
        # Ensure point has batch dimension
        if point.dim() == 1:
            point = point.unsqueeze(0)
            
        # Get base metric
        metric = self.compute_metric(point)
        
        # Add small perturbation for stability
        eps = 1e-8
        perturbed_metric = metric.values + eps * torch.eye(
            self.manifold_dim,
            device=point.device
        ).unsqueeze(0)
        
        return MetricTensor(perturbed_metric, dimension=self.manifold_dim)
        
    def sectional_curvature(
        self,
        point: Tensor,
        v1: Tensor,
        v2: Tensor
    ) -> float:
        """Mock implementation of sectional curvature with stability."""
        # Ensure inputs have proper shape
        if point.dim() == 1:
            point = point.unsqueeze(0)
        if v1.dim() == 1:
            v1 = v1.unsqueeze(0)
        if v2.dim() == 1:
            v2 = v2.unsqueeze(0)
            
        # Normalize vectors
        v1_norm = torch.norm(v1)
        v2_norm = torch.norm(v2)
        eps = 1e-8
        
        if v1_norm > eps and v2_norm > eps:
            v1 = v1 / v1_norm
            v2 = v2 / v2_norm
            
            # Return small positive curvature
            return 0.1
        else:
            # Return zero for degenerate case
            return 0.0
    
    def compute_christoffel(self, points: Tensor) -> MotivicChristoffelSymbols:
        """Compute Christoffel symbols with dynamics."""
        # Ensure points have batch dimension
        if points.dim() == 1:
            points = points.unsqueeze(0)
            
        batch_size = points.shape[0]
        
        # Get metric tensor
        metric = self.compute_metric(points)
        
        # Compute connection values
        connection_values = self.connection_map(points)  # [batch_size, manifold_dim^3]
        connection_values = connection_values.view(batch_size, self.manifold_dim, self.manifold_dim, self.manifold_dim)
        
        # Ensure symmetry in lower indices
        connection_values = 0.5 * (connection_values + connection_values.transpose(-2, -1))
        
        # Create dynamics with projected input
        dynamics = ArithmeticDynamics(
            hidden_dim=self.hidden_dim,
            motive_rank=self.motive_rank,
            num_primes=self.num_primes
        )
        
        # Create Christoffel symbols with dynamics
        christoffel = MotivicChristoffelSymbols(
            values=connection_values,
            metric=metric,
            dynamics=dynamics
        )
        
        # Project points to hidden_dim for dynamics
        projected_points = self.input_proj(points)
        christoffel.dynamics_state = projected_points
        
        return christoffel

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


class TestHeightValidation:
    """Test suite specifically for height validation."""
    
    def test_basic_height_properties(self, motivic_structure, points, test_config):
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
        
        # Check numerical stability
        assert not torch.any(torch.isnan(height_data)), "Heights contain NaN values"
        assert not torch.any(torch.isinf(height_data)), "Heights contain infinite values"
        
        # Check height distribution
        if len(height_data) > 1:
            assert height_data.std() > 0, "Heights should have non-zero variance"
            assert height_data.max() - height_data.min() > 0, "Heights should have non-zero range"
    
    def test_strictly_increasing_heights(self, motivic_structure, test_config):
        """Test that heights are strictly increasing."""
        validator = MotivicValidator()
        tolerance = test_config["validation"]["tolerances"]["state_norm"]
        manifold_dim = test_config["geometric"]["manifold_dim"]
        batch_size = test_config["performance"]["batch_sizes"]["geometric"]
        min_scale = test_config["geometric"]["min_scale"]
        max_scale = test_config["geometric"]["max_scale"]
        
        # Create sequence of points with increasing norms
        points = torch.randn(batch_size, manifold_dim)
        scale_factors = torch.linspace(min_scale, max_scale, batch_size).view(-1, 1)
        points = points * scale_factors
        points.requires_grad_(True)
        
        metric = motivic_structure.compute_metric(points)
        result = validator.validate_height(metric)
        
        assert result.is_valid
        assert result.height_data is not None, "Height data should not be None"
        height_data = result.height_data  # Store in local variable after null check
        height_diffs = height_data[1:] - height_data[:-1]  # Use local variable
        assert torch.all(height_diffs > 0), "Heights must be strictly increasing"
        
        # Check minimum gap between heights
        min_gap = height_diffs.min()
        assert min_gap > tolerance, f"Height gaps too small: {min_gap}"
        
        # Check maximum gap is reasonable
        max_gap = height_diffs.max()
        assert max_gap < 1.0, f"Height gaps too large: {max_gap}"
    
    def test_height_edge_cases(self, motivic_structure, test_config):
        """Test height validation with edge cases."""
        validator = MotivicValidator()
        manifold_dim = test_config["geometric"]["manifold_dim"]
        tolerance = test_config["validation"]["tolerances"]["state_norm"]
        
        # Test single point
        single_point = torch.randn(1, manifold_dim)
        single_point.requires_grad_(True)
        metric = motivic_structure.compute_metric(single_point)
        result = validator.validate_height(metric)
        assert result.is_valid
        
        # Test zero point
        zero_point = torch.zeros(1, manifold_dim)
        zero_point.requires_grad_(True)
        metric = motivic_structure.compute_metric(zero_point)
        result = validator.validate_height(metric)
        assert result.is_valid
        assert result.height_data is not None, "Height data should not be None"
        height_data = result.height_data
        assert height_data[0] > tolerance, "Height of zero point should be positive"
        
        # Test very small points
        tiny_points = torch.randn(5, manifold_dim) * 1e-10
        tiny_points.requires_grad_(True)
        metric = motivic_structure.compute_metric(tiny_points)
        result = validator.validate_height(metric)
        assert result.is_valid
        assert result.height_data is not None
        assert torch.all(result.height_data > tolerance), "Heights of tiny points should be positive"
        
        # Test very large points
        huge_points = torch.randn(5, manifold_dim) * 1e10
        huge_points.requires_grad_(True)
        metric = motivic_structure.compute_metric(huge_points)
        result = validator.validate_height(metric)
        assert result.is_valid
        assert result.height_data is not None
        assert torch.all(torch.isfinite(result.height_data)), "Heights should be finite for large points"
        
        # Test points with identical norms
        base_point = torch.randn(manifold_dim)
        base_norm = torch.norm(base_point)
        identical_points = torch.stack([
            base_point * (base_norm / torch.norm(base_point))
            for _ in range(5)
        ])
        identical_points.requires_grad_(True)
        metric = motivic_structure.compute_metric(identical_points)
        result = validator.validate_height(metric)
        assert result.is_valid
        assert result.height_data is not None
        assert torch.allclose(
            result.height_data,
            result.height_data[0].expand_as(result.height_data),
            rtol=tolerance,
            atol=tolerance
        ), "Points with identical norms should have identical heights"
    
    def test_invalid_height_cases(self, test_config):
        """Test cases where height validation should fail."""
        validator = MotivicValidator()
        tolerance = test_config["validation"]["tolerances"]["state_norm"]
        manifold_dim = test_config["geometric"]["manifold_dim"]
        
        # Test negative height
        invalid_height = torch.tensor([-1.0, 0.0, 1.0])
        height_structure = HeightStructure(num_primes=4)
        metric = MotivicMetricTensor(
            values=torch.eye(manifold_dim),
            dimension=manifold_dim,
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
        
        # Test NaN height
        nan_height = torch.tensor([float('nan'), 1.0, 2.0])
        metric.height_data = nan_height
        result = validator.validate_height(metric)
        assert not result.is_valid
        assert not result.local_heights_valid
        
        # Test non-monotonic heights
        non_monotonic = torch.tensor([0.1, 0.3, 0.2, 0.4])
        metric.height_data = non_monotonic
        result = validator.validate_height(metric)
        assert not result.is_valid
        assert not result.local_heights_valid
        
        # Test zero heights
        zero_heights = torch.zeros(4)
        metric.height_data = zero_heights
        result = validator.validate_height(metric)
        assert not result.is_valid
        assert not result.local_heights_valid

class TestDynamicsValidation:
    """Test suite specifically for dynamics validation."""
    
    def test_basic_dynamics_properties(self, motivic_structure, points, test_config):
        """Test basic arithmetic dynamics properties."""
        validator = MotivicValidator()
        tolerance = test_config["validation"]["tolerances"]["state_norm"]
        connection = motivic_structure.compute_christoffel(points)
        
        result = validator.validate_dynamics(connection)
        assert result.is_valid
        assert "passed" in result.message
        assert connection.dynamics_state is not None
        
        # Check numerical properties
        assert torch.all(torch.isfinite(connection.values)), "Connection values must be finite"
        assert not torch.any(torch.isnan(connection.values)), "Connection contains NaN values"
        assert not torch.any(torch.isinf(connection.values)), "Connection contains infinite values"
        
        # Check symmetry in lower indices (Christoffel symbols)
        for i in range(points.size(0)):
            for j in range(connection.values.size(1)):
                assert torch.allclose(
                    connection.values[i, j, :, :],
                    connection.values[i, j, :, :].transpose(-2, -1),
                    rtol=tolerance,
                    atol=tolerance
                ), f"Connection not symmetric at batch {i}, index {j}"
    
    def test_dynamics_consistency(self, motivic_structure, points, test_config):
        """Test consistency of arithmetic dynamics."""
        validator = MotivicValidator()
        tolerance = test_config["validation"]["tolerances"]["state_norm"]
        connection = motivic_structure.compute_christoffel(points)
        
        # Check that dynamics preserves structure
        dynamics = connection.dynamics
        assert isinstance(dynamics, ArithmeticDynamics)
        assert dynamics.hidden_dim == motivic_structure.hidden_dim
        assert dynamics.motive_rank == motivic_structure.motive_rank
        assert dynamics.num_primes == motivic_structure.num_primes
        
        # Get projected points from dynamics_state
        projected_points = connection.dynamics_state
        assert projected_points.shape[-1] == dynamics.hidden_dim
        
        # Add sequence dimension for dynamics computation
        points_seq = projected_points.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Test dynamics computation
        output, metrics = dynamics(points_seq)
        assert output.shape == points_seq.shape
        assert torch.all(torch.isfinite(output)), "Output should be finite"
        assert not torch.any(torch.isnan(output)), "Output should not contain NaN"
        
        # Test height computation
        height = dynamics.compute_height(projected_points)
        assert height.shape[-1] == dynamics.height_dim, "Height should have correct dimension"
        assert torch.all(torch.isfinite(height)), "Height should be finite"
        assert not torch.any(torch.isnan(height)), "Height should not contain NaN"
        
        # Test L-function computation
        l_value = dynamics.compute_l_function(projected_points)
        assert l_value.shape[-1] == dynamics.motive_rank, "L-value should have correct dimension"
        assert torch.all(torch.isfinite(l_value)), "L-value should be finite"
        assert not torch.any(torch.isnan(l_value)), "L-value should not contain NaN"
    
    def test_dynamics_edge_cases(self, motivic_structure, test_config):
        """Test dynamics validation with edge cases."""
        validator = MotivicValidator()
        manifold_dim = test_config["geometric"]["manifold_dim"]
        tolerance = test_config["validation"]["tolerances"]["state_norm"]
        
        # Test single point
        single_point = torch.randn(1, manifold_dim)
        single_point.requires_grad_(True)
        connection = motivic_structure.compute_christoffel(single_point)
        result = validator.validate_dynamics(connection)
        assert result.is_valid
        
        # Test zero connection
        zero_point = torch.zeros(1, manifold_dim)
        zero_point.requires_grad_(True)
        connection = motivic_structure.compute_christoffel(zero_point)
        result = validator.validate_dynamics(connection)
        assert result.is_valid
        
        # Test very small points
        tiny_points = torch.randn(5, manifold_dim) * 1e-10
        tiny_points.requires_grad_(True)
        connection = motivic_structure.compute_christoffel(tiny_points)
        result = validator.validate_dynamics(connection)
        assert result.is_valid
        assert torch.all(torch.isfinite(connection.values)), "Connection values must be finite for tiny points"
        
        # Test very large points
        huge_points = torch.randn(5, manifold_dim) * 1e10
        huge_points.requires_grad_(True)
        connection = motivic_structure.compute_christoffel(huge_points)
        result = validator.validate_dynamics(connection)
        assert result.is_valid
        assert torch.all(torch.isfinite(connection.values)), "Connection values must be finite for large points"
        
        # Test points with identical coordinates
        identical_points = torch.ones(5, manifold_dim)
        identical_points.requires_grad_(True)
        connection = motivic_structure.compute_christoffel(identical_points)
        result = validator.validate_dynamics(connection)
        assert result.is_valid
        
        # Test points with extreme condition numbers
        ill_conditioned_points = torch.cat([
            torch.ones(1, manifold_dim) * 1e10,
            torch.ones(1, manifold_dim) * 1e-10,
            torch.randn(3, manifold_dim)
        ])
        ill_conditioned_points.requires_grad_(True)
        connection = motivic_structure.compute_christoffel(ill_conditioned_points)
        result = validator.validate_dynamics(connection)
        assert result.is_valid
    
    def test_invalid_dynamics_cases(self, motivic_structure, points, test_config):
        """Test cases where dynamics validation should fail."""
        validator = MotivicValidator()
        tolerance = test_config["validation"]["tolerances"]["state_norm"]
        connection = motivic_structure.compute_christoffel(points)
        
        # Test with no dynamics state
        connection.dynamics_state = None
        result = validator.validate_dynamics(connection)
        assert not result.is_valid
        assert "No dynamics state" in result.message
        
        # Test with invalid dynamics parameters
        connection.dynamics = ArithmeticDynamics(
            hidden_dim=motivic_structure.hidden_dim + 2,  # Mismatched dimension
            motive_rank=motivic_structure.motive_rank,
            num_primes=motivic_structure.num_primes
        )
        result = validator.validate_dynamics(connection)
        assert not result.is_valid
        
        # Test with NaN values in connection
        connection = motivic_structure.compute_christoffel(points)
        connection.values[0, 0, 0, 0] = float('nan')
        result = validator.validate_dynamics(connection)
        assert not result.is_valid
        
        # Test with infinite values in connection
        connection = motivic_structure.compute_christoffel(points)
        connection.values[0, 0, 0, 0] = float('inf')
        result = validator.validate_dynamics(connection)
        assert not result.is_valid
        
        # Test with non-symmetric connection
        connection = motivic_structure.compute_christoffel(points)
        connection.values[0, 0, 0, 1] = 1.0
        connection.values[0, 0, 1, 0] = 2.0  # Break symmetry
        result = validator.validate_dynamics(connection)
        assert not result.is_valid

class TestCohomologyValidation:
    """Test suite specifically for cohomology validation."""
    
    def test_basic_cohomology_properties(self, motivic_structure, points, test_config):
        """Test basic cohomology properties."""
        validator = MotivicValidator()
        tolerance = test_config["validation"]["tolerances"]["state_norm"]
        metric = motivic_structure.compute_metric(points)
        connection = motivic_structure.compute_christoffel(points)
        curvature = motivic_structure.compute_curvature(points, connection)
        
        result = validator.validate_cohomology(curvature)
        assert result.bounds_satisfied
        assert torch.is_tensor(result.error_bounds)
        assert torch.is_tensor(result.scalar_curvatures)
        
        # Check specific curvature properties
        assert torch.all(torch.isfinite(curvature.riemann)), "Riemann tensor must be finite"
        assert torch.all(torch.isfinite(curvature.ricci)), "Ricci tensor must be finite"
        assert torch.all(torch.isfinite(curvature.scalar_curvatures)), "Scalar curvature must be finite"
        
        # Check numerical stability
        assert not torch.any(torch.isnan(curvature.riemann)), "Riemann tensor contains NaN"
        assert not torch.any(torch.isnan(curvature.ricci)), "Ricci tensor contains NaN"
        assert not torch.any(torch.isnan(curvature.scalar_curvatures)), "Scalar curvature contains NaN"
        
        # Check tensor symmetries
        batch_size = points.size(0)
        for b in range(batch_size):
            # Riemann tensor symmetries
            riemann = curvature.riemann[b]
            # R_ijkl = -R_ijlk (antisymmetry in last two indices)
            assert torch.allclose(
                riemann,
                -riemann.transpose(-1, -2),
                rtol=tolerance,
                atol=tolerance
            ), "Riemann tensor not antisymmetric in last two indices"
            
            # R_ijkl = -R_jikl (antisymmetry in first two indices)
            assert torch.allclose(
                riemann.transpose(0, 1),
                -riemann,
                rtol=tolerance,
                atol=tolerance
            ), "Riemann tensor not antisymmetric in first two indices"
            
            # R_ijkl = R_klij (pair symmetry)
            assert torch.allclose(
                riemann,
                riemann.transpose(0, 2).transpose(1, 3),
                rtol=tolerance,
                atol=tolerance
            ), "Riemann tensor does not have pair symmetry"
            
            # Ricci tensor symmetry
            ricci = curvature.ricci[b]
            assert torch.allclose(
                ricci,
                ricci.transpose(-1, -2),
                rtol=tolerance,
                atol=tolerance
            ), "Ricci tensor not symmetric"
    
    def test_curvature_bounds(self, motivic_structure, points, test_config):
        """Test that curvature satisfies required bounds."""
        validator = MotivicValidator()
        tolerance = test_config["validation"]["tolerances"]["state_norm"]
        metric = motivic_structure.compute_metric(points)
        connection = motivic_structure.compute_christoffel(points)
        curvature = motivic_structure.compute_curvature(points, connection)
        
        result = validator.validate_cohomology(curvature)
        assert result.bounds_satisfied
        
        # Check positive scalar curvature with tolerance
        assert torch.all(curvature.scalar_curvatures > -tolerance), "Scalar curvature should be non-negative"
        
        # Check error bounds are finite and reasonable
        assert torch.all(torch.isfinite(result.error_bounds)), "Error bounds must be finite"
        assert torch.all(result.error_bounds >= -tolerance), "Error bounds must be non-negative"
        assert torch.all(result.error_bounds < 1.0 + tolerance), "Error bounds too large"
        
        # Check curvature magnitudes
        riemann_norm = torch.norm(curvature.riemann.reshape(points.size(0), -1), dim=1)
        ricci_norm = torch.norm(curvature.ricci.reshape(points.size(0), -1), dim=1)
        scalar_norm = torch.abs(curvature.scalar_curvatures)
        
        max_norm = test_config["geometric"]["max_norm"]
        assert torch.all(riemann_norm < max_norm), f"Riemann tensor norm too large: {riemann_norm.max().item()}"
        assert torch.all(ricci_norm < max_norm), f"Ricci tensor norm too large: {ricci_norm.max().item()}"
        assert torch.all(scalar_norm < max_norm), f"Scalar curvature too large: {scalar_norm.max().item()}"
    
    def test_cohomology_edge_cases(self, motivic_structure, test_config):
        """Test cohomology validation with edge cases."""
        validator = MotivicValidator()
        manifold_dim = test_config["geometric"]["manifold_dim"]
        tolerance = test_config["validation"]["tolerances"]["state_norm"]
        
        # Test single point
        single_point = torch.randn(1, manifold_dim)
        single_point.requires_grad_(True)
        metric = motivic_structure.compute_metric(single_point)
        connection = motivic_structure.compute_christoffel(single_point)
        curvature = motivic_structure.compute_curvature(single_point, connection)
        result = validator.validate_cohomology(curvature)
        assert result.bounds_satisfied
        
        # Test flat space (zero curvature)
        flat_point = torch.zeros(1, manifold_dim)
        flat_point.requires_grad_(True)
        metric = motivic_structure.compute_metric(flat_point)
        connection = motivic_structure.compute_christoffel(flat_point)
        curvature = motivic_structure.compute_curvature(flat_point, connection)
        result = validator.validate_cohomology(curvature)
        assert result.bounds_satisfied
        
        # Test very small points
        tiny_points = torch.randn(5, manifold_dim) * 1e-10
        tiny_points.requires_grad_(True)
        metric = motivic_structure.compute_metric(tiny_points)
        connection = motivic_structure.compute_christoffel(tiny_points)
        curvature = motivic_structure.compute_curvature(tiny_points, connection)
        result = validator.validate_cohomology(curvature)
        assert result.bounds_satisfied
        assert torch.all(torch.isfinite(curvature.riemann)), "Riemann tensor must be finite for tiny points"
        
        # Test very large points
        huge_points = torch.randn(5, manifold_dim) * 1e10
        huge_points.requires_grad_(True)
        metric = motivic_structure.compute_metric(huge_points)
        connection = motivic_structure.compute_christoffel(huge_points)
        curvature = motivic_structure.compute_curvature(huge_points, connection)
        result = validator.validate_cohomology(curvature)
        assert result.bounds_satisfied
        assert torch.all(torch.isfinite(curvature.riemann)), "Riemann tensor must be finite for large points"
        
        # Test points with identical coordinates
        identical_points = torch.ones(5, manifold_dim)
        identical_points.requires_grad_(True)
        metric = motivic_structure.compute_metric(identical_points)
        connection = motivic_structure.compute_christoffel(identical_points)
        curvature = motivic_structure.compute_curvature(identical_points, connection)
        result = validator.validate_cohomology(curvature)
        assert result.bounds_satisfied
        
        # Test points with extreme condition numbers
        ill_conditioned_points = torch.cat([
            torch.ones(1, manifold_dim) * 1e10,
            torch.ones(1, manifold_dim) * 1e-10,
            torch.randn(3, manifold_dim)
        ])
        ill_conditioned_points.requires_grad_(True)
        metric = motivic_structure.compute_metric(ill_conditioned_points)
        connection = motivic_structure.compute_christoffel(ill_conditioned_points)
        curvature = motivic_structure.compute_curvature(ill_conditioned_points, connection)
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
        
        # Test with NaN curvature
        curvature.riemann = torch.full_like(curvature.riemann, float('nan'))
        result = validator.validate_cohomology(curvature)
        assert not result.bounds_satisfied
        
        # Test with non-symmetric Ricci tensor
        curvature = motivic_structure.compute_curvature(points, connection)
        curvature.ricci[0, 0, 1] = 1.0
        curvature.ricci[0, 1, 0] = 2.0  # Break symmetry
        result = validator.validate_cohomology(curvature)
        assert not result.bounds_satisfied
        
        # Test with negative scalar curvature
        curvature = motivic_structure.compute_curvature(points, connection)
        curvature.scalar_curvatures = -torch.abs(curvature.scalar_curvatures)
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
  