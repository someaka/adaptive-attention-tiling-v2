"""Tests for core cohomology functionality.

This module tests the core motivic cohomology implementation including:
- Arithmetic form computation
- Height theory fundamentals
- Cohomology class construction
- Integration with Riemannian geometry
"""

import pytest
import torch
from torch import Tensor
from typing import Tuple

from src.core.patterns.motivic_riemannian import (
    MotivicRiemannianStructure,
    MotivicCurvatureTensor
)
from src.core.tiling.patterns.cohomology import (
    MotivicCohomology,
    ArithmeticForm,
    HeightStructure
)
from src.core.patterns.riemannian_base import MetricTensor, VectorField


class MockMotivicRiemannianStructure(MotivicRiemannianStructure):
    """Mock implementation of MotivicRiemannianStructure for testing."""
    
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


@pytest.fixture
def height_structure():
    """Create height structure for testing."""
    return HeightStructure(num_primes=8)


@pytest.fixture
def motivic_cohomology():
    """Create motivic cohomology structure for testing."""
    base_space = MockMotivicRiemannianStructure(  # Use mock implementation
        manifold_dim=4,
        hidden_dim=8,
        motive_rank=4
    )
    return MotivicCohomology(
        base_space=base_space,
        hidden_dim=8,
        motive_rank=4,
        num_primes=8
    )


def test_arithmetic_form_creation():
    """Test creation and properties of arithmetic forms."""
    # Create simple arithmetic form
    coeffs = torch.randn(10, 16)  # [batch_size, coefficients]
    form = ArithmeticForm(degree=2, coefficients=coeffs)
    
    assert form.degree == 2
    assert torch.equal(form.coefficients, coeffs)
    assert form.coefficients.shape == (10, 16)


def test_height_computation(height_structure):
    """Test computation of height functions."""
    # Create test values
    values = torch.randn(10, 4, 4)  # [batch_size, dim, dim]
    
    # Compute heights
    heights = height_structure.compute_height(values)
    
    # Check properties
    assert heights.shape[0] == 10  # Batch dimension preserved
    assert torch.all(heights >= 0)  # Heights should be non-negative
    assert torch.all(torch.isfinite(heights))  # Heights should be finite


def test_cohomology_class_computation(motivic_cohomology):
    """Test computation of cohomology classes."""
    # Create test form
    batch_size = 10
    coeffs = torch.randn(batch_size, 16)
    form = ArithmeticForm(degree=2, coefficients=coeffs)
    
    # Compute cohomology class
    cohomology = motivic_cohomology.compute_motive(form)
    
    # Check properties
    assert cohomology.shape[0] == batch_size
    assert cohomology.shape[1] == motivic_cohomology.motive_rank
    assert torch.all(torch.isfinite(cohomology))


def test_curvature_to_cohomology(motivic_cohomology):
    """Test conversion of curvature data to cohomology classes."""
    # Create test curvature tensor
    batch_size = 10
    dim = 4
    riemann = torch.randn(batch_size, dim, dim, dim, dim)
    ricci = torch.einsum('bijji->bij', riemann)
    scalar = torch.einsum('bii->b', ricci)
    
    # Create curvature tensor
    curvature = MotivicCurvatureTensor(
        riemann=riemann,
        ricci=ricci,
        scalar_curvatures=scalar,
        motive=motivic_cohomology
    )
    
    # Check cohomology class
    assert hasattr(curvature, 'cohomology_class')
    assert curvature.cohomology_class.shape[0] == batch_size
    assert curvature.cohomology_class.shape[1] == motivic_cohomology.motive_rank
    assert torch.all(torch.isfinite(curvature.cohomology_class))


def test_height_theory_integration(height_structure, motivic_cohomology):
    """Test integration of height theory with cohomology."""
    # Create test metric values
    batch_size = 10
    dim = 4
    metric_values = torch.randn(batch_size, dim, dim)
    
    # Compute heights
    heights = height_structure.compute_height(metric_values)
    
    # Flatten heights to 2D tensor [batch_size, features]
    heights_flat = heights.reshape(batch_size, -1)
    
    # Create arithmetic form from heights
    form = ArithmeticForm(
        degree=1,  # Height is a 1-form
        coefficients=heights_flat  # [batch_size, features]
    )
    
    # Compute cohomology class
    cohomology = motivic_cohomology.compute_motive(form)
    
    # Verify properties of cohomology class
    assert cohomology.shape[0] == batch_size  # Batch dimension preserved
    assert cohomology.shape[1] == motivic_cohomology.motive_rank  # Correct rank
    assert torch.all(torch.isfinite(cohomology))  # Values should be finite
    
    # Verify basic properties of the height-cohomology relationship
    heights_mag = torch.norm(heights_flat, dim=1)
    cohomology_mag = torch.norm(cohomology, dim=1)
    
    # Properties to check:
    # 1. Both should be non-zero when input is non-zero
    assert torch.all(heights_mag > 0)
    assert torch.all(cohomology_mag > 0)
    
    # 2. Zero input should give zero output
    zero_form = ArithmeticForm(
        degree=1,
        coefficients=torch.zeros_like(heights_flat)
    )
    zero_cohomology = motivic_cohomology.compute_motive(zero_form)
    assert torch.allclose(zero_cohomology, torch.zeros_like(zero_cohomology))
    
    # 3. Scaling input should affect output monotonically
    scale = 2.0
    scaled_form = ArithmeticForm(
        degree=1,
        coefficients=scale * heights_flat
    )
    scaled_cohomology = motivic_cohomology.compute_motive(scaled_form)
    scaled_mag = torch.norm(scaled_cohomology, dim=1)
    
    # Check that scaling up increases magnitude
    assert torch.all(scaled_mag > cohomology_mag)
    
    # 4. Check that cohomology preserves linear independence
    # Take two different height vectors
    h1 = heights_flat[0]
    h2 = heights_flat[1]
    
    form1 = ArithmeticForm(degree=1, coefficients=h1.unsqueeze(0))
    form2 = ArithmeticForm(degree=1, coefficients=h2.unsqueeze(0))
    
    c1 = motivic_cohomology.compute_motive(form1)
    c2 = motivic_cohomology.compute_motive(form2)
    
    # The cohomology classes should be different
    assert not torch.allclose(c1, c2)


def test_boundary_cases(motivic_cohomology):
    """Test cohomology computation with boundary cases."""
    # Test with zero form
    zero_coeffs = torch.zeros(10, 16)
    zero_form = ArithmeticForm(degree=2, coefficients=zero_coeffs)
    zero_cohomology = motivic_cohomology.compute_motive(zero_form)
    assert torch.all(zero_cohomology == 0)
    
    # Test with very large values
    large_coeffs = torch.randn(10, 16) * 1e6
    large_form = ArithmeticForm(degree=2, coefficients=large_coeffs)
    large_cohomology = motivic_cohomology.compute_motive(large_form)
    assert torch.all(torch.isfinite(large_cohomology))  # Should handle large values
