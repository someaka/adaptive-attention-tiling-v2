"""Tests for base Riemannian geometry implementation."""

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from src.core.patterns.riemannian import BaseRiemannianStructure
from src.core.patterns.riemannian_base import MetricTensor, ChristoffelSymbols, CurvatureTensor

@pytest.fixture
def riemannian_structure():
    """Create a base Riemannian structure for testing."""
    return BaseRiemannianStructure(
        manifold_dim=3,
        device=torch.device('cpu'),
        dtype=torch.float64  # Use double precision for numerical stability
    )

@pytest.fixture
def test_points():
    """Create test points on the manifold."""
    return torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float64)

def test_metric_properties(riemannian_structure, test_points):
    """Test that metric tensor satisfies required properties."""
    metric = riemannian_structure.compute_metric(test_points)
    
    # Test symmetry
    assert torch.allclose(
        metric.values,
        metric.values.transpose(-2, -1),
        rtol=1e-10
    )
    
    # Test positive definiteness
    eigenvals = torch.linalg.eigvalsh(metric.values)
    assert torch.all(eigenvals > 0)
    
    # Test dimension
    assert metric.dimension == 3
    assert metric.values.shape == (3, 3, 3)

def test_christoffel_properties(riemannian_structure, test_points):
    """Test that Christoffel symbols satisfy required properties."""
    christoffel = riemannian_structure.compute_christoffel(test_points)
    
    # Test symmetry in lower indices
    assert torch.allclose(
        christoffel.values.transpose(-2, -1),
        christoffel.values,
        rtol=1e-10
    )
    
    # Test metric compatibility
    metric_deriv = torch.autograd.grad(
        christoffel.metric.values.sum(),
        test_points,
        create_graph=True
    )[0]
    
    # ∇_k g_ij should be zero
    assert torch.allclose(
        metric_deriv,
        torch.zeros_like(metric_deriv),
        rtol=1e-8
    )

def test_parallel_transport(riemannian_structure):
    """Test parallel transport preserves inner product."""
    # Create a simple path
    t = torch.linspace(0, 1, 10, dtype=torch.float64)
    path = torch.stack([
        torch.cos(2 * torch.pi * t),
        torch.sin(2 * torch.pi * t),
        torch.zeros_like(t)
    ], dim=1)
    
    # Initial vector
    vector = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    
    # Perform parallel transport
    transported = riemannian_structure.parallel_transport(vector, path)
    
    # Check norm preservation
    metric = riemannian_structure.compute_metric(path)
    initial_norm = torch.einsum('i,ij,j->', vector, metric.values[0], vector)
    
    for t in range(len(path)):
        current_norm = torch.einsum(
            'i,ij,j->',
            transported[t],
            metric.values[t],
            transported[t]
        )
        assert torch.allclose(current_norm, initial_norm, rtol=1e-8)

def test_curvature_identities(riemannian_structure, test_points):
    """Test that curvature tensor satisfies Bianchi identities."""
    curvature = riemannian_structure.compute_curvature(test_points)
    
    # First Bianchi identity
    # R^i_jkl + R^i_klj + R^i_ljk = 0
    bianchi_1 = (
        curvature.riemann
        + torch.einsum('...ijkl->...ikjl', curvature.riemann)
        + torch.einsum('...ijkl->...iljk', curvature.riemann)
    )
    assert torch.allclose(bianchi_1, torch.zeros_like(bianchi_1), rtol=1e-8)
    
    # Second Bianchi identity
    # ∇_m R^i_jkl + ∇_k R^i_jlm + ∇_l R^i_jmk = 0
    riemann_grad = torch.autograd.grad(
        curvature.riemann.sum(),
        test_points,
        create_graph=True
    )[0]
    
    bianchi_2 = (
        riemann_grad
        + torch.einsum('...ijklm->...ijlmk', riemann_grad)
        + torch.einsum('...ijklm->...ijmkl', riemann_grad)
    )
    assert torch.allclose(bianchi_2, torch.zeros_like(bianchi_2), rtol=1e-8)

def test_geodesic_equation(riemannian_structure):
    """Test that geodesic flow satisfies the geodesic equation."""
    # Initial conditions
    initial_point = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    initial_velocity = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    
    # Compute geodesic
    points, velocities = riemannian_structure.geodesic_flow(
        initial_point,
        initial_velocity,
        steps=100,
        step_size=0.01
    )
    
    # Check geodesic equation
    christoffel = riemannian_structure.compute_christoffel(points)
    
    for t in range(len(points) - 1):
        # Compute acceleration
        acceleration = (velocities[t + 1] - velocities[t]) / 0.01
        
        # Compute Christoffel term
        christoffel_term = -torch.einsum(
            '...ijk,...j,...k->...i',
            christoffel.values[t],
            velocities[t],
            velocities[t]
        )
        
        # Check geodesic equation
        assert torch.allclose(
            acceleration,
            christoffel_term,
            rtol=1e-6
        )

def test_sectional_curvature(riemannian_structure, test_points):
    """Test properties of sectional curvature."""
    # Create orthonormal vectors
    v1 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    v2 = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    
    # Compute sectional curvature
    K = riemannian_structure.sectional_curvature(test_points[0], v1, v2)
    
    # Test symmetry
    K_reversed = riemannian_structure.sectional_curvature(test_points[0], v2, v1)
    assert torch.allclose(K, K_reversed, rtol=1e-10)
    
    # Test scaling invariance
    K_scaled = riemannian_structure.sectional_curvature(
        test_points[0],
        2 * v1,
        3 * v2
    )
    assert torch.allclose(K, K_scaled, rtol=1e-10)

def test_lie_derivative(riemannian_structure, test_points):
    """Test properties of Lie derivative."""
    # Define a simple vector field
    def vector_field(x: Tensor) -> Tensor:
        return torch.tensor([
            -x[1],
            x[0],
            0.0
        ], dtype=torch.float64)
    
    # Compute Lie derivative
    lie_deriv = riemannian_structure.lie_derivative_metric(
        test_points[0],
        vector_field
    )
    
    # Test symmetry
    assert torch.allclose(
        lie_deriv.values,
        lie_deriv.values.transpose(-2, -1),
        rtol=1e-10
    )
    
    # Test dimension
    assert lie_deriv.dimension == 3
    assert lie_deriv.values.shape == (1, 3, 3)
