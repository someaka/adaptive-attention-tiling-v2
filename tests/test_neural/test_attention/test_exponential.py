import torch
import pytest
from src.core.attention.geometric import HyperbolicExponential

def test_exponential_map():
    """Test exponential map computation."""
    dim = 3
    exp_map = HyperbolicExponential(dim)
    
    # Test case 1: Small tangent vector
    x = torch.tensor([1.2, 0.3, 0.4])  # Point on hyperboloid
    v = torch.tensor([0.0, 1e-8, 1e-8])  # Small tangent vector
    result = exp_map.forward(x, v)
    
    # Check properties
    assert not torch.any(torch.isnan(result))
    assert result[0] >= 1.0 + 1e-7  # Time component constraint
    inner = exp_map.minkowski_inner(result, result)
    assert torch.allclose(inner, torch.tensor(-1.0), atol=1e-6)
    
    # Test case 2: Normal tangent vector
    x = torch.tensor([1.5, 0.5, 0.0])
    v = torch.tensor([0.0, 0.3, 0.4])
    result = exp_map.forward(x, v)
    assert not torch.any(torch.isnan(result))
    inner = exp_map.minkowski_inner(result, result)
    assert torch.allclose(inner, torch.tensor(-1.0), atol=1e-6)
    
    # Test case 3: Large tangent vector (should be clamped)
    x = torch.tensor([2.0, 1.0, 1.0])
    v = torch.tensor([0.0, 10.0, 10.0])
    result = exp_map.forward(x, v)
    assert not torch.any(torch.isnan(result))
    inner = exp_map.minkowski_inner(result, result)
    assert torch.allclose(inner, torch.tensor(-1.0), atol=1e-6)
    
    # Test case 4: Batch computation
    x_batch = torch.stack([
        torch.tensor([1.2, 0.3, 0.4]),
        torch.tensor([1.5, 0.5, 0.0])
    ])
    v_batch = torch.stack([
        torch.tensor([0.0, 0.1, 0.1]),
        torch.tensor([0.0, 0.2, 0.3])
    ])
    result_batch = exp_map.forward(x_batch, v_batch)
    assert not torch.any(torch.isnan(result_batch))
    assert result_batch.shape == (2, 3)
    inner_batch = exp_map.minkowski_inner(result_batch, result_batch)
    assert torch.allclose(inner_batch, torch.tensor([-1.0, -1.0]), atol=1e-6)

def test_exponential_map_properties():
    """Test mathematical properties of exponential map."""
    dim = 3
    exp_map = HyperbolicExponential(dim)
    
    # Property 1: exp_x(0) = x
    x = torch.tensor([1.2, 0.3, 0.4])
    x = exp_map.project_to_hyperboloid(x)
    v_zero = torch.zeros_like(x)
    result = exp_map.forward(x, v_zero)
    assert torch.allclose(result, x, atol=1e-6)
    
    # Property 2: exp_x(v) should preserve hyperboloid constraint
    v = torch.tensor([0.0, 0.3, 0.4])
    result = exp_map.forward(x, v)
    inner = exp_map.minkowski_inner(result, result)
    assert torch.allclose(inner, torch.tensor(-1.0), atol=1e-6)
    
    # Property 3: Scaling property for small vectors
    x = torch.tensor([1.5, 0.5, 0.0])
    v = torch.tensor([0.0, 0.1, 0.1])
    result1 = exp_map.forward(x, v)
    result2 = exp_map.forward(x, 2 * v)
    
    # The geodesic distance should scale approximately for small vectors
    dist1 = torch.sqrt(-exp_map.minkowski_inner(x, result1))
    dist2 = torch.sqrt(-exp_map.minkowski_inner(x, result2))
    assert torch.abs(dist2 - 2 * dist1) < 0.1  # Approximate due to curvature
