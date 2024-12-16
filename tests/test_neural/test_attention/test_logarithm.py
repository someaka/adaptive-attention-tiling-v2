import torch
import pytest
from src.core.attention.geometric import HyperbolicExponential, HyperbolicLogarithm

def test_logarithm_map():
    """Test logarithm map computation."""
    dim = 3
    exp_map = HyperbolicExponential(dim)
    log_map = HyperbolicLogarithm(dim)
    
    # Test case 1: Same point
    x = torch.tensor([1.2, 0.3, 0.4])
    x = exp_map.project_to_hyperboloid(x)
    result = log_map.forward(x, x)
    assert torch.allclose(result, torch.zeros_like(x), atol=1e-6)
    
    # Test case 2: Close points
    x = torch.tensor([1.2, 0.3, 0.4])
    y = torch.tensor([1.2, 0.3 + 1e-8, 0.4])
    result = log_map.forward(x, y)
    assert not torch.any(torch.isnan(result))
    assert torch.norm(result) < 1e-7  # Should be very small
    
    # Test case 3: Normal points
    x = torch.tensor([1.5, 0.5, 0.0])
    y = torch.tensor([1.5, 0.7, 0.2])
    result = log_map.forward(x, y)
    assert not torch.any(torch.isnan(result))
    
    # The result should be in the tangent space at x
    inner = exp_map.minkowski_inner(x, result)
    assert torch.abs(inner) < 1e-6
    
    # Test case 4: Batch computation
    x_batch = torch.stack([
        torch.tensor([1.2, 0.3, 0.4]),
        torch.tensor([1.5, 0.5, 0.0])
    ])
    y_batch = torch.stack([
        torch.tensor([1.2, 0.4, 0.4]),
        torch.tensor([1.5, 0.7, 0.2])
    ])
    result_batch = log_map.forward(x_batch, y_batch)
    assert not torch.any(torch.isnan(result_batch))
    assert result_batch.shape == (2, 3)

def test_logarithm_map_properties():
    """Test mathematical properties of logarithm map."""
    dim = 3
    exp_map = HyperbolicExponential(dim)
    log_map = HyperbolicLogarithm(dim)
    
    # Property 1: log_x(x) = 0
    x = torch.tensor([1.2, 0.3, 0.4])
    x = exp_map.project_to_hyperboloid(x)
    result = log_map.forward(x, x)
    assert torch.allclose(result, torch.zeros_like(x), atol=1e-6)
    
    # Property 2: log_x(exp_x(v)) ≈ v for small v
    x = torch.tensor([1.5, 0.5, 0.0])
    v = torch.tensor([0.0, 0.1, 0.1])  # Small tangent vector
    y = exp_map.forward(x, v)
    v_recovered = log_map.forward(x, y)
    assert torch.allclose(v_recovered, v, atol=1e-5)
    
    # Property 3: Distance preservation
    x = torch.tensor([1.2, 0.3, 0.4])
    y = torch.tensor([1.5, 0.5, 0.0])
    v = log_map.forward(x, y)
    
    # Normalize to exact distance
    v_norm = torch.sqrt(torch.abs(exp_map.minkowski_inner(v, v)))  # Use proper Minkowski norm
    dist = torch.acosh(-exp_map.minkowski_inner(x, y))
    assert torch.allclose(v_norm, dist, atol=1e-6)

def test_exp_log_inverse():
    """Test inverse relationship between exponential and logarithm maps."""
    dim = 3
    exp_map = HyperbolicExponential(dim)
    log_map = HyperbolicLogarithm(dim)
    
    # Test points
    x = torch.tensor([1.2, 0.3, 0.4])
    x = exp_map.project_to_hyperboloid(x)
    
    # Small tangent vectors
    vectors = [
        torch.tensor([0.0, 0.1, 0.1]),
        torch.tensor([0.0, -0.1, 0.2]),
        torch.tensor([0.0, 0.2, -0.1])
    ]
    
    for v in vectors:
        # Test exp(log(y)) = y
        y = exp_map.forward(x, v)
        v_recovered = log_map.forward(x, y)
        y_recovered = exp_map.forward(x, v_recovered)
        
        assert torch.allclose(y, y_recovered, atol=1e-5)
        assert torch.allclose(v, v_recovered, atol=1e-5)
