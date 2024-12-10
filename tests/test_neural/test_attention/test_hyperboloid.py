import torch
import pytest
from src.core.attention.geometric import HyperbolicExponential

def test_project_to_hyperboloid():
    """Test projection onto hyperboloid."""
    dim = 3
    exp_map = HyperbolicExponential(dim)
    
    # Test case 1: Point near the hyperboloid
    x = torch.tensor([1.1, 0.3, 0.4])
    x_proj = exp_map.project_to_hyperboloid(x)
    
    # Check properties of projected point
    assert not torch.any(torch.isnan(x_proj))
    assert x_proj[0] >= 1.0 + 1e-7  # Time component ≥ 1
    
    # Verify it's on the hyperboloid: -t² + x² + y² = -1
    inner = exp_map.minkowski_inner(x_proj, x_proj)
    assert torch.allclose(inner, torch.tensor(-1.0), atol=1e-6)
    
    # Test case 2: Point far from hyperboloid
    x = torch.tensor([2.0, 1.5, 1.5])
    x_proj = exp_map.project_to_hyperboloid(x)
    assert not torch.any(torch.isnan(x_proj))
    inner = exp_map.minkowski_inner(x_proj, x_proj)
    assert torch.allclose(inner, torch.tensor(-1.0), atol=1e-6)
    
    # Test case 3: Point with very small components
    x = torch.tensor([1.0 + 1e-8, 1e-8, 1e-8])
    x_proj = exp_map.project_to_hyperboloid(x)
    assert not torch.any(torch.isnan(x_proj))
    inner = exp_map.minkowski_inner(x_proj, x_proj)
    assert torch.allclose(inner, torch.tensor(-1.0), atol=1e-6)
    
    # Test case 4: Batch of points
    x_batch = torch.stack([
        torch.tensor([1.1, 0.3, 0.4]),
        torch.tensor([2.0, 1.5, 1.5])
    ])
    x_proj_batch = exp_map.project_to_hyperboloid(x_batch)
    assert not torch.any(torch.isnan(x_proj_batch))
    assert x_proj_batch.shape == (2, 3)
    
    # Check all projected points satisfy hyperboloid equation
    inner_batch = exp_map.minkowski_inner(x_proj_batch, x_proj_batch)
    assert torch.allclose(inner_batch, torch.tensor([-1.0, -1.0]), atol=1e-6)

def test_hyperboloid_projection_properties():
    """Test mathematical properties of hyperboloid projection."""
    dim = 3
    exp_map = HyperbolicExponential(dim)
    
    # Property 1: Projecting an already projected point should be identity
    x = torch.tensor([1.1, 0.3, 0.4])
    x_proj1 = exp_map.project_to_hyperboloid(x)
    x_proj2 = exp_map.project_to_hyperboloid(x_proj1)
    assert torch.allclose(x_proj1, x_proj2, atol=1e-6)
    
    # Property 2: Scaling space components should preserve hyperboloid constraint
    x = torch.tensor([1.2, 0.4, 0.3])
    x_proj = exp_map.project_to_hyperboloid(x)
    scaled_x = torch.tensor([1.2, 0.8, 0.6])  # Scale space components by 2
    scaled_proj = exp_map.project_to_hyperboloid(scaled_x)
    
    inner1 = exp_map.minkowski_inner(x_proj, x_proj)
    inner2 = exp_map.minkowski_inner(scaled_proj, scaled_proj)
    assert torch.allclose(inner1, inner2, atol=1e-6)
    
    # Property 3: Time component should always be largest in magnitude
    x = torch.tensor([0.5, 2.0, 2.0])  # Space components larger than time
    x_proj = exp_map.project_to_hyperboloid(x)
    assert x_proj[0] > torch.norm(x_proj[1:])  # Time component dominates
