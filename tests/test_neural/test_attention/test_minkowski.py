import torch
import pytest
from src.core.attention.geometric import HyperbolicExponential

def test_minkowski_inner_product():
    """Test Minkowski inner product computation."""
    dim = 3  # 1 time + 2 space dimensions
    exp_map = HyperbolicExponential(dim)
    
    # Test case 1: Basic vectors on hyperboloid
    x = torch.tensor([1.5, 0.5, 0.5])  # A point on upper hyperboloid
    y = torch.tensor([2.0, 1.0, 0.0])  # Another point
    inner = exp_map.minkowski_inner(x, y)
    assert not torch.isnan(inner)
    assert inner <= 1e3 and inner >= -1e3  # Check bounds
    
    # Test case 2: Same vector (should give -1 for normalized vectors)
    x = torch.tensor([2.0, 1.0, 1.0])
    inner_self = exp_map.minkowski_inner(x, x)
    assert not torch.isnan(inner_self)
    assert inner_self < 0  # Should be negative for timelike vectors
    
    # Test case 3: Edge case with small components
    x = torch.tensor([1.0 + 1e-8, 1e-8, 1e-8])
    y = torch.tensor([1.0 + 1e-8, -1e-8, 1e-8])
    inner = exp_map.minkowski_inner(x, y)
    assert not torch.isnan(inner)
    assert torch.abs(inner) < 1e3
    
    # Test case 4: Batch computation
    x_batch = torch.stack([torch.tensor([1.5, 0.5, 0.0]), 
                         torch.tensor([2.0, 1.0, 1.0])])
    y_batch = torch.stack([torch.tensor([1.5, -0.5, 0.0]),
                         torch.tensor([2.0, -1.0, -1.0])])
    inner_batch = exp_map.minkowski_inner(x_batch, y_batch)
    assert inner_batch.shape == (2,)
    assert not torch.any(torch.isnan(inner_batch))
    assert torch.all(inner_batch <= 1e3) and torch.all(inner_batch >= -1e3)

def test_minkowski_inner_properties():
    """Test mathematical properties of Minkowski inner product."""
    dim = 3
    exp_map = HyperbolicExponential(dim)
    
    # Create a point on the hyperboloid with small space components
    x = torch.tensor([1.1, 0.2, 0.3])
    x = exp_map.project_to_hyperboloid(x)
    
    # Create small tangent vectors
    v1 = torch.tensor([0.0, -0.05, 0.02])  # Very small tangent vector
    v2 = torch.tensor([0.0, 0.03, -0.04])  # Very small tangent vector
    
    # Test symmetry of inner product
    inner_v1v2 = exp_map.minkowski_inner(v1, v2)
    inner_v2v1 = exp_map.minkowski_inner(v2, v1)
    assert torch.allclose(inner_v1v2, inner_v2v1, atol=1e-6)
    
    # Project vectors to tangent space at x
    v1_proj = exp_map.project_to_tangent(x, v1)
    v2_proj = exp_map.project_to_tangent(x, v2)
    
    # Verify tangent space projection by checking orthogonality to x
    inner_xv1 = exp_map.minkowski_inner(x, v1_proj)
    inner_xv2 = exp_map.minkowski_inner(x, v2_proj)
    assert torch.allclose(inner_xv1, torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(inner_xv2, torch.tensor(0.0), atol=1e-6)
    
    # Test linearity in tangent space
    # For v1, v2 in TxH: <v1 + v2, w> = <v1, w> + <v2, w>
    w = torch.tensor([0.0, 0.01, -0.02])  # Test vector
    inner_sum = exp_map.minkowski_inner(v1_proj + v2_proj, w)
    sum_inner = exp_map.minkowski_inner(v1_proj, w) + exp_map.minkowski_inner(v2_proj, w)
    assert torch.allclose(inner_sum, sum_inner, atol=1e-6)
    
    # Test scaling in tangent space
    a = 0.5
    inner_scaled = exp_map.minkowski_inner(a * v1_proj, w)
    scaled_inner = a * exp_map.minkowski_inner(v1_proj, w)
    assert torch.allclose(inner_scaled, scaled_inner, atol=1e-6)

def test_minkowski_inner_fundamental_properties():
    """Test fundamental properties of the Minkowski inner product."""
    dim = 3
    exp_map = HyperbolicExponential(dim)
    
    # Test 1: Bilinearity
    x = torch.tensor([1.5, 0.5, 0.3])  # Properly normalized vectors
    y = torch.tensor([1.3, -0.4, 0.2])
    z = torch.tensor([1.2, 0.1, -0.3])
    
    a, b = 2.5, -1.3
    # First argument bilinear
    lhs = exp_map.minkowski_inner(a*x + b*y, z)
    rhs = a*exp_map.minkowski_inner(x, z) + b*exp_map.minkowski_inner(y, z)
    assert torch.allclose(lhs, rhs, atol=1e-6), "First argument should be linear"
    
    # Second argument bilinear
    lhs = exp_map.minkowski_inner(x, a*y + b*z)
    rhs = a*exp_map.minkowski_inner(x, y) + b*exp_map.minkowski_inner(x, z)
    assert torch.allclose(lhs, rhs, atol=1e-6), "Second argument should be linear"
    
    # Test 2: Signature (-,+,+,...)
    # Time component should contribute negatively
    time_vec = torch.tensor([2.0, 0.0, 0.0])  # Normalized timelike vector
    # These are proper spacelike vectors where spatial norm > time component squared
    space_vec1 = torch.tensor([1.0, 2.0, 0.0])  # Vector with spatial component
    space_vec2 = torch.tensor([1.0, 0.0, 2.0])
    
    time_inner = exp_map.minkowski_inner(time_vec, time_vec)
    space1_inner = exp_map.minkowski_inner(space_vec1, space_vec1)
    space2_inner = exp_map.minkowski_inner(space_vec2, space_vec2)
    
    assert time_inner < 0, f"Time-time inner product should be negative, got {time_inner}"
    assert space1_inner > 0, f"Space-space inner product should be positive, got {space1_inner}"
    assert space2_inner > 0, f"Space-space inner product should be positive, got {space2_inner}"
    
    # Test 3: Causal Structure
    # Timelike vector (t² > x² + y²)
    timelike = torch.tensor([2.0, 0.5, 0.5])
    assert exp_map.minkowski_inner(timelike, timelike) < 0, "Timelike vector should have negative norm"
    
    # Spacelike vector (t² < x² + y²)
    spacelike = torch.tensor([1.0, 1.5, 1.5])
    assert exp_map.minkowski_inner(spacelike, spacelike) > 0, "Spacelike vector should have positive norm"
    
    # Lightlike vector (t² = x² + y²)
    # For a lightlike vector, if t=1, then x² + y² should = 1
    lightlike = torch.tensor([1.0, 1/torch.sqrt(torch.tensor(2.0)), 1/torch.sqrt(torch.tensor(2.0))])
    inner_light = exp_map.minkowski_inner(lightlike, lightlike)
    assert torch.abs(inner_light) < 1e-6, f"Lightlike vector should have zero norm, got {inner_light}"
    
    # Test 4: Non-degeneracy (partial test)
    # If <x,y> = 0 for specific linearly independent y's, x should be zero
    zero_vec = torch.zeros(dim)
    basis = [
        torch.tensor([1.5, 0.0, 0.0]),  # Timelike basis vector
        torch.tensor([1.1, 1.0, 0.0]),  # Mixed basis vector
        torch.tensor([1.1, 0.0, 1.0])   # Mixed basis vector
    ]
    
    # Test a vector orthogonal to all basis vectors
    test_vec = torch.tensor([1e-7, 1e-7, 1e-7])
    all_orthogonal = all(torch.abs(exp_map.minkowski_inner(test_vec, b)) < 1e-6 for b in basis)
    if all_orthogonal:
        assert torch.allclose(test_vec, zero_vec, atol=1e-6), \
            "Vector orthogonal to all basis vectors should be zero"

def test_advanced_minkowski_properties():
    """Test advanced properties of Minkowski operations including normalization,
    projection stability, and edge cases."""
    dim = 3
    exp_map = HyperbolicExponential(dim)
    
    # Test 1: Verify normalization to hyperboloid
    x = torch.tensor([2.5, 1.2, 0.8])  # Arbitrary point
    x_norm = exp_map.project_to_hyperboloid(x)
    inner_xx = exp_map.minkowski_inner(x_norm, x_norm)
    assert torch.allclose(inner_xx, torch.tensor(-1.0), atol=1e-6), \
        f"Expected normalized point to have Minkowski norm -1, got {inner_xx}"
    
    # Test 2: Projection stability (projecting already projected vectors)
    v = torch.tensor([0.0, 0.5, -0.3])  # Initial vector
    v_proj1 = exp_map.project_to_tangent(x_norm, v)
    v_proj2 = exp_map.project_to_tangent(x_norm, v_proj1)
    # The second projection should not change the vector significantly
    assert torch.allclose(v_proj1, v_proj2, atol=1e-6), \
        "Projection should be stable under repeated application"
    
    # Test 3: Edge cases with extreme vectors
    # Very large vector
    v_large = torch.tensor([0.0, 1e5, 1e5])
    v_large_proj = exp_map.project_to_tangent(x_norm, v_large)
    inner_large = exp_map.minkowski_inner(x_norm, v_large_proj)
    assert torch.allclose(inner_large, torch.tensor(0.0), atol=1e-6), \
        "Large vector projection should still be orthogonal"
    
    # Very small vector
    v_small = torch.tensor([0.0, 1e-8, -1e-8])
    v_small_proj = exp_map.project_to_tangent(x_norm, v_small)
    inner_small = exp_map.minkowski_inner(x_norm, v_small_proj)
    assert torch.allclose(inner_small, torch.tensor(0.0), atol=1e-6), \
        "Small vector projection should still be orthogonal"
    
    # Test 4: Verify that projection preserves the tangent space structure
    # Two projected vectors and their linear combination should all be orthogonal to x
    v1 = torch.tensor([0.0, 0.3, 0.4])
    v2 = torch.tensor([0.0, -0.2, 0.5])
    v1_proj = exp_map.project_to_tangent(x_norm, v1)
    v2_proj = exp_map.project_to_tangent(x_norm, v2)
    
    # Linear combination
    alpha, beta = 2.5, -1.3
    v_comb = alpha * v1_proj + beta * v2_proj
    inner_comb = exp_map.minkowski_inner(x_norm, v_comb)
    assert torch.allclose(inner_comb, torch.tensor(0.0), atol=1e-6), \
        "Linear combination of tangent vectors should remain in tangent space"
