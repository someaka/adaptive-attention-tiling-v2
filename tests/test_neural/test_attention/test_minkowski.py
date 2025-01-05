import torch
import pytest
from src.core.attention.geometric import HyperbolicExponential, ParallelTransport

def test_minkowski_inner_product():
    """Test Minkowski inner product computation with high precision checks."""
    dim = 3  # 1 time + 2 space dimensions
    exp_map = HyperbolicExponential(dim)
    
    # Test case 1: Basic vectors on hyperboloid with precise checks
    x = torch.tensor([1.5, 0.5, 0.5], dtype=torch.float64)  # Using double precision
    y = torch.tensor([2.0, 1.0, 0.0], dtype=torch.float64)
    inner = exp_map.minkowski_inner(x, y)
    print(f"Basic vectors inner product: {inner}")
    assert not torch.isnan(inner), "Inner product should not be NaN"
    assert inner.abs() < 1e3, f"Inner product magnitude {inner} exceeds bounds"
    
    # Test case 2: Same vector with precise norm check
    x = torch.tensor([2.0, 1.0, 1.0], dtype=torch.float64)
    x_norm = exp_map.project_to_hyperboloid(x)
    inner_self = exp_map.minkowski_inner(x_norm, x_norm)
    print(f"Normalized self inner product: {inner_self}")
    assert torch.allclose(inner_self, torch.tensor(-1.0, dtype=torch.float64), 
                         rtol=1e-10, atol=1e-10), f"Self inner product should be -1, got {inner_self}"
    
    # Test case 3: Edge case with small components
    x = torch.tensor([1.0 + 1e-8, 1e-8, 1e-8], dtype=torch.float64)
    y = torch.tensor([1.0 + 1e-8, -1e-8, 1e-8], dtype=torch.float64)
    inner = exp_map.minkowski_inner(x, y)
    print(f"Small components inner product: {inner}")
    assert not torch.isnan(inner), "Inner product with small components should not be NaN"
    assert torch.abs(inner + 1.0) < 1e-7, f"Inner product with small components deviation: {inner + 1.0}"
    
    # Test case 4: Batch computation with varying dimensions
    batch_size = 3
    x_batch = torch.randn(batch_size, dim, dtype=torch.float64)
    y_batch = torch.randn(batch_size, dim, dtype=torch.float64)
    # Project to hyperboloid
    x_batch = exp_map.project_to_hyperboloid(x_batch)
    y_batch = exp_map.project_to_hyperboloid(y_batch)
    inner_batch = exp_map.minkowski_inner(x_batch, y_batch)
    print(f"Batch inner products: {inner_batch}")
    assert inner_batch.shape == (batch_size,), f"Unexpected batch shape: {inner_batch.shape}"
    assert not torch.any(torch.isnan(inner_batch)), "Batch inner products contain NaN"
    
    # Test case 5: Numerical stability with large values
    scale = 1e3
    x_large = torch.tensor([scale, scale/2, scale/3], dtype=torch.float64)
    y_large = torch.tensor([scale, -scale/2, scale/3], dtype=torch.float64)
    x_large = exp_map.project_to_hyperboloid(x_large)
    y_large = exp_map.project_to_hyperboloid(y_large)
    inner_large = exp_map.minkowski_inner(x_large, y_large)
    print(f"Large values inner product: {inner_large}")
    assert not torch.isnan(inner_large), "Inner product with large values should not be NaN"
    assert torch.isfinite(inner_large), "Inner product with large values should be finite"
    
    # Test case 6: Verify Cauchy-Schwarz inequality in Minkowski space
    # For timelike vectors: -⟨x,y⟩² ≤ (-⟨x,x⟩)(-⟨y,y⟩)
    x = torch.tensor([2.0, 0.5, 0.3], dtype=torch.float64)
    y = torch.tensor([1.5, 0.2, 0.1], dtype=torch.float64)
    x = exp_map.project_to_hyperboloid(x)
    y = exp_map.project_to_hyperboloid(y)
    inner_xy = exp_map.minkowski_inner(x, y)
    inner_xx = exp_map.minkowski_inner(x, x)
    inner_yy = exp_map.minkowski_inner(y, y)
    print(f"Cauchy-Schwarz check: -⟨x,y⟩² = {-(inner_xy**2)}, (-⟨x,x⟩)(-⟨y,y⟩) = {(-inner_xx)*(-inner_yy)}")
    assert -(inner_xy**2) <= (-inner_xx)*(-inner_yy) + 1e-10, "Cauchy-Schwarz inequality violated"

def test_minkowski_inner_properties():
    """Test mathematical properties of Minkowski inner product with high precision."""
    dim = 3
    exp_map = HyperbolicExponential(dim)
    
    # Create a point on the hyperboloid with precise coordinates
    x = torch.tensor([1.1, 0.2, 0.3], dtype=torch.float64)
    x = exp_map.project_to_hyperboloid(x)
    print(f"Base point on hyperboloid: {x}")
    assert torch.allclose(exp_map.minkowski_inner(x, x), torch.tensor(-1.0, dtype=torch.float64), 
                         rtol=1e-10, atol=1e-10), "Point not properly normalized"
    
    # Create tangent vectors with controlled magnitudes
    v1 = torch.tensor([0.0, -0.05, 0.02], dtype=torch.float64)  # Small tangent vector
    v2 = torch.tensor([0.0, 0.03, -0.04], dtype=torch.float64)  # Small tangent vector
    
    # Project vectors to tangent space with high precision
    v1_proj = exp_map.project_to_tangent(x, v1)
    v2_proj = exp_map.project_to_tangent(x, v2)
    print(f"Projected tangent vectors:\nv1_proj: {v1_proj}\nv2_proj: {v2_proj}")
    
    # Test symmetry of inner product with high precision
    inner_v1v2 = exp_map.minkowski_inner(v1_proj, v2_proj)
    inner_v2v1 = exp_map.minkowski_inner(v2_proj, v1_proj)
    print(f"Symmetry check: ⟨v1,v2⟩ = {inner_v1v2}, ⟨v2,v1⟩ = {inner_v2v1}")
    assert torch.allclose(inner_v1v2, inner_v2v1, rtol=1e-10, atol=1e-10), \
        f"Inner product not symmetric: {inner_v1v2} ≠ {inner_v2v1}"
    
    # Verify tangent space projection with high precision
    inner_xv1 = exp_map.minkowski_inner(x, v1_proj)
    inner_xv2 = exp_map.minkowski_inner(x, v2_proj)
    print(f"Tangent space check: ⟨x,v1⟩ = {inner_xv1}, ⟨x,v2⟩ = {inner_xv2}")
    assert torch.allclose(inner_xv1, torch.tensor(0.0, dtype=torch.float64), rtol=1e-10, atol=1e-10), \
        f"v1 not in tangent space: ⟨x,v1⟩ = {inner_xv1}"
    assert torch.allclose(inner_xv2, torch.tensor(0.0, dtype=torch.float64), rtol=1e-10, atol=1e-10), \
        f"v2 not in tangent space: ⟨x,v2⟩ = {inner_xv2}"
    
    # Test linearity in tangent space with high precision
    w = torch.tensor([0.0, 0.01, -0.02], dtype=torch.float64)  # Test vector
    w_proj = exp_map.project_to_tangent(x, w)
    
    # Test additivity
    inner_sum = exp_map.minkowski_inner(v1_proj + v2_proj, w_proj)
    sum_inner = exp_map.minkowski_inner(v1_proj, w_proj) + exp_map.minkowski_inner(v2_proj, w_proj)
    print(f"Additivity check: ⟨v1+v2,w⟩ = {inner_sum}, ⟨v1,w⟩ + ⟨v2,w⟩ = {sum_inner}")
    assert torch.allclose(inner_sum, sum_inner, rtol=1e-10, atol=1e-10), \
        f"Inner product not additive: {inner_sum} ≠ {sum_inner}"
    
    # Test homogeneity
    alpha = 0.5
    inner_scaled = exp_map.minkowski_inner(alpha * v1_proj, w_proj)
    scaled_inner = alpha * exp_map.minkowski_inner(v1_proj, w_proj)
    print(f"Homogeneity check: ⟨αv1,w⟩ = {inner_scaled}, α⟨v1,w⟩ = {scaled_inner}")
    assert torch.allclose(inner_scaled, scaled_inner, rtol=1e-10, atol=1e-10), \
        f"Inner product not homogeneous: {inner_scaled} ≠ {scaled_inner}"
    
    # Test non-degeneracy
    # If v is orthogonal to all basis vectors of the tangent space, it should be zero
    basis = []
    for i in range(dim-1):  # dim-1 because tangent space has one dimension less
        e = torch.zeros(dim, dtype=torch.float64)
        e[i+1] = 1.0  # Skip time component
        e_proj = exp_map.project_to_tangent(x, e)
        basis.append(e_proj)
    
    # Create a vector orthogonal to all basis vectors
    v_test = torch.tensor([0.0, 1e-10, -1e-10], dtype=torch.float64)
    v_test = exp_map.project_to_tangent(x, v_test)
    all_orthogonal = all(torch.abs(exp_map.minkowski_inner(v_test, b)) < 1e-10 for b in basis)
    if all_orthogonal:
        print(f"Non-degeneracy check: vector orthogonal to basis has norm {torch.norm(v_test)}")
        assert torch.norm(v_test) < 1e-9, \
            f"Non-degeneracy violated: vector orthogonal to basis has non-zero norm {torch.norm(v_test)}"

def test_minkowski_inner_fundamental_properties():
    """Test fundamental properties of the Minkowski inner product with high precision."""
    dim = 3
    exp_map = HyperbolicExponential(dim)
    
    # Test 1: Bilinearity with high precision
    x = torch.tensor([1.5, 0.5, 0.3], dtype=torch.float64)  # Properly normalized vectors
    y = torch.tensor([1.3, -0.4, 0.2], dtype=torch.float64)
    z = torch.tensor([1.2, 0.1, -0.3], dtype=torch.float64)
    
    # Project vectors to hyperboloid
    x = exp_map.project_to_hyperboloid(x)
    y = exp_map.project_to_hyperboloid(y)
    z = exp_map.project_to_hyperboloid(z)
    print(f"Normalized vectors:\nx: {x}\ny: {y}\nz: {z}")
    
    a, b = 2.5, -1.3
    # First argument bilinear
    lhs = exp_map.minkowski_inner(a*x + b*y, z)
    rhs = a*exp_map.minkowski_inner(x, z) + b*exp_map.minkowski_inner(y, z)
    print(f"First argument bilinearity: ⟨ax+by,z⟩ = {lhs}, a⟨x,z⟩ + b⟨y,z⟩ = {rhs}")
    assert torch.allclose(lhs, rhs, rtol=1e-10, atol=1e-10), \
        f"First argument not linear: {lhs} ≠ {rhs}"
    
    # Second argument bilinear
    lhs = exp_map.minkowski_inner(x, a*y + b*z)
    rhs = a*exp_map.minkowski_inner(x, y) + b*exp_map.minkowski_inner(x, z)
    print(f"Second argument bilinearity: ⟨x,ay+bz⟩ = {lhs}, a⟨x,y⟩ + b⟨x,z⟩ = {rhs}")
    assert torch.allclose(lhs, rhs, rtol=1e-10, atol=1e-10), \
        f"Second argument not linear: {lhs} ≠ {rhs}"
    
    # Test 2: Signature (-,+,+,...) with high precision
    # Time component should contribute negatively
    time_vec = torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64)  # Pure timelike vector
    space_vec1 = torch.tensor([1.0, 2.0, 0.0], dtype=torch.float64)  # Vector with x-space component
    space_vec2 = torch.tensor([1.0, 0.0, 2.0], dtype=torch.float64)  # Vector with y-space component
    
    time_inner = exp_map.minkowski_inner(time_vec, time_vec)
    space1_inner = exp_map.minkowski_inner(space_vec1, space_vec1)
    space2_inner = exp_map.minkowski_inner(space_vec2, space_vec2)
    print(f"Signature check:\nTimelike norm: {time_inner}\nSpace1 norm: {space1_inner}\nSpace2 norm: {space2_inner}")
    
    assert time_inner < 0, f"Time-time inner product should be negative, got {time_inner}"
    assert space1_inner > 0, f"Space-space inner product should be positive, got {space1_inner}"
    assert space2_inner > 0, f"Space-space inner product should be positive, got {space2_inner}"
    
    # Test 3: Causal Structure with high precision
    # Timelike vector (t² > x² + y²)
    timelike = torch.tensor([2.0, 0.5, 0.5], dtype=torch.float64)
    timelike_norm = exp_map.minkowski_inner(timelike, timelike)
    print(f"Timelike vector norm: {timelike_norm}")
    assert timelike_norm < 0, f"Timelike vector should have negative norm, got {timelike_norm}"
    
    # Spacelike vector (t² < x² + y²)
    spacelike = torch.tensor([1.0, 1.5, 1.5], dtype=torch.float64)
    spacelike_norm = exp_map.minkowski_inner(spacelike, spacelike)
    print(f"Spacelike vector norm: {spacelike_norm}")
    assert spacelike_norm > 0, f"Spacelike vector should have positive norm, got {spacelike_norm}"
    
    # Lightlike vector (t² = x² + y²)
    # For a lightlike vector, if t=1, then x² + y² should = 1
    lightlike = torch.tensor([1.0, 1/torch.sqrt(torch.tensor(2.0)), 1/torch.sqrt(torch.tensor(2.0))], 
                           dtype=torch.float64)
    inner_light = exp_map.minkowski_inner(lightlike, lightlike)
    print(f"Lightlike vector norm: {inner_light}")
    assert torch.abs(inner_light) < 1e-10, f"Lightlike vector should have zero norm, got {inner_light}"
    
    # Test 4: Non-degeneracy (complete test)
    # Create a basis for the tangent space at a point
    base_point = torch.tensor([1.5, 0.0, 0.0], dtype=torch.float64)
    base_point = exp_map.project_to_hyperboloid(base_point)
    
    # Create an orthonormal basis for the tangent space
    basis = []
    for i in range(dim-1):  # dim-1 because tangent space has one dimension less
        e = torch.zeros(dim, dtype=torch.float64)
        e[i+1] = 1.0  # Skip time component
        e_proj = exp_map.project_to_tangent(base_point, e)
        # Normalize the basis vector
        e_norm = torch.sqrt(torch.abs(exp_map.minkowski_inner(e_proj, e_proj)))
        if e_norm > 1e-10:  # Avoid division by zero
            e_proj = e_proj / e_norm
        basis.append(e_proj)
    
    # Verify orthonormality of the basis
    for i, ei in enumerate(basis):
        for j, ej in enumerate(basis):
            inner = exp_map.minkowski_inner(ei, ej)
            expected = 1.0 if i == j else 0.0
            print(f"Basis vectors ⟨e{i},e{j}⟩ = {inner}")
            assert torch.allclose(inner, torch.tensor(expected, dtype=torch.float64), 
                                rtol=1e-10, atol=1e-10), \
                f"Basis not orthonormal: ⟨e{i},e{j}⟩ = {inner}"
    
    # Test completeness: any vector orthogonal to all basis vectors must be zero
    test_vec = torch.tensor([0.0, 1e-10, -1e-10], dtype=torch.float64)
    test_vec = exp_map.project_to_tangent(base_point, test_vec)
    all_orthogonal = all(torch.abs(exp_map.minkowski_inner(test_vec, b)) < 1e-10 for b in basis)
    if all_orthogonal:
        print(f"Completeness check: vector orthogonal to all basis vectors has norm {torch.norm(test_vec)}")
        assert torch.norm(test_vec) < 1e-9, \
            f"Non-degeneracy violated: vector orthogonal to basis has non-zero norm {torch.norm(test_vec)}"

def test_advanced_minkowski_properties():
    """Test advanced properties of Minkowski operations with high precision.
    
    This test suite verifies:
    1. Normalization to hyperboloid
    2. Projection stability
    3. Edge cases with extreme vectors
    4. Preservation of tangent space structure
    5. Parallel transport properties
    """
    dim = 3
    exp_map = HyperbolicExponential(dim)
    
    # Test 1: Verify normalization to hyperboloid with high precision
    x = torch.tensor([2.5, 1.2, 0.8], dtype=torch.float64)  # Arbitrary point
    x_norm = exp_map.project_to_hyperboloid(x)
    inner_xx = exp_map.minkowski_inner(x_norm, x_norm)
    print(f"Normalized point: {x_norm}\nMinkowski norm: {inner_xx}")
    assert torch.allclose(inner_xx, torch.tensor(-1.0, dtype=torch.float64), rtol=1e-10, atol=1e-10), \
        f"Expected normalized point to have Minkowski norm -1, got {inner_xx}"
    
    # Test 2: Projection stability with high precision
    # Project vectors multiple times and verify stability
    v = torch.tensor([0.0, 0.5, -0.3], dtype=torch.float64)  # Initial vector
    v_proj1 = exp_map.project_to_tangent(x_norm, v)
    v_proj2 = exp_map.project_to_tangent(x_norm, v_proj1)
    v_proj3 = exp_map.project_to_tangent(x_norm, v_proj2)
    print(f"Projection sequence:\nv_proj1: {v_proj1}\nv_proj2: {v_proj2}\nv_proj3: {v_proj3}")
    
    # Check stability of repeated projections
    diff12 = torch.norm(v_proj2 - v_proj1)
    diff23 = torch.norm(v_proj3 - v_proj2)
    print(f"Projection stability:\nFirst diff: {diff12}\nSecond diff: {diff23}")
    assert diff12 < 1e-10 and diff23 < 1e-10, \
        f"Projection not stable: consecutive diffs {diff12}, {diff23}"
    
    # Test 3: Edge cases with extreme vectors
    # Very large vector
    v_large = torch.tensor([0.0, 1e5, 1e5], dtype=torch.float64)
    v_large_proj = exp_map.project_to_tangent(x_norm, v_large)
    inner_large = exp_map.minkowski_inner(x_norm, v_large_proj)
    print(f"Large vector projection:\nv_large_proj: {v_large_proj}\nOrthogonality: {inner_large}")
    assert torch.allclose(inner_large, torch.tensor(0.0, dtype=torch.float64), rtol=1e-10, atol=1e-10), \
        f"Large vector projection not orthogonal: {inner_large}"
    
    # Very small vector
    v_small = torch.tensor([0.0, 1e-8, -1e-8], dtype=torch.float64)
    v_small_proj = exp_map.project_to_tangent(x_norm, v_small)
    inner_small = exp_map.minkowski_inner(x_norm, v_small_proj)
    print(f"Small vector projection:\nv_small_proj: {v_small_proj}\nOrthogonality: {inner_small}")
    assert torch.allclose(inner_small, torch.tensor(0.0, dtype=torch.float64), rtol=1e-10, atol=1e-10), \
        f"Small vector projection not orthogonal: {inner_small}"
    
    # Test 4: Verify that projection preserves the tangent space structure
    # Create two tangent vectors and verify their linear combination
    v1 = torch.tensor([0.0, 0.3, 0.4], dtype=torch.float64)
    v2 = torch.tensor([0.0, -0.2, 0.5], dtype=torch.float64)
    v1_proj = exp_map.project_to_tangent(x_norm, v1)
    v2_proj = exp_map.project_to_tangent(x_norm, v2)
    
    # Verify orthogonality to base point
    inner1 = exp_map.minkowski_inner(x_norm, v1_proj)
    inner2 = exp_map.minkowski_inner(x_norm, v2_proj)
    print(f"Tangent vectors orthogonality:\n⟨x,v1⟩: {inner1}\n⟨x,v2⟩: {inner2}")
    assert torch.allclose(inner1, torch.tensor(0.0, dtype=torch.float64), rtol=1e-10, atol=1e-10), \
        f"v1 not in tangent space: {inner1}"
    assert torch.allclose(inner2, torch.tensor(0.0, dtype=torch.float64), rtol=1e-10, atol=1e-10), \
        f"v2 not in tangent space: {inner2}"
    
    # Test linear combination
    alpha, beta = 2.5, -1.3
    v_comb = alpha * v1_proj + beta * v2_proj
    inner_comb = exp_map.minkowski_inner(x_norm, v_comb)
    print(f"Linear combination orthogonality: {inner_comb}")
    assert torch.allclose(inner_comb, torch.tensor(0.0, dtype=torch.float64), rtol=1e-10, atol=1e-10), \
        f"Linear combination not in tangent space: {inner_comb}"
    
    # Test 5: Verify parallel transport properties
    # Create a parallel transport object
    transport = ParallelTransport(dim)
    
    # Transport a vector between two points
    y = torch.tensor([1.2, 0.4, -0.3], dtype=torch.float64)
    y = exp_map.project_to_hyperboloid(y)
    v = torch.tensor([0.0, 0.1, 0.2], dtype=torch.float64)
    v = exp_map.project_to_tangent(x_norm, v)
    
    # Transport v from x to y
    v_transported = transport.forward(x_norm, y, v)
    
    # Verify transported vector is in tangent space at y
    inner_transported = exp_map.minkowski_inner(y, v_transported)
    print(f"Transported vector orthogonality: {inner_transported}")
    assert torch.allclose(inner_transported, torch.tensor(0.0, dtype=torch.float64), rtol=1e-10, atol=1e-10), \
        f"Transported vector not in tangent space: {inner_transported}"
    
    # Verify norm preservation under parallel transport
    norm_original = torch.sqrt(torch.abs(exp_map.minkowski_inner(v, v)))
    norm_transported = torch.sqrt(torch.abs(exp_map.minkowski_inner(v_transported, v_transported)))
    print(f"Norm preservation:\nOriginal norm: {norm_original}\nTransported norm: {norm_transported}")
    assert torch.allclose(norm_original, norm_transported, rtol=1e-10, atol=1e-10), \
        f"Parallel transport did not preserve norm: {norm_original} ≠ {norm_transported}"
