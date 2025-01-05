import torch
import pytest
from src.core.attention.geometric import HyperbolicExponential, GeometricStructures

def print_test_case(name: str, **values):
    """Print test case information with enhanced numerical analysis."""
    print(f"\n=== {name} ===")
    print("=" * 50)
    for key, val in values.items():
        if isinstance(val, torch.Tensor):
            print(f"{key}:")
            print(f"  Shape: {val.shape}")
            print(f"  Values: {val}")
            print(f"  Device: {val.device}")
            print(f"  Dtype: {val.dtype}")
            if len(val.shape) > 0:
                print(f"  Norm (L2): {torch.norm(val)}")
                print(f"  Norm (L∞): {torch.max(torch.abs(val))}")
                if val.shape[-1] > 1:
                    print(f"  Max: {torch.max(val)}")
                    print(f"  Min: {torch.min(val)}")
                    print(f"  Mean: {torch.mean(val)}")
                    print(f"  Std: {torch.std(val)}")
                    if val.shape[-1] >= 2:
                        print(f"  Time component: {val[..., 0]}")
                        print(f"  Space components: {val[..., 1:]}")
        else:
            print(f"{key}: {val}")
    print("=" * 50)

def verify_hyperboloid_constraint(exp_map, point, name):
    """Verify that a point lies on the hyperboloid."""
    # Ensure point has correct shape
    if point.dim() == 1:
        point = point.unsqueeze(0)
    
    inner = exp_map.minkowski_inner(point, point)
    if inner.dim() == 0:
        inner = inner.unsqueeze(0)
    
    print_test_case(f"--- Hyperboloid Constraint Check {name} ---",
        point=point[0],
        components={
            "Time": float(point[0, 0]),
            "Space": point[0, 1:]
        },
        inner_product=float(inner[0]),
        deviation_from_minus_one=float(inner[0] + 1),
        numerical_properties={
            "Device": point.device,
            "Dtype": point.dtype,
            "Max abs component": float(point.abs().max()),
            "Min abs component": float(point.abs().min()),
            "L2 norm": float(point.norm()),
            "Mean": float(point.mean()),
            "Std": float(point.std())
        },
        tolerances={
            "Base": 1e-6,
            "Adaptive": float(1e-6 * (1 + point[0, 0].abs()))
        }
    )
    tol = 1e-6 * (1 + point[0, 0].abs())
    assert torch.all(torch.abs(inner + 1) < tol), \
        f"Point {name} does not satisfy hyperboloid constraint: {inner} ≠ -1"

def verify_tangent_space(exp_map, point, vector, name=""):
    """Verify that a vector lies in the tangent space at a point."""
    # Ensure both point and vector have batch dimension
    if point.dim() == 1:
        point = point.unsqueeze(0)
    if vector.dim() == 1:
        vector = vector.unsqueeze(0)
    
    # Compute inner product
    inner = exp_map.minkowski_inner(point, vector)
    if inner.dim() == 0:
        inner = inner.unsqueeze(0)
    
    # Print test case details
    print_test_case(f"--- Tangent Space Check {name} ---",
        point=point[0],
        vector=vector[0],
        inner_product=inner[0],
        numerical_properties={
            "Point norm": float(point.norm()),
            "Vector norm": float(vector.norm()),
            "Max component": float(vector.abs().max())
        }
    )
    
    # Verify orthogonality with adaptive tolerance
    tol = 1e-6 * (1 + point.norm() * vector.norm())
    assert torch.all(torch.abs(inner) < tol), \
        f"Vector {name} not in tangent space: inner product with point = {inner} ≠ 0"

def test_exponential_map():
    """Test exponential map implementation."""
    # Initialize with double precision for better numerical stability
    exp_map = HyperbolicExponential(dim=2, dtype=torch.float64)
    geom = GeometricStructures(dim=2, num_heads=1, manifold_type="hyperbolic", curvature=-1.0)
    
    # Create test points and vectors with correct dimensions
    x = torch.tensor([1.7321, 1.0, 1.0], dtype=torch.float64)  # Point on hyperboloid
    v = torch.tensor([6.9282, 6.0, 6.0], dtype=torch.float64)  # Tangent vector
    
    # Add batch dimension
    x = x.unsqueeze(0)  # Shape: [1, 3]
    v = v.unsqueeze(0)  # Shape: [1, 3]
    
    # Project point to hyperboloid and vector to tangent space
    x = exp_map.project_to_hyperboloid(x)
    v = exp_map.project_to_tangent(x, v)
    
    # Verify point is on hyperboloid and vector is in tangent space
    verify_hyperboloid_constraint(exp_map, x, "initial point")
    verify_tangent_space(exp_map, x, v, "initial vector")
    
    # Apply exponential map
    result = exp_map(x, v)
    
    # Print test case details
    print("\n=== Large Vector Result ===")
    print_test_case("Large Vector Result", result=result[0])  # Remove batch dim for printing
    
    # Verify result is on hyperboloid
    verify_hyperboloid_constraint(exp_map, result, "large vector result")
    
    # Test with batch input
    x_batch = torch.tensor([
        [1.2, 0.3, 0.4],
        [1.5, 0.5, 0.0]
    ], dtype=torch.float64)
    
    v_batch = torch.tensor([
        [0.0, 0.1, 0.1],
        [0.0, 0.2, 0.3]
    ], dtype=torch.float64)
    
    print("\n=== Batch Test ===")
    print_test_case("Batch Input", x=x_batch, v=v_batch)
    
    # Project batch points to hyperboloid and vectors to tangent space
    x_batch = exp_map.project_to_hyperboloid(x_batch)
    v_batch = exp_map.project_to_tangent(x_batch, v_batch)
    
    # Apply exponential map to batch
    result_batch = exp_map(x_batch, v_batch)
    
    # Verify each point in batch
    for i in range(result_batch.shape[0]):
        verify_hyperboloid_constraint(exp_map, result_batch[i:i+1], f"batch point {i}")
        
    # Test zero vector case
    v_zero = torch.zeros_like(x)
    result_zero = exp_map(x, v_zero)
    
    print("\n=== Zero Vector Test ===")
    print_test_case("Zero Vector Test",
        x=x[0],  # Remove batch dim for printing
        v=v_zero[0],
        result=result_zero[0],
        difference=(result_zero - x)[0],
        max_diff=torch.max(torch.abs(result_zero - x))
    )
    
    assert torch.allclose(result_zero, x, atol=1e-6)
    verify_hyperboloid_constraint(exp_map, result_zero, "zero vector result")
    
    # Property 2: Testing small vectors for numerical stability
    x_small = torch.tensor([1.2, 0.3, 0.4], dtype=torch.float64).unsqueeze(0)
    v_small = torch.tensor([0.0, 1e-7, 1e-7], dtype=torch.float64).unsqueeze(0)
    
    print("\n=== Small Vector Test ===")
    print_test_case("Small Vector Test",
        x=x_small[0],  # Remove batch dim for printing
        v=v_small[0],
        description="Testing stability with very small vectors"
    )
    
    # Project small vector to tangent space
    x_small = exp_map.project_to_hyperboloid(x_small)
    v_small = exp_map.project_to_tangent(x_small, v_small)
    verify_tangent_space(exp_map, x_small, v_small, "small vector")
    
    result_small = exp_map(x_small, v_small)
    print_test_case("Small Vector Result",
        result=result_small[0],  # Remove batch dim for printing
        distance=geom.compute_geodesic_distance(x_small, result_small),
        vector_norm=torch.sqrt(torch.abs(exp_map.minkowski_inner(v_small, v_small)))
    )
    
    # Property 3: Testing vector scaling properties
    print("\n=== Vector Scaling Test ===")
    print_test_case("Vector Scaling Test",
        description="Testing v → 2v scaling property"
    )
    
    # Create a small tangent vector for better numerical stability
    x_scale = torch.tensor([1.2, 0.3, 0.4], dtype=torch.float64).unsqueeze(0)
    v_base = torch.tensor([0.0, 0.01, 0.01], dtype=torch.float64).unsqueeze(0)
    
    # Project to hyperboloid and tangent space
    x_scale = exp_map.project_to_hyperboloid(x_scale)
    v_base = exp_map.project_to_tangent(x_scale, v_base)
    verify_tangent_space(exp_map, x_scale, v_base, "base vector")
    
    # Scale the vector
    v1 = v_base  # Original vector
    v2 = 2 * v_base  # Double the vector
    verify_tangent_space(exp_map, x_scale, v1, "v1")
    verify_tangent_space(exp_map, x_scale, v2, "v2")
    
    # Apply exponential map to both vectors
    result1 = exp_map(x_scale, v1)
    result2 = exp_map(x_scale, v2)
    
    # Verify results are on hyperboloid
    verify_hyperboloid_constraint(exp_map, result1, "result1")
    verify_hyperboloid_constraint(exp_map, result2, "result2")
    
    # Compute distances
    dist1 = geom.compute_geodesic_distance(x_scale, result1)
    dist2 = geom.compute_geodesic_distance(x_scale, result2)
    print_test_case("Distance Values",
        dist1=dist1,
        dist2=dist2,
        twice_dist1=2*dist1,
        diff=torch.abs(dist2 - 2*dist1),
        relative_error=(dist2 - 2*dist1)/(2*dist1)
    )
    
    # The scaling should be exact up to numerical precision
    scaling_tol = 1e-10  # Tighter tolerance for scaling property
    assert torch.abs(dist2 - 2 * dist1) < scaling_tol, \
        f"Distance scaling error: {torch.abs(dist2 - 2 * dist1)} > {scaling_tol}"
    
    # Additional property: Distance should equal vector norm
    v1_norm = torch.sqrt(torch.abs(exp_map.minkowski_inner(v1, v1)))
    v2_norm = torch.sqrt(torch.abs(exp_map.minkowski_inner(v2, v2)))
    print_test_case("Vector-Distance Comparison",
        v1_norm=v1_norm,
        dist1=dist1,
        v1_diff=torch.abs(v1_norm - dist1),
        v1_relative_error=(v1_norm - dist1)/dist1,
        v2_norm=v2_norm,
        dist2=dist2,
        v2_diff=torch.abs(v2_norm - dist2),
        v2_relative_error=(v2_norm - dist2)/dist2
    )

def test_exponential_map_properties():
    """Test mathematical properties of exponential map with enhanced precision checks."""
    dim = 3
    exp_map = HyperbolicExponential(dim)
    geom = GeometricStructures(dim=dim, num_heads=1, manifold_type="hyperbolic", curvature=-1.0)
    
    # Property 1: exp_x(0) = x
    print_test_case("Zero Vector Test",
        x=torch.tensor([1.2, 0.3, 0.4], dtype=torch.float64),  # Using double precision
        description="Testing exp_x(0) = x property"
    )
    x = torch.tensor([1.2, 0.3, 0.4], dtype=torch.float64)
    x = exp_map.project_to_hyperboloid(x)
    verify_hyperboloid_constraint(exp_map, x, "initial point")
    
    v_zero = torch.zeros_like(x)
    result = exp_map(x, v_zero)
    print_test_case("Zero Vector Result", 
        result=result,
        difference=result - x,
        max_diff=torch.max(torch.abs(result - x))
    )
    
    assert torch.allclose(result, x, atol=1e-6)
    verify_hyperboloid_constraint(exp_map, result, "zero vector result")
    
    # Property 2: Testing small vectors for numerical stability
    print_test_case("Small Vector Stability Test",
        x=torch.tensor([1.2, 0.3, 0.4], dtype=torch.float64),
        v=torch.tensor([0.0, 1e-7, 1e-7], dtype=torch.float64),
        description="Testing stability with very small vectors"
    )
    v_small = torch.tensor([0.0, 1e-7, 1e-7], dtype=torch.float64)
    v_small = exp_map.project_to_tangent(x, v_small)
    verify_tangent_space(exp_map, x, v_small, "small vector")
    
    result_small = exp_map(x, v_small)
    print_test_case("Small Vector Result",
        result=result_small,
        distance=geom.compute_geodesic_distance(x.unsqueeze(0), result_small.unsqueeze(0)),
        vector_norm=torch.sqrt(torch.abs(exp_map.minkowski_inner(v_small.unsqueeze(0), v_small.unsqueeze(0))))
    )
    
    # Property 3: Testing vector scaling properties
    print_test_case("Vector Scaling Test",
        description="Testing v → 2v scaling property"
    )
    
    # Create a small tangent vector for better numerical stability
    v = torch.tensor([0.0, 0.01, 0.01], dtype=torch.float64)
    v = exp_map.project_to_tangent(x, v)
    verify_tangent_space(exp_map, x, v, "base vector")
    
    # Scale the vector
    v1 = v  # Original vector
    v2 = 2 * v  # Double the vector
    verify_tangent_space(exp_map, x, v2, "scaled vector")
    
    print_test_case("Vector Norms",
        v1_norm=torch.sqrt(torch.abs(exp_map.minkowski_inner(v1.unsqueeze(0), v1.unsqueeze(0)))),
        v2_norm=torch.sqrt(torch.abs(exp_map.minkowski_inner(v2.unsqueeze(0), v2.unsqueeze(0)))),
        ratio=torch.sqrt(torch.abs(exp_map.minkowski_inner(v2.unsqueeze(0), v2.unsqueeze(0)))) / 
              torch.sqrt(torch.abs(exp_map.minkowski_inner(v1.unsqueeze(0), v1.unsqueeze(0))))
    )
    
    # Compute exponential maps
    result1 = exp_map.forward(x, v1)
    result2 = exp_map.forward(x, v2)
    print_test_case("Scaling Results",
        result1=result1,
        result2=result2,
        difference_norm=torch.norm(result2 - result1)
    )
    
    verify_hyperboloid_constraint(exp_map, result1, "result1")
    verify_hyperboloid_constraint(exp_map, result2, "result2")
    
    # Compute distances using GeometricStructures
    dist1 = geom.compute_geodesic_distance(x.unsqueeze(0), result1.unsqueeze(0))
    dist2 = geom.compute_geodesic_distance(x.unsqueeze(0), result2.unsqueeze(0))
    print_test_case("Distance Values",
        dist1=dist1,
        dist2=dist2,
        twice_dist1=2*dist1,
        diff=torch.abs(dist2 - 2*dist1),
        relative_error=(dist2 - 2*dist1)/(2*dist1)
    )
    
    # The scaling should be exact up to numerical precision
    scaling_tol = 1e-10  # Tighter tolerance for scaling property
    assert torch.abs(dist2 - 2 * dist1) < scaling_tol, \
        f"Distance scaling error: {torch.abs(dist2 - 2 * dist1)} > {scaling_tol}"
    
    # Additional property: Distance should equal vector norm
    v1_norm = torch.sqrt(torch.abs(exp_map.minkowski_inner(v1.unsqueeze(0), v1.unsqueeze(0))))
    v2_norm = torch.sqrt(torch.abs(exp_map.minkowski_inner(v2.unsqueeze(0), v2.unsqueeze(0))))
    print_test_case("Vector-Distance Comparison",
        v1_norm=v1_norm,
        dist1=dist1,
        v1_diff=torch.abs(v1_norm - dist1),
        v1_relative_error=(v1_norm - dist1)/dist1,
        v2_norm=v2_norm,
        dist2=dist2,
        v2_diff=torch.abs(v2_norm - dist2),
        v2_relative_error=(v2_norm - dist2)/dist2
    )
    
    # Check with adaptive tolerance based on magnitude
    base_tol = 1e-10  # Tighter base tolerance
    v1_tol = base_tol * (1 + v1_norm)
    v2_tol = base_tol * (1 + v2_norm)
    
    assert torch.abs(v1_norm - dist1) < v1_tol, \
        f"v1 norm-distance mismatch: diff={torch.abs(v1_norm - dist1)}, tol={v1_tol}"
    assert torch.abs(v2_norm - dist2) < v2_tol, \
        f"v2 norm-distance mismatch: diff={torch.abs(v2_norm - dist2)}, tol={v2_tol}"
    
    # Property 4: Testing transitivity of distances
    result_mid = exp_map.forward(x, 1.5 * v1)
    dist_mid = geom.compute_geodesic_distance(x.unsqueeze(0), result_mid.unsqueeze(0))
    print_test_case("Distance Transitivity",
        dist_mid=dist_mid,
        expected_mid=1.5*dist1,
        mid_diff=torch.abs(dist_mid - 1.5*dist1),
        mid_relative_error=(dist_mid - 1.5*dist1)/(1.5*dist1)
    )
    
    # Check transitivity with tight tolerance
    trans_tol = 1e-10
    assert torch.abs(dist_mid - 1.5*dist1) < trans_tol, \
        f"Distance transitivity error: {torch.abs(dist_mid - 1.5*dist1)} > {trans_tol}"

def test_project_to_hyperboloid():
    """Test the projection to hyperboloid functionality."""
    dim = 3
    exp_map = HyperbolicExponential(dim, dtype=torch.float64)

    # Test cases
    test_points = [
        torch.tensor([1.0, 0.1, 0.1], dtype=torch.float64),
        torch.tensor([2.0, 1.0, 1.5], dtype=torch.float64),
        torch.tensor([1.0, 1e-7, 1e-7], dtype=torch.float64)
    ]

    for x in test_points:
        print(f"\n=== Testing projection: {get_point_description(x)} ===")
        print("Input point:", x)
        print("Input properties:")
        print(f"  Shape: {x.shape}")
        print(f"  Dtype: {x.dtype}")
        print(f"  Norm: {torch.norm(x)}")
        if len(x.shape) > 0:
            print(f"  Time component: {x[..., 0]}")
            print(f"  Space components: {x[..., 1:]}")
            print(f"  Space norm: {torch.norm(x[..., 1:])}")

        # Project point
        result = exp_map.project_to_hyperboloid(x)

        # Compute constraint value
        constraint = exp_map.minkowski_inner(result, result) + 1.0

        print("\nProjection result:", result)
        print("Result properties:")
        print(f"  Shape: {result.shape}")
        print(f"  Dtype: {result.dtype}")
        print(f"  Norm: {torch.norm(result)}")
        print(f"  Time component: {result[..., 0]}")
        print(f"  Space components: {result[..., 1:]}")
        print(f"  Space norm: {torch.norm(result[..., 1:])}")
        print(f"  Constraint value: {constraint}")

        # Determine tolerance based on input and result magnitudes
        space_norm = torch.norm(x[..., 1:])
        result_norm = torch.norm(result)
        base_tol = 1e-8  # Increased base tolerance
        scale_factor = max(1e-8, float(space_norm))  # Minimum scale factor
        adaptive_tol = base_tol * (1.0 + float(result_norm))  # Scale with result magnitude

        # Verify properties with adaptive precision
        assert result.dtype == torch.float64, "Result should maintain double precision"
        assert result[..., 0] > 0, "Time component should be positive"
        assert torch.abs(constraint) < adaptive_tol, f"Constraint violation: {constraint} (tolerance: {adaptive_tol})"

def get_point_description(x):
    """Helper function to get a descriptive name for test points."""
    space_norm = torch.norm(x[..., 1:])
    if space_norm < 1e-6:
        return "Point with very small components"
    elif space_norm < 0.5:
        return "Point near origin"
    else:
        return "Point with larger space components"
