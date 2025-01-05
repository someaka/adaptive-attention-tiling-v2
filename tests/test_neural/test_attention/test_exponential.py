import torch
import pytest
from src.core.attention.geometric import HyperbolicExponential

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

def verify_hyperboloid_constraint(exp_map: HyperbolicExponential, x: torch.Tensor, tag: str = "") -> torch.Tensor:
    """Verify point lies on hyperboloid with enhanced precision checks.
    
    Checks the constraint ⟨x,x⟩_M = -1 with detailed numerical analysis.
    """
    inner = exp_map.minkowski_inner(x, x)
    deviation = torch.abs(inner + 1)
    
    print(f"\n--- Hyperboloid Constraint Check {tag} ---")
    print("-" * 50)
    print(f"Point: {x}")
    print(f"Components:")
    print(f"  Time: {x[..., 0]}")
    print(f"  Space: {x[..., 1:]}")
    print(f"Inner product: {inner}")
    print(f"Deviation from -1: {deviation}")
    print(f"Numerical Properties:")
    print(f"  Device: {x.device}")
    print(f"  Dtype: {x.dtype}")
    print(f"  Max abs component: {torch.max(torch.abs(x))}")
    print(f"  Min abs component: {torch.min(torch.abs(x))}")
    print(f"  L2 norm: {torch.norm(x)}")
    print(f"  Mean: {torch.mean(x)}")
    print(f"  Std: {torch.std(x)}")
    
    # Stricter tolerance for points near the origin
    base_tol = 1e-6
    adaptive_tol = base_tol * (1 + torch.max(torch.abs(x)).item())
    print(f"Tolerances:")
    print(f"  Base: {base_tol}")
    print(f"  Adaptive: {adaptive_tol}")
    print("-" * 50)
    
    assert deviation <= adaptive_tol, f"Point not on hyperboloid: inner={inner}, deviation={deviation}, tolerance={adaptive_tol}"
    assert x[..., 0] >= 1.0, f"Time component must be >= 1: {x[..., 0]}"
    
    return inner

def verify_tangent_space(exp_map: HyperbolicExponential, x: torch.Tensor, v: torch.Tensor, tag: str = "") -> torch.Tensor:
    """Verify vector lies in tangent space with enhanced precision checks.
    
    Checks the constraint ⟨x,v⟩_M = 0 with detailed numerical analysis.
    """
    inner = exp_map.minkowski_inner(x, v)
    deviation = torch.abs(inner)
    v_norm = exp_map.minkowski_norm(v)
    
    print(f"\n--- Tangent Space Check {tag} ---")
    print("-" * 50)
    print(f"Base Point:")
    print(f"  Values: {x}")
    print(f"  Time: {x[..., 0]}")
    print(f"  Space: {x[..., 1:]}")
    print(f"\nTangent Vector:")
    print(f"  Values: {v}")
    print(f"  Time: {v[..., 0]}")
    print(f"  Space: {v[..., 1:]}")
    print(f"\nConstraints:")
    print(f"  Inner product: {inner}")
    print(f"  Deviation from 0: {deviation}")
    print(f"  Vector norm: {v_norm}")
    print(f"\nNumerical Properties:")
    print(f"  Device: {v.device}")
    print(f"  Dtype: {v.dtype}")
    print(f"  Max abs component: {torch.max(torch.abs(v))}")
    print(f"  Min abs component: {torch.min(torch.abs(v))}")
    print(f"  L2 norm: {torch.norm(v)}")
    print(f"  Mean: {torch.mean(v)}")
    print(f"  Std: {torch.std(v)}")
    
    # Adaptive tolerance based on vector magnitude
    base_tol = 1e-6
    adaptive_tol = base_tol * (1 + v_norm.item())
    print(f"\nTolerances:")
    print(f"  Base: {base_tol}")
    print(f"  Adaptive: {adaptive_tol}")
    print("-" * 50)
    
    assert deviation <= adaptive_tol, f"Vector not in tangent space: inner={inner}, deviation={deviation}, tolerance={adaptive_tol}"
    
    return inner

def test_exponential_map():
    """Test exponential map computation with enhanced precision checks."""
    dim = 3
    exp_map = HyperbolicExponential(dim)
    
    # Test case 1: Small tangent vector
    print_test_case("Small Vector Test",
        x=torch.tensor([1.2, 0.3, 0.4]),
        v=torch.tensor([0.0, 1e-8, 1e-8])
    )
    x = torch.tensor([1.2, 0.3, 0.4])  # Point on hyperboloid
    v = torch.tensor([0.0, 1e-8, 1e-8])  # Small tangent vector
    
    # Verify initial point projection
    x = exp_map.project_to_hyperboloid(x)
    verify_hyperboloid_constraint(exp_map, x, "initial point")
    
    # Project vector to tangent space
    v = exp_map.project_to_tangent(x, v)
    verify_tangent_space(exp_map, x, v, "small vector")
    
    result = exp_map.forward(x, v)
    print_test_case("Small Vector Result", result=result)
    
    # Check properties
    assert not torch.any(torch.isnan(result))
    assert result[0] >= 1.0 + 1e-7  # Time component constraint
    verify_hyperboloid_constraint(exp_map, result, "small vector result")
    
    # Test case 2: Normal tangent vector
    print_test_case("Normal Vector Test",
        x=torch.tensor([1.5, 0.5, 0.0]),
        v=torch.tensor([0.0, 0.3, 0.4])
    )
    x = torch.tensor([1.5, 0.5, 0.0])
    v = torch.tensor([0.0, 0.3, 0.4])
    
    x = exp_map.project_to_hyperboloid(x)
    verify_hyperboloid_constraint(exp_map, x, "initial point")
    
    v = exp_map.project_to_tangent(x, v)
    verify_tangent_space(exp_map, x, v, "normal vector")
    
    result = exp_map.forward(x, v)
    print_test_case("Normal Vector Result", result=result)
    
    assert not torch.any(torch.isnan(result))
    verify_hyperboloid_constraint(exp_map, result, "normal vector result")
    
    # Test case 3: Large tangent vector (should be clamped)
    print_test_case("Large Vector Test",
        x=torch.tensor([2.0, 1.0, 1.0]),
        v=torch.tensor([0.0, 10.0, 10.0])
    )
    x = torch.tensor([2.0, 1.0, 1.0])
    v = torch.tensor([0.0, 10.0, 10.0])
    
    x = exp_map.project_to_hyperboloid(x)
    verify_hyperboloid_constraint(exp_map, x, "initial point")
    
    v = exp_map.project_to_tangent(x, v)
    verify_tangent_space(exp_map, x, v, "large vector")
    
    result = exp_map.forward(x, v)
    print_test_case("Large Vector Result", result=result)
    
    assert not torch.any(torch.isnan(result))
    verify_hyperboloid_constraint(exp_map, result, "large vector result")
    
    # Test case 4: Batch computation
    print_test_case("Batch Test",
        x_batch=torch.stack([
            torch.tensor([1.2, 0.3, 0.4]),
            torch.tensor([1.5, 0.5, 0.0])
        ]),
        v_batch=torch.stack([
            torch.tensor([0.0, 0.1, 0.1]),
            torch.tensor([0.0, 0.2, 0.3])
        ])
    )
    x_batch = torch.stack([
        torch.tensor([1.2, 0.3, 0.4]),
        torch.tensor([1.5, 0.5, 0.0])
    ])
    v_batch = torch.stack([
        torch.tensor([0.0, 0.1, 0.1]),
        torch.tensor([0.0, 0.2, 0.3])
    ])
    
    x_batch = exp_map.project_to_hyperboloid(x_batch)
    for i in range(x_batch.shape[0]):
        verify_hyperboloid_constraint(exp_map, x_batch[i], f"batch point {i}")
    
    v_batch = exp_map.project_to_tangent(x_batch, v_batch)
    for i in range(v_batch.shape[0]):
        verify_tangent_space(exp_map, x_batch[i], v_batch[i], f"batch vector {i}")
    
    result_batch = exp_map.forward(x_batch, v_batch)
    print_test_case("Batch Result", result=result_batch)
    
    assert not torch.any(torch.isnan(result_batch))
    assert result_batch.shape == (2, 3)
    for i in range(result_batch.shape[0]):
        verify_hyperboloid_constraint(exp_map, result_batch[i], f"batch result {i}")

def test_exponential_map_properties():
    """Test mathematical properties of exponential map with enhanced precision checks."""
    dim = 3
    exp_map = HyperbolicExponential(dim)
    
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
        distance=exp_map.compute_distance(x, result_small),
        vector_norm=exp_map.minkowski_norm(v_small)
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
        v1_norm=exp_map.minkowski_norm(v1),
        v2_norm=exp_map.minkowski_norm(v2),
        ratio=exp_map.minkowski_norm(v2) / exp_map.minkowski_norm(v1)
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
    
    # Compute distances using the new method
    dist1 = exp_map.compute_distance(x, result1)
    dist2 = exp_map.compute_distance(x, result2)
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
    v1_norm = exp_map.minkowski_norm(v1)
    v2_norm = exp_map.minkowski_norm(v2)
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
    dist_mid = exp_map.compute_distance(x, result_mid)
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
