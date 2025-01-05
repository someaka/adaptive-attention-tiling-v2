import torch
import pytest
from src.core.attention.geometric import HyperbolicExponential, HyperbolicLogarithm

def print_test_case(name: str, **values):
    """Print test case details."""
    print(f"\n=== {name} ===")
    for key, val in values.items():
        if isinstance(val, torch.Tensor):
            print(f"{key}:")
            print(f"  Shape: {val.shape}")
            print(f"  Values: {val}")
            print(f"  Dtype: {val.dtype}")
            if len(val.shape) > 0:
                print(f"  Norm (L2): {torch.norm(val)}")
                if val.shape[-1] > 1:
                    print(f"  Time component: {val[..., 0]}")
                    print(f"  Space components: {val[..., 1:]}")
        else:
            print(f"{key}: {val}")
    print("=" * 50)

def verify_hyperboloid_constraint(exp_map: HyperbolicExponential, x: torch.Tensor, tag: str, atol: float = 1e-6):
    """Verify point lies on hyperboloid."""
    constraint = exp_map.minkowski_inner(x, x) + 1.0
    print(f"\nHyperboloid constraint check ({tag}):")
    print(f"  Constraint value: {constraint}")
    print(f"  Time component: {x[..., 0]}")
    print(f"  Space components norm: {torch.norm(x[..., 1:])}")
    assert torch.allclose(constraint, torch.zeros_like(constraint), atol=atol), \
        f"Point {tag} not on hyperboloid: {constraint}"
    assert torch.all(x[..., 0] > 0), f"Time component not positive for {tag}: {x[..., 0]}"

def verify_tangent_space(exp_map: HyperbolicExponential, x: torch.Tensor, v: torch.Tensor, tag: str, atol: float = 1e-6):
    """Verify vector lies in tangent space."""
    inner = exp_map.minkowski_inner(x, v)
    print(f"\nTangent space check ({tag}):")
    print(f"  Inner product: {inner}")
    print(f"  Vector norm: {torch.norm(v)}")
    assert torch.allclose(inner, torch.zeros_like(inner), atol=atol), \
        f"Vector {tag} not in tangent space: {inner}"

def test_logarithm_map():
    """Test logarithm map computation with enhanced precision checks."""
    # Initialize with double precision for better numerical stability
    dim = 3
    exp_map = HyperbolicExponential(dim, dtype=torch.float64)
    log_map = HyperbolicLogarithm(dim, dtype=torch.float64)
    
    # Test case 1: Same point
    print_test_case("Same Point Test",
        description="Testing log_x(x) = 0 property",
        x=torch.tensor([1.2, 0.3, 0.4], dtype=torch.float64)
    )
    x = torch.tensor([1.2, 0.3, 0.4], dtype=torch.float64)
    x = exp_map.project_to_hyperboloid(x)
    verify_hyperboloid_constraint(exp_map, x, "base point")
    
    result = log_map.forward(x, x)
    print_test_case("Same Point Result",
        result=result,
        norm=torch.norm(result)
    )
    verify_tangent_space(exp_map, x, result, "zero vector")
    assert torch.allclose(result, torch.zeros_like(x), atol=1e-8)
    
    # Test case 2: Close points
    print_test_case("Close Points Test",
        description="Testing stability for nearby points",
        x=torch.tensor([1.2, 0.3, 0.4], dtype=torch.float64),
        y=torch.tensor([1.2, 0.3 + 1e-8, 0.4], dtype=torch.float64)
    )
    x = torch.tensor([1.2, 0.3, 0.4], dtype=torch.float64)
    y = torch.tensor([1.2, 0.3 + 1e-8, 0.4], dtype=torch.float64)
    x = exp_map.project_to_hyperboloid(x)
    y = exp_map.project_to_hyperboloid(y)
    verify_hyperboloid_constraint(exp_map, x, "base point")
    verify_hyperboloid_constraint(exp_map, y, "target point")
    
    result = log_map.forward(x, y)
    print_test_case("Close Points Result",
        result=result,
        norm=torch.norm(result)
    )
    verify_tangent_space(exp_map, x, result, "small vector")
    assert not torch.any(torch.isnan(result))
    assert torch.norm(result) < 1e-7
    
    # Test case 3: Normal points
    print_test_case("Normal Points Test",
        description="Testing standard case",
        x=torch.tensor([1.5, 0.5, 0.0], dtype=torch.float64),
        y=torch.tensor([1.5, 0.7, 0.2], dtype=torch.float64)
    )
    x = torch.tensor([1.5, 0.5, 0.0], dtype=torch.float64)
    y = torch.tensor([1.5, 0.7, 0.2], dtype=torch.float64)
    x = exp_map.project_to_hyperboloid(x)
    y = exp_map.project_to_hyperboloid(y)
    verify_hyperboloid_constraint(exp_map, x, "base point")
    verify_hyperboloid_constraint(exp_map, y, "target point")
    
    result = log_map.forward(x, y)
    print_test_case("Normal Points Result",
        result=result,
        norm=torch.norm(result)
    )
    verify_tangent_space(exp_map, x, result, "result vector", atol=1e-8)
    assert not torch.any(torch.isnan(result))

def test_logarithm_map_properties():
    """Test mathematical properties of logarithm map with enhanced precision."""
    dim = 3
    exp_map = HyperbolicExponential(dim, dtype=torch.float64)
    log_map = HyperbolicLogarithm(dim, dtype=torch.float64)
    
    # Property 1: log_x(x) = 0
    print_test_case("Zero Vector Property",
        description="Testing log_x(x) = 0",
        x=torch.tensor([1.2, 0.3, 0.4], dtype=torch.float64)
    )
    x = torch.tensor([1.2, 0.3, 0.4], dtype=torch.float64)
    x = exp_map.project_to_hyperboloid(x)
    verify_hyperboloid_constraint(exp_map, x, "base point")
    
    result = log_map.forward(x, x)
    print_test_case("Zero Vector Result",
        result=result,
        norm=torch.norm(result)
    )
    verify_tangent_space(exp_map, x, result, "zero vector")
    assert torch.allclose(result, torch.zeros_like(x), atol=1e-8)
    
    # Property 2: log_x(exp_x(v)) ≈ v for small v
    print_test_case("Inverse Property",
        description="Testing log_x(exp_x(v)) = v",
        x=torch.tensor([1.5, 0.5, 0.0], dtype=torch.float64),
        v=torch.tensor([0.0, 0.1, 0.1], dtype=torch.float64)
    )
    x = torch.tensor([1.5, 0.5, 0.0], dtype=torch.float64)
    v = torch.tensor([0.0, 0.1, 0.1], dtype=torch.float64)  # Small tangent vector
    x = exp_map.project_to_hyperboloid(x)
    v = exp_map.project_to_tangent(x, v)
    verify_hyperboloid_constraint(exp_map, x, "base point")
    verify_tangent_space(exp_map, x, v, "initial vector")
    
    y = exp_map.forward(x, v)
    verify_hyperboloid_constraint(exp_map, y, "exponential point")
    
    v_recovered = log_map.forward(x, y)
    print_test_case("Recovered Vector",
        v_original=v,
        v_recovered=v_recovered,
        difference=v - v_recovered,
        max_diff=torch.max(torch.abs(v - v_recovered))
    )
    verify_tangent_space(exp_map, x, v_recovered, "recovered vector")
    assert torch.allclose(v_recovered, v, atol=1e-7)
    
    # Property 3: Distance preservation
    print_test_case("Distance Preservation",
        description="Testing ‖log_x(y)‖ = d(x,y)",
        x=torch.tensor([1.2, 0.3, 0.4], dtype=torch.float64),
        y=torch.tensor([1.5, 0.5, 0.0], dtype=torch.float64)
    )
    x = torch.tensor([1.2, 0.3, 0.4], dtype=torch.float64)
    y = torch.tensor([1.5, 0.5, 0.0], dtype=torch.float64)
    x = exp_map.project_to_hyperboloid(x)
    y = exp_map.project_to_hyperboloid(y)
    verify_hyperboloid_constraint(exp_map, x, "base point")
    verify_hyperboloid_constraint(exp_map, y, "target point")
    
    v = log_map.forward(x, y)
    verify_tangent_space(exp_map, x, v, "logarithm vector")
    
    # Compute distances
    v_norm = torch.sqrt(torch.abs(exp_map.minkowski_inner(v, v)))
    dist = torch.acosh(-exp_map.minkowski_inner(x, y))
    print_test_case("Distance Check",
        vector_norm=v_norm,
        hyperbolic_distance=dist,
        difference=torch.abs(v_norm - dist)
    )
    assert torch.allclose(v_norm, dist, atol=1e-7)

def test_exp_log_inverse():
    """Test inverse relationship between exponential and logarithm maps with enhanced precision."""
    dim = 3
    exp_map = HyperbolicExponential(dim, dtype=torch.float64)
    log_map = HyperbolicLogarithm(dim, dtype=torch.float64)
    
    # Test points
    print_test_case("Base Point",
        x=torch.tensor([1.2, 0.3, 0.4], dtype=torch.float64)
    )
    x = torch.tensor([1.2, 0.3, 0.4], dtype=torch.float64)
    x = exp_map.project_to_hyperboloid(x)
    verify_hyperboloid_constraint(exp_map, x, "base point")
    
    # Small tangent vectors
    vectors = [
        torch.tensor([0.0, 0.1, 0.1], dtype=torch.float64),
        torch.tensor([0.0, -0.1, 0.2], dtype=torch.float64),
        torch.tensor([0.0, 0.2, -0.1], dtype=torch.float64)
    ]
    
    for i, v in enumerate(vectors):
        print_test_case(f"Test Vector {i+1}",
            v_original=v
        )
        v = exp_map.project_to_tangent(x, v)
        verify_tangent_space(exp_map, x, v, f"initial vector {i+1}")
        
        # Test exp(log(y)) = y
        y = exp_map.forward(x, v)
        verify_hyperboloid_constraint(exp_map, y, f"exponential point {i+1}")
        
        v_recovered = log_map.forward(x, y)
        verify_tangent_space(exp_map, x, v_recovered, f"recovered vector {i+1}")
        
        y_recovered = exp_map.forward(x, v_recovered)
        verify_hyperboloid_constraint(exp_map, y_recovered, f"recovered point {i+1}")
        
        print_test_case(f"Recovery Check {i+1}",
            y_original=y,
            y_recovered=y_recovered,
            y_difference=y - y_recovered,
            v_original=v,
            v_recovered=v_recovered,
            v_difference=v - v_recovered
        )
        
        assert torch.allclose(y, y_recovered, atol=1e-7)
        assert torch.allclose(v, v_recovered, atol=1e-7)
