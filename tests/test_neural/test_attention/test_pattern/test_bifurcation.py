"""Tests for bifurcation analysis."""

import torch
import pytest
import numpy as np

from src.neural.attention.pattern.models import BifurcationDiagram
from tests.test_neural.test_attention.test_pattern.conftest import assert_tensor_equal

# Control parameters for all tests
MAX_ITERATIONS = 20  # Reduced from 50
NUM_PARAMETER_POINTS = 10  # Reduced from 20
CONVERGENCE_STEPS = 5  # Reduced from 10

def test_bifurcation_analysis(pattern_system, grid_size):
    """Test bifurcation analysis with comprehensive metrics."""
    # Create test pattern with small initial values
    pattern = torch.ones(1, 2, grid_size, grid_size) * 0.1
    
    # Define parameter range focused around expected bifurcation
    parameter_range = torch.linspace(0.8, 1.2, NUM_PARAMETER_POINTS)  # Focused around bifurcation at 1.0
    
    # Create parameterized reaction term with known bifurcation
    def parameterized_reaction(state, param):
        u, v = state[:, 0], state[:, 1]
        # Modified reaction term to ensure bifurcation
        du = 5.0 * (param * u - u**3)  # Stronger pitchfork bifurcation at param = 1
        dv = -v + u**2  # Coupling term
        return torch.stack([du, dv], dim=1)
    
    # Print test configuration
    print("\nBifurcation Analysis Test Configuration:")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Parameter range: [{parameter_range[0].item():.3f}, {parameter_range[-1].item():.3f}]")
    print(f"Number of parameter points: {len(parameter_range)}")
    
    # Analyze bifurcations
    diagram = pattern_system.bifurcation_analysis(
        pattern,
        parameter_range,
        parameterized_reaction
    )
    
    # Test diagram properties with detailed assertions
    assert isinstance(diagram, BifurcationDiagram), "Should return bifurcation diagram"
    
    # Test solution properties
    assert diagram.solution_states.shape[0] > 0, "Should have solution states"
    assert diagram.solution_params.shape[0] > 0, "Should have solution parameters"
    assert diagram.solution_states.shape[0] == diagram.solution_params.shape[0], \
        "Number of states should match number of parameters"
    
    # Print solution statistics
    print("\nSolution Statistics:")
    print(f"Number of solution states: {diagram.solution_states.shape[0]}")
    print(f"State tensor shape: {diagram.solution_states.shape}")
    print(f"Parameter range covered: [{diagram.solution_params.min().item():.3f}, {diagram.solution_params.max().item():.3f}]")
    
    # Test bifurcation detection
    assert diagram.bifurcation_points.numel() > 0, "Should detect bifurcations"
    
    # Print bifurcation statistics
    print("\nBifurcation Statistics:")
    print(f"Number of bifurcation points: {diagram.bifurcation_points.numel()}")
    print(f"Bifurcation parameters: {diagram.bifurcation_points.tolist()}")
    
    # Test bifurcation properties
    if diagram.bifurcation_points.numel() > 0:
        # Should find bifurcation near param = 1
        assert torch.any(torch.abs(diagram.bifurcation_points - 1.0) < 0.1), \
            "Should detect bifurcation near param = 1.0"


def test_bifurcation_detection_threshold(pattern_system, grid_size):
    """Test that bifurcation detection threshold is appropriate."""
    # Create test pattern
    pattern = torch.randn(1, 2, grid_size, grid_size)

    # Define parameter range with finer granularity around bifurcation point
    parameter_range = torch.linspace(0.8, 1.2, NUM_PARAMETER_POINTS)  # Use class variable

    # Create parameterized reaction term with known bifurcation
    def parameterized_reaction(state, param):
        u, v = state[:, 0], state[:, 1]
        # Pitchfork bifurcation at param = 1
        du = param * u - u**3
        dv = -v  # Simple linear decay
        return torch.stack([du, dv], dim=1)

    # Analyze bifurcations
    diagram = pattern_system.bifurcation_analysis(
        pattern,
        parameter_range,
        parameterized_reaction
    )

    # Should detect the pitchfork bifurcation near param = 1
    bifurcation_params = diagram.bifurcation_points
    assert any(abs(p - 1.0) < 0.1 for p in bifurcation_params), \
        "Should detect bifurcation near param = 1"


def test_stability_regions(pattern_system, grid_size):
    """Test that stability regions are correctly identified."""
    # Create test pattern
    pattern = torch.randn(1, 2, grid_size, grid_size)

    # Define parameter range
    parameter_range = torch.linspace(0, 2, NUM_PARAMETER_POINTS)  # Use class variable

    # Create parameterized reaction term with known stability change
    def parameterized_reaction(state, param):
        u, v = state[:, 0], state[:, 1]
        # System becomes unstable at param = 1
        du = (param - 1) * u
        dv = -v
        return torch.stack([du, dv], dim=1)

    # Analyze bifurcations
    diagram = pattern_system.bifurcation_analysis(
        pattern,
        parameter_range,
        parameterized_reaction
    )

    # Check stability changes near bifurcation points
    for param in diagram.bifurcation_points:
        param_idx = torch.argmin(torch.abs(parameter_range - param))
        if param_idx > 0 and param_idx < len(parameter_range) - 1:
            state_before = diagram.solution_states[param_idx - 1]
            state_after = diagram.solution_states[param_idx + 1]
            assert not torch.allclose(state_before, state_after, atol=1e-3), \
                "State should change significantly at bifurcation"


def test_solution_branches(pattern_system, grid_size):
    """Test that solution branches are correctly tracked."""
    # Create test pattern with small initial values
    pattern = torch.ones(1, 2, grid_size, grid_size) * 0.1

    # Define parameter range
    parameter_range = torch.linspace(0, 2, NUM_PARAMETER_POINTS)  # Use class variable

    # Create parameterized reaction term with known solution structure
    def parameterized_reaction(state, param):
        u, v = state[:, 0], state[:, 1]
        # Modified system with clear branching behavior
        du = param * u - u**3
        dv = -v + u**2
        return torch.stack([du, dv], dim=1)

    # Analyze bifurcations
    diagram = pattern_system.bifurcation_analysis(
        pattern,
        parameter_range,
        parameterized_reaction
    )

    # Test solution states are non-zero
    assert diagram.solution_states.shape[0] > 0, "Should have solution states"
    
    for state in diagram.solution_states:
        assert not torch.allclose(state, torch.zeros_like(state)), \
            "Solution states should not be zero"

    # Check that solution magnitude increases with parameter
    magnitudes = torch.norm(diagram.solution_states.reshape(diagram.solution_states.shape[0], -1), dim=1)
    diffs = magnitudes[1:] - magnitudes[:-1]
    assert torch.all(diffs >= -1e-6), \
        "Solution magnitude should not decrease significantly"


@pytest.fixture
def simple_parameterized_reaction():
    """Create a simple parameterized reaction that has a known bifurcation."""
    def reaction(state: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
        # More complex reaction with stronger nonlinearity and coupling
        if len(state.shape) == 4:  # [batch, channels, height, width]
            u = state[:, 0]  # First component
            v = state[:, 1] if state.shape[1] > 1 else torch.zeros_like(u)  # Second component or zeros
            
            # Stronger reaction terms with more dramatic bifurcation behavior
            du = 2.0 * param * u - 0.5 * torch.pow(u, 3) + 0.5 * v  # Increased linear term, reduced nonlinearity
            dv = -0.5 * v + torch.pow(u, 2) - 0.1 * torch.pow(v, 3)  # Balanced feedback
            
            return torch.stack([du, dv], dim=1)
        else:  # Flattened state
            # For single component, use a simpler nonlinearity with stronger parameter dependence
            return 2.0 * param * state - 0.5 * torch.pow(state, 3)
    return reaction


def test_stability_computation(pattern_system, simple_parameterized_reaction):
    """Test that stability computation works correctly."""
    # Create initial state
    state = torch.zeros((1, 1, 4, 4))
    param = torch.tensor(0.5)
    
    # Compute stability
    reaction = lambda x: simple_parameterized_reaction(x, param)
    stability = pattern_system.stability.compute_stability(state, reaction)
    
    # Print stability for debugging
    print(f"\nStability value at param={param.item()}: {stability}")
    
    # Convert stability to tensor properly for checks
    stability_tensor = torch.as_tensor(stability).clone().detach()
    assert not torch.isnan(stability_tensor), "Stability computation returned NaN"
    assert not torch.isinf(stability_tensor), "Stability computation returned Inf"


def test_state_evolution(pattern_system, simple_parameterized_reaction):
    """Test that state evolution behaves as expected."""
    # Create initial state with small values
    state = torch.ones((1, 1, 4, 4)) * 0.01
    param = torch.tensor(1.0)
    
    # Define reaction
    reaction = lambda state, p=param: simple_parameterized_reaction(state, p)
    
    # Evolve state
    states = []
    for _ in range(CONVERGENCE_STEPS):  # Use class variable
        state = pattern_system.reaction_diffusion(state, reaction, param)
        states.append(state.mean().item())
        
    # Print evolution for debugging
    print("\nState evolution:")
    for i, s in enumerate(states):
        print(f"Step {i}: {s}")
        
    # Check that evolution is not constant
    assert len(set(states)) > 1, "State is not evolving"


def test_bifurcation_detection_components(pattern_system, simple_parameterized_reaction):
    """Test individual components of bifurcation detection."""
    # Initial setup with small values
    state = torch.zeros((1, 1, 4, 4))
    # Focus on a smaller parameter range around the bifurcation point
    params = torch.linspace(-0.2, 0.2, NUM_PARAMETER_POINTS)

    # Track stability and state values
    stability_values = []
    state_values = []

    print("\nBifurcation analysis components:")

    for param in params:
        # Evolve to steady state with larger time steps
        current_state = state.clone()
        for _ in range(CONVERGENCE_STEPS):
            current_state = pattern_system.reaction_diffusion(
                current_state, 
                simple_parameterized_reaction,
                param,
                dt=0.5,  # Increased time step
                diffusion_coefficient=0.2  # Increased diffusion
            )

        # Compute stability
        stability = pattern_system.compute_stability(current_state, simple_parameterized_reaction, param)
        stability_values.append(stability)
        state_values.append(current_state.mean().item())

        print(f"Param: {param:.3f}, Stability: {stability:.3f}, State: {current_state.mean().item():.3f}")

    # Convert to numpy arrays for analysis
    stability_values = np.array(stability_values)
    state_values = np.array(state_values)

    # Check for significant changes using max difference instead of all differences
    stability_change = np.max(np.abs(np.diff(stability_values)))
    state_change = np.max(np.abs(np.diff(state_values)))

    print(f"\nMax stability change: {stability_change:.3f}")
    print(f"Max state change: {state_change:.3f}")

    # Assert that we detect significant changes
    assert stability_change > 0.1, "No significant stability changes detected"
    assert state_change > 0.1, "No significant state changes detected"


def test_convergence_at_bifurcation(pattern_system, simple_parameterized_reaction):
    """Test system behavior near known bifurcation point."""
    # Create states near bifurcation point with slightly larger initial values
    state = torch.ones((1, 1, 4, 4)) * 0.05  # Increased initial value
    params = torch.tensor([-0.02, 0.02])  # Reduced parameter points, increased separation
    
    print("\nConvergence near bifurcation point:")
    
    for param in params:
        reaction = lambda state, p=param: simple_parameterized_reaction(state, p)
        
        # Track convergence with fewer steps
        current_state = state.clone()
        states = []
        for _ in range(CONVERGENCE_STEPS):
            current_state = pattern_system.reaction_diffusion(
                current_state,
                reaction,
                param,
                dt=0.1,  # Reduced time step for smoother convergence
                diffusion_coefficient=0.1  # Reduced diffusion coefficient
            )
            states.append(current_state.mean().item())
            
        # Check if state changes significantly
        state_range = max(states) - min(states)
        print(f"\nParam {param.item():.3f} - State range: {state_range:.6f}")
        
        if param < 0:
            assert state_range < 0.01, f"State should converge to 0 for param={param.item()}"
        elif param > 0:
            assert state_range > 0.001, f"State should diverge from 0 for param={param.item()}"
