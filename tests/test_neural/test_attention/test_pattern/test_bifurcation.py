"""Tests for bifurcation analysis."""

import torch
import pytest

from src.neural.attention.pattern.models import BifurcationDiagram
from tests.test_neural.test_attention.test_pattern.conftest import assert_tensor_equal


def test_bifurcation_analysis(pattern_system, grid_size):
    """Test bifurcation analysis."""
    # Create test pattern
    pattern = torch.randn(1, 2, grid_size, grid_size)

    # Define parameter range
    parameter_range = torch.linspace(0, 2, 100)

    # Create parameterized reaction term
    def parameterized_reaction(state, param):
        u, v = state[:, 0], state[:, 1]
        du = param * u**2 * v - u
        dv = u**2 - v
        return torch.stack([du, dv], dim=1)

    # Analyze bifurcations
    diagram = pattern_system.bifurcation_analysis(
        pattern, parameterized_reaction, parameter_range
    )

    # Test diagram properties
    assert isinstance(diagram, BifurcationDiagram), "Should return bifurcation diagram"
    assert diagram.bifurcation_points.numel() > 0, "Should detect bifurcations"
    assert diagram.solution_states.shape[0] > 0, "Should have solution states"
    assert diagram.solution_params.shape[0] > 0, "Should have solution parameters"

    # Test that solution parameters are within expected range
    assert torch.all(diagram.solution_params >= 0), "Parameters should be non-negative"
    assert torch.all(diagram.solution_params <= 2), "Parameters should be <= 2"

    # Test that solution states have expected shape
    assert len(diagram.solution_states.shape) == 4, "Solution states should be 4D tensor"
    assert diagram.solution_states.shape[1:] == (2, grid_size, grid_size), \
        "Solution states should have correct spatial dimensions"

    # Test solution states are non-zero
    for state in diagram.solution_states:
        assert not torch.allclose(state, torch.zeros_like(state)), \
            "Solution states should not be zero"

    # Check that solution magnitude increases with parameter
    magnitudes = torch.norm(diagram.solution_states.reshape(diagram.solution_states.shape[0], -1), dim=1)
    diffs = magnitudes[1:] - magnitudes[:-1]
    assert torch.all(diffs >= -1e-6), \
        "Solution magnitude should not decrease significantly"


def test_bifurcation_detection_threshold(pattern_system, grid_size):
    """Test that bifurcation detection threshold is appropriate."""
    # Create test pattern
    pattern = torch.randn(1, 2, grid_size, grid_size)

    # Define parameter range
    parameter_range = torch.linspace(0, 2, 100)

    # Create parameterized reaction term with known bifurcation
    def parameterized_reaction(state, param):
        u, v = state[:, 0], state[:, 1]
        # Pitchfork bifurcation at param = 1
        du = param * u - u**3
        dv = -v  # Simple linear decay
        return torch.stack([du, dv], dim=1)

    # Analyze bifurcations
    diagram = pattern_system.bifurcation_analysis(
        pattern, parameterized_reaction, parameter_range
    )

    # Should detect the pitchfork bifurcation near param = 1
    bifurcation_params = diagram.bifurcation_points
    assert any(abs(p - 1.0) < 0.1 for p in bifurcation_params), \
        "Should detect bifurcation near param = 1"

    # Check stability changes near bifurcation points
    for param in diagram.bifurcation_points:
        param_idx = torch.argmin(torch.abs(parameter_range - param))
        if param_idx > 0 and param_idx < len(parameter_range) - 1:
            state_before = diagram.solution_states[param_idx - 1]
            state_after = diagram.solution_states[param_idx + 1]
            assert not torch.allclose(state_before, state_after, atol=1e-3), \
                "State should change significantly at bifurcation"


def test_stability_regions(pattern_system, grid_size):
    """Test that stability regions are correctly identified."""
    # Create test pattern
    pattern = torch.randn(1, 2, grid_size, grid_size)

    # Define parameter range
    parameter_range = torch.linspace(0, 2, 100)

    # Create parameterized reaction term with known stability change
    def parameterized_reaction(state, param):
        u, v = state[:, 0], state[:, 1]
        # System becomes unstable at param = 1
        du = (param - 1) * u
        dv = -v
        return torch.stack([du, dv], dim=1)

    # Analyze bifurcations
    diagram = pattern_system.bifurcation_analysis(
        pattern, parameterized_reaction, parameter_range
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
    # Create test pattern
    pattern = torch.ones(1, 2, grid_size, grid_size)  # Start from uniform state

    # Define parameter range
    parameter_range = torch.linspace(0, 2, 100)

    # Create parameterized reaction term with known solution structure
    def parameterized_reaction(state, param):
        u, v = state[:, 0], state[:, 1]
        # Simple linear system with known solution u = param * u_initial
        du = (param - 1) * u
        dv = -v
        return torch.stack([du, dv], dim=1)

    # Analyze bifurcations
    diagram = pattern_system.bifurcation_analysis(
        pattern, parameterized_reaction, parameter_range
    )

    # Test solution states are non-zero
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
        # Simple cubic reaction: dx/dt = rx - x^3
        # Has a pitchfork bifurcation at r = 0
        # Scale down to prevent numerical instability
        return 0.1 * (param * state - torch.pow(state, 3))
    return reaction


def test_stability_computation(pattern_dynamics, simple_parameterized_reaction):
    """Test that stability computation works correctly."""
    # Create initial state
    state = torch.zeros((1, 1, 4, 4))
    param = torch.tensor(0.5)
    
    # Compute stability
    reaction = lambda x: simple_parameterized_reaction(x, param)
    stability = pattern_dynamics.stability_analyzer.compute_stability(state, reaction)
    
    # Print stability for debugging
    print(f"\nStability value at param={param.item()}: {stability}")
    
    assert not torch.isnan(torch.tensor(stability)), "Stability computation returned NaN"
    assert not torch.isinf(torch.tensor(stability)), "Stability computation returned Inf"


def test_state_evolution(pattern_dynamics, simple_parameterized_reaction):
    """Test that state evolution behaves as expected."""
    # Create initial state with small values
    state = torch.ones((1, 1, 4, 4)) * 0.01
    param = torch.tensor(1.0)
    
    # Define reaction
    reaction = lambda x: simple_parameterized_reaction(x, param)
    
    # Evolve state
    states = []
    for _ in range(10):
        state = pattern_dynamics.reaction_diffusion(state, reaction)
        states.append(state.mean().item())
        
    # Print evolution for debugging
    print("\nState evolution:")
    for i, s in enumerate(states):
        print(f"Step {i}: {s}")
        
    # Check that evolution is not constant
    assert len(set(states)) > 1, "State is not evolving"


def test_bifurcation_detection_components(pattern_dynamics, simple_parameterized_reaction):
    """Test individual components of bifurcation detection."""
    # Initial setup with small values
    state = torch.zeros((1, 1, 4, 4))
    params = torch.linspace(-0.5, 0.5, 10)  # Reduced parameter range
    
    # Track stability and state values
    stability_values = []
    state_values = []
    
    print("\nBifurcation analysis components:")
    
    for param in params:
        reaction = lambda x: simple_parameterized_reaction(x, param)
        
        # Evolve to steady state
        current_state = state.clone()
        for _ in range(20):  # Reduced iterations
            current_state = pattern_dynamics.reaction_diffusion(current_state, reaction)
            
        # Compute stability
        stability = pattern_dynamics.stability_analyzer.compute_stability(current_state, reaction)
        stability_values.append(stability)
        
        # Store state value
        state_values.append(current_state.mean().item())
        
        print(f"Param: {param.item():.3f}, Stability: {stability:.3f}, State: {state_values[-1]:.3f}")
    
    # Check for changes in stability and state
    stability_changes = [abs(stability_values[i+1] - stability_values[i]) 
                        for i in range(len(stability_values)-1)]
    state_changes = [abs(state_values[i+1] - state_values[i]) 
                    for i in range(len(state_values)-1)]
    
    print("\nChanges in stability:", [f"{x:.3f}" for x in stability_changes])
    print("Changes in state:", [f"{x:.3f}" for x in state_changes])
    
    # Assert that we see some significant changes
    assert max(stability_changes) > 0.001, "No significant stability changes detected"
    assert max(state_changes) > 0.001, "No significant state changes detected"


def test_convergence_at_bifurcation(pattern_dynamics, simple_parameterized_reaction):
    """Test system behavior near known bifurcation point."""
    # Create states near bifurcation point (r = 0) with small values
    state = torch.ones((1, 1, 4, 4)) * 0.01
    params = torch.tensor([-0.01, 0.0, 0.01])
    
    print("\nConvergence near bifurcation point:")
    
    for param in params:
        reaction = lambda x: simple_parameterized_reaction(x, param)
        
        # Track convergence
        current_state = state.clone()
        states = []
        for _ in range(20):
            current_state = pattern_dynamics.reaction_diffusion(current_state, reaction)
            states.append(current_state.mean().item())
            
        print(f"\nParam {param.item():.3f} evolution:")
        for i, s in enumerate(states):
            print(f"Step {i}: {s:.6f}")
            
        # Check if state changes significantly
        state_range = max(states) - min(states)
        print(f"State range: {state_range:.6f}")
        
        if param < 0:
            assert state_range < 0.01, f"State should converge to 0 for param={param.item()}"
        elif param > 0:
            assert state_range > 0.001, f"State should diverge from 0 for param={param.item()}"
