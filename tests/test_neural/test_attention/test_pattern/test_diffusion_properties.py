"""Tests for diffusion system properties."""

import torch
import pytest

from tests.test_neural.test_attention.test_pattern.conftest import assert_tensor_equal, assert_mass_conserved


@pytest.fixture
def test_params():
    """Common test parameters.
    
    For numerical stability of the diffusion equation:
    - dt * D / (dx)^2 <= 0.5  (CFL condition)
    where:
    - D is the diffusion coefficient
    - dt is the time step
    - dx is the grid spacing (1.0 in our case)
    """
    return {
        'dt': 0.01,
        'dx': 1.0,
        'D': 0.1,  # Diffusion coefficient
        'batch_size': 4,
        'grid_size': 16
    }


def create_test_state(params, pattern='uniform'):
    """Create a test state tensor.
    
    Args:
        params: Test parameters
        pattern: Type of test pattern ('uniform', 'random', 'impulse', 'checkerboard', 'gradient')
        
    Returns:
        Test state tensor
    """
    batch_size = params['batch_size']
    grid_size = params['grid_size']
    
    if pattern == 'uniform':
        return torch.ones(batch_size, 2, grid_size, grid_size)
    elif pattern == 'random':
        return torch.rand(batch_size, 2, grid_size, grid_size)
    elif pattern == 'impulse':
        state = torch.zeros(batch_size, 2, grid_size, grid_size)
        state[:, :, grid_size//2, grid_size//2] = 1.0
        return state
    elif pattern == 'checkerboard':
        x = torch.arange(grid_size)
        y = torch.arange(grid_size)
        X, Y = torch.meshgrid(x, y)
        pattern = (-1)**(X + Y)
        return pattern.expand(batch_size, 2, -1, -1).float()
    elif pattern == 'gradient':
        x = torch.linspace(0, 1, grid_size)
        y = torch.linspace(0, 1, grid_size)
        X, Y = torch.meshgrid(x, y)
        pattern = X + Y
        return pattern.expand(batch_size, 2, -1, -1).float()
    else:
        raise ValueError(f"Unknown pattern type: {pattern}")


def test_mass_conservation(pattern_system, test_params):
    """Test that diffusion conserves total mass."""
    # Test for different initial patterns
    for pattern in ['uniform', 'random', 'impulse', 'checkerboard', 'gradient']:
        state = create_test_state(test_params, pattern)
        evolved = pattern_system.apply_diffusion(state, test_params['D'], test_params['dt'])
        assert_mass_conserved(state, evolved)


def test_positivity_preservation(pattern_system, test_params):
    """Test that diffusion preserves positivity."""
    # Test for non-negative initial state
    state = torch.abs(create_test_state(test_params, 'random'))
    evolved = pattern_system.apply_diffusion(state, test_params['D'], test_params['dt'])
    assert torch.all(evolved >= 0), "Diffusion should preserve positivity"


def test_maximum_principle(pattern_system, test_params):
    """Test that diffusion satisfies the maximum principle."""
    state = create_test_state(test_params, 'random')
    evolved = pattern_system.apply_diffusion(state, test_params['D'], test_params['dt'])
    
    # Maximum principle: evolved values should be bounded by initial min/max
    assert torch.all(evolved <= state.max()), "Maximum principle violated (upper bound)"
    assert torch.all(evolved >= state.min()), "Maximum principle violated (lower bound)"


def test_symmetry_preservation(pattern_system, test_params):
    """Test that diffusion preserves symmetry."""
    # Create symmetric initial state
    state = create_test_state(test_params, 'checkerboard')
    evolved = pattern_system.apply_diffusion(state, test_params['D'], test_params['dt'])
    
    # Test rotational symmetry
    rotated_state = torch.rot90(state, k=1, dims=[-2, -1])
    rotated_evolved = pattern_system.apply_diffusion(rotated_state, test_params['D'], test_params['dt'])
    rotated_back = torch.rot90(rotated_evolved, k=-1, dims=[-2, -1])
    
    assert_tensor_equal(evolved, rotated_back, msg="Diffusion should preserve rotational symmetry")


def test_convergence_to_steady_state(pattern_system, test_params):
    """Test that diffusion converges to steady state.
    
    This test verifies three key properties of diffusion:
    1. The system converges to a uniform steady state
    2. The mean value is preserved during diffusion
    3. Total mass is conserved throughout the process
    
    For each test pattern (impulse, checkerboard, gradient), we:
    - Apply diffusion until the state is sufficiently uniform
    - Verify the final state matches theoretical expectations
    - Check that mass is conserved during the entire process
    """
    for pattern in ['impulse', 'checkerboard', 'gradient']:
        state = create_test_state(test_params, pattern)
        initial_mean = state.mean(dim=(-2, -1), keepdim=True)
        
        # Apply diffusion for multiple steps
        current = state
        for _ in range(100):
            current = pattern_system.apply_diffusion(current, test_params['D'], test_params['dt'])
            
            # Verify mass conservation at each step
            assert_mass_conserved(state, current)
            
            # Check mean preservation
            current_mean = current.mean(dim=(-2, -1), keepdim=True)
            assert_tensor_equal(initial_mean, current_mean, msg="Mean should be preserved during diffusion")
            
        # Final state should be approximately uniform
        assert torch.std(current) < 1e-3, "Should converge to uniform state"
