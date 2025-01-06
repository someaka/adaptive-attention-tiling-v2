"""Tests for quantum pattern functionality v2."""

import torch
import pytest
import logging

from  src.neural.attention.pattern.pattern_dynamics import PatternDynamics
from src.core.quantum.types import QuantumState
from src.core.quantum.state_space import HilbertSpace

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def quantum_system():
    """Create a quantum system for testing."""
    system = PatternDynamics(
        grid_size=4,  # 4x4 grid
        space_dim=2,  # Two channels for symplectic structure
        dt=0.1,
        quantum_enabled=True
    )
    return system

def test_basic_quantum_conversion(quantum_system):
    """Test basic conversion between classical and quantum states."""
    # Create test pattern
    pattern = torch.randn(1, 2, 4, 4, dtype=torch.float64)  # [batch, channel, height, width]
    pattern = pattern / torch.norm(pattern)
    
    logger.info("\n=== Initial pattern ===")
    logger.info(f"Pattern shape: {pattern.shape}")
    logger.info(f"Pattern norm: {torch.norm(pattern)}")
    for c in range(pattern.shape[1]):
        logger.info(f"\nChannel {c} magnitudes:")
        logger.info(f"\n{torch.abs(pattern[:, c])}")
    
    # Convert to quantum and back
    quantum_state = quantum_system._to_quantum_state(pattern)
    logger.info("\n=== Quantum state ===")
    logger.info(f"Quantum state shape: {quantum_state.amplitudes.shape}")
    logger.info(f"Quantum state norm: {quantum_state.norm()}")
    for c in range(quantum_state.amplitudes.shape[1]):
        logger.info(f"\nChannel {c} quantum magnitudes:")
        logger.info(f"\n{torch.abs(quantum_state.amplitudes[:, c])}")
    
    recovered = quantum_system._from_quantum_state(quantum_state)
    logger.info("\n=== Recovered pattern ===")
    logger.info(f"Recovered shape: {recovered.shape}")
    logger.info(f"Recovered norm: {torch.norm(recovered)}")
    for c in range(recovered.shape[1]):
        logger.info(f"\nChannel {c} recovered magnitudes:")
        logger.info(f"\n{torch.abs(recovered[:, c])}")
    
    # Check shape preservation
    assert recovered.shape == pattern.shape
    
    # Check that the relative magnitudes are preserved within each channel
    for c in range(pattern.shape[1]):
        pattern_channel = torch.abs(pattern[:, c])
        recovered_channel = torch.abs(recovered[:, c])
        
        # Normalize both to compare relative magnitudes within channel
        pattern_norm = pattern_channel / pattern_channel.max()
        recovered_norm = recovered_channel / recovered_channel.max()
        
        logger.info(f"\n=== Channel {c} comparison ===")
        logger.info("Pattern normalized:")
        logger.info(f"\n{pattern_norm}")
        logger.info("\nRecovered normalized:")
        logger.info(f"\n{recovered_norm}")
        logger.info("\nAbsolute difference:")
        logger.info(f"\n{torch.abs(pattern_norm - recovered_norm)}")
        
        # Log max difference for easier debugging
        max_diff = torch.max(torch.abs(pattern_norm - recovered_norm))
        logger.info(f"\nMaximum difference: {max_diff}")
        
        assert torch.allclose(pattern_norm, recovered_norm, atol=1e-2)
    
    # Check normalization
    assert torch.allclose(torch.norm(recovered), torch.tensor(1.0, dtype=torch.float64), atol=1e-6)

def test_quantum_evolution(quantum_system):
    """Test quantum state evolution."""
    # Create initial pattern
    pattern = torch.randn(1, 2, 4, 4, dtype=torch.float64)  # [batch, channel, height, width]
    pattern = pattern / torch.norm(pattern)
    
    # Evolve for a few steps
    evolved_pattern = pattern.clone()
    for _ in range(3):
        evolved_pattern = quantum_system.compute_next_state(evolved_pattern)
        # Check shape preservation
        assert evolved_pattern.shape == pattern.shape
        # Check normalization
        assert torch.allclose(torch.norm(evolved_pattern), torch.tensor(1.0, dtype=torch.float64), atol=1e-6)

def test_quantum_geometric_properties(quantum_system):
    """Test quantum geometric properties of the system."""
    # Create test pattern
    pattern = torch.randn(1, 2, 4, 4, dtype=torch.float64)  # [batch, channel, height, width]
    pattern = pattern / torch.norm(pattern)
    
    # Get quantum state
    quantum_state = quantum_system._to_quantum_state(pattern)
    
    # Flatten the state for density matrix computation
    flattened_state = quantum_state.amplitudes.reshape(1, -1)  # [batch, all_dims]
    flattened_state = flattened_state / torch.norm(flattened_state)  # Normalize
    
    # Test density matrix properties
    rho = torch.matmul(flattened_state.unsqueeze(-1), flattened_state.conj().unsqueeze(-2))
    
    # Check hermiticity
    assert torch.allclose(rho, rho.conj().transpose(-2, -1), atol=1e-6)
    
    # Check trace = 1
    trace = torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1)
    assert torch.allclose(trace.real, torch.tensor(1.0, dtype=torch.float64), atol=1e-6)

def test_quantum_flow_integration(quantum_system):
    """Test integration of quantum flow with pattern dynamics."""
    # Create initial pattern
    pattern = torch.randn(1, 2, 4, 4, dtype=torch.float64)  # [batch, channel, height, width]
    pattern = pattern / torch.norm(pattern)
    
    # Store initial energy
    initial_energy = torch.norm(pattern)
    
    # Evolve with quantum flow
    evolved = pattern.clone()
    energies = []
    
    for _ in range(3):
        evolved = quantum_system.compute_next_state(evolved)
        energies.append(torch.norm(evolved))
        
        # Check shape preservation
        assert evolved.shape == pattern.shape
        # Check approximate energy conservation
        assert torch.allclose(torch.norm(evolved), initial_energy, atol=1e-6)

def test_quantum_state_superposition(quantum_system):
    """Test quantum state superposition properties."""
    # Create two different patterns
    pattern1 = torch.randn(1, 2, 4, 4, dtype=torch.float64)  # [batch, channel, height, width]
    pattern2 = torch.randn(1, 2, 4, 4, dtype=torch.float64)
    
    # Normalize patterns
    pattern1 = pattern1 / torch.norm(pattern1)
    pattern2 = pattern2 / torch.norm(pattern2)
    
    # Create superposition
    alpha = 0.6
    beta = 0.8
    superposition = alpha * pattern1 + beta * pattern2
    superposition = superposition / torch.norm(superposition)
    
    # Convert to quantum state
    quantum_state = quantum_system._to_quantum_state(superposition)
    
    # Check normalization
    assert torch.allclose(quantum_state.norm(), torch.tensor(1.0, dtype=torch.float64), atol=1e-6)
    
    # Flatten for density matrix computation
    flattened_state = quantum_state.amplitudes.reshape(1, -1)
    flattened_state = flattened_state / torch.norm(flattened_state)  # Normalize
    
    # Compute density matrix
    rho = torch.matmul(flattened_state.unsqueeze(-1), flattened_state.conj().unsqueeze(-2))
    rho = rho.squeeze(0)  # Remove batch dimension
    
    # Check purity (Tr(ρ²) should be close to 1 for pure states)
    purity = torch.trace(torch.matmul(rho, rho)).real
    assert torch.allclose(purity, torch.tensor(1.0, dtype=torch.float64), atol=1e-6)

def test_quantum_measurement_consistency(quantum_system):
    """Test consistency of quantum measurements."""
    # Create test pattern
    pattern = torch.randn(1, 2, 4, 4, dtype=torch.float64)  # [batch, channel, height, width]
    pattern = pattern / torch.norm(pattern)
    
    # Convert to quantum state
    quantum_state = quantum_system._to_quantum_state(pattern)
    
    # Convert back to classical state
    recovered = quantum_system._from_quantum_state(quantum_state)
    
    # Check that consecutive measurements are consistent
    second_quantum = quantum_system._to_quantum_state(recovered)
    second_recovered = quantum_system._from_quantum_state(second_quantum)
    
    # The second measurement should be very close to the first
    assert torch.allclose(recovered, second_recovered, atol=1e-6) 