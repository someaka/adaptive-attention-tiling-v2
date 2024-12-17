"""Tests for quantum pattern functionality."""

import torch
import pytest
import numpy as np
from src.neural.attention.pattern.dynamics import PatternDynamics
from src.neural.attention.pattern.quantum import QuantumState, QuantumGeometricTensor


@pytest.fixture
def quantum_system():
    """Create quantum-enabled pattern system."""
    return PatternDynamics(
        grid_size=8,
        space_dim=2,
        quantum_enabled=True
    )


class TestQuantumPatterns:
    """Test suite for quantum pattern functionality."""
    
    def test_quantum_state_conversion(self, quantum_system):
        """Test conversion between classical and quantum states."""
        # Create test state
        state = torch.randn(1, quantum_system.dim, quantum_system.size, quantum_system.size)
        
        # Convert to quantum
        quantum_state = quantum_system._to_quantum_state(state)
        
        # Convert back
        classical_state = quantum_system._from_quantum_state(quantum_state)
        
        # Check shape preservation
        assert classical_state.shape == state.shape
        
        # Check approximate value preservation (allowing for complex conversion)
        assert torch.allclose(classical_state, state, rtol=1e-5, atol=1e-5)
        
    def test_quantum_evolution(self, quantum_system):
        """Test quantum state evolution."""
        # Create test state
        state = torch.randn(1, quantum_system.dim, quantum_system.size, quantum_system.size)
        
        # Evolve state
        evolved = quantum_system.compute_next_state(state)
        
        # Check shape preservation
        assert evolved.shape == state.shape
        
        # Check state is modified
        assert not torch.allclose(evolved, state)
        
    def test_quantum_geometric_tensor(self, quantum_system):
        """Test quantum geometric tensor computation."""
        # Create test state
        state = torch.randn(1, quantum_system.dim, quantum_system.size, quantum_system.size)
        
        # Convert to quantum state
        quantum_state = quantum_system._to_quantum_state(state)
        
        # Compute tensor
        Q = quantum_system.quantum_tensor.compute_tensor(quantum_state)
        
        # Check shape
        assert Q.shape == (quantum_system.dim, quantum_system.dim)
        
        # Check Hermiticity
        assert torch.allclose(Q, Q.conj().T)
        
    def test_berry_phase(self, quantum_system):
        """Test Berry phase computation."""
        # Create test state
        state = torch.randn(1, quantum_system.dim, quantum_system.size, quantum_system.size)
        
        # Create closed path in parameter space
        t = torch.linspace(0, 2*np.pi, 100)
        path = torch.stack([torch.cos(t), torch.sin(t)], dim=1)
        
        # Compute Berry phase
        phase = quantum_system.compute_berry_phase(state, path)
        
        # Check phase is real
        assert isinstance(phase, float)
        
        # Check phase is within expected range [-π, π]
        assert -np.pi <= phase <= np.pi
        
    def test_quantum_potential(self, quantum_system):
        """Test quantum potential computation."""
        # Create test state
        state = torch.randn(1, quantum_system.dim, quantum_system.size, quantum_system.size)
        
        # Compute potential
        V = quantum_system.compute_quantum_potential(state)
        
        # Check shape
        assert V.shape == state.shape
        
        # Check potential is real
        assert torch.all(torch.isreal(V))
        
    def test_quantum_disabled(self):
        """Test quantum operations fail when disabled."""
        # Create system with quantum disabled
        system = PatternDynamics(grid_size=8, space_dim=2, quantum_enabled=False)
        
        # Create test state
        state = torch.randn(1, system.dim, system.size, system.size)
        
        # Check quantum operations raise error
        with pytest.raises(RuntimeError):
            system.compute_quantum_potential(state)
            
        with pytest.raises(RuntimeError):
            path = torch.zeros((10, 2))
            system.compute_berry_phase(state, path)
            
    def test_parallel_transport(self, quantum_system):
        """Test parallel transport of quantum state."""
        # Create test state
        state = torch.randn(1, quantum_system.dim, quantum_system.size, quantum_system.size)
        quantum_state = quantum_system._to_quantum_state(state)
        
        # Create test points
        p1 = torch.tensor([0.0, 0.0])
        p2 = torch.tensor([1.0, 0.0])
        
        # Transport state
        transported = quantum_system._parallel_transport(quantum_state, p1, p2)
        
        # Check amplitude preservation
        assert torch.allclose(transported.amplitude, quantum_state.amplitude)
        
        # Check phase change
        assert not torch.allclose(transported.phase, quantum_state.phase) 