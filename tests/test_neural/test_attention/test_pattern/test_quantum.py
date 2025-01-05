"""Tests for quantum pattern functionality."""

import torch
import pytest
import numpy as np
from src.neural.attention.pattern.dynamics import PatternDynamics
from src.neural.attention.pattern.quantum import QuantumState, QuantumGeometricTensor
from src.core.quantum.state_space import HilbertSpace


def print_tensor_info(name: str, tensor: torch.Tensor):
    """Print detailed tensor information."""
    print(f"\n{name}:")
    print(f"Shape: {tensor.shape}")
    print(f"Dtype: {tensor.dtype}")
    print(f"Norm: {torch.norm(tensor)}")
    if torch.is_complex(tensor):
        print(f"Real norm: {torch.norm(tensor.real)}")
        print(f"Imag norm: {torch.norm(tensor.imag)}")
    print(f"Max abs: {torch.max(torch.abs(tensor))}")
    print(f"Min abs: {torch.min(torch.abs(tensor))}")
    if len(tensor.shape) > 2:
        print(f"Per-channel norms: {[torch.norm(tensor[:,i]).item() for i in range(tensor.shape[1])]}")


@pytest.fixture
def quantum_system():
    """Create quantum-enabled pattern system."""
    system = PatternDynamics(
        grid_size=8,
        space_dim=2,
        quantum_enabled=True
    )
    # Initialize HilbertSpace
    hilbert_space = HilbertSpace(
        dim=system.dim,
        dtype=torch.float32
    )
    system.quantum_flow.hilbert_space = hilbert_space
    return system


class TestQuantumPatterns:
    """Test suite for quantum pattern functionality."""
    
    def test_quantum_state_conversion(self, quantum_system):
        """Test conversion between classical and quantum states."""
        print("\n=== Starting quantum state conversion test ===")
        
        # Create test state with proper normalization
        state = torch.randn(1, quantum_system.dim, quantum_system.size, quantum_system.size, dtype=torch.float32)
        print_tensor_info("Initial state (before any normalization)", state)
        
        # First normalization
        state = state / torch.norm(state)
        print_tensor_info("After first normalization", state)
        print(f"Verification - norm after first normalization: {torch.norm(state)}")
        
        # Before quantum conversion
        print("\n--- Converting to quantum state ---")
        print("State before _to_quantum_state:", torch.norm(state).item())
        
        # Convert to quantum
        quantum_state = quantum_system._to_quantum_state(state)
        print("\n=== Quantum state details ===")
        print_tensor_info("Quantum amplitudes", quantum_state.amplitudes)
        print(f"Original norm stored in quantum state: {quantum_state.original_norm}")
        print(f"Phase norm: {torch.norm(quantum_state.phase)}")
        print(f"Quantum state norm (mean): {torch.mean(quantum_state.norm())}")
        print(f"Quantum state norm per channel: {[torch.norm(quantum_state.amplitudes[:,i]).item() for i in range(quantum_state.amplitudes.shape[1])]}")
        
        # Check type and shape
        assert quantum_state.amplitudes.dtype == torch.complex64
        assert quantum_state.amplitudes.shape == state.shape
        
        # Check normalization
        assert torch.allclose(quantum_state.norm(), torch.tensor(1.0, dtype=torch.float32))
        
        # Before classical conversion
        print("\n--- Converting back to classical state ---")
        print("Quantum state norm (mean) before conversion:", torch.mean(quantum_state.norm()).item())
        print("Quantum state amplitudes norm before conversion:", torch.norm(quantum_state.amplitudes).item())
        print("Quantum state phase before conversion:", quantum_state.phase[:5])  # Show first 5 phase values
        
        # Convert back to classical
        classical = quantum_system._from_quantum_state(quantum_state)
        print_tensor_info("Final classical state", classical)
        
        # Check type and shape
        assert classical.dtype == torch.float32
        assert classical.shape == state.shape
        
        # Detailed norm analysis
        final_norm = torch.norm(classical)
        print("\n=== Final norm analysis ===")
        print(f"Final norm: {final_norm}")
        print(f"Ratio to expected norm of 1.0: {final_norm/1.0}")
        print(f"Ratio to initial norm: {final_norm/torch.norm(state)}")
        print(f"Per-channel final norms: {[torch.norm(classical[:,i]).item() for i in range(classical.shape[1])]}")
        print(f"Sum of squared per-channel norms: {sum([torch.norm(classical[:,i]).item()**2 for i in range(classical.shape[1])])}")
        
        assert torch.allclose(torch.norm(classical), torch.tensor(1.0, dtype=torch.float32))
        
    def test_quantum_evolution(self, quantum_system):
        """Test quantum state evolution."""
        # Create normalized test state
        state = torch.randn(1, quantum_system.dim, quantum_system.size, quantum_system.size, dtype=torch.float32)
        state = state / torch.norm(state)
        
        # Evolve state
        evolved = quantum_system.compute_next_state(state)
        
        # Check type and shape
        assert evolved.dtype == torch.float32
        assert evolved.shape == state.shape
        
        # Check normalization
        assert torch.allclose(torch.norm(evolved), torch.tensor(1.0, dtype=torch.float32))
        
        # Check that state has changed
        assert not torch.allclose(evolved, state)
        
    def test_quantum_geometric_tensor(self, quantum_system):
        """Test quantum geometric tensor computation."""
        # Create normalized test state
        state = torch.randn(1, quantum_system.dim, quantum_system.size, quantum_system.size, dtype=torch.float32)
        state = state / torch.norm(state)
        
        # Convert to quantum state
        quantum_state = quantum_system._to_quantum_state(state)
        
        # Compute tensor
        Q = quantum_system.quantum_tensor.compute_tensor(quantum_state)
        
        # Check type and shape
        assert Q.dtype == torch.complex64
        assert Q.shape == (quantum_system.dim, quantum_system.dim)
        
        # Check Hermiticity
        assert torch.allclose(Q, Q.conj().transpose(-2, -1))
        
    def test_berry_phase(self, quantum_system):
        """Test Berry phase computation."""
        # Create normalized test state
        state = torch.randn(1, quantum_system.dim, quantum_system.size, quantum_system.size, dtype=torch.float32)
        state = state / torch.norm(state)
        
        # Create closed path in parameter space
        t = torch.linspace(0, 2*np.pi, 100, dtype=torch.float32)
        path = torch.stack([torch.cos(t), torch.sin(t)], dim=1)
        
        # Compute Berry phase
        phase = quantum_system.compute_berry_phase(state, path)
        
        # Check type and realness
        assert phase.dtype == torch.float32
        assert torch.allclose(phase.imag, torch.tensor(0.0, dtype=torch.float32))
        
        # Check range
        assert -np.pi <= phase.item() <= np.pi
        
    def test_quantum_potential(self, quantum_system):
        """Test quantum potential computation."""
        # Create normalized test state
        state = torch.randn(1, quantum_system.dim, quantum_system.size, quantum_system.size, dtype=torch.float32)
        state = state / torch.norm(state)
        
        # Compute potential
        V = quantum_system.compute_quantum_potential(state)
        
        # Check type and shape
        assert V.dtype == torch.float32
        assert V.shape == state.shape
        
        # Check realness
        assert torch.allclose(V.imag, torch.tensor(0.0, dtype=torch.float32))
        
    def test_quantum_disabled(self, quantum_system):
        """Test behavior when quantum features are disabled."""
        # Disable quantum features
        quantum_system.quantum_enabled = False
        
        # Create test state
        state = torch.randn(1, quantum_system.dim, quantum_system.size, quantum_system.size, dtype=torch.float32)
        state = state / torch.norm(state)
        
        # Check that quantum operations raise error
        with pytest.raises(RuntimeError):
            quantum_system._to_quantum_state(state)
            
    def test_parallel_transport(self, quantum_system):
        """Test parallel transport of quantum state."""
        # Create normalized test state
        state = torch.randn(1, quantum_system.dim, quantum_system.size, quantum_system.size, dtype=torch.float32)
        state = state / torch.norm(state)
        
        # Convert to quantum state
        quantum_state = quantum_system._to_quantum_state(state)
        
        # Create test points
        p1 = torch.tensor([0.0, 0.0], dtype=torch.float32)
        p2 = torch.tensor([1.0, 0.0], dtype=torch.float32)
        
        # Transport state
        transported = quantum_system._parallel_transport(quantum_state, p1, p2)
        
        # Check type and shape
        assert transported.amplitudes.dtype == torch.complex64
        assert transported.amplitudes.shape == state.shape
        
        # Check normalization
        assert torch.allclose(transported.norm(), torch.tensor(1.0, dtype=torch.float32))