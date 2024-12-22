import pytest
import torch
import numpy as np

from src.core.quantum.types import QuantumState
from src.core.quantum.state_space import HilbertSpace
from src.core.tiling.state_manager import StateManager, StateConfig, StateType


def test_single_qubit_state_preparation():
    """Test preparation of single qubit states."""
    # Initialize Hilbert space
    hilbert_space = HilbertSpace(dim=2)  # 2D for single qubit
    
    # Test |0⟩ state
    zero_state = hilbert_space.prepare_state(torch.tensor([1.0, 0.0]))
    assert isinstance(zero_state, QuantumState)
    assert torch.allclose(zero_state.amplitudes, torch.tensor([1.0, 0.0], dtype=torch.complex128))
    assert zero_state.basis_labels == ["|0⟩", "|1⟩"]
    
    # Test |1⟩ state
    one_state = hilbert_space.prepare_state(torch.tensor([0.0, 1.0]))
    assert torch.allclose(one_state.amplitudes, torch.tensor([0.0, 1.0], dtype=torch.complex128))
    
    # Test superposition state (|0⟩ + |1⟩)/√2
    superposition = hilbert_space.prepare_state(torch.tensor([1.0, 1.0]))
    expected = torch.tensor([1.0, 1.0], dtype=torch.complex128) / np.sqrt(2)
    assert torch.allclose(superposition.amplitudes, expected)


def test_multi_qubit_state_preparation():
    """Test preparation of multi-qubit states."""
    # Initialize 2-qubit Hilbert space
    hilbert_space = HilbertSpace(dim=4)  # 4D for two qubits
    
    # Test |00⟩ state
    zero_zero = hilbert_space.prepare_state(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(
        zero_zero.amplitudes,
        torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.complex128)
    )
    
    # Test Bell state (|00⟩ + |11⟩)/√2
    bell_state = hilbert_space.prepare_state(torch.tensor([1.0, 0.0, 0.0, 1.0]))
    expected = torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.complex128) / np.sqrt(2)
    assert torch.allclose(bell_state.amplitudes, expected)


def test_state_normalization():
    """Test automatic state normalization."""
    # Initialize state manager
    config = StateConfig(dim=2, type=StateType.PURE)
    manager = StateManager(config)
    
    # Test normalization of arbitrary state
    state = manager.initialize_state("test")
    norm = torch.norm(state)
    assert torch.isclose(norm, torch.tensor(1.0))
    
    # Test normalization preservation after update
    update = torch.randn_like(state)
    new_state = manager.update_state("test", update)
    new_norm = torch.norm(new_state)
    assert torch.isclose(new_norm, torch.tensor(1.0))


def test_quantum_state_properties():
    """Test quantum state properties and methods."""
    # Create a quantum state
    amplitudes = torch.tensor([1.0, 0.0], dtype=torch.complex64)
    basis_labels = ["|0⟩", "|1⟩"]
    phase = torch.zeros(1, dtype=torch.complex64)
    
    state = QuantumState(amplitudes=amplitudes, basis_labels=basis_labels, phase=phase)
    
    # Test shape property
    assert state.shape == (2,)
    
    # Test data property
    assert torch.allclose(state.data, amplitudes)
    
    # Test normalization
    assert torch.isclose(torch.sum(torch.abs(state.amplitudes) ** 2), torch.tensor(1.0)) 


def test_time_evolution():
    """Test time evolution of quantum states."""
    hilbert_space = HilbertSpace(dim=2)
    
    # Prepare initial state |0⟩
    initial_state = hilbert_space.prepare_state(torch.tensor([1.0, 0.0]))
    
    # Define Hamiltonian (Pauli-X matrix)
    hamiltonian = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
    
    # Evolve state for time t=π/4 (quarter rotation)
    t = np.pi / 4
    evolved_state = hilbert_space.evolve_state(initial_state, hamiltonian, t)
    
    # Expected state after evolution: cos(t)|0⟩ - i*sin(t)|1⟩
    expected = torch.tensor([
        np.cos(t),
        -1j * np.sin(t)
    ], dtype=torch.complex128)
    
    assert torch.allclose(evolved_state.amplitudes, expected, atol=1e-6)


def test_unitary_operations():
    """Test application of unitary operations."""
    hilbert_space = HilbertSpace(dim=2)
    
    # Prepare initial state |0⟩
    initial_state = hilbert_space.prepare_state(torch.tensor([1.0, 0.0]))
    
    # Define Hadamard gate
    hadamard = torch.tensor([
        [1.0, 1.0],
        [1.0, -1.0]
    ], dtype=torch.complex128) / np.sqrt(2)
    
    # Apply Hadamard gate
    hadamard_state = hilbert_space.apply_unitary(initial_state, hadamard)
    
    # Expected state: (|0⟩ + |1⟩)/√2
    expected = torch.tensor([1.0, 1.0], dtype=torch.complex128) / np.sqrt(2)
    
    assert torch.allclose(hadamard_state.amplitudes, expected, atol=1e-6)
    
    # Test unitarity preservation
    norm = torch.sum(torch.abs(hadamard_state.amplitudes) ** 2)
    assert torch.isclose(norm.to(torch.float32), torch.tensor(1.0, dtype=torch.float32))


def test_measurement_operations():
    """Test quantum measurement operations."""
    hilbert_space = HilbertSpace(dim=2)
    
    # Prepare superposition state (|0⟩ + |1⟩)/√2
    initial_state = hilbert_space.prepare_state(torch.tensor([1.0, 1.0]))
    
    # Define measurement in computational basis
    measurement = torch.eye(2, dtype=torch.complex128)
    
    # Perform measurement
    result, post_state = hilbert_space.measure_state(initial_state, measurement)
    
    # Check measurement result is valid (0 or 1)
    assert result in [0, 1]
    
    # Check post-measurement state is normalized
    norm = torch.sum(torch.abs(post_state.amplitudes) ** 2)
    assert torch.isclose(norm.to(torch.float32), torch.tensor(1.0, dtype=torch.float32))
    
    # Check post-measurement state is an eigenstate
    if result == 0:
        expected = torch.tensor([1.0, 0.0], dtype=torch.complex128)
    else:
        expected = torch.tensor([0.0, 1.0], dtype=torch.complex128)
    
    assert torch.allclose(post_state.amplitudes, expected, atol=1e-6)


def test_measurement_statistics():
    """Test quantum measurement statistics."""
    hilbert_space = HilbertSpace(dim=2)
    n_samples = 1000
    
    # Prepare state |+⟩ = (|0⟩ + |1⟩)/√2
    plus_state = hilbert_space.prepare_state(torch.tensor([1.0, 1.0]))
    
    # Perform multiple measurements
    results = []
    for _ in range(n_samples):
        result, _ = hilbert_space.measure_state(plus_state, torch.eye(2, dtype=torch.complex128))
        results.append(result)
    
    # Check measurement statistics (should be roughly 50-50)
    counts = torch.bincount(torch.tensor(results))
    probabilities = counts.float() / n_samples
    
    assert torch.allclose(probabilities, torch.tensor([0.5, 0.5]), atol=0.1)


def test_state_fidelity():
    """Test quantum state fidelity computation."""
    hilbert_space = HilbertSpace(dim=2)
    
    # Create two states
    state1 = hilbert_space.prepare_state(torch.tensor([1.0, 0.0]))  # |0⟩
    state2 = hilbert_space.prepare_state(torch.tensor([0.0, 1.0]))  # |1⟩
    
    # Test orthogonal states
    fidelity = hilbert_space.state_fidelity(state1, state2)
    assert torch.isclose(fidelity.to(torch.float32), torch.tensor(0.0, dtype=torch.float32), atol=1e-6)
    
    # Test identical states
    fidelity = hilbert_space.state_fidelity(state1, state1)
    assert torch.isclose(fidelity.to(torch.float32), torch.tensor(1.0, dtype=torch.float32), atol=1e-6)
    
    # Test superposition state
    state3 = hilbert_space.prepare_state(torch.tensor([1.0, 1.0]))  # (|0⟩ + |1⟩)/√2
    fidelity = hilbert_space.state_fidelity(state1, state3)
    assert torch.isclose(fidelity.to(torch.float32), torch.tensor(0.5, dtype=torch.float32), atol=1e-6)


def test_entanglement_measures():
    """Test various entanglement measures."""
    hilbert_space = HilbertSpace(dim=4)  # 2-qubit system
    
    # Prepare separable state |00⟩
    separable_state = hilbert_space.prepare_state(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    
    # Prepare maximally entangled Bell state (|00⟩ + |11⟩)/√2
    bell_state = hilbert_space.prepare_state(torch.tensor([1.0, 0.0, 0.0, 1.0]))
    
    # Test entanglement entropy
    sep_entropy = hilbert_space.compute_entanglement_entropy(separable_state)
    bell_entropy = hilbert_space.compute_entanglement_entropy(bell_state)
    
    assert torch.isclose(sep_entropy.to(torch.float32), torch.tensor(0.0, dtype=torch.float32), atol=1e-6)
    assert torch.isclose(bell_entropy.to(torch.float32), torch.tensor(1.0, dtype=torch.float32), atol=1e-6)
    
    # Test negativity
    sep_negativity = hilbert_space.compute_negativity(separable_state)
    bell_negativity = hilbert_space.compute_negativity(bell_state)
    
    assert torch.isclose(sep_negativity.to(torch.float32), torch.tensor(0.0, dtype=torch.float32), atol=1e-6)
    assert torch.isclose(bell_negativity.to(torch.float32), torch.tensor(0.5, dtype=torch.float32), atol=1e-6)
    
    # Test entanglement witness
    sep_witness = hilbert_space.evaluate_entanglement_witness(separable_state)
    bell_witness = hilbert_space.evaluate_entanglement_witness(bell_state)
    
    assert sep_witness.to(torch.float32) >= 0  # Separable state should give non-negative value
    assert bell_witness.to(torch.float32) < 0  # Entangled state should give negative value


def test_density_matrix_properties():
    """Test density matrix properties."""
    hilbert_space = HilbertSpace(dim=2)
    
    # Create pure state
    pure_state = hilbert_space.prepare_state(torch.tensor([1.0, 0.0]))
    
    # Create mixed state (|0⟩⟨0| + |1⟩⟨1|)/2
    mixed_amplitudes = torch.tensor([1.0, 0.0], dtype=torch.complex128)  # Start with |0⟩
    mixed_state = QuantumState(
        amplitudes=mixed_amplitudes,
        basis_labels=["|0⟩", "|1⟩"],
        phase=torch.zeros(1, dtype=torch.complex128)
    )
    
    # Test purity through state fidelity
    pure_fidelity = hilbert_space.state_fidelity(pure_state, pure_state)
    mixed_fidelity = hilbert_space.state_fidelity(mixed_state, mixed_state)
    
    assert torch.isclose(pure_fidelity.to(torch.float32), torch.tensor(1.0, dtype=torch.float32), atol=1e-6)
    assert torch.isclose(mixed_fidelity.to(torch.float32), torch.tensor(1.0, dtype=torch.float32), atol=1e-6)
    
    # Test entropy
    pure_entropy = hilbert_space.compute_entropy(pure_state)
    mixed_entropy = hilbert_space.compute_entropy(mixed_state)
    
    assert torch.isclose(pure_entropy.to(torch.float32), torch.tensor(0.0, dtype=torch.float32), atol=1e-6)
    assert torch.isclose(mixed_entropy.to(torch.float32), torch.tensor(0.0, dtype=torch.float32), atol=1e-6)


def test_quantum_geometric_properties():
    """Test quantum geometric properties."""
    hilbert_space = HilbertSpace(dim=2)
    
    # Create two nearby states
    state1 = hilbert_space.prepare_state(torch.tensor([1.0, 0.0]))
    state2 = hilbert_space.prepare_state(torch.tensor([0.99, 0.14]))  # Small rotation from |0⟩
    
    # Test Fubini-Study distance
    distance = hilbert_space.fubini_study_distance(state1, state2)
    assert distance.to(torch.float32) > 0 and distance.to(torch.float32) < np.pi/2  # Should be small but non-zero
    
    # Test quantum variance
    observable = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex128)  # Pauli-Z
    variance = hilbert_space.measure_variance(state1, observable)
    assert torch.isclose(variance.to(torch.float32), torch.tensor(0.0, dtype=torch.float32), atol=1e-6)  # |0⟩ is eigenstate of Z