"""
Unit tests for the quantum state space implementation.

Tests cover:
1. State preparation and manipulation
2. Quantum evolution
3. Measurement protocols
4. Entropy and entanglement
5. Geometric structure of state space
6. Quantum channels
7. State tomography
8. Decoherence
9. Geometric phase
10. Advanced entanglement measures
"""

import numpy as np
import pytest
import torch

from src.core.quantum.state_space import QuantumState, HilbertSpace


class TestHilbertSpace:
    @pytest.fixture
    def hilbert_dim(self):
        """Dimension of test Hilbert space."""
        return 4

    @pytest.fixture
    def batch_size(self):
        """Batch size for vectorized operations."""
        return 8

    @pytest.fixture
    def hilbert_space(self, hilbert_dim):
        """Create a test Hilbert space."""
        return HilbertSpace(dim=hilbert_dim)

    def test_state_preparation(self, hilbert_space, hilbert_dim, batch_size):
        """Test quantum state preparation from classical data."""
        # Create test classical data
        classical_data = torch.randn(batch_size, hilbert_dim * 2, dtype=torch.float64)  # Complex embedding

        # Prepare quantum state
        quantum_state = hilbert_space.prepare_state(classical_data)

        # Test normalization
        norms = quantum_state.norm()
        assert torch.allclose(
            norms, torch.ones_like(norms, dtype=torch.float64), rtol=1e-5
        ), "Quantum states must be normalized"

        # Test state type
        assert isinstance(
            quantum_state, QuantumState
        ), "Output should be a QuantumState instance"

        # Test batch shape
        assert (
            quantum_state.shape[0] == batch_size
        ), "Batch dimension should be preserved"

    def test_state_evolution(self, hilbert_space, hilbert_dim):
        """Test unitary evolution of quantum states."""
        # Create initial state
        initial_state = hilbert_space.prepare_state(torch.randn(hilbert_dim * 2, dtype=torch.float64))

        # Create test Hamiltonian (Hermitian matrix)
        H = torch.randn(hilbert_dim, hilbert_dim, dtype=torch.complex128) + 1j * torch.randn(
            hilbert_dim, hilbert_dim, dtype=torch.complex128
        )
        H = H + H.conj().T  # Make Hermitian

        # Evolve state
        time = torch.linspace(0, 1, 10, dtype=torch.float64)
        evolved_states = hilbert_space.evolve_state(initial_state, H, time)

        # Test unitarity
        for state in evolved_states:
            assert torch.allclose(
                state.norm(), torch.tensor(1.0, dtype=torch.float64), rtol=1e-5
            ), "Evolution must preserve normalization"

        # Test time-reversal
        reversed_state = hilbert_space.evolve_state(evolved_states[-1], -H, time)[-1]
        assert torch.allclose(
            initial_state.amplitudes, reversed_state.amplitudes, rtol=1e-4
        ), "Time reversal should recover initial state"

    def test_measurement(self, hilbert_space, hilbert_dim):
        """Test quantum measurement protocols."""
        # Create test state
        state = hilbert_space.prepare_state(torch.randn(hilbert_dim * 2, dtype=torch.float64))

        # Create observable (Hermitian operator)
        observable = torch.randn(hilbert_dim, hilbert_dim, dtype=torch.complex128)
        observable = observable + observable.conj().T  # Make Hermitian

        # Perform measurement
        expectation = hilbert_space.measure_observable(state, observable)

        # Test expectation value is real
        assert torch.allclose(
            expectation.imag, torch.zeros_like(expectation.imag, dtype=torch.float64), rtol=1e-5
        ), "Expectation values must be real"

        # Test variance is non-negative
        variance = hilbert_space.measure_variance(state, observable)
        assert variance >= 0, "Variance must be non-negative"

    def test_entropy_computation(self, hilbert_space, hilbert_dim):
        """Test von Neumann entropy computation."""
        # Create mixed state density matrix
        pure_states = [
            hilbert_space.prepare_state(torch.randn(hilbert_dim * 2, dtype=torch.float64)) for _ in range(3)
        ]
        weights = torch.softmax(torch.randn(3), dim=0)
        mixed_state = sum(w * p.density_matrix() for w, p in zip(weights, pure_states))

        # Compute entropy
        entropy = hilbert_space.compute_entropy(mixed_state)

        # Test entropy properties
        assert entropy >= 0, "Entropy must be non-negative"
        assert entropy <= np.log(hilbert_dim), "Entropy must not exceed maximum"

        # Test pure state entropy
        pure_entropy = hilbert_space.compute_entropy(pure_states[0].density_matrix())
        assert torch.allclose(
            pure_entropy, torch.tensor(0.0, dtype=torch.float64), rtol=1e-5
        ), "Pure states should have zero entropy"

    def test_geometric_structure(self, hilbert_space, hilbert_dim):
        """Test geometric structure of Hilbert space."""
        # Create test states
        state1 = hilbert_space.prepare_state(torch.randn(hilbert_dim * 2, dtype=torch.float64))
        state2 = hilbert_space.prepare_state(torch.randn(hilbert_dim * 2, dtype=torch.float64))

        # Test Fubini-Study metric
        distance = hilbert_space.fubini_study_distance(state1, state2)
        assert distance >= 0, "Distance must be non-negative"
        assert distance <= np.pi / 2, "Maximum distance in CP^n is π/2"

        # Test parallel transport
        tangent = hilbert_space.quantum_tangent_vector(state1)
        transported = hilbert_space.parallel_transport(tangent, state1, state2)

        # Test transport preserves norm
        assert torch.allclose(
            tangent.norm(), transported.norm(), rtol=1e-4
        ), "Parallel transport should preserve norm"

    def test_entanglement(self, hilbert_space):
        """Test entanglement measures."""
        # Create Bell state (maximally entangled)
        bell_state = torch.tensor(
            [1.0, 0.0, 0.0, 1.0], dtype=torch.complex128
        ) / np.sqrt(2)
        bell_density = bell_state.outer(bell_state)

        # Create separable state
        separable_state = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.complex128)
        separable_density = separable_state.outer(separable_state)

        # Test entanglement entropy
        bell_entropy = hilbert_space.entanglement_entropy(bell_density)
        separable_entropy = hilbert_space.entanglement_entropy(separable_density)

        assert torch.allclose(
            bell_entropy, torch.tensor(np.log(2), dtype=torch.float64), rtol=1e-4
        ), "Bell state should have maximum entanglement"
        assert torch.allclose(
            separable_entropy, torch.tensor(0.0, dtype=torch.float64), rtol=1e-5
        ), "Separable state should have zero entanglement"

    def test_quantum_channels(self, hilbert_space, hilbert_dim):
        """Test quantum channel operations and properties."""
        # Create test state
        initial_state = hilbert_space.prepare_state(torch.randn(hilbert_dim * 2, dtype=torch.float64))

        # Create Kraus operators for amplitude damping channel
        gamma = 0.3  # damping parameter
        K0 = torch.zeros((hilbert_dim, hilbert_dim), dtype=torch.complex128)
        K1 = torch.zeros((hilbert_dim, hilbert_dim), dtype=torch.complex128)
        K0[0, 0] = 1.0
        K0[1, 1] = np.sqrt(1 - gamma)
        K1[0, 1] = np.sqrt(gamma)

        kraus_ops = [K0, K1]

        # Apply channel
        final_state = hilbert_space.apply_quantum_channel(initial_state, kraus_ops)

        # Test complete positivity
        assert torch.all(
            torch.linalg.eigvals(final_state.density_matrix()).real >= -1e-10
        ), "Quantum channel must preserve positivity"

        # Test trace preservation
        assert torch.allclose(
            torch.trace(final_state.density_matrix()),
            torch.tensor(1.0, dtype=torch.complex128),
            rtol=1e-5,
        ), "Quantum channel must preserve trace"

    def test_state_tomography(self, hilbert_space, hilbert_dim):
        """Test quantum state tomography procedures."""
        # Create unknown test state
        true_state = hilbert_space.prepare_state(torch.randn(hilbert_dim * 2, dtype=torch.float64))

        # Generate Pauli basis measurements
        pauli_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
        pauli_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
        pauli_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)

        # Perform tomographic measurements
        measurements = {
            "X": hilbert_space.measure_observable(true_state, pauli_x),
            "Y": hilbert_space.measure_observable(true_state, pauli_y),
            "Z": hilbert_space.measure_observable(true_state, pauli_z),
        }

        # Reconstruct state
        reconstructed_state = hilbert_space.reconstruct_state(measurements)

        # Test fidelity between true and reconstructed states
        fidelity = torch.abs(torch.vdot(true_state.amplitudes, reconstructed_state.amplitudes))**2
        assert fidelity > 0.95, "Tomographic reconstruction should be accurate"

    def test_decoherence(self, hilbert_space, hilbert_dim):
        """Test decoherence effects on quantum states."""
        # Create initial superposition state
        initial_state = hilbert_space.prepare_state(
            torch.tensor([1.0, 1.0], dtype=torch.float64) / np.sqrt(2)
        )

        # Define decoherence times
        T1 = 1.0  # relaxation time
        T2 = 0.5  # dephasing time
        times = torch.linspace(0, 2.0, 10, dtype=torch.float64)

        # Evolve under decoherence
        evolved_states = hilbert_space.evolve_with_decoherence(
            initial_state, T1, T2, times
        )

        # Test monotonic decay of coherence
        coherences = [
            torch.abs(state.density_matrix()[0, 1]) for state in evolved_states
        ]
        assert all(
            c1 >= c2 for c1, c2 in zip(coherences[:-1], coherences[1:])
        ), "Coherence should decay monotonically"

    def test_geometric_phase(self, hilbert_space, hilbert_dim):
        """Test geometric (Berry) phase computation."""

        # Create cyclic evolution path
        def hamiltonian(t):
            theta = 2 * np.pi * t
            return torch.tensor(
                [[np.cos(theta), np.sin(theta)], [np.sin(theta), -np.cos(theta)]],
                dtype=torch.complex128,
            )

        # Initial state
        initial_state = hilbert_space.prepare_state(
            torch.tensor([1.0, 0.0], dtype=torch.float64)
        )

        # Compute Berry phase
        times = torch.linspace(0, 1.0, 100, dtype=torch.float64)
        berry_phase = hilbert_space.compute_berry_phase(
            initial_state, hamiltonian, times
        )

        # Test phase is real and matches theoretical value
        assert torch.abs(berry_phase.imag) < 1e-5, "Berry phase should be real"
        assert torch.allclose(
            berry_phase.real, torch.tensor(np.pi, dtype=torch.float64), rtol=1e-2
        ), "Berry phase should match theoretical value"

    def test_advanced_entanglement(self, hilbert_space):
        """Test advanced entanglement measures."""
        # Create various entangled states
        bell_plus = torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.complex128) / np.sqrt(2)
        bell_minus = torch.tensor([1.0, 0.0, 0.0, -1.0], dtype=torch.complex128) / np.sqrt(2)

        # Test concurrence
        bell_plus_dm = torch.outer(bell_plus, bell_plus.conj())
        concurrence_plus = hilbert_space.compute_concurrence(bell_plus_dm)
        assert torch.allclose(
            concurrence_plus, torch.tensor(1.0, dtype=torch.float64), rtol=1e-5
        ), "Bell state should have maximum concurrence"

        # Test negativity
        negativity_plus = hilbert_space.compute_negativity(bell_plus_dm)
        assert torch.allclose(
            negativity_plus, torch.tensor(0.5, dtype=torch.float64), rtol=1e-5
        ), "Bell state should have expected negativity"

        # Test entanglement witness
        witness_val = hilbert_space.evaluate_entanglement_witness(bell_plus_dm)
        assert witness_val < 0, "Entanglement witness should detect entanglement"
