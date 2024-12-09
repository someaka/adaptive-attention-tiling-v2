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

from src.core.quantum.state_space import QuantumState, QuantumStateSpace


class TestQuantumStateSpace:
    @pytest.fixture
    def hilbert_dim(self):
        """Dimension of test Hilbert space."""
        return 4

    @pytest.fixture
    def batch_size(self):
        """Batch size for vectorized operations."""
        return 8

    @pytest.fixture
    def quantum_space(self, hilbert_dim):
        """Create a test quantum state space."""
        return QuantumStateSpace(
            dimension=hilbert_dim,
            metric_type="Fubini-Study"
        )

    def test_state_preparation(self, quantum_space, hilbert_dim, batch_size):
        """Test quantum state preparation from classical data."""
        # Create test classical data
        classical_data = torch.randn(batch_size, hilbert_dim * 2)  # Complex embedding

        # Prepare quantum state
        quantum_state = quantum_space.prepare_state(classical_data)

        # Test normalization
        norms = quantum_state.norm()
        assert torch.allclose(
            norms, torch.ones_like(norms), rtol=1e-5
        ), "Quantum states must be normalized"

        # Test state type
        assert isinstance(
            quantum_state, QuantumState
        ), "Output should be a QuantumState instance"

        # Test batch shape
        assert quantum_state.shape[0] == batch_size, "Batch dimension should be preserved"

    def test_state_evolution(self, quantum_space, hilbert_dim):
        """Test unitary evolution of quantum states."""
        # Create initial state
        initial_state = quantum_space.prepare_state(torch.randn(hilbert_dim * 2))

        # Create test Hamiltonian (Hermitian matrix)
        H = torch.randn(hilbert_dim, hilbert_dim) + 1j * torch.randn(hilbert_dim, hilbert_dim)
        H = H + H.conj().T  # Make Hermitian

        # Evolve state
        time = torch.linspace(0, 1, 10)
        evolved_states = quantum_space.evolve_state(initial_state, H, time)

        # Test unitarity
        for state in evolved_states:
            assert torch.allclose(
                state.norm(), torch.tensor(1.0), rtol=1e-5
            ), "Evolution must preserve normalization"

        # Test time-reversal
        reversed_state = quantum_space.evolve_state(evolved_states[-1], -H, time)[-1]
        assert torch.allclose(
            initial_state.data, reversed_state.data, rtol=1e-4
        ), "Time reversal should recover initial state"

    def test_measurement(self, quantum_space, hilbert_dim):
        """Test quantum measurement protocols."""
        # Create test state
        state = quantum_space.prepare_state(torch.randn(hilbert_dim * 2))

        # Create observable (Hermitian operator)
        observable = torch.randn(hilbert_dim, hilbert_dim)
        observable = observable + observable.T  # Make Hermitian

        # Perform measurement
        expectation = quantum_space.measure_observable(state, observable)

        # Test expectation value is real
        assert torch.allclose(
            expectation.imag, torch.zeros_like(expectation.imag), rtol=1e-5
        ), "Expectation values must be real"

        # Test variance is non-negative
        variance = quantum_space.measure_variance(state, observable)
        assert variance >= 0, "Variance must be non-negative"

    def test_entropy_computation(self, quantum_space, hilbert_dim):
        """Test von Neumann entropy computation."""
        # Create mixed state density matrix
        pure_states = [
            quantum_space.prepare_state(torch.randn(hilbert_dim * 2))
            for _ in range(3)
        ]
        weights = torch.softmax(torch.randn(3), dim=0)
        mixed_state = sum(w * p.density_matrix() for w, p in zip(weights, pure_states))

        # Compute entropy
        entropy = quantum_space.compute_entropy(mixed_state)

        # Test entropy properties
        assert entropy >= 0, "Entropy must be non-negative"
        assert entropy <= np.log(hilbert_dim), "Entropy must not exceed maximum"

        # Test pure state entropy
        pure_entropy = quantum_space.compute_entropy(pure_states[0].density_matrix())
        assert torch.allclose(
            pure_entropy, torch.tensor(0.0), rtol=1e-5
        ), "Pure states should have zero entropy"

    def test_geometric_structure(self, quantum_space, hilbert_dim):
        """Test geometric structure of quantum state space."""
        # Create test states
        state1 = quantum_space.prepare_state(torch.randn(hilbert_dim * 2))
        state2 = quantum_space.prepare_state(torch.randn(hilbert_dim * 2))

        # Test Fubini-Study metric
        distance = quantum_space.fubini_study_distance(state1, state2)
        assert distance >= 0, "Distance must be non-negative"
        assert distance <= np.pi/2, "Maximum distance in CP^n is π/2"

        # Test parallel transport
        tangent = quantum_space.quantum_tangent_vector(state1)
        transported = quantum_space.parallel_transport(tangent, state1, state2)

        # Test transport preserves norm
        assert torch.allclose(
            tangent.norm(), transported.norm(), rtol=1e-4
        ), "Parallel transport should preserve norm"

    def test_entanglement(self, quantum_space):
        """Test entanglement measures."""
        # Create Bell state (maximally entangled)
        bell_state = (
            torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.complex64) / np.sqrt(2)
        )
        bell_density = bell_state.outer(bell_state)

        # Create separable state
        separable_state = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], dtype=torch.complex64
        )
        separable_density = separable_state.outer(separable_state)

        # Test entanglement entropy
        bell_entropy = quantum_space.entanglement_entropy(bell_density)
        separable_entropy = quantum_space.entanglement_entropy(separable_density)

        assert torch.allclose(
            bell_entropy, torch.tensor(np.log(2)), rtol=1e-4
        ), "Bell state should have maximum entanglement"
        assert torch.allclose(
            separable_entropy, torch.tensor(0.0), rtol=1e-5
        ), "Separable state should have zero entanglement"

    def test_quantum_channels(self, quantum_space, hilbert_dim):
        """Test quantum channel operations and properties."""
        # Create test state
        initial_state = quantum_space.prepare_state(torch.randn(hilbert_dim * 2))

        # Create Kraus operators for amplitude damping channel
        gamma = 0.3  # damping parameter
        K0 = torch.zeros((hilbert_dim, hilbert_dim), dtype=torch.complex64)
        K1 = torch.zeros((hilbert_dim, hilbert_dim), dtype=torch.complex64)
        K0[0, 0] = 1.0
        K0[1, 1] = np.sqrt(1 - gamma)
        K1[0, 1] = np.sqrt(gamma)

        kraus_ops = [K0, K1]

        # Apply channel
        final_state = quantum_space.apply_quantum_channel(initial_state, kraus_ops)

        # Test complete positivity
        assert torch.all(
            torch.linalg.eigvals(final_state.density_matrix()).real >= -1e-10
        ), "Quantum channel must preserve positivity"

        # Test trace preservation
        assert torch.allclose(
            torch.trace(final_state.density_matrix()),
            torch.tensor(1.0, dtype=torch.complex64),
            rtol=1e-5
        ), "Quantum channel must preserve trace"

    def test_state_tomography(self, quantum_space, hilbert_dim):
        """Test quantum state tomography procedures."""
        # Create unknown test state
        true_state = quantum_space.prepare_state(torch.randn(hilbert_dim * 2))

        # Generate Pauli basis measurements
        pauli_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
        pauli_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
        pauli_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)

        # Perform tomographic measurements
        measurements = {
            "X": quantum_space.measure_observable(true_state, pauli_x),
            "Y": quantum_space.measure_observable(true_state, pauli_y),
            "Z": quantum_space.measure_observable(true_state, pauli_z)
        }

        # Reconstruct state
        reconstructed_state = quantum_space.reconstruct_state(measurements)

        # Test fidelity between true and reconstructed states
        fidelity = quantum_space.state_fidelity(true_state, reconstructed_state)
        assert fidelity > 0.95, "Tomographic reconstruction should be accurate"

    def test_decoherence(self, quantum_space, hilbert_dim):
        """Test decoherence effects on quantum states."""
        # Create initial superposition state
        initial_state = quantum_space.prepare_state(
            torch.tensor([1.0, 1.0], dtype=torch.complex64) / np.sqrt(2)
        )

        # Define decoherence times
        T1 = 1.0  # relaxation time
        T2 = 0.5  # dephasing time
        times = torch.linspace(0, 2.0, 10)

        # Evolve under decoherence
        evolved_states = quantum_space.evolve_with_decoherence(
            initial_state, T1, T2, times
        )

        # Test monotonic decay of coherence
        coherences = [
            torch.abs(state.density_matrix()[0, 1]) for state in evolved_states
        ]
        assert all(
            c1 >= c2 for c1, c2 in zip(coherences[:-1], coherences[1:])
        ), "Coherence should decay monotonically"

    def test_geometric_phase(self, quantum_space, hilbert_dim):
        """Test geometric (Berry) phase computation."""
        # Create cyclic evolution path
        def hamiltonian(t):
            theta = 2 * np.pi * t
            return torch.tensor([
                [np.cos(theta), np.sin(theta)],
                [np.sin(theta), -np.cos(theta)]
            ], dtype=torch.complex64)

        # Initial state
        initial_state = quantum_space.prepare_state(
            torch.tensor([1.0, 0.0], dtype=torch.complex64)
        )

        # Compute Berry phase
        times = torch.linspace(0, 1.0, 100)
        berry_phase = quantum_space.compute_berry_phase(
            initial_state, hamiltonian, times
        )

        # Test phase is real and matches theoretical value
        assert torch.abs(berry_phase.imag) < 1e-5, "Berry phase should be real"
        assert torch.allclose(
            berry_phase.real,
            torch.tensor(np.pi),
            rtol=1e-2
        ), "Berry phase should match theoretical value"

    def test_advanced_entanglement(self, quantum_space):
        """Test advanced entanglement measures."""
        # Create various entangled states
        bell_plus = torch.tensor(
            [1.0, 0.0, 0.0, 1.0], dtype=torch.complex64
        ) / np.sqrt(2)
        bell_minus = torch.tensor(
            [1.0, 0.0, 0.0, -1.0], dtype=torch.complex64
        ) / np.sqrt(2)

        # Test concurrence
        concurrence_plus = quantum_space.compute_concurrence(
            bell_plus.outer(bell_plus)
        )
        assert torch.allclose(
            concurrence_plus,
            torch.tensor(1.0),
            rtol=1e-5
        ), "Bell state should have maximum concurrence"

        # Test negativity
        negativity_plus = quantum_space.compute_negativity(
            bell_plus.outer(bell_plus)
        )
        assert torch.allclose(
            negativity_plus,
            torch.tensor(0.5),
            rtol=1e-5
        ), "Bell state should have expected negativity"

        # Test entanglement witnesses
        witness_val = quantum_space.evaluate_entanglement_witness(
            bell_minus.outer(bell_minus)
        )
        assert witness_val < 0, "Witness should detect entanglement"
