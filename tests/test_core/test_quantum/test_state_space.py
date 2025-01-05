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
import torch.nn.functional as F

from src.core.quantum.state_space import QuantumState, HilbertSpace


class TestStateSpace:
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

    @pytest.fixture
    def test_state(self, hilbert_space):
        """Create test quantum state."""
        amplitudes = torch.randn(8, hilbert_space.dim, dtype=torch.complex128)  # [batch_size, dim]
        amplitudes = F.normalize(amplitudes, p=2, dim=-1)
        return QuantumState(
            amplitudes=amplitudes,
            basis_labels=[f"|{i}⟩" for i in range(hilbert_space.dim)],
            phase=torch.zeros(hilbert_space.dim, dtype=torch.complex128)
        )

    def test_state_preparation(self, hilbert_space, hilbert_dim, batch_size):
        """Test quantum state preparation from classical data."""
        # Create test classical data
        classical_data = torch.randn(batch_size, hilbert_dim * 2, dtype=torch.float64)  # Complex embedding

        # Prepare quantum state
        quantum_state = hilbert_space.prepare_state(classical_data)

        # Test normalization
        norms = torch.norm(quantum_state.amplitudes, p=2, dim=-1)
        assert torch.allclose(
            norms, torch.ones_like(norms, dtype=torch.float64), rtol=1e-5
        ), "Quantum states must be normalized"

        # Test state type
        assert isinstance(
            quantum_state, QuantumState
        ), "Output should be a QuantumState instance"

        # Test batch shape
        assert (
            quantum_state.amplitudes.shape[0] == batch_size
        ), "Batch dimension should be preserved"

    def test_state_evolution(self, hilbert_space, hilbert_dim):
        """Test unitary evolution of quantum states."""
        # Create initial state
        initial_state = QuantumState(
            amplitudes=F.normalize(torch.randn(hilbert_dim, dtype=torch.complex128).unsqueeze(0), p=2, dim=-1).squeeze(),
            basis_labels=[f"|{i}⟩" for i in range(hilbert_dim)],
            phase=torch.zeros(hilbert_dim, dtype=torch.complex128)
        )

        # Create test Hamiltonian (Hermitian matrix)
        H = torch.randn(hilbert_dim, hilbert_dim, dtype=torch.complex128) + 1j * torch.randn(
            hilbert_dim, hilbert_dim, dtype=torch.complex128
        )
        H = H + H.conj().T  # Make Hermitian

        # Evolve state
        time = torch.linspace(0, 1, 5, dtype=torch.float64)
        evolved_states = hilbert_space.evolve_state(initial_state, H, time)

        # Test unitarity
        for state in evolved_states:
            assert torch.allclose(
                state.norm(), torch.tensor(1.0, dtype=torch.float64), rtol=1e-5
            ), "Evolution must preserve normalization"

        # Test time-reversal
        assert isinstance(evolved_states, list), "Expected list of states for multiple time points"
        last_state = evolved_states[-1]  # Get the last state from the list
        assert isinstance(last_state, QuantumState), "Expected QuantumState"
        reversed_states = hilbert_space.evolve_state(last_state, -H, time)
        assert isinstance(reversed_states, list), "Expected list of states for multiple time points"
        reversed_state = reversed_states[-1]  # Get the last state from reversed evolution
        assert isinstance(reversed_state, QuantumState), "Expected QuantumState"
        assert torch.allclose(
            initial_state.amplitudes, reversed_state.amplitudes, rtol=1e-4
        ), "Time reversal should recover initial state"

    def test_measurement(self, hilbert_space, hilbert_dim):
        """Test quantum measurement protocols."""
        # Create test state
        state = QuantumState(
            amplitudes=F.normalize(torch.randn(hilbert_dim, dtype=torch.complex128).unsqueeze(0), p=2, dim=-1).squeeze(),
            basis_labels=[f"|{i}⟩" for i in range(hilbert_dim)],
            phase=torch.zeros(hilbert_dim, dtype=torch.complex128)
        )

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
            QuantumState(
                amplitudes=F.normalize(torch.randn(hilbert_dim, dtype=torch.complex128).unsqueeze(0), p=2, dim=-1).squeeze(),
                basis_labels=[f"|{i}⟩" for i in range(hilbert_dim)],
                phase=torch.zeros(hilbert_dim, dtype=torch.complex128)
            )
            for _ in range(3)
        ]
        weights = F.softmax(torch.randn(3), dim=0)
        mixed_state = sum(w * torch.outer(p.amplitudes, p.amplitudes.conj()) for w, p in zip(weights, pure_states))

        # Compute entropy
        entropy = hilbert_space.compute_entropy(mixed_state)

        # Test entropy properties
        assert entropy >= 0, "Entropy must be non-negative"
        assert entropy <= np.log(hilbert_dim), "Entropy must not exceed maximum"

        # Test pure state entropy
        pure_entropy = hilbert_space.compute_entropy(pure_states[0])
        assert torch.allclose(
            pure_entropy, torch.tensor(0.0, dtype=torch.float64), rtol=1e-5
        ), "Pure states should have zero entropy"

    def test_geometric_structure(self, hilbert_space, hilbert_dim):
        """Test geometric structure of Hilbert space."""
        # Create test states
        state1 = QuantumState(
            amplitudes=F.normalize(torch.randn(hilbert_dim, dtype=torch.complex128).unsqueeze(0), p=2, dim=-1).squeeze(),
            basis_labels=[f"|{i}⟩" for i in range(hilbert_dim)],
            phase=torch.zeros(hilbert_dim, dtype=torch.complex128)
        )
        state2 = QuantumState(
            amplitudes=F.normalize(torch.randn(hilbert_dim, dtype=torch.complex128).unsqueeze(0), p=2, dim=-1).squeeze(),
            basis_labels=[f"|{i}⟩" for i in range(hilbert_dim)],
            phase=torch.zeros(hilbert_dim, dtype=torch.complex128)
        )

        # Test Fubini-Study metric
        distance = hilbert_space.fubini_study_distance(state1, state2)
        assert distance >= 0, "Distance must be non-negative"
        assert distance <= np.pi / 2, "Maximum distance in CP^n is π/2"

        # Test parallel transport
        tangent = hilbert_space.quantum_tangent_vector(state1)
        transported = hilbert_space.parallel_transport(tangent, state1, state2)

        # Test transport preserves norm
        assert torch.allclose(
            torch.norm(tangent), torch.norm(transported), rtol=1e-4
        ), "Parallel transport should preserve norm"

    def test_entanglement(self, hilbert_space):
        """Test entanglement measures."""
        # Create Bell state (maximally entangled)
        bell_state = QuantumState(
            amplitudes=torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.complex128) / np.sqrt(2),
            basis_labels=["|00⟩", "|01⟩", "|10⟩", "|11⟩"],
            phase=torch.zeros(4, dtype=torch.complex128)
        )
        bell_density = torch.outer(bell_state.amplitudes, bell_state.amplitudes.conj())

        # Create separable state
        separable_state = QuantumState(
            amplitudes=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.complex128),
            basis_labels=["|00⟩", "|01⟩", "|10⟩", "|11⟩"],
            phase=torch.zeros(4, dtype=torch.complex128)
        )
        separable_density = torch.outer(separable_state.amplitudes, separable_state.amplitudes.conj())

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
        initial_state = QuantumState(
            amplitudes=F.normalize(torch.randn(hilbert_dim, dtype=torch.complex128).unsqueeze(0), p=2, dim=-1).squeeze(),
            basis_labels=[f"|{i}⟩" for i in range(hilbert_dim)],
            phase=torch.zeros(hilbert_dim, dtype=torch.complex128)
        )

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
        final_density = torch.outer(final_state.amplitudes, final_state.amplitudes.conj())
        eigenvals = torch.linalg.eigvalsh(final_density).real
        assert torch.all(eigenvals >= -1e-10), "Quantum channel must preserve positivity"

        # Test trace preservation
        assert torch.allclose(
            torch.trace(final_density),
            torch.tensor(1.0, dtype=torch.complex128),
            rtol=1e-5
        ), "Quantum channel must preserve trace"

    def test_state_tomography(self, hilbert_space, hilbert_dim):
        """Test quantum state tomography procedures."""
        # Create unknown test state with controlled randomness
        torch.manual_seed(42)  # For reproducibility
        true_state = QuantumState(
            amplitudes=F.normalize(torch.randn(hilbert_dim, dtype=torch.complex128).unsqueeze(0), p=2, dim=-1).squeeze(),
            basis_labels=[f"|{i}⟩" for i in range(hilbert_dim)],
            phase=torch.zeros(hilbert_dim, dtype=torch.complex128)
        )

        # Generate Pauli basis measurements for each qubit
        pauli_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
        pauli_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
        pauli_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
        identity = torch.eye(2, dtype=torch.complex128)

        # Perform tomographic measurements for each qubit
        measurements = {}
        n_qubits = int(np.log2(hilbert_dim))

        # Create measurement operators including identity
        measurement_ops = []
        for q in range(n_qubits):
            # Create extended operators for each qubit
            for basis, label in [(pauli_x, "X"), (pauli_y, "Y"), (pauli_z, "Z"), (identity, "I")]:
                # Build the full operator using tensor products
                if q == 0:
                    full_op = basis
                    for _ in range(1, n_qubits):
                        full_op = torch.kron(full_op, identity)
                else:
                    full_op = identity
                    for i in range(1, n_qubits):
                        if i == q:
                            full_op = torch.kron(full_op, basis)
                        else:
                            full_op = torch.kron(full_op, identity)
                
                measurement_ops.append(full_op)
                measurements[f"{label}{q}"] = hilbert_space.measure_observable(true_state, full_op)

        # Add joint measurements for better reconstruction
        for q1 in range(n_qubits):
            for q2 in range(q1 + 1, n_qubits):
                for b1, l1 in [(pauli_x, "X"), (pauli_y, "Y"), (pauli_z, "Z")]:
                    for b2, l2 in [(pauli_x, "X"), (pauli_y, "Y"), (pauli_z, "Z")]:
                        # Build joint measurement operator
                        full_op = torch.ones(1, dtype=torch.complex128)
                        for i in range(n_qubits):
                            if i == q1:
                                full_op = torch.kron(full_op, b1)
                            elif i == q2:
                                full_op = torch.kron(full_op, b2)
                            else:
                                full_op = torch.kron(full_op, identity)
                        measurements[f"{l1}{q1}{l2}{q2}"] = hilbert_space.measure_observable(true_state, full_op)

        # Reconstruct state using all measurements
        reconstructed_state = hilbert_space.reconstruct_state(measurements)

        # Test fidelity between true and reconstructed states
        fidelity = hilbert_space.state_fidelity(true_state, reconstructed_state)
        assert fidelity > 0.90, f"Tomographic reconstruction should be reasonably accurate (got {fidelity:.4f})"

    def test_decoherence(self, hilbert_space, hilbert_dim):
        """Test decoherence effects on quantum states."""
        # Create initial superposition state
        initial_state = QuantumState(
            amplitudes=F.normalize(torch.tensor([1.0, 1.0], dtype=torch.complex128).unsqueeze(0), p=2, dim=-1).squeeze(),
            basis_labels=[f"|{i}⟩" for i in range(2)],
            phase=torch.zeros(2, dtype=torch.complex128)
        )

        # Define decoherence times
        T1 = 1.0  # relaxation time
        T2 = 0.5  # dephasing time
        times = torch.linspace(0, 2.0, 10, dtype=torch.float64)

        # Evolve under decoherence
        evolved_states = hilbert_space.evolve_with_decoherence(initial_state, T1, T2, times)

        # Test monotonic decay of coherence
        coherences = [torch.abs(torch.outer(state.amplitudes, state.amplitudes.conj())[0, 1]) for state in evolved_states]
        assert all(c1 >= c2 for c1, c2 in zip(coherences[:-1], coherences[1:])), "Coherence should decay monotonically"

    def test_geometric_phase(self, hilbert_space, hilbert_dim):
        """Test geometric (Berry) phase computation."""
        # Create initial state
        initial_state = QuantumState(
            amplitudes=F.normalize(torch.tensor([1.0, 0.0], dtype=torch.complex128).unsqueeze(0), p=2, dim=-1).squeeze(),
            basis_labels=[f"|{i}⟩" for i in range(2)],
            phase=torch.zeros(2, dtype=torch.complex128)
        )

        # Define time-dependent Hamiltonian
        def hamiltonian(t: float) -> torch.Tensor:
            theta = 2 * np.pi * t
            return torch.tensor(
                [[np.cos(theta), np.sin(theta)], [np.sin(theta), -np.cos(theta)]],
                dtype=torch.complex128
            )

        # Compute Berry phase
        times = torch.linspace(0, 1.0, 20, dtype=torch.float64)
        berry_phase = hilbert_space.compute_berry_phase(initial_state, hamiltonian, times)

        # Test phase is real
        assert torch.abs(berry_phase.imag) < 1e-5, "Berry phase should be real"
        assert torch.allclose(
            berry_phase.real, torch.tensor(np.pi, dtype=torch.float64), rtol=1e-2
        ), "Berry phase should match theoretical value"

    def test_advanced_entanglement(self, hilbert_space):
        """Test advanced entanglement measures."""
        # Create various entangled states
        bell_plus = QuantumState(
            amplitudes=torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.complex128) / np.sqrt(2),
            basis_labels=["|00⟩", "|01⟩", "|10⟩", "|11⟩"],
            phase=torch.zeros(4, dtype=torch.complex128)
        )
        bell_minus = QuantumState(
            amplitudes=torch.tensor([1.0, 0.0, 0.0, -1.0], dtype=torch.complex128) / np.sqrt(2),
            basis_labels=["|00⟩", "|01⟩", "|10⟩", "|11⟩"],
            phase=torch.zeros(4, dtype=torch.complex128)
        )

        # Test concurrence
        bell_plus_dm = torch.outer(bell_plus.amplitudes, bell_plus.amplitudes.conj())
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
        witness_val = hilbert_space.evaluate_entanglement_witness(bell_plus)
        assert witness_val < 0, "Entanglement witness should detect entanglement"

    def test_scale_state(self, hilbert_space, test_state):
        """Test quantum state scaling."""
        # Test upscaling
        scale_factor = 2.0
        scaled_state = hilbert_space.scale_state(test_state, scale_factor)
        
        # Check properties
        assert isinstance(scaled_state, QuantumState)
        assert scaled_state.amplitudes.shape == test_state.amplitudes.shape
        assert torch.all(torch.isfinite(scaled_state.amplitudes))
        
        # Check normalization is preserved
        norms = torch.norm(scaled_state.amplitudes, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms))
        
        # Test downscaling
        scale_factor = 0.5
        scaled_state = hilbert_space.scale_state(test_state, scale_factor)
        
        # Check properties
        assert isinstance(scaled_state, QuantumState)
        assert scaled_state.amplitudes.shape == test_state.amplitudes.shape
        assert torch.all(torch.isfinite(scaled_state.amplitudes))
        
        # Check normalization is preserved
        norms = torch.norm(scaled_state.amplitudes, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms))
        
    def test_invalid_scale_factors(self, hilbert_space, test_state):
        """Test scaling with invalid factors."""
        # Test zero scale
        with pytest.raises(ValueError):
            hilbert_space.scale_state(test_state, 0.0)
            
        # Test negative scale
        with pytest.raises(ValueError):
            hilbert_space.scale_state(test_state, -1.0)

    def test_state_conversion_fidelity(self, hilbert_space, test_state):
        """Test fidelity preservation during state conversion."""
        # Initial state properties
        initial_norm = torch.linalg.vector_norm(test_state.amplitudes, dim=-1)
        initial_phase = test_state.phase.clone()
        
        # Convert to neural representation and back
        neural_repr = hilbert_space.state_to_neural(test_state)
        reconstructed = hilbert_space.neural_to_state(neural_repr)
        
        # Check norm preservation
        final_norm = torch.linalg.vector_norm(reconstructed.amplitudes, dim=-1)
        norm_diff = torch.abs(final_norm - initial_norm)
        assert torch.all(norm_diff < 1e-6), "Norm not preserved during conversion"
        
        # Check phase preservation
        phase_diff = torch.abs(reconstructed.phase - initial_phase)
        assert torch.all(phase_diff < 1e-6), "Phase not preserved during conversion"
        
        # Check fidelity (overlap) between initial and reconstructed states
        fidelity = torch.abs(torch.sum(
            torch.conj(test_state.amplitudes) * reconstructed.amplitudes, 
            dim=-1
        ))
        assert torch.all(fidelity > 0.99), "Low fidelity in state reconstruction"

    def test_batch_conversion_consistency(self, hilbert_space):
        """Test consistency of batch conversion operations."""
        # Create batch of states with different properties
        batch_size = 8
        states = []
        for i in range(batch_size):
            amplitudes = torch.randn(hilbert_space.dim, dtype=torch.complex128)
            amplitudes = F.normalize(amplitudes, p=2, dim=-1)
            phase = torch.exp(1j * torch.rand(hilbert_space.dim, dtype=torch.complex128))
            states.append(QuantumState(
                amplitudes=amplitudes,
                basis_labels=[f"|{j}⟩" for j in range(hilbert_space.dim)],
                phase=phase
            ))
            
        # Convert batch to neural form
        neural_batch = torch.stack([
            hilbert_space.state_to_neural(state) 
            for state in states
        ])
        
        # Convert back and check consistency
        reconstructed_states = [
            hilbert_space.neural_to_state(neural)
            for neural in neural_batch
        ]
        
        for orig, recon in zip(states, reconstructed_states):
            # Check amplitude preservation
            amp_diff = torch.abs(orig.amplitudes - recon.amplitudes)
            assert torch.all(amp_diff < 1e-6), "Amplitude not preserved in batch"
            
            # Check phase consistency
            phase_diff = torch.abs(orig.phase - recon.phase)
            assert torch.all(phase_diff < 1e-6), "Phase not preserved in batch"

    def test_conversion_edge_cases(self, hilbert_space):
        """Test state conversion for edge cases."""
        # Test zero state
        zero_amplitudes = torch.zeros(hilbert_space.dim, dtype=torch.complex128)
        zero_state = QuantumState(
            amplitudes=zero_amplitudes,
            basis_labels=[f"|{i}⟩" for i in range(hilbert_space.dim)],
            phase=torch.ones(hilbert_space.dim, dtype=torch.complex128)
        )
        
        # Should raise ValueError for zero state
        with pytest.raises(ValueError):
            hilbert_space.state_to_neural(zero_state)
            
        # Test maximum superposition state
        max_super = torch.ones(hilbert_space.dim, dtype=torch.complex128) / np.sqrt(hilbert_space.dim)
        super_state = QuantumState(
            amplitudes=max_super,
            basis_labels=[f"|{i}⟩" for i in range(hilbert_space.dim)],
            phase=torch.ones(hilbert_space.dim, dtype=torch.complex128)
        )
        
        neural_super = hilbert_space.state_to_neural(super_state)
        reconstructed = hilbert_space.neural_to_state(neural_super)
        
        # Check equal superposition preserved
        amp_diff = torch.abs(reconstructed.amplitudes - max_super)
        assert torch.all(amp_diff < 1e-6), "Equal superposition not preserved"

    def test_evolution_consistency(self, hilbert_space, test_state):
        """Test consistency of quantum evolution over time."""
        # Create test Hamiltonian
        H = torch.randn(hilbert_space.dim, hilbert_space.dim, dtype=torch.complex128)
        H = H + H.conj().T  # Make Hermitian
        
        # Test different time steps
        time_steps = [5, 10, 15]
        final_states = []
        
        for steps in time_steps:
            time = torch.linspace(0, 1.0, steps, dtype=torch.float64)
            evolved = hilbert_space.evolve_state(test_state, H, time)
            final_states.append(evolved[-1])
        
        # Check consistency across different time discretizations
        for state1, state2 in zip(final_states[:-1], final_states[1:]):
            fidelity = torch.abs(torch.sum(
                torch.conj(state1.amplitudes) * state2.amplitudes,
                dim=-1
            ))
            assert torch.all(fidelity > 0.99), "Evolution should be consistent across time discretizations"

    def test_energy_conservation(self, hilbert_space, test_state):
        """Test energy conservation during evolution."""
        # Create test Hamiltonian
        H = torch.randn(hilbert_space.dim, hilbert_space.dim, dtype=torch.complex128)
        H = H + H.conj().T  # Make Hermitian
        
        # Initial energy
        initial_energy = hilbert_space.measure_observable(test_state, H)
        
        # Evolve state
        time = torch.linspace(0, 2.0, 20, dtype=torch.float64)
        evolved_states = hilbert_space.evolve_state(test_state, H, time)
        
        # Check energy conservation
        for state in evolved_states:
            energy = hilbert_space.measure_observable(state, H)
            energy_diff = torch.abs(energy - initial_energy)
            assert torch.all(energy_diff < 1e-6), "Energy should be conserved during evolution"

    def test_geometric_phase_consistency(self, hilbert_space, test_state):
        """Test consistency of geometric phase accumulation."""
        # Create cyclic Hamiltonian
        def cyclic_hamiltonian(t: float) -> torch.Tensor:
            theta = 2 * np.pi * t
            H = torch.tensor([
                [np.cos(theta), np.sin(theta)],
                [np.sin(theta), -np.cos(theta)]
            ], dtype=torch.complex128)
            return H
        
        # Create a simpler test state in 2D subspace
        simple_state = QuantumState(
            amplitudes=torch.tensor([1.0, 0.0], dtype=torch.complex128),
            basis_labels=["|0⟩", "|1⟩"],
            phase=torch.zeros(2, dtype=torch.complex128)
        )
        
        # Compute geometric phase for different time discretizations
        time_steps = [10, 20, 40]
        phases = []
        
        for steps in time_steps:
            times = torch.linspace(0, 1.0, steps, dtype=torch.float64)
            phase = hilbert_space.compute_berry_phase(simple_state, cyclic_hamiltonian, times)
            phases.append(phase)
        
        # Check phase consistency with slightly relaxed tolerance
        for phase1, phase2 in zip(phases[:-1], phases[1:]):
            phase_diff = torch.abs(phase1 - phase2)
            assert phase_diff < 2e-4, "Geometric phase should be consistent across time discretizations"

    def test_entanglement_preservation(self, hilbert_space):
        """Test preservation of entanglement during evolution."""
        # Create maximally entangled Bell state
        bell_state = QuantumState(
            amplitudes=torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.complex128) / np.sqrt(2),
            basis_labels=["|00⟩", "|01⟩", "|10⟩", "|11⟩"],
            phase=torch.zeros(4, dtype=torch.complex128)
        )
        
        # Create local Hamiltonian (acts separately on each qubit)
        H_local = torch.kron(
            torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex128),
            torch.eye(2, dtype=torch.complex128)
        )
        
        # Evolve under local Hamiltonian
        time = torch.linspace(0, 1.0, 10, dtype=torch.float64)
        evolved_states = hilbert_space.evolve_state(bell_state, H_local, time)
        
        # Check entanglement preservation
        initial_concurrence = hilbert_space.compute_concurrence(
            torch.outer(bell_state.amplitudes, bell_state.amplitudes.conj())
        )
        
        for state in evolved_states:
            density = torch.outer(state.amplitudes, state.amplitudes.conj())
            concurrence = hilbert_space.compute_concurrence(density)
            assert torch.abs(concurrence - initial_concurrence) < 1e-6, "Local evolution should preserve entanglement"

    def test_validation_framework_integration(self, hilbert_space, test_state):
        """Test integration with quantum state validation framework."""
        from src.validation.quantum.state import StateValidator, StateValidationErrorType

        # Create validator
        validator = StateValidator()

        # Create a superposition state with well-defined uncertainty relations
        # Using a Gaussian-like state which naturally satisfies uncertainty relations
        n = hilbert_space.dim
        x = torch.linspace(-2, 2, n, dtype=torch.float64)
        gaussian = torch.exp(-x**2 / 2)
        amplitudes = F.normalize(gaussian, p=2, dim=-1).to(torch.complex128)
        
        single_state = QuantumState(
            amplitudes=amplitudes,
            basis_labels=[f"|{i}⟩" for i in range(n)],
            phase=torch.zeros(n, dtype=torch.complex128)
        )

        # Test basic state properties
        properties = validator.validate_state(single_state)
        assert properties.is_normalized, "State should be normalized"
        assert properties.is_pure, "Test state should be pure"
        assert abs(properties.trace - 1.0) < 1e-6, "Trace should be 1"
        assert properties.rank > 0, "Rank should be positive"
        assert torch.all(properties.eigenvalues >= -1e-6), "Eigenvalues should be non-negative"
        assert abs(properties.purity - 1.0) < 1e-6, "Pure state should have purity 1"

        # Test uncertainty relations
        uncertainties = validator.validate_uncertainty(single_state)
        assert uncertainties.heisenberg_product >= 0.5, f"Should satisfy Heisenberg uncertainty (got {uncertainties.heisenberg_product:.4f})"
        assert uncertainties.position_uncertainty > 0, "Position uncertainty should be positive"
        assert uncertainties.momentum_uncertainty > 0, "Momentum uncertainty should be positive"

    def test_state_conversion_edge_cases(self, hilbert_space):
        """Test edge cases in state conversion."""
        # Test zero state handling
        zero_state = QuantumState(
            amplitudes=torch.zeros(hilbert_space.dim, dtype=torch.complex128),
            basis_labels=[f"|{i}⟩" for i in range(hilbert_space.dim)],
            phase=torch.zeros(hilbert_space.dim, dtype=torch.complex128)
        )
        with pytest.raises(ValueError, match="Cannot convert zero state"):
            hilbert_space.state_to_neural(zero_state)

        # Test maximum superposition state
        max_super = torch.ones(hilbert_space.dim, dtype=torch.complex128) / np.sqrt(hilbert_space.dim)
        super_state = QuantumState(
            amplitudes=max_super,
            basis_labels=[f"|{i}⟩" for i in range(hilbert_space.dim)],
            phase=torch.ones(hilbert_space.dim, dtype=torch.complex128)
        )
        neural_repr = hilbert_space.state_to_neural(super_state)
        reconstructed = hilbert_space.neural_to_state(neural_repr)
        assert torch.allclose(
            reconstructed.amplitudes, super_state.amplitudes, rtol=1e-5
        ), "Should preserve maximum superposition state"

        # Test phase wrapping
        phase_state = QuantumState(
            amplitudes=torch.ones(hilbert_space.dim, dtype=torch.complex128) / np.sqrt(hilbert_space.dim),
            basis_labels=[f"|{i}⟩" for i in range(hilbert_space.dim)],
            phase=torch.exp(1j * torch.ones(hilbert_space.dim, dtype=torch.float64) * 4 * np.pi)
        )
        neural_phase = hilbert_space.state_to_neural(phase_state)
        reconstructed_phase = hilbert_space.neural_to_state(neural_phase)
        phase_diff = torch.angle(reconstructed_phase.phase) - torch.angle(phase_state.phase)
        assert torch.allclose(
            torch.remainder(phase_diff, 2 * np.pi),
            torch.zeros_like(phase_diff),
            rtol=1e-5
        ), "Should handle phase wrapping correctly"

    def test_geometric_phase_advanced(self, hilbert_space):
        """Test advanced geometric phase properties."""
        # Create a 2D Hilbert space for this test
        hilbert_space_2d = HilbertSpace(dim=2)

        # Create cyclic Hamiltonian evolution
        def cyclic_hamiltonian(t: float) -> torch.Tensor:
            theta = 2 * np.pi * t
            H = torch.tensor([
                [np.cos(theta), np.sin(theta)],
                [np.sin(theta), -np.cos(theta)]
            ], dtype=torch.complex128)
            return H

        # Test geometric phase for different paths
        initial_state = QuantumState(
            amplitudes=torch.tensor([1.0, 0.0], dtype=torch.complex128),
            basis_labels=["|0⟩", "|1⟩"],
            phase=torch.zeros(2, dtype=torch.complex128)
        )

        # Test phase accumulation for different paths
        paths = [
            torch.linspace(0, 1.0, 10),  # Standard path
            torch.linspace(0, 2.0, 20),  # Double loop
            torch.cat([  # Composite path
                torch.linspace(0, 0.5, 5),
                torch.linspace(0.5, 0.5, 5),
                torch.linspace(0.5, 1.0, 5)
            ])
        ]

        phases = []
        for path in paths:
            phase = hilbert_space_2d.compute_geometric_phase(initial_state, path)
            phases.append(phase)

        # Check phase additivity
        assert torch.allclose(
            phases[0] * 2,
            phases[1],
            rtol=1e-4
        ), "Geometric phase should be additive for multiple loops"

        # Test phase consistency
        assert hilbert_space_2d.geometric_phase_consistency(initial_state, paths[0]), \
            "Geometric phase should be consistent under time reversal"

        # Test parallel transport consistency
        evolved_states = []
        for t in torch.linspace(0, 1.0, 10):
            H = cyclic_hamiltonian(t.item())
            state = hilbert_space_2d.evolve_state(initial_state, H, torch.tensor([0.1]))
            if isinstance(state, list):
                evolved_states.extend(state)
            else:
                evolved_states.append(state)

        # Check parallel transport between adjacent states
        for i in range(len(evolved_states) - 1):
            tangent = hilbert_space_2d.quantum_tangent_vector(evolved_states[i])
            transported = hilbert_space_2d.parallel_transport(
                tangent, evolved_states[i], evolved_states[i+1]
            )
            # Verify transport preserves norm
            assert torch.allclose(
                torch.norm(tangent),
                torch.norm(transported),
                rtol=1e-4
            ), "Parallel transport should preserve norm"


class TestStateConversion:
    """Tests for quantum state conversion operations."""
    
    @pytest.fixture
    def hilbert_space(self, hilbert_dim):
        """Create a test Hilbert space."""
        return HilbertSpace(dim=hilbert_dim)

    @pytest.fixture
    def hilbert_dim(self):
        """Dimension of test Hilbert space."""
        return 4

    @pytest.fixture
    def test_state(self, hilbert_space):
        """Create test quantum state."""
        amplitudes = torch.randn(8, hilbert_space.dim, dtype=torch.complex128)  # [batch_size, dim]
        amplitudes = F.normalize(amplitudes, p=2, dim=-1)
        return QuantumState(
            amplitudes=amplitudes,
            basis_labels=[f"|{i}⟩" for i in range(hilbert_space.dim)],
            phase=torch.zeros(hilbert_space.dim, dtype=torch.complex128)
        )

    def test_classical_to_quantum(self, hilbert_space):
        """Test conversion from classical to quantum state."""
        classical_data = torch.randn(8, hilbert_space.dim * 2)  # Complex embedding
        quantum_state = hilbert_space.prepare_state(classical_data)

        assert isinstance(quantum_state, QuantumState)
        assert torch.allclose(quantum_state.norm(), torch.tensor(1.0, dtype=torch.float64))

    def test_quantum_to_classical(self, hilbert_space, test_state):
        """Test conversion from quantum to classical states."""
        classical_data = hilbert_space.measure_state(test_state, "Z")  # Add measurement basis

        assert isinstance(classical_data, torch.Tensor)
        assert classical_data.shape[-1] == hilbert_space.dim * 2  # Real and imaginary parts

    def test_unitary_evolution(self, hilbert_space, test_state):
        """Test consistency of unitary evolution."""
        # Create test Hamiltonian
        H = torch.randn(hilbert_space.dim, hilbert_space.dim, dtype=torch.complex128)
        H = H + H.conj().T  # Make Hermitian

        # Evolve state
        time = torch.linspace(0, 1.0, 10)
        evolved_states = hilbert_space.evolve_state(test_state, H, time)

        # Check unitarity preservation
        for state in evolved_states:
            assert torch.allclose(state.norm(), torch.tensor(1.0, dtype=torch.float64), rtol=1e-5)

    def test_observable_expectation(self, hilbert_space, test_state):
        """Test consistency of observable expectations during evolution."""
        # Create test observable
        O = torch.randn(hilbert_space.dim, hilbert_space.dim, dtype=torch.complex128)
        O = O + O.conj().T  # Make Hermitian

        # Initial expectation
        initial_exp = hilbert_space.measure_observable(test_state, O)  # Use measure_observable instead of expectation_value


class TestEvolutionConsistency:
    """Tests for quantum evolution consistency."""
    
    @pytest.fixture
    def hilbert_space(self, hilbert_dim):
        """Create a test Hilbert space."""
        return HilbertSpace(dim=hilbert_dim)

    @pytest.fixture
    def hilbert_dim(self):
        """Dimension of test Hilbert space."""
        return 4

    @pytest.fixture
    def test_state(self, hilbert_space):
        """Create test quantum state."""
        amplitudes = torch.randn(8, hilbert_space.dim, dtype=torch.complex128)  # [batch_size, dim]
        amplitudes = F.normalize(amplitudes, p=2, dim=-1)
        return QuantumState(
            amplitudes=amplitudes,
            basis_labels=[f"|{i}⟩" for i in range(hilbert_space.dim)],
            phase=torch.zeros(hilbert_space.dim, dtype=torch.complex128)
        )

    def test_unitary_evolution(self, hilbert_space, test_state):
        """Test consistency of unitary evolution."""
        # Create test Hamiltonian
        H = torch.randn(hilbert_space.dim, hilbert_space.dim, dtype=torch.complex128)
        H = H + H.conj().T  # Make Hermitian
        
        # Evolve state
        time = torch.linspace(0, 1.0, 10)
        evolved_states = hilbert_space.evolve_state(test_state, H, time)
        
        # Check unitarity preservation
        for state in evolved_states:
            assert torch.allclose(state.norm(), torch.tensor(1.0, dtype=torch.float64), rtol=1e-5)
            
    def test_observable_expectation(self, hilbert_space, test_state):
        """Test consistency of observable expectations during evolution."""
        # Create test observable
        O = torch.randn(hilbert_space.dim, hilbert_space.dim, dtype=torch.complex128)
        O = O + O.conj().T  # Make Hermitian
        
        # Initial expectation
        initial_exp = hilbert_space.measure_observable(test_state, O)  # Use measure_observable instead of expectation_value
        
        # Create compatible Hamiltonian that commutes with O
        H = torch.matmul(O, O)  # Guaranteed to commute
        
        # Evolve and check expectation preservation
        time = torch.linspace(0, 1.0, 10)
        evolved_states = hilbert_space.evolve_state(test_state, H, time)
        
        for state in evolved_states:
            exp_val = hilbert_space.measure_observable(state, O)
            assert torch.allclose(exp_val, initial_exp, rtol=1e-5)
