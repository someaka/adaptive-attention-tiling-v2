"""Quantum State Space Implementation.

This module implements the quantum mechanical framework for attention patterns:
- Hilbert space structure with quantum states
- State preparation and measurement
- Quantum evolution operators
- Density matrix formalism
"""

from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import torch
import torch.nn as nn
import numpy as np

from src.core.quantum.types import QuantumState
from src.validation.quantum.state import (
    StateValidator,
    StatePreparationValidator,
    EntanglementMetrics
)

class HilbertSpace:
    """Quantum Hilbert space implementation."""
    
    def __init__(self, dim: int):
        """Initialize Hilbert space.
        
        Args:
            dim: Dimension of Hilbert space
        """
        self.dim = dim
        
        # Initialize quantum space components
        self.basis_states = self._initialize_basis()
        self.hamiltonian = nn.Parameter(torch.eye(dim, dtype=torch.complex128))
        self.observables = self._initialize_observables()

    def _initialize_basis(self) -> List[str]:
        """Initialize computational basis states.
        
        Returns:
            List of basis state labels
        """
        return [f"|{i}⟩" for i in range(self.dim)]

    def _initialize_observables(self) -> Dict[str, torch.Tensor]:
        """Initialize standard observables.
        
        Returns:
            Dictionary of observable operators
        """
        # Single qubit Pauli matrices
        sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
        sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
        sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
        
        # Initialize observables dictionary
        observables = {
            "X": sigma_x,
            "Y": sigma_y, 
            "Z": sigma_z
        }
        
        return observables

    def prepare_state(self, amplitudes: torch.Tensor) -> QuantumState:
        """Prepare quantum state with given amplitudes.
        
        Args:
            amplitudes: Complex amplitudes in computational basis
            
        Returns:
            Prepared quantum state
        """
        # Convert input to complex128
        if amplitudes.dtype != torch.complex128:
            if amplitudes.dtype == torch.complex64:
                amplitudes = amplitudes.to(torch.complex128)
            else:
                amplitudes = torch.complex(amplitudes.to(torch.float64), torch.zeros_like(amplitudes, dtype=torch.float64))

        # Handle different input dimensions
        if amplitudes.shape[-1] == self.dim * 2:
            # Convert real and imaginary parts to complex
            real_part = amplitudes[..., :self.dim].to(torch.float64)
            imag_part = amplitudes[..., self.dim:].to(torch.float64)
            amplitudes = torch.complex(real_part, imag_part)
        elif amplitudes.shape[-1] == 2 and self.dim == 4:
            # Special case: Convert 2-qubit product state to 4-dim state
            amplitudes = torch.kron(amplitudes[:1], amplitudes[1:])
        elif amplitudes.shape[-1] != self.dim:
            raise ValueError(f"Amplitudes must have dimension {self.dim} or {self.dim * 2}")
            
        # Normalize state
        norm = torch.sqrt(torch.sum(torch.abs(amplitudes)**2, dim=-1, keepdim=True))
        if torch.any(norm > 1e-10):
            amplitudes = amplitudes / norm
            
        return QuantumState(
            amplitudes=amplitudes.to(torch.complex128),
            basis_labels=self.basis_states,
            phase=torch.zeros_like(amplitudes, dtype=torch.complex128)
        )

    def entanglement_entropy(self, state: Union[QuantumState, torch.Tensor]) -> torch.Tensor:
        """Compute entanglement entropy of a bipartite quantum state.
        
        Args:
            state: Quantum state or density matrix
            
        Returns:
            Entanglement entropy
        """
        if isinstance(state, QuantumState):
            # Convert pure state to density matrix
            state_vector = state.amplitudes.to(torch.complex128)
            if len(state_vector.shape) == 1:
                state_vector = state_vector.unsqueeze(0)
            rho = torch.matmul(state_vector.conj().transpose(-2,-1), state_vector)
        else:
            rho = state.to(torch.complex128)
            
        # Compute reduced density matrix by partial trace
        n = int(torch.sqrt(torch.tensor(rho.shape[0])))  # Assuming 2-qubit system
        rho_reshaped = rho.reshape(n, n, n, n)
        rho_reduced = torch.einsum('ijik->jk', rho_reshaped)
        
        # Compute von Neumann entropy
        eigenvals = torch.linalg.eigvalsh(rho_reduced).to(torch.float64)
        eigenvals = torch.clamp(eigenvals, min=1e-10)  # Remove small negative eigenvalues
        eigenvals = eigenvals / torch.sum(eigenvals)  # Normalize eigenvalues
        entropy = -torch.sum(eigenvals * torch.log2(eigenvals))
        return entropy.to(torch.float64)

    def evolve_state(self, initial_state: QuantumState, hamiltonian: torch.Tensor, t: Union[float, torch.Tensor]) -> Union[QuantumState, List[QuantumState]]:
        """Evolve quantum state under Hamiltonian.
        
        Args:
            initial_state: Initial quantum state
            hamiltonian: Hamiltonian operator
            t: Evolution time or vector of times
            
        Returns:
            Evolved quantum state(s)
        """
        # Ensure complex types
        hamiltonian = hamiltonian.to(torch.complex128)
        state_vector = initial_state.amplitudes.to(torch.complex128)
        
        if len(state_vector.shape) == 1:
            state_vector = state_vector.unsqueeze(0)
            
        if isinstance(t, (float, int)):
            # Single time evolution
            evolution_operator = torch.matrix_exp(-1j * hamiltonian * t)
            evolved = torch.matmul(state_vector, evolution_operator.T)
            return QuantumState(
                amplitudes=evolved.squeeze(),
                basis_labels=initial_state.basis_labels,
                phase=initial_state.phase
            )
        else:
            # Multiple time evolution
            states = []
            for time in t:
                evolution_operator = torch.matrix_exp(-1j * hamiltonian * time)
                evolved = torch.matmul(state_vector, evolution_operator.T)
                states.append(QuantumState(
                    amplitudes=evolved.squeeze(),
                    basis_labels=initial_state.basis_labels,
                    phase=initial_state.phase
                ))
            return states

    def measure_observable(self, state: QuantumState, observable: torch.Tensor) -> torch.Tensor:
        """Measure quantum observable on state.
        
        Args:
            state: Quantum state to measure
            observable: Observable operator
            
        Returns:
            Expectation value of the observable
        """
        # Ensure complex types
        state_vector = state.amplitudes.to(torch.complex128)
        observable = observable.to(torch.complex128)
        
        # Handle dimension mismatch for single-qubit observables
        if observable.shape[0] < self.dim:
            # Expand observable to full Hilbert space dimension
            n_qubits = int(torch.log2(torch.tensor(self.dim)))
            target_qubit = 0  # Default to first qubit
            expanded_obs = torch.eye(2, dtype=torch.complex128)
            for i in range(n_qubits):
                if i == target_qubit:
                    expanded_obs = torch.kron(expanded_obs, observable)
                else:
                    expanded_obs = torch.kron(expanded_obs, torch.eye(2, dtype=torch.complex128))
            observable = expanded_obs[:self.dim, :self.dim]  # Truncate to match dimension
            
        # Reshape state if needed
        if len(state_vector.shape) == 1:
            state_vector = state_vector.unsqueeze(0)
            
        # Compute expectation value
        expectation = torch.einsum('bi,ij,bj->b', state_vector.conj(), observable, state_vector)
        return expectation

    def compute_entropy(self, state: Union[QuantumState, torch.Tensor]) -> torch.Tensor:
        """Compute von Neumann entropy of quantum state.
        
        Args:
            state: Quantum state or density matrix
            
        Returns:
            Von Neumann entropy
        """
        if isinstance(state, QuantumState):
            # Convert pure state to density matrix
            state_vector = state.amplitudes.to(torch.complex128)
            if len(state_vector.shape) == 1:
                state_vector = state_vector.unsqueeze(0)
            rho = torch.matmul(state_vector.conj().transpose(-2,-1), state_vector)
        else:
            rho = state.to(torch.complex128)
            
        # Compute eigenvalues
        eigenvals = torch.linalg.eigvalsh(rho)
        # Remove small negative eigenvalues due to numerical errors
        eigenvals = torch.clamp(eigenvals, min=0)
        # Normalize eigenvalues
        eigenvals = eigenvals / torch.sum(eigenvals)
        # Compute entropy
        entropy = -torch.sum(eigenvals * torch.log2(eigenvals + 1e-10))
        return entropy.to(torch.float64)

    def compute_negativity(self, state: Union[QuantumState, torch.Tensor]) -> torch.Tensor:
        """Compute negativity measure of entanglement.
        
        Args:
            state: Quantum state or density matrix
            
        Returns:
            Negativity measure
        """
        if isinstance(state, QuantumState):
            # Convert pure state to density matrix
            state_vector = state.amplitudes.to(torch.complex128)
            if len(state_vector.shape) == 1:
                state_vector = state_vector.unsqueeze(0)
            rho = torch.matmul(state_vector.conj().transpose(-2,-1), state_vector)
        else:
            rho = state.to(torch.complex128)
            
        # Compute partial transpose
        n = int(torch.sqrt(torch.tensor(rho.shape[0])))  # Assuming 2-qubit system
        rho_reshaped = rho.reshape(n, n, n, n)
        rho_pt = rho_reshaped.permute(0, 3, 2, 1).reshape(n*n, n*n)
        
        # Compute eigenvalues of partial transpose
        eigenvals = torch.linalg.eigvalsh(rho_pt)
        # Negativity is sum of absolute values of negative eigenvalues
        negativity = torch.sum(torch.abs(torch.clamp(eigenvals, max=0)))
        return negativity.to(torch.float64)

    def parallel_transport(self, tangent: torch.Tensor, state1: QuantumState, state2: QuantumState) -> torch.Tensor:
        """Parallel transport a tangent vector between quantum states."""
        # Get state vectors
        v1 = state1.amplitudes
        v2 = state2.amplitudes
        
        # Compute overlap
        overlap = torch.vdot(v1, v2)
        phase = torch.angle(overlap)
        
        # Apply phase correction
        v2_aligned = v2 * torch.exp(-1j * phase)
        
        # Compute parallel transport operator
        P = torch.eye(self.dim, dtype=torch.complex128) - torch.outer(v2_aligned, v2_aligned.conj())
        
        # Transport the tangent vector
        transported = torch.matmul(P, tangent)
        
        # Normalize to preserve norm
        original_norm = torch.norm(tangent)
        transported = transported * (original_norm / (torch.norm(transported) + 1e-10))
        
        return transported

    def compute_berry_phase(
        self, initial_state: QuantumState, hamiltonian_fn: Callable[[float], torch.Tensor], times: torch.Tensor
    ) -> torch.Tensor:
        """Compute Berry phase for cyclic evolution."""
        state = initial_state
        phase = torch.zeros(1, dtype=torch.complex128)
        
        for t, next_t in zip(times[:-1], times[1:]):
            # Get Hamiltonian at current time
            H = hamiltonian_fn(t.item())
            
            # Pad Hamiltonian if needed
            if H.shape[0] < self.dim:
                H_padded = torch.zeros((self.dim, self.dim), dtype=torch.complex128)
                H_padded[:H.shape[0], :H.shape[1]] = H
                H = H_padded
            
            # Time step
            dt = next_t - t
            
            # Evolve state
            U = torch.matrix_exp(-1j * H * dt)
            state_vector = state.amplitudes.squeeze()
            evolved = torch.matmul(U, state_vector)
            
            # Compute overlap
            overlap = torch.vdot(state_vector, evolved)
            
            # Update phase
            phase = phase - torch.log(overlap / (torch.abs(overlap) + 1e-10))
            
            # Update state
            state = QuantumState(
                amplitudes=evolved,
                basis_labels=state.basis_labels,
                phase=state.phase
            )
        
        return phase.real

    def apply_quantum_channel(self, state: QuantumState, kraus_ops: List[torch.Tensor]) -> QuantumState:
        """Apply quantum channel using Kraus operators."""
        # Convert all operators to complex128
        kraus_ops = [K.to(torch.complex128) for K in kraus_ops]
        state_vector = state.amplitudes.to(torch.complex128)
        
        if len(state_vector.shape) == 1:
            state_vector = state_vector.unsqueeze(0)
            
        # Apply channel
        output_dm = torch.zeros((self.dim, self.dim), dtype=torch.complex128)
        for K in kraus_ops:
            # Pad operator if needed
            if K.shape[0] < self.dim:
                K_padded = torch.zeros((self.dim, self.dim), dtype=torch.complex128)
                K_padded[:K.shape[0], :K.shape[1]] = K
                K = K_padded
                
            evolved = torch.matmul(K, state_vector.T)
            output_dm += torch.matmul(evolved, evolved.conj().T)
            
        # Normalize
        output_dm = output_dm / torch.trace(output_dm).real
        
        # Convert back to state vector (take first eigenvector)
        eigenvals, eigenvecs = torch.linalg.eigh(output_dm)
        max_idx = torch.argmax(eigenvals.real)
        output_state = eigenvecs[:, max_idx]
        
        # Ensure proper phase
        phase = torch.angle(output_state[torch.argmax(torch.abs(output_state))])
        output_state = output_state * torch.exp(-1j * phase)
        
        return QuantumState(
            amplitudes=output_state,
            basis_labels=state.basis_labels,
            phase=state.phase
        )

    def reconstruct_state(self, measurements: Dict[str, torch.Tensor]) -> QuantumState:
        """Reconstruct quantum state from tomographic measurements."""
        # Initialize density matrix
        rho = torch.zeros((self.dim, self.dim), dtype=torch.complex128)
        
        # Add contributions from each measurement
        for basis, result in measurements.items():
            if basis in self.observables:
                observable = self.observables[basis].to(torch.complex128)
                # Expand 2x2 observable to full dimension if needed
                if observable.shape[0] < self.dim:
                    expanded = torch.eye(self.dim, dtype=torch.complex128)
                    expanded[:2, :2] = observable
                    observable = expanded
                rho += result.real * observable
                
        # Ensure Hermiticity
        rho = 0.5 * (rho + rho.conj().T)
        
        # Ensure positive semidefiniteness
        eigenvals, eigenvecs = torch.linalg.eigh(rho)
        eigenvals = torch.clamp(eigenvals.real, min=0).to(torch.complex128)
        rho = torch.matmul(eigenvecs, torch.matmul(torch.diag(eigenvals), eigenvecs.conj().T))
        
        # Normalize
        rho = rho / torch.trace(rho).real
        
        # Convert to pure state (take eigenvector with largest eigenvalue)
        eigenvals, eigenvecs = torch.linalg.eigh(rho)
        max_idx = torch.argmax(eigenvals.real)
        state_vector = eigenvecs[:, max_idx]
        
        # Ensure proper phase
        phase = torch.angle(state_vector[torch.argmax(torch.abs(state_vector))])
        state_vector = state_vector * torch.exp(-1j * phase)
        
        return QuantumState(
            amplitudes=state_vector,
            basis_labels=self.basis_states,
            phase=torch.zeros_like(state_vector)
        )

    def evolve_with_decoherence(self, state: QuantumState, T1: float, T2: float, times: torch.Tensor) -> List[QuantumState]:
        """Evolve quantum state under decoherence.
        
        Args:
            state: Initial quantum state
            T1: Relaxation time
            T2: Dephasing time
            times: Evolution time points
            
        Returns:
            List of evolved quantum states
        """
        states = []
        state_vector = state.amplitudes.to(torch.complex128)
        
        # Ensure state vector has correct dimension
        if len(state_vector.shape) == 1:
            state_vector = state_vector.unsqueeze(0)
            
        # Pad state vector if needed
        if state_vector.shape[1] < self.dim:
            padded = torch.zeros((state_vector.shape[0], self.dim), dtype=torch.complex128)
            padded[:, :state_vector.shape[1]] = state_vector
            state_vector = padded
        
        for t in times:
            # Compute decoherence factors
            gamma1 = 1 - torch.exp(-t/torch.tensor(T1, dtype=torch.float64))
            gamma2 = 1 - torch.exp(-t/torch.tensor(T2, dtype=torch.float64))
            
            # Apply amplitude damping and phase damping
            evolved = state_vector.clone()
            for i in range(1, self.dim):  # Apply to all excited states
                evolved[:, i] *= torch.sqrt(1 - gamma1)
                evolved[:, i] *= torch.exp(-gamma2)
            
            # Ensure conservation of probability
            ground_state_amp = torch.sqrt(1 - torch.sum(torch.abs(evolved[:, 1:])**2, dim=1, keepdim=True))
            evolved[:, 0] = ground_state_amp
            
            # Normalize
            norm = torch.sqrt(torch.sum(torch.abs(evolved)**2, dim=1, keepdim=True))
            evolved = evolved / (norm + 1e-10)
                
            states.append(QuantumState(
                amplitudes=evolved.squeeze(),
                basis_labels=state.basis_labels,
                phase=state.phase
            ))
            
        return states

    def state_fidelity(self, state1: QuantumState, state2: QuantumState) -> torch.Tensor:
        """Compute fidelity between two quantum states.
        
        Args:
            state1: First quantum state
            state2: Second quantum state
            
        Returns:
            Fidelity between states
        """
        v1 = state1.amplitudes.to(torch.complex128)
        v2 = state2.amplitudes.to(torch.complex128)
        
        if len(v1.shape) == 1:
            v1 = v1.unsqueeze(0)
        if len(v2.shape) == 1:
            v2 = v2.unsqueeze(0)
            
        overlap = torch.abs(torch.matmul(v1.conj(), v2.T))**2
        return overlap.squeeze().real

    def measure_variance(self, state: QuantumState, observable: torch.Tensor) -> torch.Tensor:
        """Compute variance of an observable in a quantum state.
        
        Args:
            state: Quantum state
            observable: Observable operator
            
        Returns:
            Variance of the observable
        """
        # Compute expectation value
        expectation = self.measure_observable(state, observable)
        
        # Compute expectation of squared observable
        squared = torch.matmul(observable, observable)
        squared_expectation = self.measure_observable(state, squared)
        
        # Compute variance
        variance = squared_expectation - expectation**2
        return variance.real.to(torch.float64)

    def fubini_study_distance(self, state1: QuantumState, state2: QuantumState) -> torch.Tensor:
        """Compute Fubini-Study distance between quantum states.
        
        Args:
            state1: First quantum state
            state2: Second quantum state
            
        Returns:
            Fubini-Study distance
        """
        # Ensure complex types
        v1 = state1.amplitudes.to(torch.complex128)
        v2 = state2.amplitudes.to(torch.complex128)
        
        if len(v1.shape) == 1:
            v1 = v1.unsqueeze(0)
        if len(v2.shape) == 1:
            v2 = v2.unsqueeze(0)
            
        # Compute overlap
        overlap = torch.abs(torch.matmul(v1.conj(), v2.T))
        # Clamp to avoid numerical issues
        overlap = torch.clamp(overlap, min=0.0, max=1.0)
        
        # Compute distance
        distance = torch.arccos(overlap)
        return distance.to(torch.float64)

    def quantum_tangent_vector(self, state: QuantumState) -> torch.Tensor:
        """Compute tangent vector at a quantum state.
        
        Args:
            state: Quantum state to compute tangent at
            
        Returns:
            Tangent vector at the state
        """
        state_vector = state.amplitudes.to(torch.complex128)
        if len(state_vector.shape) == 1:
            state_vector = state_vector.unsqueeze(0)
            
        # Compute projector onto state
        projector = torch.matmul(state_vector.conj().transpose(-2,-1), state_vector)
        
        # Tangent space is orthogonal complement
        tangent = torch.eye(self.dim, dtype=torch.complex128) - projector
        
        return tangent

    def compute_concurrence(self, density_matrix: torch.Tensor) -> torch.Tensor:
        """Compute concurrence measure of entanglement.
        
        Args:
            density_matrix: Density matrix of two-qubit state
            
        Returns:
            Concurrence measure
        """
        if density_matrix.shape != (4, 4):
            raise ValueError("Concurrence is only defined for 2-qubit states")
            
        # Compute spin-flipped state
        sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
        R = torch.kron(sigma_y, sigma_y)
        density_matrix = density_matrix.to(torch.complex128)
        rho_tilde = torch.matmul(R, torch.matmul(density_matrix.conj(), R))
        
        # Compute eigenvalues of rho * rho_tilde
        M = torch.matmul(density_matrix, rho_tilde)
        eigenvals = torch.linalg.eigvalsh(M).to(torch.float64)
        eigenvals = torch.sqrt(torch.clamp(eigenvals, min=0))
        
        # Sort eigenvalues in descending order
        eigenvals, _ = torch.sort(eigenvals, descending=True)
        
        # Compute concurrence
        concurrence = torch.maximum(
            eigenvals[0] - eigenvals[1] - eigenvals[2] - eigenvals[3],
            torch.tensor(0.0, dtype=torch.float64)
        )
        
        return concurrence.to(torch.float64)

    def evaluate_entanglement_witness(self, state: Union[QuantumState, torch.Tensor]) -> torch.Tensor:
        """Evaluate entanglement witness on quantum state."""
        if isinstance(state, QuantumState):
            rho = torch.outer(state.amplitudes, state.amplitudes.conj())
        else:
            rho = state.to(torch.complex128)
            
        # Use PPT criterion as witness
        d = int(np.sqrt(self.dim))
        reshaped = rho.reshape(d, d, d, d)
        partial_transpose = reshaped.permute(0, 3, 2, 1).reshape(self.dim, self.dim)
        
        # Get minimum eigenvalue
        eigenvals = torch.linalg.eigvalsh(partial_transpose)
        return eigenvals[0].real
