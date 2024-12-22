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
import torch.nn.functional as F

from src.core.quantum.types import QuantumState
from src.validation.quantum.state import (
    StateValidator,
    StatePreparationValidator,
    EntanglementMetrics
)

class HilbertSpace:
    """Quantum Hilbert space implementation."""
    
    def __init__(self, dim: int, dtype=torch.float32):
        """Initialize Hilbert space.
        
        Args:
            dim: Dimension of Hilbert space
            dtype: Data type for tensors
        """
        self.dim = dim
        self.dtype = dtype
        
        # Initialize quantum space components
        self.basis_states = self._initialize_basis()
        self.hamiltonian = nn.Parameter(torch.eye(dim, dtype=self._get_complex_dtype()))
        self.observables = self._initialize_observables()

    def _get_complex_dtype(self) -> torch.dtype:
        """Get corresponding complex dtype."""
        if self.dtype == torch.float32:
            return torch.complex64
        elif self.dtype == torch.float64:
            return torch.complex128
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")

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
        complex_dtype = self._get_complex_dtype()
        
        # Single qubit Pauli matrices
        sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=complex_dtype)
        sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=complex_dtype)
        sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=complex_dtype)
        
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
        entropy = -torch.sum(eigenvals * torch.log(eigenvals))  # Using natural log
        return entropy.to(torch.float64)

    def evolve_state(self, initial_state: QuantumState, hamiltonian: torch.Tensor, t: Union[float, torch.Tensor]) -> QuantumState:
        """Evolve quantum state under Hamiltonian.
        
        Args:
            initial_state: Initial quantum state
            hamiltonian: Hamiltonian operator
            t: Evolution time
            
        Returns:
            Evolved quantum state
        """
        # Ensure complex types
        hamiltonian = hamiltonian.to(torch.complex128)
        state_vector = initial_state.amplitudes.to(torch.complex128)
        
        if len(state_vector.shape) == 1:
            state_vector = state_vector.unsqueeze(0)
            
        # Single time evolution
        evolution_operator = torch.matrix_exp(-1j * hamiltonian * t)
        # Add phase correction to preserve pattern orientation
        first_row = evolution_operator[0] if len(evolution_operator.shape) == 2 else evolution_operator[0, 0]
        phase_correction = torch.exp(1j * torch.angle(torch.vdot(state_vector.squeeze(), first_row)))
        evolution_operator = evolution_operator * phase_correction
        evolved = torch.matmul(state_vector, evolution_operator.mT)
        
        return QuantumState(
            amplitudes=evolved.squeeze(),
            basis_labels=initial_state.basis_labels,
            phase=initial_state.phase
        )

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
        # Initialize phase accumulation
        phase = torch.tensor(0.0, dtype=torch.float64)
        
        # Calculate time step
        dt = times[1] - times[0]
        
        # Initialize state vector with proper shape and type
        state_vector = initial_state.amplitudes.to(torch.complex128)
        if len(state_vector.shape) == 0 or (len(state_vector.shape) == 1 and state_vector.shape[0] == 1):
            state_vector = torch.zeros(self.dim, dtype=torch.complex128)
            state_vector[0] = 1.0
            
        # Normalize initial state
        state_vector = state_vector / torch.sqrt(torch.vdot(state_vector, state_vector))
        
        # Store initial state for final overlap
        initial_vector = state_vector.clone()
        
        # Use smaller time steps for better accuracy
        fine_times = torch.linspace(times[0], times[-1], len(times) * 1000, dtype=torch.float64)
        dt_fine = fine_times[1] - fine_times[0]
        
        # Previous state for phase tracking
        prev_state = state_vector.clone()
        
        for t in fine_times[:-1]:
            # Get Hamiltonian at current time
            H = hamiltonian_fn(t.item())
            
            # Pad Hamiltonian if needed
            if H.shape[0] < self.dim:
                H_padded = torch.zeros((self.dim, self.dim), dtype=torch.complex128)
                H_padded[:H.shape[0], :H.shape[1]] = H
                H = H_padded
            
            # Ensure Hamiltonian is Hermitian
            H = (H + H.conj().T) / 2
            
            # Get instantaneous eigenstates
            eigenvals, eigenvecs = torch.linalg.eigh(H)
            
            # Project state onto instantaneous eigenbasis
            overlaps = torch.matmul(eigenvecs.conj().T, state_vector)
            max_idx = torch.argmax(torch.abs(overlaps))
            
            # Evolve state using exponential map with smaller time step
            U = torch.matrix_exp(-1j * H * dt_fine)
            next_state = torch.matmul(U, state_vector)
            
            # Project onto instantaneous eigenstate
            next_state = eigenvecs[:, max_idx] * torch.vdot(eigenvecs[:, max_idx], next_state)
            
            # Normalize next state
            next_state = next_state / torch.sqrt(torch.vdot(next_state, next_state))
            
            # Compute geometric phase contribution
            overlap = torch.vdot(prev_state, next_state)
            if torch.abs(overlap) > 1e-10:  # Avoid division by zero
                phase_factor = overlap / torch.abs(overlap)
                phase += torch.log(phase_factor).imag
            
            # Update states
            prev_state = next_state.clone()
            state_vector = next_state
        
        # Compute final overlap with initial state for cyclic correction
        final_overlap = torch.vdot(initial_vector, state_vector)
        if torch.abs(final_overlap) > 1e-10:
            final_phase = torch.log(final_overlap / torch.abs(final_overlap)).imag
            phase -= final_phase
            
        # Take absolute value since phase can be negative
        phase = torch.abs(phase)
        
        # Convert to complex tensor for output
        return torch.complex(phase, torch.zeros_like(phase))

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
        
        # Add contribution from identity
        rho += torch.eye(self.dim, dtype=torch.complex128) / self.dim
        
        # Define Pauli matrices
        pauli_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
        pauli_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
        pauli_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
        
        # Process each qubit pair
        n_qubits = int(np.log2(self.dim))
        for i in range(0, n_qubits, 2):
            # Extract measurements for this qubit pair
            x_val = measurements.get('X', torch.zeros(1, dtype=torch.complex128))[i//2]
            y_val = measurements.get('Y', torch.zeros(1, dtype=torch.complex128))[i//2]
            z_val = measurements.get('Z', torch.zeros(1, dtype=torch.complex128))[i//2]
            
            # Construct local density matrix
            local_rho = torch.eye(4, dtype=torch.complex128) / 4
            local_rho += torch.kron(pauli_x, torch.eye(2, dtype=torch.complex128)) * x_val / 2
            local_rho += torch.kron(pauli_y, torch.eye(2, dtype=torch.complex128)) * y_val / 2
            local_rho += torch.kron(pauli_z, torch.eye(2, dtype=torch.complex128)) * z_val / 2
            
            # Ensure Hermiticity
            local_rho = (local_rho + local_rho.conj().T) / 2
            
            # Ensure positive semidefiniteness
            eigenvals, eigenvecs = torch.linalg.eigh(local_rho)
            eigenvals = torch.clamp(eigenvals, min=0).to(torch.complex128)
            local_rho = torch.matmul(torch.matmul(eigenvecs, torch.diag(eigenvals)), eigenvecs.conj().T)
            
            # Normalize
            local_rho = local_rho / torch.trace(local_rho)
            
            # Update global density matrix
            if i == 0:
                rho = local_rho
            else:
                rho = torch.kron(rho, local_rho)
        
        # Extract state vector from density matrix
        eigenvals, eigenvecs = torch.linalg.eigh(rho)
        max_idx = torch.argmax(eigenvals.real)
        state_vector = eigenvecs[:, max_idx]
        
        # Normalize and align phase
        state_vector = state_vector / torch.sqrt(torch.vdot(state_vector, state_vector))
        max_amp_idx = torch.argmax(torch.abs(state_vector))
        phase = torch.angle(state_vector[max_amp_idx])
        state_vector = state_vector * torch.exp(-1j * phase)
        
        return QuantumState(
            amplitudes=state_vector,
            basis_labels=[f"|{i}⟩" for i in range(self.dim)],
            phase=torch.zeros(self.dim, dtype=torch.complex128)
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
        v1 = state1.amplitudes
        v2 = state2.amplitudes
        
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
        # Convert input to density matrix
        if isinstance(state, QuantumState):
            state_vector = state.amplitudes.to(torch.complex128)
            if len(state_vector.shape) == 1:
                state_vector = state_vector.unsqueeze(0)
            rho = torch.matmul(state_vector.conj().transpose(-2,-1), state_vector)
        else:
            rho = state.to(torch.complex128)
            
        # Compute partial transpose
        d = int(np.sqrt(self.dim))  # Local dimension
        rho_reshaped = rho.reshape(d, d, d, d)
        rho_pt = rho_reshaped.permute(0, 3, 2, 1).reshape(self.dim, self.dim)
        
        # Compute minimum eigenvalue of partial transpose
        eigenvals = torch.linalg.eigvalsh(rho_pt).real
        min_eigenval = torch.min(eigenvals)
        
        # Return minimum eigenvalue (negative indicates entanglement)
        return min_eigenval.to(torch.float64)

    def scale_state(
        self,
        state: QuantumState,
        scale_factor: float
    ) -> QuantumState:
        """Scale a quantum state by adjusting its amplitudes.
        
        This method implements quantum state scaling by:
        1. Adjusting amplitudes according to scale factor
        2. Preserving normalization
        3. Maintaining quantum properties
        
        Args:
            state: Input quantum state
            scale_factor: Factor to scale the state by
            
        Returns:
            Scaled quantum state
            
        Raises:
            ValueError: If scale_factor is invalid
        """
        if scale_factor <= 0:
            raise ValueError("Scale factor must be positive")
            
        # Scale amplitudes while preserving normalization
        scale_tensor = torch.tensor(scale_factor, dtype=torch.float64)
        scaled_amplitudes = state.amplitudes * torch.sqrt(scale_tensor)
        scaled_amplitudes = F.normalize(scaled_amplitudes, p=2, dim=-1)
        
        # Create new quantum state with scaled amplitudes
        return QuantumState(
            amplitudes=scaled_amplitudes,
            basis_labels=state.basis_labels,
            phase=state.phase
        )

    def compute_entanglement_entropy(
        self,
        state: QuantumState
    ) -> torch.Tensor:
        """Compute the entanglement entropy of a quantum state.
        
        Args:
            state: Input quantum state
            
        Returns:
            Entanglement entropy tensor
        """
        # Reshape amplitudes into bipartite system (2 x N/2)
        shape = state.amplitudes.shape
        mid_dim = shape[-1] // 2
        amplitudes = state.amplitudes.view(2, mid_dim)
        
        # Compute reduced density matrix
        rho = torch.einsum('ij,ik->jk', amplitudes, torch.conj(amplitudes))
        
        # Compute eigenvalues
        eigenvalues = torch.linalg.eigvalsh(rho)
        
        # Remove small negative eigenvalues due to numerical errors
        eigenvalues = torch.clamp(eigenvalues, min=0.0)
        
        # Normalize eigenvalues
        eigenvalues = eigenvalues / torch.sum(eigenvalues)
        
        # Compute von Neumann entropy
        entropy = -torch.sum(
            eigenvalues * torch.log2(eigenvalues + 1e-10)
        )
        
        return entropy

    def apply_unitary(self, state: QuantumState, unitary: torch.Tensor) -> QuantumState:
        """Apply unitary operation to quantum state.
        
        Args:
            state: Input quantum state
            unitary: Unitary operator to apply
            
        Returns:
            Transformed quantum state
        """
        # Ensure complex types
        state_vector = state.amplitudes.to(torch.complex128)
        unitary = unitary.to(torch.complex128)
        
        # Verify unitarity
        if not torch.allclose(
            torch.matmul(unitary, unitary.conj().T),
            torch.eye(unitary.shape[0], dtype=torch.complex128),
            atol=1e-6
        ):
            raise ValueError("Operator is not unitary")
            
        # Apply unitary transformation
        if len(state_vector.shape) == 1:
            state_vector = state_vector.unsqueeze(0)
        transformed = torch.matmul(state_vector, unitary.T)
        
        return QuantumState(
            amplitudes=transformed.squeeze(),
            basis_labels=state.basis_labels,
            phase=state.phase
        )

    def measure_state(
        self,
        state: QuantumState,
        measurement: torch.Tensor
    ) -> Tuple[int, QuantumState]:
        """Perform projective measurement on quantum state.
        
        Args:
            state: Quantum state to measure
            measurement: Measurement operator (projector)
            
        Returns:
            Tuple of (measurement result, post-measurement state)
        """
        # Ensure complex types
        state_vector = state.amplitudes.to(torch.complex128)
        measurement = measurement.to(torch.complex128)
        
        # Compute measurement probabilities
        if len(state_vector.shape) == 1:
            state_vector = state_vector.unsqueeze(0)
            
        probabilities = []
        for i in range(measurement.shape[0]):
            projector = measurement[i:i+1].T @ measurement[i:i+1].conj()
            prob = torch.abs(
                torch.einsum('bi,ij,bj->b', state_vector.conj(), projector, state_vector)
            ).real
            probabilities.append(prob)
            
        probabilities = torch.cat(probabilities)
        
        # Sample measurement result
        result = int(torch.multinomial(probabilities, 1).item())
        
        # Compute post-measurement state
        post_state = measurement[result:result+1].T
        post_state = post_state / torch.sqrt(probabilities[result])
        
        return result, QuantumState(
            amplitudes=post_state.squeeze(),
            basis_labels=state.basis_labels,
            phase=state.phase
        )
