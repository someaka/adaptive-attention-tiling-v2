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

from src.validation.quantum.state import (
    StateValidator,
    StatePreparationValidator,
    EntanglementMetrics
)

# Import the base QuantumState type
from src.core.quantum.types import QuantumState

# Re-export QuantumState for backward compatibility
__all__ = ['QuantumState', 'HilbertSpace']

class HilbertSpace:
    """Quantum Hilbert space implementation."""
    
    def __init__(self, dim: int, dtype=torch.float64):
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
        return torch.complex128  # Always use complex128 for quantum states

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

    def _ensure_complex128(self, tensor: torch.Tensor) -> torch.Tensor:
        """Helper method to ensure tensor is complex128."""
        if tensor.dtype != torch.complex128:
            if tensor.dtype == torch.complex64:
                return tensor.to(torch.complex128)
            else:
                return torch.complex(tensor.to(torch.float64), torch.zeros_like(tensor, dtype=torch.float64))
        return tensor

    def _compute_norm(self, tensor: torch.Tensor) -> torch.Tensor:
        """Helper method to compute norm consistently."""
        norm = torch.sqrt(torch.sum(torch.abs(tensor)**2, dim=-1, keepdim=True))
        return norm.to(torch.float64)

    def prepare_state(self, amplitudes: torch.Tensor) -> QuantumState:
        """Prepare quantum state with given amplitudes."""
        # Convert input to complex128
        if amplitudes.dtype != torch.complex128:
            if amplitudes.dtype == torch.complex64:
                amplitudes = amplitudes.to(torch.complex128)
            else:
                # Handle real inputs by converting to complex
                amplitudes = amplitudes.to(torch.float64)
                if amplitudes.shape[-1] == self.dim * 2:
                    # Convert real and imaginary parts to complex
                    real_part = amplitudes[..., :self.dim].to(torch.float64)
                    imag_part = amplitudes[..., self.dim:].to(torch.float64)
                    amplitudes = torch.complex(real_part, imag_part)
                else:
                    # Single real input
                    amplitudes = torch.complex(amplitudes, torch.zeros_like(amplitudes, dtype=torch.float64))

        # Handle different input dimensions
        if amplitudes.shape[-1] == 2 and self.dim == 4:
            # Special case: Convert 2-qubit product state to 4-dim state
            amplitudes = torch.kron(amplitudes[:1], amplitudes[1:])
        elif amplitudes.shape[-1] != self.dim:
            raise ValueError(f"Amplitudes must have dimension {self.dim} or {self.dim * 2}")
            
        # Store original norm
        original_norm = self._compute_norm(amplitudes)
            
        # Normalize state
        if torch.any(original_norm > 1e-10):
            amplitudes = (amplitudes / original_norm).to(torch.complex128)
            
        return QuantumState(
            amplitudes=amplitudes,
            basis_labels=self.basis_states,
            phase=torch.zeros_like(amplitudes, dtype=torch.complex128),
            original_norm=original_norm
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
        
        # For a lattice state, we'll use the first dimension as the subsystem
        if len(rho.shape) == 2:
            # Create density matrix from pure state if needed
            if not torch.allclose(torch.matmul(rho.conj().transpose(-2,-1), rho), rho):
                rho = torch.matmul(rho.conj().transpose(-2,-1), rho)

            # Handle 2-qubit systems (4x4 density matrix) specially
            if rho.shape == (4, 4):
                # Reshape into 2x2x2x2 tensor for qubit subsystems
                rho = rho.reshape(2, 2, 2, 2)
                # Perform partial trace over second subsystem
                reduced_rho = torch.einsum('ijik->jk', rho)
            else:
                # For larger systems, perform partial trace using matrix operations
                # Calculate subsystem dimensions
                dim1 = rho.shape[0]
                dim2 = rho.shape[1]
                
                # Verify matrix is square
                if dim1 != dim2:
                    raise ValueError(f"Input matrix must be square, got {dim1}x{dim2}")
                
                # Calculate subsystem dimensions (try to find factors)
                factors = []
                for i in range(1, int(np.sqrt(dim1)) + 1):
                    if dim1 % i == 0:
                        factors.append((i, dim1 // i))
                
                if not factors:
                    raise ValueError(f"Cannot factorize matrix dimension {dim1}")
                
                # Use largest possible subsystem dimensions
                d1, d2 = factors[-1]
                
                # Reshape into [d1, d2, d1, d2] tensor
                rho = rho.reshape(d1, d2, d1, d2)
                # Perform partial trace over second subsystem
                reduced_rho = torch.einsum('ijik->jk', rho)
            
            # Compute eigenvalues
            eigenvalues = torch.linalg.eigvalsh(reduced_rho)
            
            # Remove small negative eigenvalues (numerical artifacts)
            eigenvalues = torch.where(eigenvalues > 1e-10, eigenvalues, torch.zeros_like(eigenvalues))
            
            # Normalize eigenvalues
            eigenvalues = eigenvalues / torch.sum(eigenvalues)
            
            # Compute von Neumann entropy
            entropy = -torch.sum(eigenvalues * torch.log(eigenvalues + 1e-10))
            
            return entropy
        else:
            raise ValueError("Input state must be 2-dimensional")

    def evolve_state(self, initial_state: QuantumState, hamiltonian: torch.Tensor, t: Union[float, torch.Tensor]) -> Union[QuantumState, List[QuantumState]]:
        """Evolve quantum state under Hamiltonian."""
        # Ensure complex types
        hamiltonian = self._ensure_complex128(hamiltonian)
        state_vector = self._ensure_complex128(initial_state.amplitudes)
        
        if len(state_vector.shape) == 1:
            state_vector = state_vector.unsqueeze(0)
            
        # Handle single time point vs multiple time points
        if isinstance(t, (float, int)):
            t = torch.tensor([t], dtype=torch.float64)
        else:
            t = t.to(torch.float64)
            
        evolved_states = []
        for time_point in t:
            # Compute evolution operator for this time point
            evolution_operator = torch.matrix_exp(-1j * hamiltonian * time_point)
            evolution_operator = evolution_operator.to(torch.complex128)
            
            # Apply evolution operator
            evolved = torch.matmul(state_vector, evolution_operator.transpose(-2, -1))
            evolved = evolved.to(torch.complex128)
            
            # Add phase correction to preserve pattern orientation
            first_row = evolution_operator[0]
            phase_corrections = []
            for state in evolved:
                # Compute overlap with initial state for phase reference
                overlap = torch.vdot(state_vector[0], state)
                phase_correction = torch.exp(-1j * torch.angle(overlap))
                phase_corrections.append(phase_correction)
            phase_correction = torch.stack(phase_corrections).to(torch.complex128)
            
            # Apply phase correction
            evolved = evolved * phase_correction.unsqueeze(-1)
            
            # Normalize state
            norm = self._compute_norm(evolved)
            evolved = (evolved / norm).to(torch.complex128)
            
            # Create quantum state for this time point
            evolved_state = QuantumState(
                amplitudes=evolved.squeeze(),
                basis_labels=initial_state.basis_labels,
                phase=initial_state.phase.to(torch.complex128)
            )
            evolved_states.append(evolved_state)
            
        # Return single state or list depending on input
        if len(t) == 1:
            return evolved_states[0]
        return evolved_states

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
        eigenvals = eigenvals / (torch.sum(eigenvals) + 1e-10)
        # Remove zero eigenvalues to avoid log(0)
        eigenvals = eigenvals[eigenvals > 1e-10]
        # Compute entropy
        entropy = -torch.sum(eigenvals * torch.log(eigenvals))
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
        
        # Initialize state vector with proper shape and type
        state_vector = initial_state.amplitudes.to(torch.complex128)
        if len(state_vector.shape) == 2:
            state_vector = state_vector[0]  # Take first batch element
            
        # Get dimension of input Hamiltonian
        H_test = hamiltonian_fn(0.0)
        H_dim = H_test.shape[0]
        
        # If state dimension doesn't match Hamiltonian, truncate or pad state
        if state_vector.shape[0] > H_dim:
            state_vector = state_vector[:H_dim]
        elif state_vector.shape[0] < H_dim:
            padded = torch.zeros(H_dim, dtype=torch.complex128)
            padded[:state_vector.shape[0]] = state_vector
            state_vector = padded
            
        # Normalize initial state
        norm = torch.sqrt(torch.sum(torch.abs(state_vector)**2))
        state_vector = state_vector / norm
        
        # Store initial state for final overlap
        initial_vector = state_vector.clone()
        
        # Use adaptive time steps for better accuracy
        n_steps = len(times) * 1000
        fine_times = torch.linspace(times[0], times[-1], n_steps, dtype=torch.float64)
        dt_fine = fine_times[1] - fine_times[0]
        
        # Previous state for phase tracking
        prev_state = state_vector.clone()
        
        # Initialize phase accumulation arrays
        phase_factors = torch.zeros(n_steps - 1, dtype=torch.complex128)
        
        for i, t in enumerate(fine_times[:-1]):
            # Get Hamiltonian at current time
            H = hamiltonian_fn(t.item())
                
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
                phase_factors[i] = phase_factor
            
            # Update states
            prev_state = next_state.clone()
            state_vector = next_state
            
        # Compute cumulative phase
        phase = torch.sum(torch.log(phase_factors + 1e-10).imag)
            
        # Compute final overlap with initial state for cyclic correction
        final_overlap = torch.vdot(initial_vector, state_vector)
        if torch.abs(final_overlap) > 1e-10:
            final_phase = torch.log(final_overlap / torch.abs(final_overlap)).imag
            phase -= final_phase
            
        # Take absolute value since phase can be negative
        phase = torch.abs(phase)
        
        # Convert to complex tensor for output
        return torch.complex(phase, torch.zeros_like(phase))

    def apply_quantum_channel(self, state: QuantumState, kraus_operators: List[torch.Tensor]) -> QuantumState:
        """Apply quantum channel to state using Kraus operators."""
        # Convert state to complex128
        state_vector = self._ensure_complex128(state.amplitudes)
        if len(state_vector.shape) == 1:
            state_vector = state_vector.unsqueeze(0)
            
        # Initialize density matrix
        rho = torch.outer(state_vector.squeeze(), state_vector.squeeze().conj()).to(torch.complex128)
        
        # Apply each Kraus operator
        output_dm = torch.zeros_like(rho, dtype=torch.complex128)
        for kraus in kraus_operators:
            kraus = self._ensure_complex128(kraus)
            # Ensure proper dimensions
            if kraus.shape[0] < self.dim:
                padded = torch.zeros((self.dim, self.dim), dtype=torch.complex128)
                padded[:kraus.shape[0], :kraus.shape[1]] = kraus
                kraus = padded
            output_dm += torch.matmul(torch.matmul(kraus, rho), kraus.conj().T)
            
        # Ensure Hermiticity and positivity
        output_dm = (output_dm + output_dm.conj().T) / 2
        eigenvals, eigenvecs = torch.linalg.eigh(output_dm)
        eigenvals = torch.clamp(eigenvals.real, min=1e-15).to(torch.float64)  # Use small positive threshold
        eigenvals = eigenvals / torch.sum(eigenvals)
        
        # Reconstruct density matrix with positive eigenvalues
        diag_eigenvals = torch.diag(eigenvals).to(torch.complex128)  # Convert eigenvalues to complex128
        output_dm = torch.matmul(
            torch.matmul(eigenvecs, diag_eigenvals),
            eigenvecs.conj().T
        )
        
        # Get pure state from density matrix
        max_idx = torch.argmax(eigenvals)
        output_state = eigenvecs[:, max_idx].to(torch.complex128)
        
        # Normalize
        norm = self._compute_norm(output_state)
        output_state = (output_state / norm).to(torch.complex128)
        
        # Store original norm
        original_norm = norm.clone()
        
        return QuantumState(
            amplitudes=output_state,
            basis_labels=state.basis_labels,
            phase=state.phase.to(torch.complex128),
            original_norm=original_norm
        )

    def reconstruct_state(self, measurements: Dict[str, torch.Tensor], bases: Optional[List[torch.Tensor]] = None) -> QuantumState:
        """Reconstruct quantum state from tomographic measurements using a hybrid approach.
        
        Args:
            measurements: Dictionary of measurement results for different bases
            bases: Optional list of measurement bases (defaults to Pauli bases)
            
        Returns:
            Reconstructed quantum state
        """
        # Default to Pauli bases if not provided
        if bases is None:
            bases = [
                torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128),  # X
                torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128),  # Y
                torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)  # Z
            ]
            
        # Convert measurements to list format
        meas_list = []
        basis_list = []
        for basis_name, meas_val in measurements.items():
            meas_list.append(meas_val.to(torch.complex128))
            if basis_name == "X":
                basis_list.append(bases[0])
            elif basis_name == "Y":
                basis_list.append(bases[1])
            elif basis_name == "Z":
                basis_list.append(bases[2])
                
        # Initialize state vector based on Z measurement
        z_meas = measurements["Z"].real
        x_meas = measurements["X"].real
        y_meas = measurements["Y"].real
        
        # Construct Bloch vector for first qubit
        bloch_vector = torch.tensor([x_meas, y_meas, z_meas], dtype=torch.float64)
        bloch_norm = torch.norm(bloch_vector)
        if bloch_norm > 1:
            bloch_vector = bloch_vector / bloch_norm
            
        # Convert Bloch vector to state vector for first qubit
        theta = torch.arccos(bloch_vector[2])
        phi = torch.atan2(bloch_vector[1], bloch_vector[0])
        state = torch.zeros(self.dim, dtype=torch.complex128)
        state[0] = torch.cos(theta/2)
        state[1] = torch.sin(theta/2) * torch.exp(1j * phi)
        
        # Set remaining amplitudes to small random values
        if self.dim > 2:
            state[2:] = torch.randn(self.dim - 2, dtype=torch.complex128) * 0.01
        state = state / torch.sqrt(torch.sum(torch.abs(state) ** 2))
        
        # Iterative optimization with density matrix projection
        n_iterations = 2000
        learning_rate = 0.01
        best_state = state.clone()
        best_error = float('inf')
        
        for iter_idx in range(n_iterations):
            # Convert state to density matrix
            rho = torch.outer(state, state.conj())
            
            # Compute error and gradient for each measurement
            total_error = 0.0
            grad = torch.zeros_like(state)
            
            for basis, target_meas in zip(basis_list, meas_list):
                # Expand basis operator if needed
                if basis.shape[0] < self.dim:
                    expanded = torch.eye(self.dim, dtype=torch.complex128)
                    expanded[:2, :2] = basis
                    basis = expanded
                    
                # Compute expectation value using density matrix
                expect = torch.trace(torch.matmul(basis, rho)).real
                
                # Compute error
                error = (target_meas - expect).real
                total_error += error ** 2
                
                # Gradient for state vector
                grad_expect = 2 * (torch.matmul(basis, state))
                grad = grad - error * grad_expect
                
            # Update state vector
            state = state - learning_rate * grad
            
            # Project onto unit sphere
            state = state / torch.sqrt(torch.sum(torch.abs(state) ** 2))
            
            # Project density matrix onto positive semidefinite cone
            rho = torch.outer(state, state.conj())
            eigenvals, eigenvecs = torch.linalg.eigh(rho)
            eigenvals = torch.clamp(eigenvals.real, min=0).to(torch.complex128)
            eigenvals = eigenvals / torch.sum(eigenvals)
            diag_eigenvals = torch.diag(eigenvals)
            rho = torch.matmul(torch.matmul(eigenvecs, diag_eigenvals), eigenvecs.conj().T)
            
            # Extract principal eigenvector
            eigenvals, eigenvecs = torch.linalg.eigh(rho)
            max_idx = torch.argmax(eigenvals.real)
            state = eigenvecs[:, max_idx].to(torch.complex128)
            
            # Update best solution
            if total_error < best_error:
                best_error = total_error
                best_state = state.clone()
                
            # Adaptive learning rate
            if iter_idx % 100 == 0:
                learning_rate *= 0.95
                
            # Early stopping
            if total_error < 1e-10:
                break
                
        # Use best solution found
        state = best_state
        
        # Ensure proper normalization
        norm = torch.sqrt(torch.sum(torch.abs(state) ** 2))
        state = (state / norm).to(torch.complex128)
        
        # Store original norm
        original_norm = norm.clone()
        
        return QuantumState(
            amplitudes=state,
            basis_labels=self.basis_states,
            phase=torch.zeros_like(state, dtype=torch.complex128),
            original_norm=original_norm
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
        """Compute fidelity between two quantum states."""
        v1 = self._ensure_complex128(state1.amplitudes)
        v2 = self._ensure_complex128(state2.amplitudes)
        
        if len(v1.shape) == 1:
            v1 = v1.unsqueeze(0)
        if len(v2.shape) == 1:
            v2 = v2.unsqueeze(0)
            
        overlap = torch.abs(torch.matmul(v1.conj(), v2.T))**2
        return overlap.squeeze().real.to(torch.float64)

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
        measurement: Union[str, torch.Tensor]
    ) -> torch.Tensor:
        """Perform projective measurement on quantum state.

        Args:
            state: Quantum state to measure
            measurement: Measurement basis name ("X", "Y", "Z") or operator

        Returns:
            Measurement result as a tensor with shape (2 * dim,) containing real and imaginary parts
        """
        # Ensure complex types
        state_vector = state.amplitudes.to(torch.complex128)
        if len(state_vector.shape) == 1:
            state_vector = state_vector.unsqueeze(0)
        elif len(state_vector.shape) > 2:
            # Take only the first state if we have multiple batches
            state_vector = state_vector[0].unsqueeze(0)
        
        # Convert string basis to operator
        if isinstance(measurement, str):
            if measurement == "X":
                measurement = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
            elif measurement == "Y":
                measurement = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
            elif measurement == "Z":
                measurement = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
            else:
                raise ValueError(f"Unknown measurement basis {measurement}")
            
            # Expand to full dimension if needed
            if measurement.shape[0] < self.dim:
                expanded = torch.eye(self.dim, dtype=torch.complex128)
                expanded[:2, :2] = measurement
                measurement = expanded
        else:
            measurement = measurement.to(torch.complex128)
        
        # Project state onto measurement basis
        eigenvals, eigenvecs = torch.linalg.eigh(measurement)
        
        # Get probabilities for each eigenstate
        probs = torch.zeros(len(eigenvals), dtype=torch.float64)
        for i in range(len(eigenvals)):
            proj = torch.outer(eigenvecs[:, i], eigenvecs[:, i].conj())
            overlap = torch.abs(torch.einsum('bi,ij,bj->b', state_vector.conj(), proj, state_vector))
            probs[i] = overlap.mean().real  # Average over batch
        
        # Sample measurement result
        result = torch.multinomial(probs, 1)[0]
        
        # Project state onto measurement result
        proj = torch.outer(eigenvecs[:, result], eigenvecs[:, result].conj())
        new_state = torch.matmul(proj, state_vector[0])  # Use first state in batch
        
        # Convert to real vector with real and imaginary parts
        result_vector = torch.zeros(2 * self.dim, dtype=torch.float64)
        result_vector[:self.dim] = new_state.real
        result_vector[self.dim:] = new_state.imag
        
        return result_vector

    def state_to_neural(self, state: QuantumState) -> torch.Tensor:
        """Convert quantum state to neural network representation."""
        # Check for zero state
        if torch.all(torch.abs(state.amplitudes) < 1e-10):
            raise ValueError("Cannot convert zero state to neural representation")
            
        # Get state vector
        state_vector = state.amplitudes.to(torch.complex128)
        if len(state_vector.shape) == 1:
            state_vector = state_vector.unsqueeze(0)
            
        # Convert to real representation
        real_part = state_vector.real
        imag_part = state_vector.imag
        
        # Include phase information
        phase = state.phase.to(torch.complex128)
        if len(phase.shape) == 1:
            phase = phase.unsqueeze(0).expand(state_vector.shape[0], -1)
        phase_real = phase.real
        phase_imag = phase.imag
        
        # Concatenate all components
        neural_repr = torch.cat([real_part, imag_part, phase_real, phase_imag], dim=-1)
        
        return neural_repr
        
    def neural_to_state(self, neural_repr: torch.Tensor) -> QuantumState:
        """Convert neural network representation back to quantum state."""
        # Split components
        split_point = neural_repr.shape[-1] // 4
        real_part = neural_repr[..., :split_point]
        imag_part = neural_repr[..., split_point:2*split_point]
        phase_real = neural_repr[..., 2*split_point:3*split_point]
        phase_imag = neural_repr[..., 3*split_point:]
        
        # Combine into complex amplitudes
        amplitudes = torch.complex(real_part, imag_part)
        phase = torch.complex(phase_real, phase_imag)
        
        # Normalize
        norm = torch.sqrt(torch.sum(torch.abs(amplitudes)**2, dim=-1, keepdim=True))
        if torch.any(norm > 1e-10):
            amplitudes = amplitudes / norm
            
        # Handle batch dimension
        if len(amplitudes.shape) > 1:
            amplitudes = amplitudes.squeeze(0)
            phase = phase.squeeze(0)
            
        return QuantumState(
            amplitudes=amplitudes,
            basis_labels=self.basis_states,
            phase=phase
        )

    def compute_geometric_phase(self, state: QuantumState, path: torch.Tensor) -> torch.Tensor:
        """Compute geometric phase along a closed path."""
        # Ensure path is properly shaped and typed
        path = path.to(torch.float64)
        if len(path.shape) == 1:
            path = path.unsqueeze(0)
            
        # Initialize phase accumulation
        total_phase = torch.zeros(1, dtype=torch.float64)
        current_state = state
        
        # Create time-dependent Hamiltonian
        def get_hamiltonian(t: float) -> torch.Tensor:
            # Create a time-dependent Hamiltonian based on the path parameter
            H = self.hamiltonian.clone()  # Use the stored Hamiltonian as base
            # Add time dependence through rotation
            theta = 2 * np.pi * t
            rotation = torch.tensor([
                [np.cos(theta), np.sin(theta)],
                [-np.sin(theta), np.cos(theta)]
            ], dtype=torch.complex128)
            if H.shape[0] > 2:
                # Extend rotation to full Hilbert space dimension
                full_rotation = torch.eye(H.shape[0], dtype=torch.complex128)
                full_rotation[:2, :2] = rotation
                rotation = full_rotation
            return torch.matmul(rotation, H)
        
        # Compute phase for each segment with adaptive step size
        for i in range(len(path) - 1):
            # Get current and next points
            current = path[i]
            next_point = path[i + 1]
            dt = next_point - current
            
            # Use smaller steps for better accuracy
            n_substeps = 10
            dt_sub = dt / n_substeps
            
            # Evolve through substeps
            for j in range(n_substeps):
                t = current + j * dt_sub
                H = get_hamiltonian(t.item())
                
                # Evolve state for small time step
                evolved = self.evolve_state(current_state, H, dt_sub)
                # Handle both single state and list return types
                evolved_state = evolved[0] if isinstance(evolved, list) else evolved
                
                # Compute phase difference using parallel transport
                overlap = torch.vdot(current_state.amplitudes, evolved_state.amplitudes)
                phase_diff = torch.angle(overlap)
                
                # Ensure phase difference is in [-π, π]
                if phase_diff > np.pi:
                    phase_diff -= 2 * np.pi
                elif phase_diff < -np.pi:
                    phase_diff += 2 * np.pi
                    
                # Accumulate phase
                total_phase += phase_diff
                
                # Update state for next substep
                current_state = evolved_state
            
        # Ensure final phase is gauge invariant
        total_phase = torch.remainder(total_phase, 2 * np.pi)
        if total_phase > np.pi:
            total_phase -= 2 * np.pi
            
        return total_phase.real

    def batch_convert_states(self, states: List[QuantumState]) -> torch.Tensor:
        """Convert batch of states to tensor representation."""
        # Convert states to tensors and stack
        state_tensors = []
        for state in states:
            # Ensure complex128 dtype
            amplitudes = state.amplitudes.to(torch.complex128)
            phase = state.phase.to(torch.complex128)
            
            # Combine amplitude and phase
            full_state = amplitudes * torch.exp(1j * torch.angle(phase))
            state_tensors.append(full_state)
            
        # Stack states
        batch_tensor = torch.stack(state_tensors)
        
        # Normalize batch
        norms = torch.sqrt(torch.sum(torch.abs(batch_tensor)**2, dim=-1, keepdim=True)).to(torch.float64)
        batch_tensor = batch_tensor / norms
        
        return batch_tensor

    def geometric_phase_consistency(self, state: QuantumState, path: torch.Tensor) -> bool:
        """Check geometric phase consistency along path."""
        # Ensure path is properly shaped and typed
        path = path.to(torch.float64)
        if len(path.shape) == 1:
            path = path.unsqueeze(0)
            
        # Compute forward phase
        forward_phase = self.compute_geometric_phase(state, path)
        
        # Compute reverse phase
        reverse_path = torch.flip(path, [0])
        reverse_phase = self.compute_geometric_phase(state, reverse_path)
        
        # Check consistency (phases should sum to zero modulo 2π)
        phase_sum = torch.remainder(forward_phase + reverse_phase, 2 * np.pi)
        if phase_sum > np.pi:
            phase_sum -= 2 * np.pi
            
        return bool(torch.abs(phase_sum) < 1e-4)
