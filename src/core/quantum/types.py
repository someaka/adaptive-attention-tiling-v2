"""Quantum type definitions.

This module contains core quantum type definitions used across the framework.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch


@dataclass
class QuantumState:
    """Represents a quantum state in the attention Hilbert space."""

    amplitudes: torch.Tensor  # State vector amplitudes
    basis_labels: List[str]  # Labels for basis states
    phase: torch.Tensor  # Quantum phase information
    original_norm: Optional[torch.Tensor] = None  # Original norm of the input tensor

    def __post_init__(self):
        """Ensure state normalization and proper tensor types."""
        # Convert to complex128 if not already
        if not torch.is_complex(self.amplitudes):
            self.amplitudes = self.amplitudes.to(torch.complex128)
        elif self.amplitudes.dtype != torch.complex128:
            self.amplitudes = self.amplitudes.to(torch.complex128)
            
        if not torch.is_complex(self.phase):
            self.phase = self.phase.to(torch.complex128)
        elif self.phase.dtype != torch.complex128:
            self.phase = self.phase.to(torch.complex128)

        # Normalize state vector
        if len(self.amplitudes.shape) == 1:
            # Single state vector
            norm = torch.sqrt(torch.sum(torch.abs(self.amplitudes) ** 2))
            self.amplitudes = self.amplitudes / (norm + 1e-8)
        else:
            # Batch of state vectors
            norm = torch.sqrt(torch.sum(torch.abs(self.amplitudes) ** 2, dim=1, keepdim=True))
            self.amplitudes = self.amplitudes / (norm + 1e-8)

        # Store original norm if not provided
        if self.original_norm is None:
            self.original_norm = norm.to(torch.float64)
        else:
            self.original_norm = self.original_norm.to(torch.float64)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return shape of state amplitudes."""
        return self.amplitudes.shape

    @property
    def data(self) -> torch.Tensor:
        """Get state data."""
        return self.amplitudes

    def norm(self) -> torch.Tensor:
        """Compute the norm of the quantum state."""
        if len(self.amplitudes.shape) == 1:
            return torch.sqrt(torch.sum(torch.abs(self.amplitudes) ** 2)).to(torch.float64)
        return torch.sqrt(torch.sum(torch.abs(self.amplitudes) ** 2, dim=1)).to(torch.float64)

    def density_matrix(self) -> torch.Tensor:
        """Compute the density matrix representation of the state."""
        # For a pure state, density matrix is |ψ⟩⟨ψ|
        if len(self.amplitudes.shape) == 1:
            return torch.outer(self.amplitudes, self.amplitudes.conj())
        
        # For batched states
        batch_size = self.amplitudes.shape[0]
        if len(self.amplitudes.shape) == 3:
            # Handle case where amplitudes have sequence dimension
            seq_len = self.amplitudes.shape[1]
            state_dim = self.amplitudes.shape[2]
            rho = torch.zeros((batch_size, seq_len, state_dim, state_dim), 
                             dtype=torch.complex128, 
                             device=self.amplitudes.device)
            for i in range(batch_size):
                for j in range(seq_len):
                    rho[i, j] = torch.outer(self.amplitudes[i, j], self.amplitudes[i, j].conj())
            return rho
        else:
            # Regular batch case
            state_dim = self.amplitudes.shape[1]
            rho = torch.zeros((batch_size, state_dim, state_dim), 
                             dtype=torch.complex128, 
                             device=self.amplitudes.device)
            for i in range(batch_size):
                rho[i] = torch.outer(self.amplitudes[i], self.amplitudes[i].conj())
            return rho

    def inner_product(self, other: 'QuantumState') -> torch.Tensor:
        """Compute inner product with another state."""
        if len(self.amplitudes.shape) == 1:
            return torch.sum(self.amplitudes.conj() * other.amplitudes)
        # For batched states
        return torch.sum(self.amplitudes.conj() * other.amplitudes, dim=1)

    def outer_product(self, other: Optional['QuantumState'] = None) -> torch.Tensor:
        """Compute outer product with another state or itself."""
        if other is None:
            other = self
        if len(self.amplitudes.shape) == 1:
            return torch.outer(self.amplitudes, other.amplitudes.conj())
        # For batched states
        batch_size = self.amplitudes.shape[0]
        state_dim = self.amplitudes.shape[1]
        result = torch.zeros((batch_size, state_dim, state_dim), dtype=self.amplitudes.dtype, device=self.amplitudes.device)
        for i in range(batch_size):
            result[i] = torch.outer(self.amplitudes[i], other.amplitudes[i].conj())
        return result

    def evolve(self, unitary: torch.Tensor) -> 'QuantumState':
        """Evolve the state under a unitary operator."""
        if len(self.amplitudes.shape) == 1:
            new_amplitudes = unitary @ self.amplitudes
        else:
            # For batched states
            if len(self.amplitudes.shape) == 3:  # [batch_size, seq_len, dim]
                batch_size, seq_len, dim = self.amplitudes.shape
                amplitudes_flat = self.amplitudes.reshape(-1, dim)  # [batch_size * seq_len, dim]
                
                # Ensure unitary has correct shape [batch_size * seq_len, dim, dim]
                if len(unitary.shape) > 3:
                    unitary = unitary.reshape(-1, dim, dim)
                unitary_expanded = unitary.expand(batch_size * seq_len, dim, dim)
                
                new_amplitudes_flat = torch.bmm(unitary_expanded, amplitudes_flat.unsqueeze(-1)).squeeze(-1)
                new_amplitudes = new_amplitudes_flat.reshape(batch_size, seq_len, dim)
            else:
                new_amplitudes = unitary @ self.amplitudes
            
        return QuantumState(
            amplitudes=new_amplitudes,
            basis_labels=self.basis_labels,
            phase=self.phase,
            original_norm=self.original_norm
        )

    def partial_trace(self, subsystem_dims: List[int], trace_out: int) -> torch.Tensor:
        """Compute reduced density matrix by tracing out specified subsystem."""
        total_dims = len(subsystem_dims)
        if total_dims < 2:
            raise ValueError("Need at least 2 subsystems to perform partial trace")
        
        # Reshape state vector into tensor product structure
        state_tensor = self.amplitudes.view(*subsystem_dims)
        
        # Compute full density matrix
        rho = self.density_matrix()
        reshaped_rho = rho.view(*subsystem_dims, *subsystem_dims)
        
        # Trace out specified subsystem
        trace_dims = list(range(total_dims))
        trace_dims.pop(trace_out)
        reduced_rho = torch.einsum(reshaped_rho, [*trace_dims, *trace_dims])
        
        return reduced_rho

    def to_device(self, device: torch.device) -> 'QuantumState':
        """Move quantum state to specified device."""
        return QuantumState(
            amplitudes=self.amplitudes.to(device),
            basis_labels=self.basis_labels,
            phase=self.phase.to(device)
        )

    @property
    def hilbert_space(self) -> int:
        """Get the dimension of the Hilbert space."""
        return self.amplitudes.shape[-1]

    @property
    def num_qubits(self) -> int:
        """Get the number of qubits in the state."""
        return int(torch.log2(torch.tensor(self.hilbert_space)).item())

    def is_pure(self, tolerance: float = 1e-6) -> bool:
        """Check if the state is pure by computing the purity of its density matrix.
        
        A pure state has Tr(ρ²) = 1, while mixed states have Tr(ρ²) < 1.
        
        Args:
            tolerance: Numerical tolerance for comparison with 1.0
            
        Returns:
            bool: True if the state is pure, False otherwise
        """
        rho = self.density_matrix()
        if len(rho.shape) == 3:  # Batched case
            # Compute purity for each state in batch
            purity = torch.diagonal(torch.matmul(rho, rho), dim1=-2, dim2=-1).sum(-1).real
            return bool(torch.all(torch.abs(purity - 1.0) < tolerance))
        else:  # Single state case
            purity = torch.trace(torch.matmul(rho, rho)).real
            return bool(abs(purity - 1.0) < tolerance)

    def state_vector(self) -> torch.Tensor:
        """Get the state vector representation.
        
        For pure states, this is just the amplitudes.
        For mixed states, this raises an error.
        
        Returns:
            torch.Tensor: The state vector
            
        Raises:
            ValueError: If the state is not pure
        """
        if not self.is_pure():
            raise ValueError("Cannot get state vector for mixed state")
        return self.amplitudes
