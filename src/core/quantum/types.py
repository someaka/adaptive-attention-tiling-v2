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

    def __post_init__(self):
        """Ensure state normalization and proper tensor types."""
        # Convert to complex if not already
        if not torch.is_complex(self.amplitudes):
            self.amplitudes = self.amplitudes.to(torch.complex64)
        if not torch.is_complex(self.phase):
            self.phase = self.phase.to(torch.complex64)

        # Normalize state vector
        if len(self.amplitudes.shape) == 1:
            # Single state vector
            norm = torch.sqrt(torch.sum(torch.abs(self.amplitudes) ** 2))
            self.amplitudes = self.amplitudes / (norm + 1e-8)
        else:
            # Batch of state vectors
            norm = torch.sqrt(torch.sum(torch.abs(self.amplitudes) ** 2, dim=1, keepdim=True))
            self.amplitudes = self.amplitudes / (norm + 1e-8)

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
            return torch.sqrt(torch.sum(torch.abs(self.amplitudes) ** 2))
        return torch.sqrt(torch.sum(torch.abs(self.amplitudes) ** 2, dim=1))

    def density_matrix(self) -> torch.Tensor:
        """Compute the density matrix representation of the state."""
        # For a pure state, density matrix is |ψ⟩⟨ψ|
        if len(self.amplitudes.shape) == 1:
            return torch.outer(self.amplitudes, self.amplitudes.conj())
        # For batched states
        batch_size = self.amplitudes.shape[0]
        state_dim = self.amplitudes.shape[1]
        rho = torch.zeros((batch_size, state_dim, state_dim), dtype=torch.complex64, device=self.amplitudes.device)
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
            new_amplitudes = torch.bmm(unitary.unsqueeze(0).expand(self.amplitudes.shape[0], -1, -1),
                                     self.amplitudes.unsqueeze(-1)).squeeze(-1)
        return QuantumState(new_amplitudes, self.basis_labels, self.phase)

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