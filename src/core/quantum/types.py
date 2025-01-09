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
    original_energy: Optional[torch.Tensor] = None  # Original energy of the input tensor

    def __post_init__(self):
        """Ensure state normalization and proper tensor types."""
        # Convert to complex while preserving input dtype
        if not torch.is_complex(self.amplitudes):
            dtype = torch.complex64 if self.amplitudes.dtype == torch.float32 else torch.complex128
            self.amplitudes = torch.complex(self.amplitudes, torch.zeros_like(self.amplitudes)).to(dtype)
        if not torch.is_complex(self.phase):
            dtype = torch.complex64 if self.phase.dtype == torch.float32 else torch.complex128
            self.phase = torch.complex(self.phase, torch.zeros_like(self.phase)).to(dtype)

        # Store original norm before normalization
        if len(self.amplitudes.shape) == 1:
            # Single state vector
            norm = torch.sqrt(torch.sum(torch.abs(self.amplitudes) ** 2))
        else:
            # Batch of state vectors - normalize globally across all dimensions except batch
            norm = torch.sqrt(torch.sum(torch.abs(self.amplitudes) ** 2, dim=tuple(range(1, len(self.amplitudes.shape))), keepdim=True))
        
        # Store original norm if not provided
        if self.original_norm is None:
            self.original_norm = norm.real
        
        # Normalize state vector globally
        self.amplitudes = self.amplitudes / (norm + 1e-8)

    # Add PyTorch tensor compatibility properties
    @property
    def layout(self) -> torch.layout:
        """Get the memory layout of the amplitudes tensor."""
        return self.amplitudes.layout

    @property
    def device(self) -> torch.device:
        """Get the device of the amplitudes tensor."""
        return self.amplitudes.device

    @property
    def requires_grad(self) -> bool:
        """Get the requires_grad flag of the amplitudes tensor."""
        return self.amplitudes.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        """Set the requires_grad flag of the amplitudes tensor."""
        self.amplitudes.requires_grad = value

    def to(self, *args, **kwargs) -> 'QuantumState':
        """Move the quantum state to a device or change its dtype."""
        new_amplitudes = self.amplitudes.to(*args, **kwargs)
        new_phase = self.phase.to(*args, **kwargs)
        new_original_norm = self.original_norm.to(*args, **kwargs) if self.original_norm is not None else None
        return QuantumState(
            amplitudes=new_amplitudes,
            basis_labels=self.basis_labels,
            phase=new_phase,
            original_norm=new_original_norm
        )

    def __torch_function__(self, func, types, args=(), kwargs=None):
        """Enable PyTorch function compatibility."""
        if kwargs is None:
            kwargs = {}
        # Convert QuantumState instances to their amplitude tensors
        processed_args = []
        for arg in args:
            if isinstance(arg, QuantumState):
                processed_args.append(arg.amplitudes)
            else:
                processed_args.append(arg)
        return func(*processed_args, **kwargs)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return shape of state amplitudes."""
        return self.amplitudes.shape

    @property
    def data(self) -> torch.Tensor:
        """Get state data."""
        return self.amplitudes

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of the state amplitudes."""
        return self.amplitudes.dtype

    def norm(self) -> torch.Tensor:
        """Compute the norm of the quantum state."""
        if len(self.amplitudes.shape) == 1:
            return torch.sqrt(torch.sum(torch.abs(self.amplitudes) ** 2))
        # For multi-dimensional tensors, compute norm across all dimensions except batch
        return torch.sqrt(torch.sum(torch.abs(self.amplitudes) ** 2, dim=tuple(range(1, len(self.amplitudes.shape)))))

    def density_matrix(self) -> torch.Tensor:
        """Compute the density matrix representation of the quantum state."""
        # Compute outer product while preserving dtype
        density = torch.outer(self.amplitudes, self.amplitudes.conj())
        # Apply phase factor
        return density * self.phase

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

    def abs(self) -> torch.Tensor:
        """Compute absolute value of quantum state amplitudes.
        
        Returns:
            Tensor containing absolute values of amplitudes
        """
        return torch.abs(self.amplitudes)
