"""U(1) Covariant Utilities.

This module provides utilities for handling U(1) symmetry and phase tracking
in quantum geometric computations.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Union, cast
import numpy as np

def normalize_with_phase(
    tensor: torch.Tensor,
    eps: float = 1e-8,
    return_phase: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Normalize a complex tensor while tracking its U(1) phase.
    
    This function implements U(1)-covariant normalization:
    1. Preserves the complex phase structure
    2. Returns both normalized tensor and phase factor
    3. Handles numerical stability with proper epsilon
    
    Args:
        tensor: Input tensor to normalize
        eps: Small value for numerical stability
        return_phase: Whether to return the phase factor
        
    Returns:
        If return_phase is False:
            Normalized tensor
        If return_phase is True:
            Tuple of (normalized tensor, phase factor)
    """
    if not tensor.is_complex():
        tensor = torch.complex(tensor, torch.zeros_like(tensor))
        
    # Compute norm with proper complex handling
    norm = torch.sqrt(torch.sum(tensor.conj() * tensor, dim=-1, keepdim=True).real)
    
    # Add epsilon for stability
    safe_norm = norm + eps
    
    # Normalize
    normalized = tensor / safe_norm
    
    if return_phase:
        # Extract phase using principal value
        phases = torch.angle(normalized)
        # Use mean phase for better stability
        phase = torch.mean(phases, dim=-1)
        # Wrap to [-π, π]
        phase = torch.angle(torch.exp(1j * phase))
        return normalized, phase
    
    return normalized

def u1_inner_product(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute U(1)-covariant inner product between tensors.
    
    This implements a proper inner product that respects U(1) symmetry:
    1. Uses conjugate for complex tensors
    2. Preserves phase information
    3. Handles numerical stability
    
    Args:
        x: First tensor
        y: Second tensor
        eps: Small value for numerical stability
        
    Returns:
        Complex inner product
    """
    if not (x.is_complex() and y.is_complex()):
        x = torch.complex(x, torch.zeros_like(x))
        y = torch.complex(y, torch.zeros_like(y))
    
    # Normalize inputs first
    x_norm = cast(torch.Tensor, normalize_with_phase(x))
    y_norm = cast(torch.Tensor, normalize_with_phase(y))
    
    # Compute inner product with conjugate
    inner = torch.sum(x_norm.conj() * y_norm, dim=-1)
    
    # Ensure result is in unit circle
    return inner / (torch.abs(inner) + eps)

def track_phase_evolution(
    initial: torch.Tensor,
    evolved: torch.Tensor,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Track phase evolution between two states.
    
    This function:
    1. Computes the relative phase between states
    2. Ensures phase continuity
    3. Handles numerical stability
    
    Args:
        initial: Initial state
        evolved: Evolved state
        eps: Small value for numerical stability
        
    Returns:
        Tuple of (phase-corrected evolved state, phase difference)
    """
    # Normalize both states
    initial_norm, initial_phase = cast(Tuple[torch.Tensor, torch.Tensor], normalize_with_phase(initial, eps=eps, return_phase=True))
    evolved_norm, evolved_phase = cast(Tuple[torch.Tensor, torch.Tensor], normalize_with_phase(evolved, eps=eps, return_phase=True))
    
    # Compute phase difference using inner product
    inner = u1_inner_product(initial_norm, evolved_norm)
    phase_diff = torch.angle(inner)
    
    # Ensure phase difference is continuous
    phase_diff = torch.where(
        phase_diff > np.pi,
        phase_diff - 2*np.pi,
        phase_diff
    )
    phase_diff = torch.where(
        phase_diff < -np.pi,
        phase_diff + 2*np.pi,
        phase_diff
    )
    
    # Apply phase correction
    correction = torch.exp(-1j * phase_diff)
    corrected = evolved_norm * correction.unsqueeze(-1)
    
    return corrected, phase_diff

def compose_phases(
    phase1: torch.Tensor,
    phase2: torch.Tensor
) -> torch.Tensor:
    """Compose two U(1) phases.
    
    This function:
    1. Properly handles phase composition
    2. Ensures result is in [-π, π]
    3. Maintains differentiability
    
    Args:
        phase1: First phase
        phase2: Second phase
        
    Returns:
        Composed phase
    """
    # Convert to complex numbers on unit circle
    z1 = torch.exp(1j * phase1)
    z2 = torch.exp(1j * phase2)
    
    # Compose by multiplication
    composed = z1 * z2
    
    # Extract phase in [-π, π]
    return torch.angle(composed)

def compute_winding_number(
    state: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """Compute winding number of a U(1) state.
    
    This function:
    1. Computes total phase accumulation
    2. Normalizes to get integer winding
    3. Handles numerical stability
    
    For constant vectors (same value repeated), we compute the winding
    number from the phase of a single component.
    
    Args:
        state: Input state
        eps: Small value for numerical stability
        
    Returns:
        Winding number (in units of 2π)
    """
    # Normalize state
    state_norm = cast(torch.Tensor, normalize_with_phase(state))
    
    # Check if state is constant (same value repeated)
    # Use a more robust check for constant phase
    first_component = state_norm[..., :1]
    max_diff = torch.max(torch.abs(state_norm - first_component))
    if max_diff < 1e-5:
        # For constant phase states, winding number is 0
        return torch.tensor(0.0, device=state.device, dtype=state.dtype)
    
    # Compute phases
    phases = torch.angle(state_norm)
    
    # Compute phase differences
    phase_diffs = phases[..., 1:] - phases[..., :-1]
    
    # Ensure differences are in [-π, π]
    phase_diffs = torch.where(
        phase_diffs > np.pi,
        phase_diffs - 2*np.pi,
        phase_diffs
    )
    phase_diffs = torch.where(
        phase_diffs < -np.pi,
        phase_diffs + 2*np.pi,
        phase_diffs
    )
    
    # Total phase accumulation
    total_phase = torch.sum(phase_diffs, dim=-1)
    
    # Convert to winding number (divide by 2π)
    return total_phase / (2 * np.pi) 