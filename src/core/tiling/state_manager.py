"""State Manager for Quantum Geometric Attention.

This module manages the state evolution and transitions in the quantum geometric attention framework.
It handles state initialization, updates, and validation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict

import torch


class StateType(Enum):
    """Types of quantum geometric states."""

    PURE = "pure"
    MIXED = "mixed"
    ENTANGLED = "entangled"


@dataclass
class StateConfig:
    """Configuration for quantum geometric states."""

    dim: int
    type: StateType
    epsilon: float = 1e-6
    max_entanglement: float = 1.0


class StateManager:
    """Manager for quantum geometric states."""

    def __init__(self, config: StateConfig, device: Optional[torch.device] = None):
        """Initialize the state manager.

        Args:
            config: Configuration for state management
            device: Torch device for computations
        """
        self.config = config
        self.device = device if device is not None else torch.device("cpu")
        self.states = {}
        self.history = []

    def initialize_state(self, key: str, dim: Optional[int] = None) -> torch.Tensor:
        """Initialize a new quantum state.

        Args:
            key: Unique identifier for the state
            dim: Optional dimension override

        Returns:
            Initialized quantum state tensor
        """
        dim = dim or self.config.dim

        if self.config.type == StateType.PURE:
            state = torch.randn(dim, device=self.device)
            state = state / torch.norm(state)
        elif self.config.type == StateType.MIXED:
            state = torch.eye(dim, device=self.device)
            state = state / torch.trace(state)
        else:  # ENTANGLED
            state = torch.randn(dim, dim, device=self.device)
            state = state / torch.norm(state)

        self.states[key] = state
        return state

    def update_state(
        self, key: str, update: torch.Tensor, learning_rate: float = 0.01
    ) -> torch.Tensor:
        """Update an existing quantum state.

        Args:
            key: State identifier
            update: Update tensor
            learning_rate: Learning rate for update

        Returns:
            Updated state tensor
        """
        if key not in self.states:
            raise KeyError(f"State {key} not found")

        state = self.states[key]
        new_state = state + learning_rate * update

        # Normalize based on state type
        if self.config.type == StateType.PURE:
            new_state = new_state / torch.norm(new_state)
        elif self.config.type == StateType.MIXED:
            new_state = 0.5 * (new_state + new_state.T)
            new_state = new_state / torch.trace(new_state)
        else:  # ENTANGLED
            new_state = new_state / torch.norm(new_state)

        self.states[key] = new_state
        self.history.append((key, new_state.detach().clone()))
        return new_state

    def calculate_fidelity(self, state1: torch.Tensor, state2: torch.Tensor) -> float:
        """Calculate fidelity between two quantum states.

        Args:
            state1: First quantum state
            state2: Second quantum state

        Returns:
            Fidelity between states
        """
        if self.config.type == StateType.PURE:
            return float((torch.abs(torch.dot(state1, state2)) ** 2.0).item())
        if self.config.type == StateType.MIXED:
            sqrt1 = torch.matrix_power(state1, 1)  # Use identity for numerical stability
            sqrt1 = torch.sqrt(sqrt1)  # Element-wise sqrt is more stable
            inner_term = sqrt1 @ state2 @ sqrt1
            fidelity = torch.trace(torch.sqrt(inner_term)) ** 2.0
            return float(fidelity.item())
        # ENTANGLED
        return float(torch.abs(torch.sum(state1 * state2.conj())) ** 2.0)

    def get_state_history(self, key: str) -> List[torch.Tensor]:
        """Get history of state updates.

        Args:
            key: State identifier

        Returns:
            List of historical states
        """
        return [state for k, state in self.history if k == key]

    def validate_state(self, state: torch.Tensor) -> bool:
        """Validate a quantum state.

        Args:
            state: Quantum state to validate

        Returns:
            True if state is valid
        """
        if self.config.type == StateType.PURE:
            norm_diff = abs(torch.norm(state).item() - 1.0)
            return norm_diff < self.config.epsilon
        if self.config.type == StateType.MIXED:
            # Check hermiticity and trace
            is_hermitian = torch.allclose(
                state, state.T.conj(), rtol=self.config.epsilon
            )
            has_unit_trace = abs(torch.trace(state).item() - 1.0) < self.config.epsilon
            return is_hermitian and has_unit_trace
        # ENTANGLED
        # Check normalization and entanglement
        norm_ok = abs(torch.norm(state).item() - 1.0) < self.config.epsilon
        entanglement = torch.abs(torch.det(state)).item()
        return norm_ok and entanglement <= self.config.max_entanglement

    def update_entanglement(
        self,
        source_scale: float,
        target_scale: float,
        entropy: torch.Tensor
    ) -> None:
        """Update entanglement tracking between scales.
        
        Args:
            source_scale: Source scale factor
            target_scale: Target scale factor
            entropy: Entanglement entropy tensor
        """
        # Create scale transition key
        transition_key = f"{source_scale:.2f}->{target_scale:.2f}"
        
        # Initialize entanglement tracking if needed
        if not hasattr(self, "_entanglement_tracking"):
            self._entanglement_tracking = {}
            
        # Update tracking
        if transition_key not in self._entanglement_tracking:
            self._entanglement_tracking[transition_key] = []
            
        self._entanglement_tracking[transition_key].append(entropy.item())
        
        # Keep only recent history
        max_history = 100
        if len(self._entanglement_tracking[transition_key]) > max_history:
            self._entanglement_tracking[transition_key] = (
                self._entanglement_tracking[transition_key][-max_history:]
            )
            
    def get_entanglement_history(
        self,
        source_scale: Optional[float] = None,
        target_scale: Optional[float] = None
    ) -> Dict[str, List[float]]:
        """Get entanglement history for scale transitions.
        
        Args:
            source_scale: Optional source scale to filter
            target_scale: Optional target scale to filter
            
        Returns:
            Dictionary mapping transition keys to entropy histories
        """
        if not hasattr(self, "_entanglement_tracking"):
            return {}
            
        if source_scale is None and target_scale is None:
            return self._entanglement_tracking
            
        # Filter transitions
        filtered = {}
        for key, history in self._entanglement_tracking.items():
            src, tgt = map(float, key.split("->"))
            if (source_scale is None or src == source_scale) and \
               (target_scale is None or tgt == target_scale):
                filtered[key] = history
                
        return filtered
