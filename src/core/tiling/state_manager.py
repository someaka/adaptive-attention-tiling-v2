"""State Manager for Quantum Geometric Attention.

This module manages the state evolution and transitions in the quantum geometric attention framework.
It handles state initialization, updates, and validation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

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
        # Prefer Vulkan if available, otherwise CPU
        if device is None:
            if hasattr(torch, "vulkan") and torch.vulkan.is_available():
                device = torch.device("vulkan")
            else:
                device = torch.device("cpu")
        self.device = device
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

    def compute_fidelity(self, state1: torch.Tensor, state2: torch.Tensor) -> float:
        """Compute fidelity between two quantum states.

        Args:
            state1: First quantum state
            state2: Second quantum state

        Returns:
            Fidelity between states
        """
        if self.config.type == StateType.PURE:
            return torch.abs(torch.dot(state1, state2)) ** 2
        if self.config.type == StateType.MIXED:
            sqrt1 = torch.matrix_power(state1, 0.5)
            fidelity = torch.trace(torch.matrix_power(sqrt1 @ state2 @ sqrt1, 0.5)) ** 2
            return fidelity.item()
        # ENTANGLED
        return torch.abs(torch.sum(state1 * state2.conj())) ** 2

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
