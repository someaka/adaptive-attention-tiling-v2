"""Scale Transition System Implementation.

This module implements the scale transition system that connects different scales
in the neural framework, handling:
- Scale transitions between layers
- Pattern scale connections
- Quantum scale bridges
- Scale validation
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Mapping, cast

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from .crystal.scale import ScaleSystem
from .quantum.neural_quantum_bridge import NeuralQuantumBridge


@dataclass
class ScaleTransitionConfig:
    """Configuration for scale transitions."""

    min_scale: float = 0.25
    max_scale: float = 4.0
    num_scales: int = 4
    dim: int = 64
    use_quantum_bridge: bool = True
    hidden_dim: int = 64
    dtype: torch.dtype = torch.float32  # Data type for tensors


class ScaleTransitionLayer(nn.Module):
    """Layer that handles transitions between different scales."""
    
    def __init__(self, config: ScaleTransitionConfig):
        super().__init__()
        self.config = config
        
        # Initialize quantum bridge if enabled
        self.quantum_bridge = (
            NeuralQuantumBridge(
                hidden_dim=config.hidden_dim,
                dtype=config.dtype
            )
            if config.use_quantum_bridge
            else None
        )
        
        # Scale transition networks - use simple linear layers
        self.scale_up = nn.ModuleList([
            nn.Linear(config.dim, config.dim, dtype=config.dtype)
            for _ in range(config.num_scales - 1)
        ])
        
        self.scale_down = nn.ModuleList([
            nn.Linear(config.dim, config.dim, dtype=config.dtype)
            for _ in range(config.num_scales - 1)
        ])
        
        # Initialize weights for norm preservation and reversibility
        for linear in self.scale_up:
            # Initialize as close to identity as possible
            nn.init.eye_(linear.weight)
            nn.init.zeros_(linear.bias)
            # Disable bias to improve reversibility
            linear.bias.requires_grad = False
            
        for linear in self.scale_down:
            # Initialize as close to identity as possible
            nn.init.eye_(linear.weight)
            nn.init.zeros_(linear.bias)
            # Disable bias to improve reversibility
            linear.bias.requires_grad = False
    
    def _get_scale_idx(self, source_scale: float, target_scale: float) -> int:
        """Get scale transition index."""
        ratio = target_scale / source_scale
        # Handle all possible scale ratios
        if ratio >= 1:
            idx = int(np.round(np.log2(ratio)))
        else:
            idx = int(np.round(np.log2(1 / ratio)))
        
        # Validate scale difference
        if idx >= len(self.scale_up):
            raise ValueError(f"Scale difference too large (index {idx} >= {len(self.scale_up)})")
        return idx
    
    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Normalize state to unit norm."""
        # Compute current norm
        curr_norm = torch.linalg.vector_norm(state, dim=-1, keepdim=True)
        # Normalize to unit norm with a small offset to prevent division by zero
        # and improve stability
        return F.normalize(state, p=2, dim=-1)
    
    def _apply_scale_transition(
        self,
        state: torch.Tensor,
        scale_idx: int,
        transition_modules: nn.ModuleList
    ) -> torch.Tensor:
        """Apply scale transition with multiple steps if needed."""
        # Apply transitions one step at a time
        remaining_steps = scale_idx
        curr_state = state
        while remaining_steps > 0:
            curr_step = min(remaining_steps, len(transition_modules))
            # Apply transition
            next_state = transition_modules[curr_step - 1](curr_state)
            # Ensure stability by clamping the change
            delta = next_state - curr_state
            delta_norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
            max_delta = 0.0001
            # Apply soft clamping using sigmoid with steeper slope
            scale = max_delta * torch.sigmoid(2 * delta_norm / max_delta) / (delta_norm + 1e-6)
            curr_state = curr_state + delta * scale
            # Renormalize after each step to maintain stability
            curr_state = self._normalize_state(curr_state)
            remaining_steps -= curr_step
        return curr_state
    
    def transition_up(
        self, 
        state: torch.Tensor,
        source_scale: float,
        target_scale: float
    ) -> torch.Tensor:
        """Transition to a larger scale."""
        # Validate scales
        if target_scale <= source_scale:
            raise ValueError("Target scale must be larger than source scale")
            
        # Get scale index
        scale_idx = self._get_scale_idx(source_scale, target_scale)
            
        # Store original norm
        orig_norm = torch.linalg.vector_norm(state, dim=-1, keepdim=True)
        
        # Normalize input state
        state = self._normalize_state(state)
        
        # Apply scale transition
        state = self._apply_scale_transition(state, scale_idx, self.scale_up)
        
        # Apply scale factor
        scale_factor = target_scale / source_scale
        state = state * (orig_norm * scale_factor)
        
        # Apply quantum bridge if enabled
        if self.quantum_bridge is not None:
            state = self.quantum_bridge.bridge_scales(
                state,
                source_scale,
                target_scale
            )
            
        return state
    
    def transition_down(
        self,
        state: torch.Tensor,
        source_scale: float,
        target_scale: float
    ) -> torch.Tensor:
        """Transition to a smaller scale."""
        # Validate scales
        if target_scale >= source_scale:
            raise ValueError("Target scale must be smaller than source scale")
            
        # Get scale index
        scale_idx = self._get_scale_idx(source_scale, target_scale)
            
        # Store original norm
        orig_norm = torch.linalg.vector_norm(state, dim=-1, keepdim=True)
        
        # Normalize input state
        state = self._normalize_state(state)
        
        # Apply scale transition
        state = self._apply_scale_transition(state, scale_idx, self.scale_down)
        
        # Apply scale factor
        scale_factor = target_scale / source_scale
        state = state * (orig_norm * scale_factor)
        
        # Apply quantum bridge if enabled
        if self.quantum_bridge is not None:
            state = self.quantum_bridge.bridge_scales(
                state,
                source_scale,
                target_scale
            )
            
        return state
    
    def forward(
        self,
        state: torch.Tensor,
        source_scale: float,
        target_scale: float
    ) -> torch.Tensor:
        """Forward pass handling scale transitions in either direction."""
        if target_scale > source_scale:
            return self.transition_up(state, source_scale, target_scale)
        elif target_scale < source_scale:
            return self.transition_down(state, source_scale, target_scale)
        else:
            return state  # No transition needed


class ScaleTransitionSystem:
    """System that manages scale transitions across the neural framework."""
    
    def __init__(self, config: ScaleTransitionConfig):
        self.config = config
        self.transition_layer = ScaleTransitionLayer(config)
        
    def connect_scales(
        self,
        states: List[torch.Tensor],
        scales: List[float]
    ) -> List[torch.Tensor]:
        """Connect states at different scales."""
        if len(states) != len(scales):
            raise ValueError("Number of states must match number of scales")
            
        connected_states = []
        for i, (state, scale) in enumerate(zip(states, scales)):
            # Connect to next scale if not last
            if i < len(states) - 1:
                next_state = self.transition_layer(
                    state,
                    scale,
                    scales[i + 1]
                )
                connected_states.append(next_state)
            else:
                connected_states.append(state)
                
        return connected_states
    
    def validate_transitions(
        self,
        states: List[torch.Tensor],
        scales: List[float]
    ) -> Mapping[str, torch.Tensor]:
        """Validate scale transitions."""
        if len(states) != len(scales):
            raise ValueError("Number of states must match number of scales")
            
        metrics = {}
        
        # Compute scale consistency
        scale_consistency = []
        for i in range(len(states) - 1):
            # Compute norm ratio between consecutive scales
            norm_ratio = (
                torch.linalg.vector_norm(states[i + 1], dim=-1) /
                (torch.linalg.vector_norm(states[i], dim=-1) + 1e-8)
            )
            # Compare with expected scale ratio
            expected_ratio = scales[i + 1] / scales[i]
            scale_consistency.append(
                torch.mean(torch.abs(norm_ratio - expected_ratio))
            )
        metrics["scale_consistency"] = torch.tensor(scale_consistency)
        
        # Compute information preservation
        info_preservation = []
        for i in range(len(states) - 1):
            # Compute cosine similarity between consecutive states
            similarity = F.cosine_similarity(
                states[i + 1],
                states[i],
                dim=-1
            )
            info_preservation.append(torch.mean(similarity))
        metrics["information_preservation"] = torch.tensor(info_preservation)
        
        # Add quantum coherence if quantum bridge is enabled
        if self.transition_layer.quantum_bridge is not None:
            quantum_coherence = []
            for i in range(len(states) - 1):
                coherence = self.transition_layer.quantum_bridge.compute_coherence(
                    states[i],
                    states[i + 1]
                )
                quantum_coherence.append(torch.mean(coherence))
            metrics["quantum_coherence"] = torch.tensor(quantum_coherence)
            
        return metrics


class ScaleFlowIntegrator(nn.Module):
    """Integrates scale transitions with quantum geometric flow."""
    
    def __init__(self, config: ScaleTransitionConfig):
        super().__init__()
        self.config = config
        
        # Scale-aware quantum components
        self.scale_quantum_proj = nn.Linear(config.dim, config.hidden_dim)
        self.scale_classical_proj = nn.Linear(config.hidden_dim, config.dim)
        
        # Scale-dependent quantum operations
        self.scale_dependent_ops = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.ReLU()
            )
            for _ in range(config.num_scales)
        ])
        
        # Entanglement tracking
        self.entanglement_tracker = nn.Parameter(
            torch.zeros(config.num_scales, config.hidden_dim)
        )
        
    def compute_scale_quantum_state(
        self,
        state: torch.Tensor,
        scale_idx: int
    ) -> torch.Tensor:
        """Compute scale-dependent quantum state."""
        # Project to quantum space
        quantum_state = self.scale_quantum_proj(state)
        
        # Apply scale-dependent operations
        quantum_state = self.scale_dependent_ops[scale_idx](quantum_state)
        
        # Track entanglement
        self.entanglement_tracker.data[scale_idx] = torch.mean(
            torch.abs(quantum_state), dim=0
        )
        
        return quantum_state
        
    def integrate_flow(
        self,
        state: torch.Tensor,
        source_scale: float,
        target_scale: float,
        quantum_bridge: Optional[NeuralQuantumBridge] = None
    ) -> torch.Tensor:
        """Integrate quantum geometric flow between scales."""
        # Get scale indices
        source_idx = int(np.log2(source_scale / self.config.min_scale))
        target_idx = int(np.log2(target_scale / self.config.min_scale))
        
        # Compute quantum states
        source_quantum = self.compute_scale_quantum_state(state, source_idx)
        
        if quantum_bridge is not None:
            # Use quantum bridge for evolution
            quantum_result = quantum_bridge.neural_to_quantum(source_quantum)
            if isinstance(quantum_result, tuple):
                quantum_state, _ = quantum_result
            else:
                quantum_state = quantum_result
                
            evolved = quantum_bridge.evolve_quantum_state(
                quantum_state,
                time=abs(target_scale - source_scale)
            )
            target_quantum = quantum_bridge.quantum_to_neural(evolved)
        else:
            # Direct evolution in quantum space
            target_quantum = self.compute_scale_quantum_state(state, target_idx)
        
        # Project back to classical space
        evolved_state = self.scale_classical_proj(target_quantum)
        
        return evolved_state
        
    def get_entanglement_metrics(self) -> Dict[str, torch.Tensor]:
        """Get entanglement tracking metrics."""
        return {
            "mean_entanglement": torch.mean(self.entanglement_tracker, dim=1),
            "max_entanglement": torch.max(self.entanglement_tracker, dim=1)[0],
            "entanglement_std": torch.std(self.entanglement_tracker, dim=1)
        }