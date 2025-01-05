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
        return F.normalize(state, p=2, dim=-1)
    
    def _apply_scale_transition(
        self,
        state: torch.Tensor,
        scale_idx: int,
        transition_modules: nn.ModuleList,
        source_scale: float,
        target_scale: float,
        apply_final_scale: bool = True
    ) -> torch.Tensor:
        """Apply scale transition with multiple steps if needed."""
        # Apply transitions one step at a time
        remaining_steps = scale_idx
        curr_state = state
        
        # Calculate total scale factor and per-step factor
        scale_factor = target_scale / source_scale
        # Adjust step factor to be more conservative for stability
        step_factor = scale_factor ** (1.0 / (2 * scale_idx)) if scale_idx > 0 else 1.0
        
        # Store original norm for preservation
        orig_norm = torch.linalg.vector_norm(state, dim=-1, keepdim=True)
        
        # Initialize reference state and stability history
        ref_state = curr_state.clone()
        stability_history = []
        
        while remaining_steps > 0:
            curr_step = min(remaining_steps, len(transition_modules))
            
            # Apply transition
            next_state = transition_modules[curr_step - 1](curr_state)
            
            # Compute change and update stability history
            delta = next_state - curr_state
            delta_norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
            stability_history.append(delta_norm)
            
            # Compute adaptive stability threshold that scales with transition magnitude
            base_threshold = 0.0001 * orig_norm * (abs(scale_factor - 1.0) + 1.0)
            if len(stability_history) > 1:
                # Use exponential moving average of past changes with decay
                history_tensor = torch.stack(stability_history, dim=0)
                weights = torch.exp(-torch.arange(len(stability_history), device=delta.device))
                weights = weights / weights.sum()
                avg_change = torch.sum(history_tensor * weights.view(-1, 1, 1), dim=0)
                max_change = torch.maximum(avg_change, base_threshold)
            else:
                max_change = base_threshold
            
            # Apply adaptive stability control with smoother transition
            scale = torch.where(
                delta_norm > max_change,
                torch.sqrt(max_change / (delta_norm + 1e-6)),  # Smoother scaling
                torch.ones_like(delta_norm)
            )
            
            # Update state with controlled change
            curr_state = curr_state + delta * scale * step_factor
            curr_state = self._normalize_state(curr_state)
            ref_state = curr_state.clone()
            
            remaining_steps -= curr_step
        
        # Apply final scale factor with stability check only if requested
        if apply_final_scale:
            final_scale = orig_norm * scale_factor
            return curr_state * final_scale
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
            
        # Apply scale transition without final scaling if using quantum bridge
        state = self._apply_scale_transition(
            state,
            scale_idx,
            self.scale_up,
            source_scale,
            target_scale,
            apply_final_scale=self.quantum_bridge is None
        )
        
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
            
        # Apply scale transition without final scaling if using quantum bridge
        state = self._apply_scale_transition(
            state,
            scale_idx,
            self.scale_down,
            source_scale,
            target_scale,
            apply_final_scale=self.quantum_bridge is None
        )
        
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
        
    def _validate_scales(self, source_scale: float, target_scale: float) -> None:
        """Validate scale inputs."""
        if source_scale <= 0 or target_scale <= 0:
            raise ValueError("Scales must be positive")
            
        if source_scale == target_scale:
            raise ValueError("Source and target scales must be different")
            
        # Check if scales are valid powers of 2
        source_log2 = np.log2(source_scale)
        target_log2 = np.log2(target_scale)
        
        if not (np.isclose(source_log2, np.round(source_log2)) and 
                np.isclose(target_log2, np.round(target_log2))):
            raise ValueError("Scales must be powers of 2")
            
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
    
    def transition_layer(
        self,
        state: torch.Tensor,
        source_scale: float,
        target_scale: float
    ) -> torch.Tensor:
        """Apply scale transition."""
        # Validate inputs
        self._validate_scales(source_scale, target_scale)
        
        # Store original norm and handle complex states
        if torch.is_complex(state):
            orig_norm = torch.linalg.vector_norm(state, dim=-1, keepdim=True)
            # Normalize input state while preserving phase
            state_norm = torch.linalg.vector_norm(state, dim=-1, keepdim=True)
            state = state / (state_norm + 1e-8)
        else:
            orig_norm = torch.linalg.vector_norm(state, dim=-1, keepdim=True)
            state = F.normalize(state, p=2, dim=-1)
        
        # Apply appropriate transition
        if target_scale > source_scale:
            state = self.transition_layer.transition_up(
                state,
                source_scale,
                target_scale
            )
        else:
            state = self.transition_layer.transition_down(
                state,
                source_scale,
                target_scale
            )
            
        # Apply final scale factor with improved stability compensation
        scale_factor = target_scale / source_scale
        ref_norm = torch.linalg.vector_norm(state, dim=-1, keepdim=True)
        target_norm = orig_norm * scale_factor
        
        # Compute scale adjustment with adaptive threshold
        delta_norm = (ref_norm - target_norm).abs()
        # Scale threshold with transition magnitude
        base_threshold = 0.0001 * target_norm
        max_delta = base_threshold * (1.0 + 0.5 * abs(np.log2(scale_factor)))
        
        # Smooth interpolation between norms
        interp_factor = torch.clamp(delta_norm / max_delta, 0.0, 1.0)
        scale = target_norm * (1.0 - interp_factor) + ref_norm * interp_factor
        
        # Apply adjusted scale with additional stability constraint
        scale_ratio = scale / ref_norm
        max_ratio = 1.0 + 0.1 * abs(np.log2(scale_factor))
        min_ratio = 1.0 / max_ratio
        scale_ratio = torch.clamp(scale_ratio, min_ratio, max_ratio)
        
        return state * (scale_ratio * ref_norm)


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