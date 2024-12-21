"""Central Initialization System.

This module provides a unified initialization system that coordinates:
1. State Management
2. Pattern Processing
3. Quantum Attention
4. Scale Transitions
5. Neural-Quantum Bridging

The system ensures proper setup and interaction between all components.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from .tiling.state_manager import StateManager, StateConfig, StateType
from .patterns.pattern_processor import PatternProcessor
from .tiling.quantum_attention_tile import QuantumMotivicTile
from .scale_transition import ScaleTransitionLayer, ScaleTransitionConfig
from .quantum.neural_quantum_bridge import NeuralQuantumBridge


@dataclass
class InitializationConfig:
    """Configuration for the initialization system."""
    
    # Core dimensions
    hidden_dim: int = 64
    num_heads: int = 8
    
    # State configuration
    state_type: StateType = StateType.PURE
    max_entanglement: float = 1.0
    epsilon: float = 1e-6
    
    # Scale configuration
    min_scale: float = 0.25
    max_scale: float = 4.0
    num_scales: int = 4
    
    # Pattern configuration
    motive_rank: int = 4
    num_primes: int = 8
    
    # Device configuration
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None


class InitializationSystem(nn.Module):
    """Central initialization system coordinating all components."""

    def __init__(self, config: InitializationConfig):
        super().__init__()
        self.config = config
        
        # Set device and dtype
        self.device = config.device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = config.dtype or torch.float32
        
        # Initialize components
        self._initialize_state_manager()
        self._initialize_pattern_processor()
        self._initialize_quantum_tile()
        self._initialize_scale_transition()
        self._initialize_quantum_bridge()
        
        # Connect components
        self._connect_components()

    def _initialize_state_manager(self) -> None:
        """Initialize state management system."""
        state_config = StateConfig(
            dim=self.config.hidden_dim,
            type=self.config.state_type,
            epsilon=self.config.epsilon,
            max_entanglement=self.config.max_entanglement
        )
        self.state_manager = StateManager(
            config=state_config,
            device=self.device
        )

    def _initialize_pattern_processor(self) -> None:
        """Initialize pattern processing system."""
        self.pattern_processor = PatternProcessor(
            manifold_dim=self.config.hidden_dim,
            hidden_dim=self.config.hidden_dim,
            motive_rank=self.config.motive_rank,
            num_primes=self.config.num_primes,
            num_heads=self.config.num_heads,
            device=self.device,
            dtype=self.dtype
        )

    def _initialize_quantum_tile(self) -> None:
        """Initialize quantum attention tile."""
        self.quantum_tile = QuantumMotivicTile(
            size=self.config.hidden_dim,
            hidden_dim=self.config.hidden_dim,
            num_heads=self.config.num_heads,
            dropout=0.1,
            resolution=1.0,
            cohomology_dim=self.config.hidden_dim,
            motive_rank=self.config.motive_rank
        )

    def _initialize_scale_transition(self) -> None:
        """Initialize scale transition system."""
        scale_config = ScaleTransitionConfig(
            min_scale=self.config.min_scale,
            max_scale=self.config.max_scale,
            num_scales=self.config.num_scales,
            dim=self.config.hidden_dim,
            use_quantum_bridge=True
        )
        self.scale_transition = ScaleTransitionLayer(scale_config)

    def _initialize_quantum_bridge(self) -> None:
        """Initialize neural-quantum bridge."""
        self.quantum_bridge = NeuralQuantumBridge(
            hidden_dim=self.config.hidden_dim,
            num_heads=self.config.num_heads,
            dropout=0.1
        )

    def _connect_components(self) -> None:
        """Connect initialized components."""
        # Connect state manager to quantum bridge
        self.quantum_bridge.state_manager = self.state_manager
        
        # Connect pattern processor to quantum tile
        self.pattern_processor.quantum_tile = self.quantum_tile
        
        # Connect scale transition to quantum bridge
        self.scale_transition.quantum_bridge = self.quantum_bridge

    def get_component_states(self) -> Dict[str, Any]:
        """Get current states of all components."""
        return {
            'state_manager': self.state_manager.states,
            'pattern_processor': {
                'quantum_state': self.pattern_processor.quantum_bridge.last_state,
                'geometric_state': self.pattern_processor.riemannian.last_metric
            },
            'quantum_tile': {
                'resolution': self.quantum_tile.resolution,
                'metrics': self.quantum_tile.get_metrics()
            },
            'scale_transition': self.scale_transition._entanglement_tracking
        }

    def validate_initialization(self) -> Dict[str, bool]:
        """Validate initialization of all components."""
        validations = {}
        
        # Validate state manager
        validations['state_manager'] = all(
            self.state_manager.validate_state(state)
            for state in self.state_manager.states.values()
        )
        
        # Validate pattern processor
        validations['pattern_processor'] = (
            self.pattern_processor.quantum_bridge is not None and
            self.pattern_processor.quantum_tile is not None
        )
        
        # Validate quantum tile
        validations['quantum_tile'] = (
            self.quantum_tile.size == self.config.hidden_dim and
            self.quantum_tile.hidden_dim == self.config.hidden_dim
        )
        
        # Validate scale transition
        validations['scale_transition'] = (
            self.scale_transition.quantum_bridge is not None
        )
        
        return validations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the initialized components.
        
        This method demonstrates the flow of data through the connected components:
        1. Pattern processing
        2. Quantum state preparation
        3. Scale transition
        4. Attention computation
        
        Args:
            x: Input tensor
            
        Returns:
            Processed tensor
        """
        # Process pattern
        pattern_out = self.pattern_processor(x)
        
        # Prepare quantum state
        quantum_state = self.quantum_bridge.neural_to_quantum(pattern_out)
        
        # Handle scale transition
        scaled_state = self.scale_transition(
            pattern_out,
            source_scale=1.0,
            target_scale=self.config.max_scale
        )
        
        # Apply quantum attention
        attention_out = self.quantum_tile(scaled_state)
        
        return attention_out 