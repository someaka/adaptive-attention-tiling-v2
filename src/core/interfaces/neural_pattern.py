"""Neural Pattern Interfaces.

This module defines interfaces for neural pattern processing:
1. Pattern Recognition - Core pattern recognition and analysis
2. Attention Mechanisms - Specialized attention for pattern processing
3. Neural Dynamics - Neural network dynamics and training
"""

from typing import Protocol, TypeVar, List, Dict, Optional, Tuple
from typing_extensions import runtime_checkable
import torch
from torch import nn

from .pattern_space import IFiberBundle, IRiemannianStructure
from .quantum import IQuantumState
from .crystal import ICrystal

T = TypeVar('T', bound=torch.Tensor)

@runtime_checkable
class IPatternRecognition(Protocol[T]):
    """Core pattern recognition interface."""
    
    def extract_features(self, input_pattern: T) -> T:
        """Extract features from input pattern.
        
        Args:
            input_pattern: Raw input pattern
            
        Returns:
            Extracted features
        """
        ...
    
    def compute_pattern_similarity(self, pattern1: T, pattern2: T) -> float:
        """Compute similarity between patterns.
        
        Args:
            pattern1: First pattern
            pattern2: Second pattern
            
        Returns:
            Similarity score
        """
        ...
    
    def classify_pattern(self, pattern: T) -> Dict[str, float]:
        """Classify pattern into categories.
        
        Args:
            pattern: Pattern to classify
            
        Returns:
            Classification probabilities
        """
        ...
    
    def detect_subpatterns(self, pattern: T) -> List[T]:
        """Detect subpatterns within pattern.
        
        Args:
            pattern: Input pattern
            
        Returns:
            List of detected subpatterns
        """
        ...
    
    def pattern_decomposition(self, pattern: T) -> Dict[str, T]:
        """Decompose pattern into components.
        
        Args:
            pattern: Pattern to decompose
            
        Returns:
            Component dictionary
        """
        ...

@runtime_checkable
class IAttentionMechanism(Protocol[T]):
    """Specialized attention for pattern processing."""
    
    def compute_attention_weights(self, query: T, key: T, value: T) -> Tuple[T, T]:
        """Compute attention weights and output.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            
        Returns:
            (attention_weights, attention_output)
        """
        ...
    
    def multi_head_attention(self, 
                           query: T, 
                           key: T, 
                           value: T,
                           num_heads: int) -> T:
        """Compute multi-head attention.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            num_heads: Number of attention heads
            
        Returns:
            Multi-head attention output
        """
        ...
    
    def self_attention(self, input_tensor: T) -> T:
        """Compute self-attention.
        
        Args:
            input_tensor: Input tensor
            
        Returns:
            Self-attention output
        """
        ...
    
    def cross_attention(self, source: T, target: T) -> T:
        """Compute cross-attention between source and target.
        
        Args:
            source: Source tensor
            target: Target tensor
            
        Returns:
            Cross-attention output
        """
        ...
    
    def attention_pattern_analysis(self, attention_weights: T) -> Dict[str, float]:
        """Analyze attention pattern.
        
        Args:
            attention_weights: Attention weight tensor
            
        Returns:
            Analysis metrics
        """
        ...

@runtime_checkable
class INeuralDynamics(Protocol[T]):
    """Neural network dynamics and training."""
    
    def forward_pass(self, input_tensor: T) -> T:
        """Perform forward pass through network.
        
        Args:
            input_tensor: Input tensor
            
        Returns:
            Network output
        """
        ...
    
    def backward_pass(self, loss: T) -> Dict[str, T]:
        """Perform backward pass through network.
        
        Args:
            loss: Loss tensor
            
        Returns:
            Gradients dictionary
        """
        ...
    
    def update_parameters(self, gradients: Dict[str, T], 
                         learning_rate: float) -> None:
        """Update network parameters.
        
        Args:
            gradients: Parameter gradients
            learning_rate: Learning rate
        """
        ...
    
    def compute_loss(self, predictions: T, targets: T) -> T:
        """Compute loss between predictions and targets.
        
        Args:
            predictions: Model predictions
            targets: Target values
            
        Returns:
            Loss value
        """
        ...
    
    def regularization_term(self) -> T:
        """Compute regularization term.
        
        Returns:
            Regularization loss
        """
        ...
    
    def activation_patterns(self) -> Dict[str, T]:
        """Get activation patterns across layers.
        
        Returns:
            Layer activation patterns
        """
        ...
    
    def weight_analysis(self) -> Dict[str, Dict[str, float]]:
        """Analyze network weights.
        
        Returns:
            Weight analysis metrics
        """
        ...

@runtime_checkable
class IPatternNetwork(Protocol[T]):
    """Combined pattern network interface."""
    
    recognition: IPatternRecognition[T]
    attention: IAttentionMechanism[T]
    dynamics: INeuralDynamics[T]
    geometric: IRiemannianStructure[T]
    quantum: IQuantumState[T]
    crystal: ICrystal[T]
    
    def process_pattern(self, pattern: T) -> Dict[str, T]:
        """Process pattern through full pipeline.
        
        Args:
            pattern: Input pattern
            
        Returns:
            Processing results
        """
        ...
    
    def adapt_attention(self, pattern: T, context: Optional[T] = None) -> T:
        """Adapt attention based on pattern and context.
        
        Args:
            pattern: Input pattern
            context: Optional context tensor
            
        Returns:
            Adapted attention output
        """
        ...
    
    def geometric_analysis(self, pattern: T) -> Dict[str, float]:
        """Analyze geometric properties of pattern.
        
        Args:
            pattern: Input pattern
            
        Returns:
            Geometric analysis metrics
        """
        ...
    
    def quantum_analysis(self, pattern: T) -> Dict[str, float]:
        """Analyze quantum properties of pattern.
        
        Args:
            pattern: Input pattern
            
        Returns:
            Quantum analysis metrics
        """
        ...
    
    def crystal_analysis(self, pattern: T) -> Dict[str, float]:
        """Analyze crystalline properties of pattern.
        
        Args:
            pattern: Input pattern
            
        Returns:
            Crystal analysis metrics
        """
        ... 