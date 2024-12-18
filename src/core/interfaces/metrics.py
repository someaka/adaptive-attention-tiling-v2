"""Metrics Interfaces.

This module defines interfaces for system metrics:
1. Pattern Metrics - Pattern quality and characteristics
2. Performance Metrics - System performance measurements
3. Analysis Metrics - Deep analysis of system behavior
"""

from typing import Protocol, TypeVar, Dict, List, Optional, Any, Tuple
from typing_extensions import runtime_checkable
import torch
from dataclasses import dataclass

from .pattern_space import IFiberBundle, IRiemannianStructure
from .neural_pattern import IPatternNetwork
from .quantum import IQuantumState
from .crystal import ICrystal

# Invariant type variable for protocols that use T as parameter
T = TypeVar('T', bound=torch.Tensor)
# Covariant type variable for protocols that only return T
T_co = TypeVar('T_co', bound=torch.Tensor, covariant=True)

@dataclass
class MetricResult:
    """Metric result container."""
    value: float
    confidence: Optional[float] = None
    components: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None

@runtime_checkable
class IPatternMetrics(Protocol[T]):
    """Pattern quality metrics interface."""
    
    def pattern_complexity(self, pattern: T) -> MetricResult:
        """Compute pattern complexity.
        
        Args:
            pattern: Input pattern
            
        Returns:
            Complexity metric
        """
        ...
    
    def pattern_stability(self, 
                         pattern: T,
                         perturbations: List[T]) -> MetricResult:
        """Compute pattern stability.
        
        Args:
            pattern: Input pattern
            perturbations: Pattern perturbations
            
        Returns:
            Stability metric
        """
        ...
    
    def pattern_coherence(self, pattern: T) -> MetricResult:
        """Compute pattern coherence.
        
        Args:
            pattern: Input pattern
            
        Returns:
            Coherence metric
        """
        ...
    
    def geometric_metrics(self, 
                         pattern: T,
                         metric: IRiemannianStructure[T]) -> Dict[str, MetricResult]:
        """Compute geometric metrics.
        
        Args:
            pattern: Input pattern
            metric: Geometric metric
            
        Returns:
            Geometric metrics
        """
        ...
    
    def quantum_metrics(self,
                       pattern: T,
                       state: IQuantumState[T]) -> Dict[str, MetricResult]:
        """Compute quantum metrics.
        
        Args:
            pattern: Input pattern
            state: Quantum state
            
        Returns:
            Quantum metrics
        """
        ...
    
    def crystal_metrics(self,
                       pattern: T,
                       crystal: ICrystal[T]) -> Dict[str, MetricResult]:
        """Compute crystal metrics.
        
        Args:
            pattern: Input pattern
            crystal: Crystal structure
            
        Returns:
            Crystal metrics
        """
        ...

@runtime_checkable
class IPerformanceMetrics(Protocol[T_co]):
    """System performance metrics interface."""
    
    def compute_time(self, 
                    operation: str,
                    inputs: Dict[str, Any]) -> MetricResult:
        """Measure computation time.
        
        Args:
            operation: Operation name
            inputs: Operation inputs
            
        Returns:
            Time metric
        """
        ...
    
    def memory_usage(self, 
                    operation: str,
                    inputs: Dict[str, Any]) -> MetricResult:
        """Measure memory usage.
        
        Args:
            operation: Operation name
            inputs: Operation inputs
            
        Returns:
            Memory metric
        """
        ...
    
    def throughput(self,
                  operation: str,
                  batch_size: int,
                  duration: float) -> MetricResult:
        """Measure operation throughput.
        
        Args:
            operation: Operation name
            batch_size: Processing batch size
            duration: Measurement duration
            
        Returns:
            Throughput metric
        """
        ...
    
    def resource_efficiency(self,
                          operation: str,
                          resources: Dict[str, float]) -> MetricResult:
        """Measure resource efficiency.
        
        Args:
            operation: Operation name
            resources: Resource usage
            
        Returns:
            Efficiency metric
        """
        ...

@runtime_checkable
class IAnalysisMetrics(Protocol[T]):
    """Deep analysis metrics interface."""
    
    def attention_analysis(self,
                         network: IPatternNetwork[T],
                         input_data: T) -> Dict[str, MetricResult]:
        """Analyze attention patterns.
        
        Args:
            network: Pattern network
            input_data: Input data
            
        Returns:
            Attention metrics
        """
        ...
    
    def training_dynamics(self,
                        network: IPatternNetwork[T],
                        history: Dict[str, List[float]]) -> Dict[str, MetricResult]:
        """Analyze training dynamics.
        
        Args:
            network: Pattern network
            history: Training history
            
        Returns:
            Training metrics
        """
        ...
    
    def pattern_evolution(self,
                        patterns: List[T],
                        timestamps: List[float]) -> Dict[str, MetricResult]:
        """Analyze pattern evolution.
        
        Args:
            patterns: Pattern sequence
            timestamps: Evolution timestamps
            
        Returns:
            Evolution metrics
        """
        ...
    
    def system_stability(self,
                       network: IPatternNetwork[T],
                       test_cases: List[Tuple[T, T]]) -> Dict[str, MetricResult]:
        """Analyze system stability.
        
        Args:
            network: Pattern network
            test_cases: (input, expected) pairs
            
        Returns:
            Stability metrics
        """
        ...
    
    def error_analysis(self,
                     predictions: T,
                     targets: T) -> Dict[str, MetricResult]:
        """Analyze prediction errors.
        
        Args:
            predictions: Model predictions
            targets: Target values
            
        Returns:
            Error metrics
        """
        ...
    
    def generate_analysis_report(self,
                              metrics: Dict[str, Dict[str, MetricResult]]) -> Dict[str, Any]:
        """Generate comprehensive analysis report.
        
        Args:
            metrics: Collected metrics
            
        Returns:
            Analysis report
        """
        ... 