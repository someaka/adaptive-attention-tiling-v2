"""Validation Interfaces.

This module defines interfaces for validation and testing:
1. Pattern Validation - Validate pattern properties and transformations
2. Metric Validation - Validate metric computations and geometric properties
3. System Validation - Validate system-wide properties and invariants
"""

from typing import Protocol, TypeVar, Dict, List, Optional, Any
from typing_extensions import runtime_checkable
import torch
from dataclasses import dataclass

from .pattern_space import IFiberBundle, IRiemannianStructure, ICohomologyStructure
from .neural_pattern import IPatternNetwork
from .quantum import IQuantumState
from .crystal import ICrystal

T = TypeVar('T', bound=torch.Tensor)

@dataclass
class ValidationResult:
    """Validation result container."""
    passed: bool
    metrics: Dict[str, float]
    failures: List[str]
    warnings: List[str]
    details: Dict[str, Any]

@runtime_checkable
class IPatternValidation(Protocol[T]):
    """Pattern validation interface."""
    
    def validate_pattern_structure(self, pattern: T) -> ValidationResult:
        """Validate pattern structure.
        
        Args:
            pattern: Pattern to validate
            
        Returns:
            Validation results
        """
        ...
    
    def validate_transformations(self, 
                               pattern: T,
                               transformed: T) -> ValidationResult:
        """Validate pattern transformations.
        
        Args:
            pattern: Original pattern
            transformed: Transformed pattern
            
        Returns:
            Validation results
        """
        ...
    
    def validate_decomposition(self, 
                             pattern: T,
                             components: Dict[str, T]) -> ValidationResult:
        """Validate pattern decomposition.
        
        Args:
            pattern: Original pattern
            components: Decomposed components
            
        Returns:
            Validation results
        """
        ...
    
    def validate_invariants(self, pattern: T) -> ValidationResult:
        """Validate pattern invariants.
        
        Args:
            pattern: Pattern to validate
            
        Returns:
            Validation results
        """
        ...

@runtime_checkable
class IMetricValidation(Protocol[T]):
    """Metric validation interface."""
    
    def validate_metric_properties(self, 
                                 metric: IRiemannianStructure[T]) -> ValidationResult:
        """Validate metric properties.
        
        Args:
            metric: Metric to validate
            
        Returns:
            Validation results
        """
        ...
    
    def validate_curvature(self, 
                          metric: IRiemannianStructure[T],
                          point: T) -> ValidationResult:
        """Validate curvature computation.
        
        Args:
            metric: Metric to validate
            point: Point to validate at
            
        Returns:
            Validation results
        """
        ...
    
    def validate_parallel_transport(self,
                                  bundle: IFiberBundle[T],
                                  section: T,
                                  path: T) -> ValidationResult:
        """Validate parallel transport.
        
        Args:
            bundle: Fiber bundle
            section: Section to transport
            path: Transport path
            
        Returns:
            Validation results
        """
        ...
    
    def validate_cohomology(self,
                           cohomology: ICohomologyStructure[T]) -> ValidationResult:
        """Validate cohomology structure.
        
        Args:
            cohomology: Cohomology to validate
            
        Returns:
            Validation results
        """
        ...

@runtime_checkable
class ISystemValidation(Protocol[T]):
    """System-wide validation interface."""
    
    def validate_network(self, 
                        network: IPatternNetwork[T]) -> ValidationResult:
        """Validate pattern network.
        
        Args:
            network: Network to validate
            
        Returns:
            Validation results
        """
        ...
    
    def validate_quantum_state(self,
                             state: IQuantumState[T]) -> ValidationResult:
        """Validate quantum state.
        
        Args:
            state: State to validate
            
        Returns:
            Validation results
        """
        ...
    
    def validate_crystal_structure(self,
                                 crystal: ICrystal[T]) -> ValidationResult:
        """Validate crystal structure.
        
        Args:
            crystal: Crystal to validate
            
        Returns:
            Validation results
        """
        ...
    
    def validate_attention_mechanism(self,
                                   network: IPatternNetwork[T]) -> ValidationResult:
        """Validate attention mechanism.
        
        Args:
            network: Network with attention to validate
            
        Returns:
            Validation results
        """
        ...
    
    def validate_training_dynamics(self,
                                 network: IPatternNetwork[T],
                                 data: T) -> ValidationResult:
        """Validate training dynamics.
        
        Args:
            network: Network to validate
            data: Training data
            
        Returns:
            Validation results
        """
        ...
    
    def system_health_check(self) -> ValidationResult:
        """Perform system-wide health check.
        
        Returns:
            Validation results
        """
        ... 