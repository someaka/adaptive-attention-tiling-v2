"""Validation for operadic structures and their geometric properties.

This module provides validators for:
1. Operadic structure type validation
2. Dimension compatibility checks
3. Structure preservation validation
4. Operation creation and composition validation
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Set
import torch
from torch import Tensor

from ...core.patterns.operadic_handler import OperadicStructureHandler
from ..base import ValidationResult


@dataclass
class OperadicValidationResult(ValidationResult[Dict[str, Any]]):
    """Result of operadic structure validation.
    
    This class handles validation results specific to operadic structures,
    with special handling for structure types and geometric properties.
    """
    
    def merge(self, other: ValidationResult) -> 'OperadicValidationResult':
        """Merge with another validation result."""
        if not isinstance(other, ValidationResult):
            raise TypeError(f"Cannot merge with {type(other)}")
            
        return OperadicValidationResult(
            is_valid=self.is_valid and other.is_valid,
            message=f"{self.message}; {other.message}",
            data={**(self.data or {}), **(other.data or {})}
        )
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'is_valid': self.is_valid,
            'message': self.message,
            'data': self.data
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OperadicValidationResult':
        """Create from dictionary representation.
        
        Args:
            data: Dictionary containing validation data
            
        Returns:
            New OperadicValidationResult instance
            
        Raises:
            ValueError: If required fields are missing
        """
        required = {'is_valid', 'message'}
        if not all(k in data for k in required):
            raise ValueError(f"Missing required fields: {required - set(data.keys())}")
            
        return cls(
            is_valid=data['is_valid'],
            message=data['message'],
            data=data.get('data', {})
        )


class OperadicStructureValidator:
    """Validator for operadic structures and their geometric properties."""

    VALID_STRUCTURES: Set[str] = {'quantum', 'classical', 'geometric', 'arithmetic'}

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance

    def validate_structure_type(
        self,
        structure_type: str,
        handler: OperadicStructureHandler
    ) -> OperadicValidationResult:
        """Validate operadic structure type and basic properties.
        
        Args:
            structure_type: Type of structure to validate
            handler: Operadic structure handler
            
        Returns:
            Validation result
        """
        # Validate structure type
        if structure_type not in self.VALID_STRUCTURES:
            return OperadicValidationResult(
                is_valid=False,
                message=f"Invalid structure type: {structure_type}. Must be one of {self.VALID_STRUCTURES}",
                data={
                    'structure_type': structure_type,
                    'valid_types': list(self.VALID_STRUCTURES)
                }
            )
            
        # Validate structure preservation based on type
        if structure_type == 'quantum' and not handler.preserve_symplectic:
            return OperadicValidationResult(
                is_valid=False,
                message="Quantum structure requires preserved symplectic form",
                data={
                    'structure_type': structure_type,
                    'preserve_symplectic': False
                }
            )
            
        if structure_type in {'geometric', 'arithmetic'} and not handler.preserve_metric:
            return OperadicValidationResult(
                is_valid=False,
                message=f"{structure_type} structure requires preserved metric",
                data={
                    'structure_type': structure_type,
                    'preserve_metric': False
                }
            )
            
        return OperadicValidationResult(
            is_valid=True,
            message=f"Structure type '{structure_type}' is valid",
            data={
                'structure_type': structure_type,
                'preserves_symplectic': handler.preserve_symplectic,
                'preserves_metric': handler.preserve_metric
            }
        )

    def validate_dimension_compatibility(
        self,
        handler: OperadicStructureHandler,
        expected_dim: int
    ) -> OperadicValidationResult:
        """Validate dimension compatibility.
        
        Args:
            handler: Operadic structure handler
            expected_dim: Expected dimension
            
        Returns:
            Validation result
        """
        if handler.base_dim != expected_dim:
            return OperadicValidationResult(
                is_valid=False,
                message=f"Structure dimension {handler.base_dim} does not match expected dimension {expected_dim}",
                data={
                    'structure_dim': handler.base_dim,
                    'expected_dim': expected_dim
                }
            )
            
        return OperadicValidationResult(
            is_valid=True,
            message="Dimension compatibility validated",
            data={
                'dimension': expected_dim
            }
        )

    def validate_operation(
        self,
        handler: OperadicStructureHandler,
        structure_type: str,
        dim: int,
        dtype: torch.dtype
    ) -> OperadicValidationResult:
        """Validate operation creation and application.
        
        Args:
            handler: Operadic structure handler
            structure_type: Type of structure to preserve
            dim: Dimension for test tensors
            dtype: Data type for test tensors
            
        Returns:
            Validation result
        """
        try:
            # Create test tensor with batch dimension
            test_tensor = torch.zeros((1, dim), dtype=dtype)
            
            # Test operation application via dimension transition
            result, metrics = handler.handle_dimension_transition(
                tensor=test_tensor,
                source_dim=dim,
                target_dim=dim,
                preserve_structure=structure_type
            )
            
            # Validate result dimension
            if result.shape[-1] != dim:
                return OperadicValidationResult(
                    is_valid=False,
                    message=f"Operation result has incorrect dimension: {result.shape[-1]}, expected {dim}",
                    data={
                        'result_dim': result.shape[-1],
                        'expected_dim': dim
                    }
                )
                
            return OperadicValidationResult(
                is_valid=True,
                message="Operation creation and application validated",
                data={
                    'result_shape': result.shape,
                    'metrics': metrics
                }
            )
            
        except Exception as e:
            return OperadicValidationResult(
                is_valid=False,
                message=f"Operation validation failed: {str(e)}",
                data={'error': str(e)}
            )

    def validate_all(
        self,
        handler: OperadicStructureHandler,
        structure_type: str,
        expected_dim: int,
        dtype: torch.dtype
    ) -> OperadicValidationResult:
        """Perform complete validation of operadic structure.
        
        Args:
            handler: Operadic structure handler to validate
            structure_type: Type of structure to validate
            expected_dim: Expected dimension
            dtype: Data type for test tensors
            
        Returns:
            Combined validation result
        """
        results = []
        
        # Validate structure type
        results.append(
            self.validate_structure_type(structure_type, handler)
        )
        
        # Validate dimension compatibility
        results.append(
            self.validate_dimension_compatibility(handler, expected_dim)
        )
        
        # Validate operations
        results.append(
            self.validate_operation(handler, structure_type, expected_dim, dtype)
        )
        
        # Combine results
        final_result = results[0]
        for result in results[1:]:
            final_result = final_result.merge(result)
            
        return final_result 