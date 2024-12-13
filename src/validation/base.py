"""Base validation classes and utilities."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, TypeVar, Generic

import torch

T = TypeVar('T')  # For specific validation data types

@dataclass
class ValidationResult(Generic[T], ABC):
    """Abstract base class for validation results."""

    is_valid: bool
    message: str
    data: Optional[T | Dict[str, Any]] = None

    def __init__(self, is_valid: bool, message: str, data: Optional[T | Dict[str, Any]] = None):
        """Initialize validation result.
        
        Args:
            is_valid: Whether validation passed
            message: Description of validation result
            data: Optional validation data or metrics
        """
        self.is_valid = is_valid
        self.message = message
        self.data = data if data is not None else {}

    @abstractmethod
    def merge(self, other: 'ValidationResult') -> 'ValidationResult':
        """Merge two validation results.
        
        Args:
            other: Another validation result to merge with
            
        Returns:
            New validation result combining both results
        """
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary representation.
        
        Returns:
            Dictionary containing validation data
        """
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationResult':
        """Create validation result from dictionary.
        
        Args:
            data: Dictionary containing validation data
            
        Returns:
            New validation result instance
        """
        pass

    def __str__(self) -> str:
        """String representation of validation result."""
        return f"ValidationResult(valid={self.is_valid}, message='{self.message}', data={self.data})"


@dataclass
class BasicValidationResult(ValidationResult[Dict[str, Any]]):
    """Basic implementation of validation result for simple cases."""

    def merge(self, other: ValidationResult) -> 'BasicValidationResult':
        """Merge with another validation result."""
        return BasicValidationResult(
            is_valid=self.is_valid and other.is_valid,
            message=f"{self.message}; {other.message}",
            data={**(self.data or {}), **(other.data or {})}
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "message": self.message,
            "data": self.data
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BasicValidationResult':
        """Create from dictionary."""
        return cls(
            is_valid=data["is_valid"],
            message=data["message"],
            data=data.get("data", {})
        )