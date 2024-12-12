"""Base validation classes and utilities."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass
class ValidationResult:
    """Result of validation operation."""

    is_valid: bool
    message: str
    data: Dict[str, Any]

    def __init__(self, is_valid: bool, message: str, data: Optional[Dict[str, Any]] = None):
        """Initialize validation result.
        
        Args:
            is_valid: Whether validation passed
            message: Description of validation result
            data: Optional dictionary of validation metrics/data
        """
        self.is_valid = is_valid
        self.message = message
        self.data = data if data is not None else {}
