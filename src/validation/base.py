"""Base validation classes and utilities."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass
class ValidationResult:
    """Result of validation operation."""

    is_valid: bool
    message: str
    stable: bool
    error: float
    initial_energy: Optional[float] = None
    final_energy: Optional[float] = None
    relative_error: Optional[float] = None
    initial_states: Optional[torch.Tensor] = None
    final_states: Optional[torch.Tensor] = None
    metrics: Optional[Dict[str, Any]] = None
