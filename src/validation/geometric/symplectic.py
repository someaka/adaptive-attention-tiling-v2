"""Validation for symplectic structures and related geometric components.

This module provides validators for:
1. Symplectic structure preservation
2. Wave packet properties and evolution
3. Operadic structure transitions
4. Quantum geometric tensor properties
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Protocol
import torch
from torch import Tensor

from ...core.patterns.symplectic import SymplecticStructure, SymplecticForm
from ...core.patterns.operadic_structure import AttentionOperad, EnrichedAttention
from ..flow.hamiltonian import HamiltonianFlowValidationResult

@dataclass
class SymplecticValidationResult:
    """Result of symplectic structure validation."""
    is_valid: bool
    message: str
    data: Dict[str, Any]

    def merge(self, other: 'SymplecticValidationResult') -> 'SymplecticValidationResult':
        """Merge with another validation result."""
        return SymplecticValidationResult(
            is_valid=self.is_valid and other.is_valid,
            message=f"{self.message}; {other.message}",
            data={**self.data, **other.data}
        )

class WavePacketValidator:
    """Validator for wave packet properties and evolution."""

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance

    def validate_wave_packet(
        self,
        structure: SymplecticStructure,
        packet: Tensor,
        position: Optional[Tensor] = None,
        momentum: Optional[Tensor] = None
    ) -> SymplecticValidationResult:
        """Validate wave packet properties.
        
        Args:
            structure: Symplectic structure
            packet: Wave packet tensor
            position: Expected position (optional)
            momentum: Expected momentum (optional)
            
        Returns:
            Validation result
        """
        try:
            # Check wave packet normalization
            norm = torch.norm(packet)
            is_normalized = bool(torch.abs(norm - 1.0) < self.tolerance)

            # Get position and momentum expectation values
            if hasattr(structure.enriched, 'get_position'):
                computed_position = structure.enriched.get_position(packet)
                computed_momentum = structure.enriched.get_momentum(packet)
                
                # Validate position and momentum if provided
                pos_valid = True
                mom_valid = True
                if position is not None:
                    pos_valid = bool(torch.allclose(computed_position, position, rtol=self.tolerance))
                if momentum is not None:
                    mom_valid = bool(torch.allclose(computed_momentum, momentum, rtol=self.tolerance))
            else:
                computed_position = None
                computed_momentum = None
                pos_valid = position is None
                mom_valid = momentum is None

            # Overall validation
            is_valid = bool(is_normalized and pos_valid and mom_valid)

            return SymplecticValidationResult(
                is_valid=is_valid,
                message=f"Wave packet validation {'passed' if is_valid else 'failed'}",
                data={
                    'is_normalized': is_normalized,
                    'norm': norm.item(),
                    'position_valid': pos_valid,
                    'momentum_valid': mom_valid,
                    'computed_position': computed_position,
                    'computed_momentum': computed_momentum
                }
            )

        except Exception as e:
            return SymplecticValidationResult(
                is_valid=False,
                message=f"Error validating wave packet: {str(e)}",
                data={'error': str(e)}
            )

class OperadicValidator:
    """Validator for operadic structure transitions."""

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance

    def validate_operadic_transition(
        self,
        structure: SymplecticStructure,
        source: Tensor,
        target: Tensor,
        operation: Optional[Any] = None
    ) -> SymplecticValidationResult:
        """Validate operadic structure preservation during transitions.
        
        Args:
            structure: Symplectic structure
            source: Source tensor
            target: Target tensor
            operation: Optional operadic operation
            
        Returns:
            Validation result
        """
        try:
            # Get or create operadic operation
            if operation is None:
                operation = structure.operadic.create_operation(
                    source_dim=source.shape[-1],
                    target_dim=target.shape[-1],
                    preserve_structure='symplectic'
                )

            # Apply operation
            result = structure.enriched.create_morphism(
                pattern=source,
                operation=operation,
                include_wave=structure.wave_enabled
            )

            # Check dimension matching
            dims_match = result.shape[-1] == target.shape[-1]

            # Check structure preservation
            source_form = structure.compute_form(source)
            target_form = structure.compute_form(target)
            result_form = structure.compute_form(result)

            # Compare symplectic forms
            form_preserved = torch.allclose(
                target_form.evaluate(target, target),
                result_form.evaluate(result, result),
                rtol=self.tolerance
            )

            is_valid = dims_match and form_preserved

            return SymplecticValidationResult(
                is_valid=is_valid,
                message=f"Operadic transition validation {'passed' if is_valid else 'failed'}",
                data={
                    'dimensions_match': dims_match,
                    'form_preserved': form_preserved,
                    'source_dim': source.shape[-1],
                    'target_dim': target.shape[-1],
                    'result_dim': result.shape[-1]
                }
            )

        except Exception as e:
            return SymplecticValidationResult(
                is_valid=False,
                message=f"Error validating operadic transition: {str(e)}",
                data={'error': str(e)}
            )

class QuantumGeometricValidator:
    """Validator for quantum geometric tensor properties."""

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance

    def validate_quantum_geometric(
        self,
        structure: SymplecticStructure,
        point: Tensor
    ) -> SymplecticValidationResult:
        """Validate quantum geometric tensor properties.
        
        Args:
            structure: Symplectic structure
            point: Point to validate at
            
        Returns:
            Validation result
        """
        try:
            # Compute quantum geometric tensor
            Q = structure.compute_quantum_geometric_tensor(point)
            g = Q.real  # Metric part
            omega = Q.imag  # Symplectic part

            # Check metric properties
            is_symmetric = torch.allclose(g, g.transpose(-2, -1), rtol=self.tolerance)
            eigenvals = torch.linalg.eigvalsh(g)
            is_positive = bool(eigenvals.min() > 0)

            # Check symplectic form properties
            is_antisymmetric = torch.allclose(omega, -omega.transpose(-2, -1), rtol=self.tolerance)
            
            # Check compatibility
            compatibility = torch.allclose(
                torch.matmul(g, omega),
                torch.matmul(omega, g),
                rtol=self.tolerance
            )

            is_valid = is_symmetric and is_positive and is_antisymmetric and compatibility

            return SymplecticValidationResult(
                is_valid=is_valid,
                message=f"Quantum geometric tensor validation {'passed' if is_valid else 'failed'}",
                data={
                    'metric_symmetric': is_symmetric,
                    'metric_positive': is_positive,
                    'form_antisymmetric': is_antisymmetric,
                    'compatible': compatibility,
                    'eigenvalues': eigenvals,
                    'tensor_shape': Q.shape
                }
            )

        except Exception as e:
            return SymplecticValidationResult(
                is_valid=False,
                message=f"Error validating quantum geometric tensor: {str(e)}",
                data={'error': str(e)}
            )

class SymplecticStructureValidator:
    """Complete validation for symplectic structures with enriched features."""

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        self.wave_validator = WavePacketValidator(tolerance)
        self.operadic_validator = OperadicValidator(tolerance)
        self.quantum_validator = QuantumGeometricValidator(tolerance)

    def validate_all(
        self,
        structure: SymplecticStructure,
        point: Tensor,
        wave_packet: Optional[Tensor] = None,
        target_point: Optional[Tensor] = None
    ) -> SymplecticValidationResult:
        """Perform complete validation of symplectic structure.
        
        Args:
            structure: Symplectic structure to validate
            point: Point to validate at
            wave_packet: Optional wave packet to validate
            target_point: Optional target point for operadic transition
            
        Returns:
            Combined validation result
        """
        results = []

        # Validate quantum geometric tensor
        results.append(
            self.quantum_validator.validate_quantum_geometric(structure, point)
        )

        # Validate wave packet if provided
        if wave_packet is not None:
            results.append(
                self.wave_validator.validate_wave_packet(structure, wave_packet)
            )

        # Validate operadic transition if target provided
        if target_point is not None:
            results.append(
                self.operadic_validator.validate_operadic_transition(
                    structure, point, target_point
                )
            )

        # Combine results
        final_result = results[0]
        for result in results[1:]:
            final_result = final_result.merge(result)

        return final_result 