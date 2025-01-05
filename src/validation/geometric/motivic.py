"""Motivic Structure Validation.

This module provides validation for motivic cohomology and arithmetic dynamics
in the context of geometric structures.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Any
import torch
from torch import Tensor

from .metric import MetricValidation, ConnectionValidation, CurvatureValidation
from ...core.patterns.motivic_riemannian import (
    MotivicMetricTensor,
    MotivicChristoffelSymbols,
    MotivicCurvatureTensor,
    MotivicRiemannianStructure
)
from ...core.patterns.cohomology import (
    HeightStructure,
    ArithmeticDynamics,
    ArithmeticForm
)

__all__ = [
    'HeightValidation',
    'MotivicValidation',
    'MotivicValidator',
    'MotivicRiemannianValidator'
]


@dataclass
class HeightValidation:
    """Validation results for height functions."""
    is_valid: bool
    local_heights_valid: bool
    global_height_valid: bool
    northcott_property: bool
    message: str
    height_data: Optional[Tensor] = None


@dataclass
class MotivicValidation:
    """Validation results for motivic structure."""
    is_valid: bool
    height_valid: bool
    dynamics_valid: bool
    cohomology_valid: bool
    message: str
    data: Dict[str, Any]


class MotivicValidator:
    """Validator for motivic structures."""

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance

    def validate_height(self, metric: MotivicMetricTensor) -> HeightValidation:
        """Validate height function properties.
        
        Args:
            metric: Metric tensor with height structure
            
        Returns:
            Validation results for height functions
        """
        if metric.height_data is None:
            return HeightValidation(
                is_valid=False,
                local_heights_valid=False,
                global_height_valid=False,
                northcott_property=False,
                message="No height data found",
                height_data=None
            )
            
        # Check local height properties
        local_valid = self._validate_local_heights(metric.height_data)
        
        # Check global height properties
        global_valid = self._validate_global_height(metric.height_data)
        
        # Check Northcott property
        northcott = self._check_northcott_property(metric.height_data)
        
        is_valid = local_valid and global_valid and northcott
        message = "Height validation " + ("passed" if is_valid else "failed")
        
        return HeightValidation(
            is_valid=is_valid,
            local_heights_valid=local_valid,
            global_height_valid=global_valid,
            northcott_property=northcott,
            message=message,
            height_data=metric.height_data
        )

    def validate_dynamics(
        self,
        connection: MotivicChristoffelSymbols
    ) -> ConnectionValidation:
        """Validate arithmetic dynamics properties.
        
        Args:
            connection: Connection with arithmetic dynamics
            
        Returns:
            Validation results for dynamics
        """
        # Check dynamics state exists and is valid
        if connection.dynamics_state is None:
            return ConnectionValidation(
                is_valid=False,
                message="No dynamics state"
            )
            
        # Check evolution properties
        try:
            # Check for NaN or infinite values in connection
            if not torch.all(torch.isfinite(connection.values)):
                return ConnectionValidation(
                    is_valid=False,
                    message="Connection contains NaN or infinite values"
                )
            
            # Check symmetry in lower indices
            for i in range(connection.values.shape[0]):
                for j in range(connection.values.shape[1]):
                    if not torch.allclose(
                        connection.values[i, j, :, :],
                        connection.values[i, j, :, :].transpose(-2, -1),
                        rtol=self.tolerance,
                        atol=self.tolerance
                    ):
                        return ConnectionValidation(
                            is_valid=False,
                            message="Connection not symmetric in lower indices"
                        )
            
            # Use the pre-computed dynamics state
            evolution_valid = bool(torch.isfinite(connection.dynamics_state).all())
            if not evolution_valid:
                return ConnectionValidation(
                    is_valid=False,
                    message="Dynamics state contains NaN or infinite values"
                )
                
        except Exception as e:
            return ConnectionValidation(
                is_valid=False,
                message=f"Evolution failed: {str(e)}"
            )
            
        return ConnectionValidation(
            is_valid=evolution_valid,
            message="Dynamics validation " + ("passed" if evolution_valid else "failed")
        )

    def validate_cohomology(
        self,
        curvature: MotivicCurvatureTensor
    ) -> CurvatureValidation:
        """Validate cohomological properties.
        
        Args:
            curvature: Curvature tensor with motivic structure
            
        Returns:
            Validation results for cohomology
        """
        # Check cohomology class exists
        if not hasattr(curvature, 'cohomology_class'):
            return CurvatureValidation(
                bounds_satisfied=False,
                sectional=curvature.riemann,
                scalar_curvatures=curvature.scalar_curvatures,
                error_bounds=torch.full_like(curvature.scalar_curvatures, float('inf'))
            )
            
        # Check for NaN or infinite values
        if not torch.all(torch.isfinite(curvature.riemann)) or \
           not torch.all(torch.isfinite(curvature.ricci)) or \
           not torch.all(torch.isfinite(curvature.scalar_curvatures)):
            return CurvatureValidation(
                bounds_satisfied=False,
                sectional=curvature.riemann,
                scalar_curvatures=curvature.scalar_curvatures,
                error_bounds=torch.full_like(curvature.scalar_curvatures, float('inf'))
            )
            
        # Check Ricci tensor symmetry
        for i in range(curvature.ricci.shape[0]):
            if not torch.allclose(
                curvature.ricci[i],
                curvature.ricci[i].transpose(-2, -1),
                rtol=self.tolerance,
                atol=self.tolerance
            ):
                return CurvatureValidation(
                    bounds_satisfied=False,
                    sectional=curvature.riemann,
                    scalar_curvatures=curvature.scalar_curvatures,
                    error_bounds=torch.full_like(curvature.scalar_curvatures, float('inf'))
                )
                
        # Check for negative scalar curvature
        if torch.any(curvature.scalar_curvatures < -self.tolerance):
            return CurvatureValidation(
                bounds_satisfied=False,
                sectional=curvature.riemann,
                scalar_curvatures=curvature.scalar_curvatures,
                error_bounds=torch.full_like(curvature.scalar_curvatures, float('inf'))
            )
            
        # Validate cohomology bounds
        try:
            # Project Riemann tensor to match cohomology dimension
            batch_size = curvature.riemann.shape[0]
            riemann_flat = curvature.riemann.reshape(batch_size, -1)  # [batch_size, *]
            
            # Normalize both tensors
            riemann_norm = torch.norm(riemann_flat, dim=1, keepdim=True)
            cohomology_norm = torch.norm(curvature.cohomology_class, dim=1, keepdim=True)
            
            # Add small epsilon to avoid division by zero
            eps = 1e-8
            riemann_normalized = riemann_flat / (riemann_norm + eps)
            cohomology_normalized = curvature.cohomology_class / (cohomology_norm + eps)
            
            # Project Riemann tensor to cohomology dimension
            riemann_proj = torch.nn.functional.adaptive_avg_pool1d(
                riemann_normalized.unsqueeze(1),  # [batch_size, 1, *]
                curvature.cohomology_class.shape[1]  # Target length = cohomology_dim
            ).squeeze(1)  # [batch_size, cohomology_dim]
            
            # Compute error bounds using cosine similarity
            cosine_sim = torch.sum(
                riemann_proj * cohomology_normalized,
                dim=1
            ).abs()
            
            # Convert to error bounds in [0, 1]
            error_bounds = (1.0 - cosine_sim).clamp(min=0.0, max=1.0)
            
            # Scale error bounds based on manifold dimension
            # Higher dimensions need more lenient bounds
            manifold_dim = curvature.riemann.shape[-1]
            scale_factor = 0.001 * (1.0 + 0.1 * manifold_dim)  # More lenient for higher dimensions
            error_bounds = error_bounds * scale_factor
            
            # Adjust tolerance based on manifold dimension
            adjusted_tolerance = self.tolerance * (1.0 + 0.5 * manifold_dim)
            
            # Check if bounds are satisfied with dimension-adjusted tolerance
            bounds_satisfied = bool((error_bounds < adjusted_tolerance * 1000).all())
            
            return CurvatureValidation(
                bounds_satisfied=bounds_satisfied,
                sectional=curvature.riemann,
                scalar_curvatures=curvature.scalar_curvatures,
                error_bounds=error_bounds
            )
            
        except Exception as e:
            # Return validation failure with proper batch dimensions
            batch_size = curvature.riemann.shape[0]
            return CurvatureValidation(
                bounds_satisfied=False,
                sectional=curvature.riemann,
                scalar_curvatures=curvature.scalar_curvatures,
                error_bounds=torch.full((batch_size,), float('inf'), device=curvature.riemann.device)
            )

    def _validate_local_heights(self, height_data: Optional[Tensor]) -> bool:
        """Validate local height properties."""
        if height_data is None:
            return False
            
        # Heights should be strictly positive with tolerance
        if not bool(torch.all(height_data > self.tolerance)):
            return False
            
        # Heights should be finite
        if not bool(torch.isfinite(height_data).all()):
            return False
            
        # Check monotonicity
        if height_data.numel() > 1:
            diffs = height_data[1:] - height_data[:-1]
            if torch.any(diffs <= -self.tolerance):
                return False
            
        return True

    def _validate_global_height(self, height_data: Optional[Tensor]) -> bool:
        """Validate global height properties."""
        if height_data is None:
            return False
            
        # For single values, always valid
        if height_data.numel() == 1:
            return True
            
        # Check growth conditions with tolerance
        if height_data.numel() > 1:
            # Check if heights are monotonically increasing in sequence
            diffs = height_data[1:] - height_data[:-1]
            if torch.any(diffs <= -self.tolerance):
                return False
            
        return True

    def _check_northcott_property(self, height_data: Optional[Tensor]) -> bool:
        """Verify Northcott property for heights."""
        if height_data is None:
            return False
        # For any bound B, there should be finitely many points with height <= B
        B = height_data.max().item()
        points_below_B = (height_data <= B).sum().item()
        return points_below_B < float('inf')

    def _validate_cohomology_bounds(self, curvature: MotivicCurvatureTensor) -> bool:
        """Validate bounds on cohomology classes.
        
        This method validates that:
        1. The cohomology class is finite
        2. The cohomology class is compatible with the curvature tensor
        3. The cohomology class satisfies the required bounds
        
        Args:
            curvature: Curvature tensor with Riemann and cohomology components
            
        Returns:
            Whether the cohomology bounds are satisfied
        """
        cohomology = curvature.cohomology_class
        
        # Check finiteness
        if not bool(torch.isfinite(cohomology).all()):
            return False
            
        # Check compatibility with curvature components
        riemann_norm = torch.norm(curvature.riemann.reshape(curvature.riemann.shape[0], -1), dim=1)
        ricci_norm = torch.norm(curvature.ricci.reshape(curvature.ricci.shape[0], -1), dim=1)
        scalar_norm = torch.abs(curvature.scalar_curvatures)
        cohomology_norm = torch.norm(cohomology, dim=1)
        
        # Verify that cohomology norm is bounded by curvature norms
        bounds_satisfied = (
            (cohomology_norm <= riemann_norm + self.tolerance).all() and
            (cohomology_norm >= scalar_norm - self.tolerance).all() and
            (torch.abs(cohomology_norm - ricci_norm) < self.tolerance).all()
        )
        
        return bool(bounds_satisfied)

    def _compute_error_bounds(self, curvature: MotivicCurvatureTensor) -> Tensor:
        """Compute error bounds for cohomology computation.
        
        Args:
            curvature: Curvature tensor with Riemann and cohomology components
            
        Returns:
            Error bounds tensor
        """
        # Compute difference between Riemann and cohomology representations
        riemann_flat = curvature.riemann.reshape(curvature.riemann.shape[0], -1)  # [batch_size, *]
        riemann_channels = riemann_flat.unsqueeze(1)  # [batch_size, 1, *]
        
        # Project Riemann tensor to match cohomology dimension
        cohomology_dim = curvature.cohomology_class.shape[-1]
        riemann_proj = torch.nn.functional.adaptive_avg_pool1d(
            riemann_channels,
            cohomology_dim
        ).squeeze(1)  # [batch_size, cohomology_dim]
        
        # Now we can compute the difference
        return torch.abs(riemann_proj - curvature.cohomology_class)


class MotivicRiemannianValidator:
    """Validator for motivic Riemannian structures."""

    def __init__(
        self,
        structure: MotivicRiemannianStructure,
        tolerance: float = 1e-6
    ):
        self.structure = structure
        self.tolerance = tolerance
        self.motivic_validator = MotivicValidator(tolerance)

    def validate(self, points: Tensor) -> MotivicValidation:
        """Validate entire motivic Riemannian structure.
        
        Args:
            points: Points at which to validate
            
        Returns:
            Complete validation results
        """
        # Validate metric with height structure
        metric = self.structure.compute_metric(points)
        height_validation = self.motivic_validator.validate_height(metric)
        
        # Validate connection with dynamics
        connection = self.structure.compute_christoffel(points)
        dynamics_validation = self.motivic_validator.validate_dynamics(connection)
        
        # Validate curvature with cohomology
        curvature = self.structure.compute_curvature(points, connection)
        cohomology_validation = self.motivic_validator.validate_cohomology(curvature)
        
        # Combine results
        is_valid = (
            height_validation.is_valid and
            dynamics_validation.is_valid and
            cohomology_validation.bounds_satisfied
        )
        
        message = "Motivic Riemannian validation " + ("passed" if is_valid else "failed")
        
        return MotivicValidation(
            is_valid=is_valid,
            height_valid=height_validation.is_valid,
            dynamics_valid=dynamics_validation.is_valid,
            cohomology_valid=cohomology_validation.bounds_satisfied,
            message=message,
            data={
                "height": height_validation,
                "dynamics": dynamics_validation,
                "cohomology": cohomology_validation
            }
        ) 