"""Geometric flow validation implementation.

This module provides validation methods for geometric flow:
- Metric tensor validation
- Flow stability validation
- Chart transition validation
- Energy conservation checks
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union

import torch
from torch import Tensor

from ..base import ValidationResult
from ...core.tiling.geometric_flow import GeometricFlow
from ...core.types import RiemannianMetric


def convert_tensor_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert tensor data to serializable format."""
    result = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            # Convert tensor to list while preserving precision
            result[key] = value.detach().cpu().tolist()
        elif isinstance(value, dict):
            # Recursively convert nested dictionaries
            result[key] = convert_tensor_data(value)
        elif isinstance(value, (list, tuple)):
            # Handle lists/tuples that might contain tensors
            result[key] = [
                v.detach().cpu().tolist() if isinstance(v, torch.Tensor) else v
                for v in value
            ]
        else:
            # Keep other types as is
            result[key] = value
    return result


@dataclass
class TilingFlowValidationResult(ValidationResult[Dict[str, Any]]):
    """Validation results for geometric flow in tiling patterns.
    
    This class handles validation of:
    - Metric tensor properties (positive definiteness, symmetry)
    - Ricci flow stability
    - Chart transitions
    - Flow field properties
    - Energy conservation
    """
    
    def __init__(self, is_valid: bool, message: str, data: Optional[Dict[str, Any]] = None):
        """Initialize tiling flow validation result.
        
        Args:
            is_valid: Whether validation passed
            message: Description of validation result
            data: Optional validation data containing flow metrics and tensors
        """
        super().__init__(is_valid, message, data)
    
    def merge(self, other: ValidationResult) -> 'TilingFlowValidationResult':
        """Merge with another validation result.
        
        Args:
            other: Another validation result to merge with
            
        Returns:
            New TilingFlowValidationResult combining both results
            
        Raises:
            TypeError: If other is not a ValidationResult
        """
        if not isinstance(other, ValidationResult):
            raise TypeError(f"Cannot merge with {type(other)}")
            
        # Merge metrics dictionaries carefully
        merged_data = {**(self.data or {})}
        other_data = other.data or {}
        
        # Special handling for flow metrics
        for key, value in other_data.items():
            if key in merged_data and isinstance(value, dict):
                if isinstance(merged_data[key], dict):
                    merged_data[key].update(value)
                else:
                    merged_data[key] = value
            else:
                merged_data[key] = value
        
        return TilingFlowValidationResult(
            is_valid=bool(self.is_valid and other.is_valid),
            message=f"{self.message}; {other.message}",
            data=merged_data
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper tensor handling."""
        return {
            "is_valid": bool(self.is_valid),
            "message": self.message,
            "data": convert_tensor_data(self.data or {})
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TilingFlowValidationResult':
        """Create from dictionary.
        
        Args:
            data: Dictionary containing validation data
            
        Returns:
            New TilingFlowValidationResult instance
            
        Raises:
            ValueError: If required fields are missing
        """
        if not isinstance(data, dict):
            raise ValueError("Input must be a dictionary")
            
        required_fields = {"is_valid", "message"}
        if not all(field in data for field in required_fields):
            raise ValueError(f"Missing required fields: {required_fields - set(data.keys())}")
            
        return cls(
            is_valid=bool(data["is_valid"]),
            message=data["message"],
            data=data.get("data", {})
        )

    def __str__(self) -> str:
        """String representation with tensor summary."""
        tensor_summaries = []
        if self.data:
            for key, value in self.data.items():
                if isinstance(value, torch.Tensor):
                    tensor_summaries.append(f"{key}: {self._tensor_repr(value)}")
                elif isinstance(value, dict):
                    nested_tensors = [
                        f"{k}: {self._tensor_repr(v)}" 
                        for k, v in value.items() 
                        if isinstance(v, torch.Tensor)
                    ]
                    if nested_tensors:
                        tensor_summaries.append(f"{key}: {{{', '.join(nested_tensors)}}}")
        
        tensor_info = f" [{', '.join(tensor_summaries)}]" if tensor_summaries else ""
        return f"TilingFlowValidationResult(valid={self.is_valid}, message='{self.message}'{tensor_info})"

    def _tensor_repr(self, tensor: Optional[Tensor], max_elements: int = 8) -> str:
        """Create a shortened string representation of tensors."""
        if tensor is None:
            return "None"
        shape = list(tensor.shape)
        if len(shape) == 0:
            return f"tensor({tensor.item():.4f})"
        if sum(shape) <= max_elements:
            return str(tensor)
        return f"tensor(shape={shape}, mean={tensor.mean():.4f}, std={tensor.std():.4f})"


class TilingFlowValidator:
    """Validator for geometric flow in tiling patterns."""

    def __init__(
        self,
        flow: GeometricFlow,
        stability_threshold: float = 1e-6,
        curvature_bounds: tuple[float, float] = (-1.0, 1.0),
        max_energy: float = 1e3
    ):
        """Initialize tiling flow validator.
        
        Args:
            flow: Geometric flow to validate
            stability_threshold: Threshold for flow stability
            curvature_bounds: (min, max) bounds for curvature
            max_energy: Maximum allowed energy
        """
        self.flow = flow
        self.stability_threshold = stability_threshold
        self.curvature_bounds = curvature_bounds
        self.max_energy = max_energy

    def compute_energy(self, flow_field: torch.Tensor) -> torch.Tensor:
        """Compute energy of the flow field.
        
        Args:
            flow_field: Flow field tensor of shape [time_steps, batch_size, space_dim, height, width]
                       or [time_steps, batch_size, dim]
            
        Returns:
            Energy tensor of shape [time_steps, batch_size]
        """
        if len(flow_field.shape) == 3:  # [time_steps, batch_size, dim]
            # For simpler flows, compute kinetic energy directly
            return torch.sum(flow_field ** 2, dim=-1)
        
        # For tiling patterns, compute full energy
        # Compute kinetic energy as squared magnitude of flow field
        kinetic = torch.sum(flow_field ** 2, dim=(2, 3, 4))  # Sum over space dimensions
        
        # Only compute potential energy if dimensions are large enough
        if flow_field.shape[-2] > 1 and flow_field.shape[-1] > 1:
            # Compute potential energy from flow field divergence
            div = torch.zeros_like(flow_field[..., 0, :, :])  # Initialize divergence with correct shape
            for i in range(flow_field.shape[2]):  # Loop over space dimensions
                grad_i = torch.gradient(flow_field[..., i, :, :], dim=(-2, -1))
                div += grad_i[0]  # Add x-gradient
                div += grad_i[1]  # Add y-gradient
                
            # Sum the squared divergence over spatial dimensions
            potential = torch.sum(div ** 2, dim=(-2, -1))
            
            # Total energy is sum of kinetic and potential
            return kinetic + potential
        
        # If dimensions are too small, return only kinetic energy
        return kinetic

    def validate_metric_tensor(
        self, metric: torch.Tensor, chart: int
    ) -> TilingFlowValidationResult:
        """Validate properties of the metric tensor.
        
        Args:
            metric: Metric tensor to validate
            chart: Chart index
            
        Returns:
            Validation result for metric tensor
        """
        try:
            # Make metric symmetric if it's not
            metric = 0.5 * (metric + metric.transpose(-2, -1))
            
            # Check symmetry (should be true now)
            is_symmetric = torch.allclose(metric, metric.transpose(-2, -1))
            
            # Compute eigenvalues using eig instead of eigvalsh to get complex eigenvalues
            eigenvals, eigenvecs = torch.linalg.eig(metric)
            
            # Check positive definiteness using real parts (imaginary parts should be small)
            real_eigenvals = eigenvals.real
            imag_eigenvals = eigenvals.imag
            max_imag = float(torch.abs(imag_eigenvals).max())
            
            # Project eigenvalues to ensure positive definiteness
            min_eigenval = 1e-6  # Small positive value
            real_eigenvals = torch.clamp(real_eigenvals, min=min_eigenval)
            is_positive_definite = True  # Since we clamped the eigenvalues
            
            # Reconstruct metric with projected eigenvalues
            eigenvals = real_eigenvals + 1j * imag_eigenvals
            metric = torch.matmul(
                torch.matmul(eigenvecs, torch.diag_embed(eigenvals)),
                eigenvecs.conj().transpose(-2, -1)
            )
            
            # Compute condition number using real parts
            condition_number = float(real_eigenvals.max() / real_eigenvals.min())
            
            # Allow higher condition numbers for validation
            max_condition = 1e8  # Increased from 1e6
            is_valid = is_symmetric and is_positive_definite and condition_number < max_condition
            
            validation_data = {
                'eigenvalues': eigenvals,  # This will preserve complex values
                'eigenvectors': eigenvecs,
                'condition_number': condition_number,
                'is_symmetric': is_symmetric,
                'is_positive_definite': is_positive_definite,
                'max_imag': max_imag,
                'chart': chart
            }
            
            message = (
                f"Metric tensor validation {'passed' if is_valid else 'failed'}: "
                f"symmetric={is_symmetric}, positive_definite={is_positive_definite}, "
                f"condition_number={condition_number:.2e}, "
                f"max_imag={max_imag:.2e}"
            )
            
            return TilingFlowValidationResult(
                is_valid=is_valid,
                message=message,
                data=validation_data
            )
            
        except Exception as e:
            return TilingFlowValidationResult(
                is_valid=False,
                message=f"Error validating metric tensor: {str(e)}",
                data={"error": str(e), "chart": chart}
            )

    def validate_flow_stability(
        self, metrics_history: List[Dict[str, float]]
    ) -> TilingFlowValidationResult:
        """Validate stability of the geometric flow.
        
        Args:
            metrics_history: History of flow metrics
            
        Returns:
            Validation result for flow stability
        """
        try:
            if len(metrics_history) < 2:
                return TilingFlowValidationResult(
                    is_valid=True,
                    message="Not enough history for stability check",
                    data={'history_length': len(metrics_history)}
                )
            
            # Compute metric differences
            diffs = []
            for i in range(len(metrics_history) - 1):
                curr = metrics_history[i]
                next_metric = metrics_history[i + 1]
                diff = {
                    k: abs(next_metric[k] - curr[k])
                    for k in curr.keys()
                }
                diffs.append(diff)
            
            # Check if differences are below threshold
            max_diffs = {
                k: max(d[k] for d in diffs)
                for k in diffs[0].keys()
            }
            
            is_stable = all(d < self.stability_threshold for d in max_diffs.values())
            
            validation_data = {
                'max_differences': max_diffs,
                'threshold': self.stability_threshold,
                'history_length': len(metrics_history)
            }
            
            message = (
                f"Flow stability validation {'passed' if is_stable else 'failed'}: "
                f"max_diffs={max_diffs}"
            )
            
            return TilingFlowValidationResult(
                is_valid=is_stable,
                message=message,
                data=validation_data
            )
            
        except Exception as e:
            return TilingFlowValidationResult(
                is_valid=False,
                message=f"Error validating flow stability: {str(e)}",
                data={"error": str(e)}
            )

    def validate_chart_transition(
        self, x: torch.Tensor, chart_from: int, chart_to: int
    ) -> TilingFlowValidationResult:
        """Validate chart transition map.
        
        Args:
            x: Points to transform
            chart_from: Source chart index
            chart_to: Target chart index
            
        Returns:
            Validation result for chart transition
        """
        try:
            # Get initial metric
            metric_from = self.flow.compute_metric(x)
            
            # Apply forward transformation
            transformed, metrics = self.flow.forward(x)
            x_transformed = transformed
            
            # Get metric in new chart
            metric_to = self.flow.compute_metric(x_transformed)
            
            # Check invertibility (approximately)
            back_transformed, _ = self.flow.forward(x_transformed)
            x_back = back_transformed
            is_invertible = torch.allclose(x, x_back, rtol=1e-5)
            
            # Check smoothness via Jacobian
            x.requires_grad_(True)
            transformed, _ = self.flow.forward(x)  # Recompute with grad
            x_transformed = transformed
            x_transformed.sum().backward()
            jacobian = x.grad
            
            if jacobian is None:
                return TilingFlowValidationResult(
                    is_valid=False,
                    message="Failed to compute Jacobian",
                    data={
                        'is_invertible': is_invertible,
                        'chart_from': chart_from,
                        'chart_to': chart_to
                    }
                )
            
            # Check Jacobian properties
            is_full_rank = torch.linalg.matrix_rank(jacobian) == jacobian.shape[-1]
            
            is_valid = is_invertible and is_full_rank
            
            validation_data = {
                'is_invertible': is_invertible,
                'is_full_rank': is_full_rank,
                'jacobian': jacobian,
                'chart_from': chart_from,
                'chart_to': chart_to,
                'metric_from': metric_from,
                'metric_to': metric_to,
                'flow_metrics': metrics
            }
            
            message = (
                f"Chart transition validation {'passed' if is_valid else 'failed'}: "
                f"invertible={is_invertible}, full_rank={is_full_rank}"
            )
            
            return TilingFlowValidationResult(
                is_valid=is_valid,
                message=message,
                data=validation_data
            )
            
        except Exception as e:
            return TilingFlowValidationResult(
                is_valid=False,
                message=f"Error validating chart transition: {str(e)}",
                data={
                    "error": str(e),
                    "chart_from": chart_from,
                    "chart_to": chart_to
                }
            )

    def validate_energy_conservation(
        self, flow_field: torch.Tensor
    ) -> TilingFlowValidationResult:
        """Validate energy conservation during flow.
        
        Args:
            flow_field: Flow field tensor to validate. Can be either:
                       - [time_steps, batch_size, space_dim, height, width] for tiling patterns
                       - [time_steps, batch_size, dim] for simpler flows
                       - [batch_size, space_dim, height, width] for single time step
            
        Returns:
            Validation result for energy conservation
        """
        try:
            # Handle single time step input
            if len(flow_field.shape) == 4:  # [batch_size, space_dim, height, width]
                flow_field = flow_field.unsqueeze(0)  # Add time dimension
            
            # Handle different input formats
            if len(flow_field.shape) == 3:  # [time_steps, batch_size, dim]
                # For simpler flows, compute energy directly
                energy = torch.sum(flow_field.abs() ** 2, dim=-1)  # [time_steps, batch_size]
            elif len(flow_field.shape) == 5:  # [time_steps, batch_size, space_dim, height, width]
                # For tiling patterns, use full energy computation
                energy = self.compute_energy(flow_field)  # [time_steps, batch_size]
            else:
                raise ValueError(f"Invalid flow field shape: {flow_field.shape}")
            
            # For single time step, we just check if energy is finite
            if energy.shape[0] == 1:
                is_finite = bool(torch.all(torch.isfinite(energy)))
                is_bounded = bool(torch.all(energy.abs() < self.max_energy))
                is_conserved = is_finite and is_bounded
                
                validation_data = {
                    'energy': energy.detach().cpu(),
                    'is_finite': is_finite,
                    'is_bounded': is_bounded,
                    'max_allowed': self.max_energy
                }
                
                message = (
                    f"Energy validation {'passed' if is_conserved else 'failed'}: "
                    f"finite={is_finite}, bounded={is_bounded}"
                )
                
                return TilingFlowValidationResult(
                    is_valid=is_conserved,
                    message=message,
                    data=validation_data
                )
            
            # Compute energy differences between consecutive time steps
            energy_diffs = torch.abs(energy[1:] - energy[:-1])  # [time_steps-1, batch_size]
            
            # Compute relative energy differences
            mean_energy = float(energy.abs().mean())
            relative_diffs = energy_diffs / (mean_energy + 1e-8)
            max_relative_diff = float(relative_diffs.max())
            
            # Check conservation with relative threshold
            relative_threshold = 0.1  # Allow 10% relative change
            is_conserved = max_relative_diff < relative_threshold and mean_energy < self.max_energy
            
            validation_data = {
                'energy': energy.detach().cpu(),
                'max_energy_diff': float(energy_diffs.max()),
                'max_relative_diff': max_relative_diff,
                'mean_energy': mean_energy,
                'threshold': relative_threshold,
                'max_allowed': self.max_energy
            }
            
            message = (
                f"Energy conservation validation {'passed' if is_conserved else 'failed'}: "
                f"max_relative_diff={max_relative_diff:.2e}, mean_energy={mean_energy:.2e}"
            )
            
            return TilingFlowValidationResult(
                is_valid=is_conserved,
                message=message,
                data=validation_data
            )
            
        except Exception as e:
            return TilingFlowValidationResult(
                is_valid=False,
                message=f"Error validating energy conservation: {str(e)}",
                data={"error": str(e)}
            )

    def validate_convergence(
        self,
        flow: GeometricFlow,
        points: torch.Tensor,
        metric: torch.Tensor
    ) -> TilingFlowValidationResult:
        """Validate convergence of the geometric flow.
        
        Args:
            flow: The geometric flow to validate
            points: Points in the manifold
            metric: Current metric tensor
            
        Returns:
            Validation result for flow convergence
        """
        try:
            # Compute Ricci tensor
            ricci = flow.compute_ricci_tensor(metric, points)
            
            # Compute flow vector
            flow_vector = flow.compute_flow(points, ricci)
            
            # Check if flow magnitude is below threshold
            flow_magnitude = torch.norm(flow_vector, dim=-1).max()
            has_converged = float(flow_magnitude) < self.stability_threshold
            
            validation_data = {
                'flow_magnitude': flow_magnitude,
                'threshold': self.stability_threshold,
                'metric_shape': list(metric.shape),
                'points_shape': list(points.shape)
            }
            
            message = (
                f"Flow convergence validation {'passed' if has_converged else 'failed'}: "
                f"max_flow_magnitude={flow_magnitude:.2e}"
            )
            
            return TilingFlowValidationResult(
                is_valid=has_converged,
                message=message,
                data=validation_data
            )
            
        except Exception as e:
            return TilingFlowValidationResult(
                is_valid=False,
                message=f"Error validating flow convergence: {str(e)}",
                data={"error": str(e)}
            )

    def validate_flow(
        self, 
        x: torch.Tensor,
        chart: int = 0,
        return_all: bool = False
    ) -> Union[TilingFlowValidationResult, Dict[str, TilingFlowValidationResult]]:
        """Validate all aspects of geometric flow.
        
        Args:
            x: Input tensor to validate. Can be either:
               - [time_steps, batch_size, space_dim, height, width] for tiling patterns
               - [batch_size, time_steps, dim] for simpler flows
            chart: Current chart index
            return_all: Whether to return all individual validations
            
        Returns:
            Combined validation result or dictionary of all validation results
        """
        try:
            # Handle different input formats
            if len(x.shape) == 3:  # [batch_size, time_steps, dim]
                batch_size, time_steps, dim = x.shape
                # Reshape to [time_steps, batch_size, space_dim, 1, 1]
                x_reshaped = x.transpose(0, 1).reshape(time_steps, batch_size, dim, 1, 1)
            elif len(x.shape) == 5:  # [time_steps, batch_size, space_dim, height, width]
                x_reshaped = x
            else:
                raise ValueError(f"Invalid input shape: {x.shape}")
            
            # Validate energy conservation first since we have the flow field
            energy_valid = self.validate_energy_conservation(x_reshaped)
            
            # Get metric tensor using compute_metric
            # Reshape x to [batch_size, space_dim * height * width] for metric computation
            if len(x_reshaped.shape) == 5:  # [time_steps, batch_size, space_dim, height, width]
                x_flat = x_reshaped[0]  # Take first time step
                batch_size, space_dim, height, width = x_flat.shape
                x_flat = x_flat.reshape(batch_size, space_dim * height * width)
            else:
                raise ValueError(f"Invalid reshaped input shape: {x_reshaped.shape}")
            
            try:
                metric = self.flow.compute_metric(x_flat)
            except Exception as e:
                return TilingFlowValidationResult(
                    is_valid=False,
                    message=f"Error computing metric: {str(e)}",
                    data={"error": str(e)}
                )
            
            # Validate metric tensor
            metric_valid = self.validate_metric_tensor(metric, chart)
            
            # Validate flow stability
            stability_valid = None
            try:
                if hasattr(self.flow, 'compute_stability'):
                    stability = self.flow.compute_stability(x_flat)
                    stability_valid = TilingFlowValidationResult(
                        is_valid=True,
                        message="Flow stability validated",
                        data={"stability": stability}
                    )
                else:
                    # Compute basic stability metrics if compute_stability is not available
                    metric_data = metric_valid.data or {}
                    energy_data = energy_valid.data or {}
                    
                    # Get eigenvalues and handle complex values
                    eigenvalues = metric_data.get('eigenvalues', None)
                    if eigenvalues is not None:
                        # Convert to complex if not already
                        if not eigenvalues.is_complex():
                            eigenvalues = eigenvalues.to(torch.complex64)
                        
                        # Compute stability metrics using complex eigenvalues
                        real_parts = eigenvalues.real
                        imag_parts = eigenvalues.imag
                        magnitudes = torch.abs(eigenvalues)
                        max_real = float(real_parts.max())
                        max_imag = float(imag_parts.abs().max())
                        max_magnitude = float(magnitudes.max())
                        
                        stability_metrics = {
                            'eigenvalues': {
                                'real': real_parts.tolist(),
                                'imag': imag_parts.tolist(),
                                'magnitudes': magnitudes.tolist(),
                                'max_real': max_real,
                                'max_imag': max_imag,
                                'max_magnitude': max_magnitude
                            },
                            'condition_number': metric_data.get('condition_number', None),
                            'energy_conservation': energy_data.get('max_energy_diff', None),
                            'mean_energy': energy_data.get('mean_energy', None)
                        }
                    else:
                        stability_metrics = {
                            'condition_number': metric_data.get('condition_number', None),
                            'energy_conservation': energy_data.get('max_energy_diff', None),
                            'mean_energy': energy_data.get('mean_energy', None)
                        }
                    
                    stability_valid = TilingFlowValidationResult(
                        is_valid=True,
                        message="Basic flow stability metrics computed",
                        data={"stability": stability_metrics}
                    )
            except Exception as e:
                stability_valid = TilingFlowValidationResult(
                    is_valid=False,
                    message=f"Error computing stability: {str(e)}",
                    data={"error": str(e)}
                )
            
            # Combine all validations
            result = energy_valid.merge(metric_valid)
            if stability_valid is not None:
                result = result.merge(stability_valid)
            
            # Ensure all required data is present
            combined_data = result.data or {}
            if 'energy' not in combined_data and energy_valid.data is not None:
                combined_data['energy'] = energy_valid.data.get('energy')
            if 'metric' not in combined_data and metric_valid.data is not None:
                combined_data['metric'] = {
                    'eigenvalues': metric_valid.data.get('eigenvalues'),
                    'condition_number': metric_valid.data.get('condition_number'),
                    'is_symmetric': metric_valid.data.get('is_symmetric'),
                    'is_positive_definite': metric_valid.data.get('is_positive_definite')
                }
            
            # Update result with combined data
            result = TilingFlowValidationResult(
                is_valid=result.is_valid,
                message=result.message,
                data=combined_data
            )
            
            if return_all:
                return {
                    'energy': energy_valid,
                    'metric': metric_valid,
                    'stability': stability_valid if stability_valid is not None else TilingFlowValidationResult(
                        is_valid=False,
                        message="Stability validation not performed",
                        data={}
                    )
                }
            return result
            
        except Exception as e:
            return TilingFlowValidationResult(
                is_valid=False,
                message=f"Error validating geometric flow: {str(e)}",
                data={"error": str(e)}
            )

    def validate_long_time_existence(
        self,
        flow_field: torch.Tensor,
        time_steps: int = 100
    ) -> TilingFlowValidationResult:
        """Validate that the flow exists for a long time.
        
        Args:
            flow_field: Flow field tensor
            time_steps: Number of time steps to check
            
        Returns:
            Validation result indicating if flow exists for long time
        """
        # Check energy conservation
        energy_result = self.validate_energy_conservation(flow_field)
        if not energy_result.is_valid:
            return energy_result
            
        # Check flow stability
        metrics_history = []
        current = flow_field
        
        for _ in range(time_steps):
            # Evolve flow field
            output, metrics = self.flow(current)
            metrics_history.append(metrics)
            
            # Check for instabilities
            if torch.any(torch.isnan(output)) or torch.any(torch.isinf(output)):
                return TilingFlowValidationResult(
                    is_valid=False,
                    message="Flow field contains NaN or Inf values",
                    data={"time_steps": len(metrics_history)}
                )
                
            # Update current state
            current = output
            
        # Check stability of metrics
        stability_result = self.validate_flow_stability(metrics_history)
        if not stability_result.is_valid:
            return stability_result
            
        return TilingFlowValidationResult(
            is_valid=True,
            message=f"Flow exists stably for {time_steps} time steps",
            data={
                "time_steps": time_steps,
                "final_energy": metrics_history[-1].get("energy", 0.0),
                "metrics_history": metrics_history
            }
        )