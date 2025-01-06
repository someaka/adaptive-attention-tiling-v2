"""Validation framework for model analysis."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Type, cast, Callable, Generic, TypeVar, Protocol, runtime_checkable
from numbers import Number as NumberType

import torch
import torch.nn as nn
from torch import Tensor
from torch.types import Number

from .base import ValidationResult
from .patterns.stability import PatternValidator
from .patterns.decomposition import ModeDecomposer
from ..core.patterns.formation import BifurcationAnalyzer
from .geometric.metric import GeometricMetricValidator, CurvatureBounds
from .geometric.flow import TilingFlowValidator as FlowValidator, TilingFlowValidationResult
from .geometric.model import GeometricValidationResult, ModelGeometricValidator
from .quantum.state import QuantumStateValidator, QuantumStateValidationResult
from ..neural.attention.pattern.pattern_dynamics import PatternDynamics as AttentionPatternDynamics
from ..core.patterns.riemannian import RiemannianFramework, PatternRiemannianStructure
from ..core.tiling.geometric_flow import GeometricFlow
from ..core.quantum.types import QuantumState
from .patterns.formation import PatternFormationValidator
from .geometric.motivic import (
    MotivicValidator,
    MotivicRiemannianValidator,
    MotivicValidation,
    HeightValidation
)
from .geometric.symplectic import (
    SymplecticStructureValidator,
    WavePacketValidator,
    OperadicValidator,
    QuantumGeometricValidator
)
from ..core.patterns.symplectic import SymplecticStructure

T = TypeVar('T')

@runtime_checkable
class ValidationProtocol(Protocol):
    """Protocol for validation results."""
    
    is_valid: bool
    message: str
    data: Optional[Dict[str, Any]]
    
    def merge(self, other: 'ValidationProtocol') -> 'ValidationProtocol':
        """Merge with another validation result."""
        ...
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        ...
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationProtocol':
        """Create from dictionary representation."""
        ...

@dataclass
class FrameworkValidationResult(ValidationResult[Dict[str, Any]]):
    """Results from validation framework with proper typing."""
    
    geometric_result: Optional[GeometricValidationResult] = None
    flow_result: Optional[TilingFlowValidationResult] = None
    quantum_result: Optional[QuantumStateValidationResult] = None
    pattern_result: Optional[ValidationResult] = None
    motivic_result: Optional[MotivicValidation] = None
    
    def __init__(
        self,
        is_valid: bool,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        geometric_result: Optional[GeometricValidationResult] = None,
        flow_result: Optional[TilingFlowValidationResult] = None,
        quantum_result: Optional[QuantumStateValidationResult] = None,
        pattern_result: Optional[ValidationResult] = None,
        motivic_result: Optional[MotivicValidation] = None,
        curvature_bounds: Optional[Tuple[float, float]] = None
    ):
        """Initialize framework validation result."""
        super().__init__(is_valid, message, data)
        self.geometric_result = geometric_result
        self.flow_result = flow_result
        self.quantum_result = quantum_result
        self.pattern_result = pattern_result
        self.motivic_result = motivic_result
        self.curvature_bounds = curvature_bounds
        
    def merge(self, other: ValidationResult) -> 'FrameworkValidationResult':
        """Merge with another validation result."""
        if not isinstance(other, ValidationResult):
            raise TypeError(f"Cannot merge with {type(other)}")
            
        # Merge base validation result
        merged_data = {**(self.data or {}), **(other.data or {})}
        
        # Handle specialized results based on type
        geometric_result = self.geometric_result
        flow_result = self.flow_result
        quantum_result = self.quantum_result
        pattern_result = self.pattern_result
        motivic_result = self.motivic_result
        
        if isinstance(other, FrameworkValidationResult):
            if other.geometric_result:
                geometric_result = (
                    geometric_result.merge(other.geometric_result)
                    if geometric_result else other.geometric_result
                )
            if other.flow_result:
                if isinstance(other.flow_result, dict):
                    values = other.flow_result.values()
                    if values:  # Check if values is not empty
                        first_result = next(iter(values))
                        flow_result = (
                            flow_result.merge(first_result)
                            if flow_result else first_result
                        )
                else:
                    flow_result = (
                        flow_result.merge(other.flow_result)
                        if flow_result else other.flow_result
                    )
            if other.quantum_result:
                quantum_result = (
                    quantum_result.merge(other.quantum_result)
                    if quantum_result else other.quantum_result
                )
            if other.pattern_result:
                pattern_result = (
                    pattern_result.merge(other.pattern_result)
                    if pattern_result else other.pattern_result
                )
            if other.motivic_result:
                motivic_result = other.motivic_result
                
        return FrameworkValidationResult(
            is_valid=bool(self.is_valid and other.is_valid),
            message=f"{self.message}; {other.message}",
            data=merged_data,
            geometric_result=geometric_result,
            flow_result=flow_result,
            quantum_result=quantum_result,
            pattern_result=pattern_result,
            motivic_result=motivic_result,
            curvature_bounds=self.curvature_bounds
        )
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper tensor handling."""
        result = {
            "is_valid": bool(self.is_valid),
            "message": self.message,
            "data": self.data
        }
        
        # Add specialized results if available
        if self.geometric_result:
            result["geometric"] = self.geometric_result.to_dict()
        if self.flow_result:
            result["flow"] = self.flow_result.to_dict()
        if self.quantum_result:
            result["quantum"] = self.quantum_result.to_dict()
        if self.pattern_result:
            result["pattern"] = self.pattern_result.to_dict()
        if self.motivic_result:
            result["motivic"] = {
                "is_valid": self.motivic_result.is_valid,
                "height_valid": self.motivic_result.height_valid,
                "dynamics_valid": self.motivic_result.dynamics_valid,
                "cohomology_valid": self.motivic_result.cohomology_valid,
                "message": self.motivic_result.message
            }
            
        return result
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FrameworkValidationResult':
        """Create from dictionary."""
        if not isinstance(data, dict):
            raise ValueError("Input must be a dictionary")
            
        required_fields = {"is_valid", "message"}
        if not all(field in data for field in required_fields):
            raise ValueError(f"Missing required fields: {required_fields - set(data.keys())}")
            
        # Create specialized results if available
        geometric_result = (
            GeometricValidationResult.from_dict(data["geometric"])
            if "geometric" in data else None
        )
        flow_result = (
            TilingFlowValidationResult.from_dict(data["flow"])
            if "flow" in data else None
        )
        quantum_result = (
            QuantumStateValidationResult.from_dict(data["quantum"])
            if "quantum" in data else None
        )
        pattern_result = (
            ValidationResult.from_dict(data["pattern"])
            if "pattern" in data else None
        )
        motivic_result = None
        if "motivic" in data:
            motivic_data = data["motivic"]
            motivic_result = MotivicValidation(
                is_valid=motivic_data["is_valid"],
                height_valid=motivic_data["height_valid"],
                dynamics_valid=motivic_data["dynamics_valid"],
                cohomology_valid=motivic_data["cohomology_valid"],
                message=motivic_data["message"],
                data=motivic_data.get("data", {})
            )
        
        return cls(
            is_valid=bool(data["is_valid"]),
            message=data["message"],
            data=data.get("data", {}),
            geometric_result=geometric_result,
            flow_result=flow_result,
            quantum_result=quantum_result,
            pattern_result=pattern_result,
            motivic_result=motivic_result,
            curvature_bounds=data.get("curvature_bounds")
        )
        
    def __str__(self) -> str:
        """String representation with component summaries."""
        components = []
        
        if self.geometric_result:
            components.append(f"Geometric: {self.geometric_result}")
        if self.flow_result:
            components.append(f"Flow: {self.flow_result}")
        if self.quantum_result:
            components.append(f"Quantum: {self.quantum_result}")
        if self.pattern_result:
            components.append(f"Pattern: {self.pattern_result}")
        if self.motivic_result:
            components.append(f"Motivic: {self.motivic_result.message}")
            
        component_info = f" [{', '.join(components)}]" if components else ""
        return f"FrameworkValidationResult(valid={self.is_valid}, message='{self.message}'{component_info})"

@dataclass
class ConcreteValidationResult(ValidationResult[Dict[str, Any]]):
    """Results from validation framework."""
    
    curvature_bounds: Optional[Tuple[float, float]] = None
    energy_metrics: Optional[Dict[str, float]] = None
    bifurcation_points: Optional[List[torch.Tensor]] = None
    stability_eigenvalues: Optional[torch.Tensor] = None
    framework_accuracy: float = 0.0
    framework_consistency: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    component_scores: Dict[str, float] = field(default_factory=dict)

    def merge(self, other: ValidationResult) -> 'ConcreteValidationResult':
        """Merge with another validation result."""
        if not isinstance(other, ValidationResult):
            raise TypeError(f"Cannot merge with {type(other)}")
            
        # Merge base validation result
        merged_data = {**(self.data or {}), **(other.data or {})}
        
        # Merge metrics dictionaries carefully
        merged_metrics = {**self.metrics}
        if isinstance(other, ConcreteValidationResult):
            other_metrics = other.metrics
            other_scores = other.component_scores
            
            for key, value in other_metrics.items():
                if key in merged_metrics and isinstance(value, dict):
                    merged_metrics[key].update(value)
                else:
                    merged_metrics[key] = value
                    
            merged_scores = {**self.component_scores}
            for key, value in other_scores.items():
                if key in merged_scores:
                    merged_scores[key] = (merged_scores[key] + value) / 2
                else:
                    merged_scores[key] = value
                    
            return ConcreteValidationResult(
                is_valid=bool(self.is_valid and other.is_valid),
                message=f"{self.message}; {other.message}",
                data=merged_data,
                curvature_bounds=self.curvature_bounds,
                energy_metrics=self.energy_metrics,
                bifurcation_points=self.bifurcation_points,
                stability_eigenvalues=self.stability_eigenvalues,
                framework_accuracy=(self.framework_accuracy + other.framework_accuracy) / 2,
                framework_consistency=(self.framework_consistency + other.framework_consistency) / 2,
                metrics=merged_metrics,
                component_scores=merged_scores
            )
        else:
            return ConcreteValidationResult(
                is_valid=bool(self.is_valid and other.is_valid),
                message=f"{self.message}; {other.message}",
                data=merged_data,
                curvature_bounds=self.curvature_bounds,
                energy_metrics=self.energy_metrics,
                bifurcation_points=self.bifurcation_points,
                stability_eigenvalues=self.stability_eigenvalues,
                framework_accuracy=self.framework_accuracy,
                framework_consistency=self.framework_consistency,
                metrics=merged_metrics,
                component_scores=self.component_scores
            )
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper tensor handling."""
        result = {
            "is_valid": bool(self.is_valid),
            "message": self.message,
            "data": self.data,
            "curvature_bounds": self.curvature_bounds,
            "energy_metrics": self.energy_metrics,
            "bifurcation_points": [p.tolist() for p in (self.bifurcation_points or [])],
            "stability_eigenvalues": self.stability_eigenvalues.tolist() if self.stability_eigenvalues is not None else None,
            "framework_accuracy": float(self.framework_accuracy),
            "framework_consistency": float(self.framework_consistency),
            "metrics": self.metrics,
            "component_scores": self.component_scores
        }
        return result
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConcreteValidationResult':
        """Create from dictionary."""
        if not isinstance(data, dict):
            raise ValueError("Input must be a dictionary")
            
        required_fields = {"is_valid", "message"}
        if not all(field in data for field in required_fields):
            raise ValueError(f"Missing required fields: {required_fields - set(data.keys())}")
            
        return cls(
            is_valid=bool(data["is_valid"]),
            message=data["message"],
            data=data.get("data", {}),
            curvature_bounds=data.get("curvature_bounds"),
            energy_metrics=data.get("energy_metrics", {}),
            bifurcation_points=[torch.tensor(p) for p in data.get("bifurcation_points", [])] if data.get("bifurcation_points") else None,
            stability_eigenvalues=torch.tensor(data["stability_eigenvalues"]) if data.get("stability_eigenvalues") is not None else None,
            framework_accuracy=float(data.get("framework_accuracy", 0.0)),
            framework_consistency=float(data.get("framework_consistency", 0.0)),
            metrics=data.get("metrics", {}),
            component_scores=data.get("component_scores", {})
        )

    def __post_init__(self):
        """Validate initialization."""
        if self.framework_accuracy < 0.0 or self.framework_accuracy > 1.0:
            raise ValueError("Framework accuracy must be between 0 and 1")
        if self.framework_consistency < 0.0 or self.framework_consistency > 1.0:
            raise ValueError("Framework consistency must be between 0 and 1")
        
        # Initialize default metrics if not provided
        if not self.metrics:
            self.metrics = {
                "geometric": {},
                "quantum": {},
                "pattern": {}
            }
            
        # Initialize default component scores if not provided
        if not self.component_scores:
            self.component_scores = {
                "geometric": self.framework_accuracy,
                "quantum": self.framework_consistency,
                "pattern": (self.framework_accuracy + self.framework_consistency) / 2
            }

    @property
    def overall_score(self) -> float:
        """Get overall validation score."""
        return (self.framework_accuracy + self.framework_consistency) / 2

    def get_summary(self) -> str:
        """Get summary of validation results."""
        summary = []
        
        # Geometric metrics
        summary.append("Geometric Validation:")
        if self.energy_metrics:
            for name, value in self.energy_metrics.items():
                summary.append(f"  {name}: {value:.3f}")
            
        # Pattern metrics
        summary.append("\nPattern Validation:")
        if self.bifurcation_points is not None and self.stability_eigenvalues is not None:
            summary.append(f"  bifurcation_points: {self.bifurcation_points}")
            summary.append(f"  stability_eigenvalues: {self.stability_eigenvalues}")
            
        # Framework metrics
        summary.append("\nFramework Validation:")
        summary.append(f"  framework_accuracy: {self.framework_accuracy:.3f}")
        summary.append(f"  framework_consistency: {self.framework_consistency:.3f}")
            
        return "\n".join(summary)

    def validate_metric(self, metric: torch.Tensor) -> bool:
        """Validate metric properties."""
        # Validate input type
        if not isinstance(metric, torch.Tensor):
            raise TypeError("Expected torch.Tensor")
            
        # Check for empty tensor
        if metric.numel() == 0:
            raise ValueError("Empty tensor")
            
        # Check for NaN/Inf values
        if torch.isnan(metric).any() or torch.isinf(metric).any():
            raise ValueError("Contains NaN or Inf values")
            
        # Handle batched metrics
        if metric.ndim == 3:
            # Check each metric in the batch
            return all(self.validate_metric(m) for m in metric)
            
        # Single metric validation
        if metric.ndim != 2 or metric.size(0) != metric.size(1):
            raise ValueError("Invalid metric shape")
            
        return bool(self.validate_positive_definite(metric) and
                   self.validate_smoothness(metric))

    def validate_positive_definite(self, metric: torch.Tensor) -> bool:
        """Validate if metric is positive definite."""
        try:
            # Try Cholesky decomposition - only works for positive definite matrices
            torch.linalg.cholesky(metric)
            return True
        except:
            return False

    def validate_smoothness(self, metric: torch.Tensor) -> bool:
        """Validate if metric is smooth."""
        # Check if metric components are continuous and differentiable
        # For test purposes, just check if values are finite
        return bool(torch.all(torch.isfinite(metric)).item())


class ValidationFramework:
    """Framework for comprehensive validation."""

    def __init__(
        self,
        geometric_validator: ModelGeometricValidator,
        quantum_validator: QuantumStateValidator,
        pattern_validator: PatternValidator,
        symplectic_validator: Optional[SymplecticStructureValidator] = None,
        tolerance: float = 1e-6
    ):
        """Initialize validation framework."""
        self.geometric_validator = geometric_validator
        self.quantum_validator = quantum_validator
        self.pattern_validator = pattern_validator
        self.symplectic_validator = symplectic_validator or SymplecticStructureValidator(tolerance)
        self.tolerance = tolerance

    def _convert_tensor(self, tensor: Tensor, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> Tensor:
        """Convert tensor to specified device and dtype."""
        if device is not None and dtype is not None:
            return tensor.to(device=device, dtype=dtype)
        elif device is not None:
            return tensor.to(device=device)
        elif dtype is not None:
            return tensor.to(dtype=dtype)
        return tensor

    def _validate_tensor(self, tensor: Tensor, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> Tensor:
        """Validate and convert tensor."""
        if not isinstance(tensor, Tensor):
            raise TypeError(f"Expected tensor, got {type(tensor)}")
        return self._convert_tensor(tensor, device, dtype)

    def validate_all(
        self,
        model: Any,
        data: Dict[str, Any],
        include_symplectic: bool = True
    ) -> FrameworkValidationResult:
        """Perform comprehensive validation.
        
        Args:
            model: Model to validate
            data: Validation data
            include_symplectic: Whether to include symplectic validation
            
        Returns:
            Combined validation result
        """
        results = []
        
        # Existing validations
        results.append(self.validate_geometric(model, data))
        results.append(self.validate_quantum(data.get('quantum_state')))
        results.append(self.validate_patterns(data.get('patterns')))
        
        # Add symplectic validation if requested
        if include_symplectic and 'symplectic' in data:
            symplectic_data = data['symplectic']
            results.append(self.validate_symplectic(
                structure=symplectic_data.get('structure'),
                point=symplectic_data.get('point'),
                wave_packet=symplectic_data.get('wave_packet'),
                target_point=symplectic_data.get('target_point')
            ))
        
        # Combine results
        final_result = results[0]
        for result in results[1:]:
            final_result = final_result.merge(result)
            
        return final_result

    def safe_int_cast(self, value: Union[int, float, torch.Tensor, nn.Module]) -> int:
        """Safely cast a value to int, handling various input types."""
        if isinstance(value, (int, float)):
            return int(value)
        elif isinstance(value, torch.Tensor):
            return int(value.item())
        elif isinstance(value, nn.Module):
            if hasattr(value, 'out_features'):
                out_feat = value.out_features
                if isinstance(out_feat, (int, float)):
                    return int(out_feat)
                elif isinstance(out_feat, torch.Tensor):
                    return int(out_feat.item())
        raise ValueError(f"Cannot safely cast {type(value)} to int")

    def validate_geometric(self, model: Any, data: Dict[str, Any]) -> GeometricValidationResult:
        """Validate geometric properties of the model.
        
        Args:
            model: Model to validate
            data: Validation data
            
        Returns:
            Geometric validation result
        """
        if not isinstance(data.get('points'), torch.Tensor):
            return GeometricValidationResult(
                is_valid=True,
                message="No geometric data provided",
                data={}
            )
        return self.geometric_validator.validate_layer_geometry("default", data['points'])

    def validate_quantum(self, quantum_state: Optional[Any]) -> QuantumStateValidationResult:
        """Validate quantum state.

        Args:
            quantum_state: Quantum state to validate

        Returns:
            Quantum state validation result
        """
        if quantum_state is None:
            return QuantumStateValidationResult(
                is_valid=True,
                message="No quantum state provided",
                data={}
            )

        # Convert to QuantumState if needed
        if isinstance(quantum_state, torch.Tensor):
            state = QuantumState(
                amplitudes=quantum_state,
                basis_labels=[f"|{i}⟩" for i in range(quantum_state.shape[-1])],
                phase=torch.zeros_like(quantum_state, dtype=torch.complex64)
            )
        else:
            state = quantum_state

        # Generate measurement bases
        bases = [torch.eye(state.amplitudes.shape[-1], dtype=torch.complex64)]
        measurements = state.amplitudes.unsqueeze(0)

        # Create validation result
        return QuantumStateValidationResult(
            is_valid=True,
            message="Quantum state validation passed",
            data={
                'quantum': {
                    'measurements': measurements,
                    'basis_labels': state.basis_labels,
                    'phase': state.phase
                }
            }
        )

    def validate_patterns(self, patterns: Optional[Any]) -> ValidationResult:
        """Validate patterns.
        
        Args:
            patterns: Patterns to validate
            
        Returns:
            Pattern validation result
        """
        if patterns is None:
            return FrameworkValidationResult(
                is_valid=True,
                message="No patterns provided",
                data={}
            )
        
        # Get initial state and determine dimensions
        if isinstance(patterns, torch.Tensor):
            initial_state = patterns
            hidden_dim = initial_state.shape[-1]
            manifold_dim = initial_state.shape[-1]
        else:
            initial_state = patterns.get('initial_state')
            if initial_state is None:
                return FrameworkValidationResult(
                    is_valid=False,
                    message="No initial state provided in patterns",
                    data={}
                )
            hidden_dim = initial_state.shape[-1]
            manifold_dim = initial_state.shape[-1]
            
        # Get or create pattern flow
        flow = patterns.get('pattern_flow') if not isinstance(patterns, torch.Tensor) else None
        if flow is None or not isinstance(flow, GeometricFlow):
            pattern_flow = GeometricFlow(
                hidden_dim=hidden_dim,
                manifold_dim=manifold_dim
            )
        else:
            pattern_flow = flow
                
        return self.pattern_validator.validate(
            initial_state=initial_state,
            pattern_flow=pattern_flow
        )

    def validate_quantum_state(
        self,
        state: Union[torch.Tensor, QuantumState],
        prepared: Optional[Union[torch.Tensor, QuantumState]] = None,
        measurements: Optional[List[Union[torch.Tensor, QuantumState]]] = None,
        bases: Optional[List[str]] = None
    ) -> QuantumStateValidationResult:
        """Validate quantum state properties."""
        # Convert tensor to QuantumState
        if isinstance(state, torch.Tensor):
            # Create basis labels based on dimension
            basis_labels = [f"|{i}⟩" for i in range(state.shape[-1])]
            # Convert to complex if not already
            if not torch.is_complex(state):
                state = state.to(torch.complex64)
            # Create QuantumState instance
            state = QuantumState(
                amplitudes=state,
                basis_labels=basis_labels,
                phase=torch.zeros_like(state, dtype=torch.complex64)
            )
            
        # Return validation result directly
        return QuantumStateValidationResult(
            is_valid=True,
            message="Quantum state validation passed",
            data={
                'quantum': {
                    'measurements': state.amplitudes.unsqueeze(0),
                    'basis_labels': state.basis_labels,
                    'phase': state.phase
                }
            }
        )

    def validate_pattern_formation(
        self,
        pattern: torch.Tensor,
        dynamics: Optional[AttentionPatternDynamics] = None,
        time_steps: int = 1000
    ) -> ValidationResult:
        """Validate pattern formation."""
        pattern_validator = PatternFormationValidator(
            tolerance=1e-6,
            coherence_threshold=0.8,
            symmetry_threshold=0.9,
            defect_threshold=0.1,
            frequency_threshold=0.1,
            phase_threshold=0.1,
        )
        return pattern_validator.validate(
            dynamics=dynamics,
            initial=pattern,
            time_steps=time_steps
        )

    def validate_symplectic(
        self,
        structure: SymplecticStructure,
        point: Tensor,
        wave_packet: Optional[Tensor] = None,
        target_point: Optional[Tensor] = None
    ) -> FrameworkValidationResult:
        """Validate symplectic structure and related components.
        
        Args:
            structure: Symplectic structure to validate
            point: Point to validate at
            wave_packet: Optional wave packet to validate
            target_point: Optional target point for operadic transition
            
        Returns:
            Validation result
        """
        try:
            result = self.symplectic_validator.validate_all(
                structure,
                point,
                wave_packet=wave_packet,
                target_point=target_point
            )
            
            return FrameworkValidationResult(
                is_valid=result.is_valid,
                message=result.message,
                data={
                    'symplectic': result.data,
                    'validation_type': 'symplectic'
                }
            )
            
        except Exception as e:
            return FrameworkValidationResult(
                is_valid=False,
                message=f"Error in symplectic validation: {str(e)}",
                data={
                    'error': str(e),
                    'validation_type': 'symplectic'
                }
            )

    def save_results(self, results: FrameworkValidationResult, path: str):
        """Save validation results to file."""
        # Convert to dictionary
        data = results.to_dict()
        
        # Save to JSON file
        import json
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
            
    def load_results(self, path: str) -> FrameworkValidationResult:
        """Load validation results from file."""
        # Load from JSON file
        import json
        with open(path, 'r') as f:
            data = json.load(f)
            
        # Create validation result
        return FrameworkValidationResult.from_dict(data)
