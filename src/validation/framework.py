"""Validation framework for model analysis."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch import nn

from ..core.patterns import RiemannianFramework, PatternDynamics
from .geometric.flow import EnergyMetrics
from .geometric.metric import CurvatureBounds
from .patterns.formation import BifurcationAnalyzer
from .patterns.stability import LinearStabilityAnalyzer, NonlinearStabilityAnalyzer
from .patterns.decomposition import ModeDecomposer


@dataclass
class ValidationMetrics:
    """Collection of validation metrics."""
    
    # Geometric metrics
    curvature_bounds: CurvatureBounds
    energy_metrics: EnergyMetrics
    
    # Pattern metrics
    bifurcation_points: List[torch.Tensor]
    stability_eigenvalues: torch.Tensor
    
    # Framework metrics
    framework_accuracy: float
    framework_consistency: float


@dataclass
class GeometricValidator:
    """Validator for geometric properties."""
    
    def __init__(
        self,
        curvature_tolerance: float = 1e-6,
        energy_tolerance: float = 1e-6
    ):
        """Initialize geometric validator.
        
        Args:
            curvature_tolerance: Tolerance for curvature bounds
            energy_tolerance: Tolerance for energy metrics
        """
        self.curvature_tolerance = curvature_tolerance
        self.energy_tolerance = energy_tolerance
        
    def validate_geometry(
        self,
        model: nn.Module,
        data: torch.Tensor,
        riemannian: RiemannianFramework
    ) -> Tuple[CurvatureBounds, EnergyMetrics]:
        """Validate geometric properties.
        
        Args:
            model: Model to validate
            data: Input data tensor
            riemannian: Riemannian framework
            
        Returns:
            Tuple of curvature bounds and energy metrics
        """
        # Compute curvature bounds
        bounds = self._compute_curvature(model, data, riemannian)
        
        # Compute energy metrics
        metrics = self._compute_energy(model, data, riemannian)
        
        return bounds, metrics
        
    def _compute_curvature(
        self,
        model: nn.Module,
        data: torch.Tensor,
        riemannian: RiemannianFramework
    ) -> CurvatureBounds:
        """Compute curvature bounds.
        
        Args:
            model: Model to validate
            data: Input data tensor
            riemannian: Riemannian framework
            
        Returns:
            Curvature bounds
        """
        # Get manifold dimension
        dim = data.shape[1]
        
        # Compute sectional curvatures
        sections = []
        for i in range(dim):
            for j in range(i + 1, dim):
                K = riemannian.sectional_curvature(
                    model,
                    data,
                    plane=(i, j)
                )
                sections.append(K)
                
        sections = torch.stack(sections)
        
        # Compute Ricci curvatures
        ricci = []
        for i in range(dim):
            Ric = riemannian.ricci_curvature(
                model,
                data,
                direction=i
            )
            ricci.append(Ric)
            
        ricci = torch.stack(ricci)
        
        # Compute scalar curvature
        scalar = riemannian.scalar_curvature(model, data)
        
        return CurvatureBounds(
            sectional_min=float(torch.min(sections)),
            sectional_max=float(torch.max(sections)),
            ricci_min=float(torch.min(ricci)),
            ricci_max=float(torch.max(ricci)),
            scalar=float(scalar),
            tolerance=self.curvature_tolerance
        )
        
    def _compute_energy(
        self,
        model: nn.Module,
        data: torch.Tensor,
        riemannian: RiemannianFramework
    ) -> EnergyMetrics:
        """Compute energy metrics.
        
        Args:
            model: Model to validate
            data: Input data tensor
            riemannian: Riemannian framework
            
        Returns:
            Energy metrics
        """
        # Compute total energy
        total = riemannian.total_energy(model, data)
        
        # Compute kinetic energy
        kinetic = riemannian.kinetic_energy(model, data)
        
        # Compute potential energy
        potential = riemannian.potential_energy(model, data)
        
        # Check conservation
        conserved = torch.abs(total[1:] - total[:-1]) < self.energy_tolerance
        
        return EnergyMetrics(
            total_energy=total,
            kinetic_energy=kinetic,
            potential_energy=potential,
            is_conserved=bool(torch.all(conserved)),
            tolerance=self.energy_tolerance
        )


@dataclass
class PatternValidator:
    """Validator for pattern formation and dynamics."""
    
    def __init__(
        self,
        stability_tolerance: float = 1e-6,
        bifurcation_tolerance: float = 1e-6,
        mode_tolerance: float = 1e-6
    ):
        """Initialize pattern validator.
        
        Args:
            stability_tolerance: Tolerance for stability analysis
            bifurcation_tolerance: Tolerance for bifurcation analysis
            mode_tolerance: Tolerance for mode analysis
        """
        self.stability_tolerance = stability_tolerance
        self.bifurcation_tolerance = bifurcation_tolerance
        self.mode_tolerance = mode_tolerance
        
    def validate_patterns(
        self,
        model: nn.Module,
        data: torch.Tensor,
        param_range: torch.Tensor,
        dynamics: PatternDynamics
    ) -> Dict[str, Any]:
        """Validate pattern formation and dynamics.
        
        Args:
            model: Model to validate
            data: Input data tensor
            param_range: Parameter range tensor
            dynamics: Pattern dynamics
            
        Returns:
            Dictionary with validation results
        """
        # Analyze stability
        stability = self._analyze_stability(model, data, dynamics)
        
        # Analyze bifurcations
        bifurcations = self._analyze_bifurcations(model, data, param_range)
        
        # Analyze modes
        modes = self._analyze_modes(model, data)
        
        return {
            "stability": stability,
            "bifurcations": bifurcations,
            "modes": modes
        }
        
    def _analyze_stability(
        self,
        model: nn.Module,
        data: torch.Tensor,
        dynamics: PatternDynamics
    ) -> Dict[str, Any]:
        """Analyze pattern stability.
        
        Args:
            model: Model to validate
            data: Input data tensor
            dynamics: Pattern dynamics
            
        Returns:
            Dictionary with stability results
        """
        # Initialize analyzers
        linear = LinearStabilityAnalyzer(self.stability_tolerance)
        nonlinear = NonlinearStabilityAnalyzer(self.stability_tolerance)
        
        # Analyze stability
        linear_results = linear.analyze_stability(model, data)
        nonlinear_results = nonlinear.analyze_stability(model, data, dynamics)
        
        return {
            "linear": linear_results,
            "nonlinear": nonlinear_results
        }
        
    def _analyze_bifurcations(
        self,
        model: nn.Module,
        data: torch.Tensor,
        param_range: torch.Tensor
    ) -> Dict[str, Any]:
        """Analyze pattern bifurcations.
        
        Args:
            model: Model to validate
            data: Input data tensor
            param_range: Parameter range tensor
            
        Returns:
            Dictionary with bifurcation results
        """
        # Initialize analyzer
        analyzer = BifurcationAnalyzer(
            tolerance=self.bifurcation_tolerance
        )
        
        # Analyze bifurcations
        results = analyzer.analyze_bifurcations(
            model,
            data,
            param_range
        )
        
        return results
        
    def _analyze_modes(
        self,
        model: nn.Module,
        data: torch.Tensor
    ) -> Dict[str, Any]:
        """Analyze pattern modes.
        
        Args:
            model: Model to validate
            data: Input data tensor
            
        Returns:
            Dictionary with mode analysis results
        """
        # Initialize analyzer
        analyzer = ModeDecomposer(
            tolerance=self.mode_tolerance
        )
        
        # Analyze modes
        results = analyzer.analyze_modes(model, data)
        
        return results


@dataclass
class QuantumValidator:
    """Validator for quantum properties."""
    
    def __init__(
        self,
        tolerance: float = 1e-6,
        n_samples: int = 1000
    ):
        """Initialize validator.
        
        Args:
            tolerance: Tolerance for quantum metrics
            n_samples: Number of samples for quantum measurements
        """
        self.tolerance = tolerance
        self.n_samples = n_samples
        
    def validate_quantum_properties(
        self,
        model: nn.Module,
        data: torch.Tensor
    ) -> Dict[str, Any]:
        """Validate quantum properties.
        
        Args:
            model: Model to validate
            data: Input data tensor
            
        Returns:
            Dictionary with validation results
        """
        # Compute quantum metrics
        metrics = self._compute_quantum_metrics(model, data)
        
        # Analyze entanglement
        entanglement = self._analyze_entanglement(model, data)
        
        # Analyze coherence
        coherence = self._analyze_coherence(model, data)
        
        return {
            "metrics": metrics,
            "entanglement": entanglement,
            "coherence": coherence
        }
        
    def _compute_quantum_metrics(
        self,
        model: nn.Module,
        data: torch.Tensor
    ) -> Dict[str, float]:
        """Compute quantum metrics.
        
        Args:
            model: Model to validate
            data: Input data tensor
            
        Returns:
            Dictionary with quantum metrics
        """
        # Get model output
        output = model(data)
        
        # Compute purity
        purity = torch.sum(output ** 2, dim=1).mean()
        
        # Compute von Neumann entropy
        entropy = -torch.sum(
            output * torch.log(output + 1e-10),
            dim=1
        ).mean()
        
        # Compute quantum Fisher information
        qfi = self._compute_qfi(output)
        
        return {
            "purity": float(purity),
            "entropy": float(entropy),
            "qfi": float(qfi)
        }
        
    def _analyze_entanglement(
        self,
        model: nn.Module,
        data: torch.Tensor
    ) -> Dict[str, float]:
        """Analyze quantum entanglement.
        
        Args:
            model: Model to validate
            data: Input data tensor
            
        Returns:
            Dictionary with entanglement metrics
        """
        # Get model output
        output = model(data)
        
        # Compute reduced density matrices
        rho_a = self._partial_trace(output, [0])
        rho_b = self._partial_trace(output, [1])
        
        # Compute entanglement entropy
        s_a = -torch.sum(
            rho_a * torch.log(rho_a + 1e-10)
        ).mean()
        
        s_b = -torch.sum(
            rho_b * torch.log(rho_b + 1e-10)
        ).mean()
        
        # Compute mutual information
        mutual_info = s_a + s_b - self._compute_entropy(output)
        
        return {
            "entanglement_entropy_a": float(s_a),
            "entanglement_entropy_b": float(s_b),
            "mutual_information": float(mutual_info)
        }
        
    def _analyze_coherence(
        self,
        model: nn.Module,
        data: torch.Tensor
    ) -> Dict[str, float]:
        """Analyze quantum coherence.
        
        Args:
            model: Model to validate
            data: Input data tensor
            
        Returns:
            Dictionary with coherence metrics
        """
        # Get model output
        output = model(data)
        
        # Compute l1 norm of coherence
        l1_coherence = torch.sum(
            torch.abs(output - torch.diag_embed(torch.diagonal(output, dim1=-2, dim2=-1)))
        ).mean()
        
        # Compute relative entropy of coherence
        diag = torch.diag_embed(torch.diagonal(output, dim1=-2, dim2=-1))
        rel_entropy = self._compute_entropy(diag) - self._compute_entropy(output)
        
        return {
            "l1_coherence": float(l1_coherence),
            "relative_entropy": float(rel_entropy)
        }
        
    def _compute_qfi(
        self,
        state: torch.Tensor
    ) -> torch.Tensor:
        """Compute quantum Fisher information.
        
        Args:
            state: Quantum state tensor
            
        Returns:
            Quantum Fisher information
        """
        # Compute eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(state)
        
        # Compute QFI matrix elements
        qfi = 0
        for i in range(len(eigenvalues)):
            for j in range(len(eigenvalues)):
                if eigenvalues[i] + eigenvalues[j] > self.tolerance:
                    qfi += (eigenvalues[i] - eigenvalues[j])**2 / (eigenvalues[i] + eigenvalues[j])
                    
        return qfi
        
    def _partial_trace(
        self,
        state: torch.Tensor,
        dims: List[int]
    ) -> torch.Tensor:
        """Compute partial trace of quantum state.
        
        Args:
            state: Quantum state tensor
            dims: Dimensions to trace out
            
        Returns:
            Reduced density matrix
        """
        # Reshape state
        shape = state.shape
        n_dims = len(shape) // 2
        
        # Compute remaining dimensions
        remain_dims = [i for i in range(n_dims) if i not in dims]
        
        # Trace out specified dimensions
        reduced = torch.einsum(
            'ii' + ''.join(chr(ord('a') + i) for i in remain_dims),
            state
        )
        
        return reduced
        
    def _compute_entropy(
        self,
        state: torch.Tensor
    ) -> torch.Tensor:
        """Compute von Neumann entropy.
        
        Args:
            state: Quantum state tensor
            
        Returns:
            von Neumann entropy
        """
        # Compute eigenvalues
        eigenvalues = torch.linalg.eigvalsh(state)
        
        # Compute entropy
        entropy = -torch.sum(
            eigenvalues * torch.log(eigenvalues + 1e-10)
        )
        
        return entropy


@dataclass
class ValidationResult:
    """Results of validation framework."""
    
    geometric_metrics: Dict[str, float]
    """Geometric validation metrics."""
    
    pattern_metrics: Dict[str, float]
    """Pattern validation metrics."""
    
    quantum_metrics: Dict[str, float]
    """Quantum validation metrics."""
    
    framework_metrics: Dict[str, float]
    """Framework validation metrics."""
    
    def __post_init__(self):
        """Validate initialization."""
        if not self.geometric_metrics:
            raise ValueError("Geometric metrics cannot be empty")
            
        if not self.pattern_metrics:
            raise ValueError("Pattern metrics cannot be empty")
            
        if not self.quantum_metrics:
            raise ValueError("Quantum metrics cannot be empty")
            
        if not self.framework_metrics:
            raise ValueError("Framework metrics cannot be empty")
            
    def get_summary(self) -> str:
        """Get summary of validation results.
        
        Returns:
            Summary string
        """
        summary = []
        
        # Geometric metrics
        summary.append("Geometric Validation:")
        for name, value in self.geometric_metrics.items():
            summary.append(f"  {name}: {value:.3f}")
            
        # Pattern metrics
        summary.append("\nPattern Validation:")
        for name, value in self.pattern_metrics.items():
            summary.append(f"  {name}: {value:.3f}")
            
        # Quantum metrics
        summary.append("\nQuantum Validation:")
        for name, value in self.quantum_metrics.items():
            summary.append(f"  {name}: {value:.3f}")
            
        # Framework metrics
        summary.append("\nFramework Validation:")
        for name, value in self.framework_metrics.items():
            summary.append(f"  {name}: {value:.3f}")
            
        return "\n".join(summary)
        
    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """Convert results to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "geometric_metrics": self.geometric_metrics,
            "pattern_metrics": self.pattern_metrics,
            "quantum_metrics": self.quantum_metrics,
            "framework_metrics": self.framework_metrics
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Dict[str, float]]) -> "ValidationResult":
        """Create results from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            ValidationResult instance
        """
        return cls(
            geometric_metrics=data["geometric_metrics"],
            pattern_metrics=data["pattern_metrics"],
            quantum_metrics=data["quantum_metrics"],
            framework_metrics=data["framework_metrics"]
        )
        
    def save(self, path: str):
        """Save validation results to file.
        
        Args:
            path: Path to save file
        """
        import json
        
        # Convert to dictionary
        data = self.to_dict()
        
        # Save to JSON file
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
            
    @classmethod
    def load(cls, path: str) -> "ValidationResult":
        """Load validation results from file.
        
        Args:
            path: Path to load file
            
        Returns:
            ValidationResult instance
        """
        import json
        
        # Load from JSON file
        with open(path, 'r') as f:
            data = json.load(f)
            
        # Create instance
        return cls.from_dict(data)


class ValidationFramework:
    """Framework for model validation."""
    
    def __init__(
        self,
        riemannian_framework: RiemannianFramework,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    ):
        """Initialize validation framework.
        
        Args:
            riemannian_framework: Riemannian framework for geometric analysis
            device: Computation device
        """
        self.riemannian = riemannian_framework
        self.device = device
        
        # Initialize analyzers
        self.bifurcation = BifurcationAnalyzer()
        self.stability = LinearStabilityAnalyzer()
        
    def validate_geometry(
        self,
        model: nn.Module,
        test_data: torch.Tensor,
    ) -> Tuple[CurvatureBounds, EnergyMetrics]:
        """Validate geometric properties.
        
        Args:
            model: Model to validate
            test_data: Test data tensor
            
        Returns:
            Curvature bounds and energy metrics
        """
        # Compute curvature bounds
        curvature = self.riemannian.curvature_tensor(test_data)
        bounds = CurvatureBounds(
            ricci_lower=torch.min(curvature.ricci),
            ricci_upper=torch.max(curvature.ricci),
            sectional_lower=torch.min(curvature.riemann),
            sectional_upper=torch.max(curvature.riemann),
        )
        
        # Compute energy metrics
        christoffel = self.riemannian.christoffel_symbols(test_data)
        metrics = EnergyMetrics(
            kinetic=torch.mean(christoffel.first_kind ** 2),
            potential=torch.mean(curvature.scalar),
            total=None,  # Will be computed in EnergyMetrics
        )
        
        return bounds, metrics
        
    def validate_patterns(
        self,
        model: nn.Module,
        param_range: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Validate pattern formation.
        
        Args:
            model: Model to validate
            param_range: Parameter range tensor
            
        Returns:
            Bifurcation points and stability eigenvalues
        """
        # Find bifurcation points
        bifurcation_points = self.bifurcation.find_bifurcations(
            model, param_range
        )
        
        # Analyze stability
        eigenvalues = self.stability.compute_stability(
            model, bifurcation_points
        )
        
        return bifurcation_points, eigenvalues
        
    def validate_framework(
        self,
        model: nn.Module,
        test_data: torch.Tensor,
        param_range: torch.Tensor,
    ) -> ValidationMetrics:
        """Complete framework validation.
        
        Args:
            model: Model to validate
            test_data: Test data tensor
            param_range: Parameter range tensor
            
        Returns:
            Validation metrics
        """
        # Validate geometry
        bounds, energy = self.validate_geometry(model, test_data)
        
        # Validate patterns
        bifurcations, stability = self.validate_patterns(model, param_range)
        
        # Compute framework metrics
        accuracy = self._compute_accuracy(model, test_data)
        consistency = self._compute_consistency(bounds, energy)
        
        return ValidationMetrics(
            curvature_bounds=bounds,
            energy_metrics=energy,
            bifurcation_points=bifurcations,
            stability_eigenvalues=stability,
            framework_accuracy=accuracy,
            framework_consistency=consistency,
        )
        
    def _compute_accuracy(self, model: nn.Module, data: torch.Tensor) -> float:
        """Compute framework accuracy."""
        with torch.no_grad():
            predictions = model(data)
            accuracy = torch.mean((predictions > 0.5).float())
        return accuracy.item()
        
    def _compute_consistency(
        self,
        bounds: CurvatureBounds,
        energy: EnergyMetrics,
    ) -> float:
        """Compute framework consistency."""
        # Check if energy satisfies bounds
        energy_consistent = (
            energy.total >= bounds.ricci_lower
            and energy.total <= bounds.ricci_upper
        )
        
        # Check if sectional curvature bounds are consistent
        curvature_consistent = (
            bounds.sectional_lower <= bounds.sectional_upper
            and bounds.ricci_lower <= bounds.ricci_upper
        )
        
        return float(energy_consistent and curvature_consistent)
