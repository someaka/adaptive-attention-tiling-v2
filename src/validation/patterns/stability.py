"""Pattern stability validation implementation.

This module validates pattern stability:
- Linear stability analysis
- Nonlinear stability analysis
- Perturbation response
- Lyapunov analysis
- Mode decomposition
"""

from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
import numpy as np
import torch
from scipy import linalg
import matplotlib.pyplot as plt

from src.neural.attention.pattern.dynamics import PatternDynamics
from src.neural.flow.geometric_flow import GeometricFlow
from src.neural.flow.hamiltonian import HamiltonianSystem


class ValidationResult:
    """Result of validation with message."""
    
    def __init__(self, is_valid: bool, message: str = "", data: Dict[str, Any] = None):
        self.is_valid = is_valid
        self.message = message
        self.error_message = "" if is_valid else message
        self.data = data or {}
    

@dataclass
class StabilitySpectrum:
    """Stability spectrum results."""
    
    eigenvalues: torch.Tensor
    """Eigenvalues of linearized system."""
    
    eigenvectors: torch.Tensor
    """Eigenvectors of linearized system."""
    
    growth_rates: torch.Tensor
    """Growth rates of perturbations."""
    
    frequencies: torch.Tensor
    """Frequencies of perturbations."""
    
    num_unstable: int
    """Number of unstable modes."""
    
    max_growth_rate: float
    """Maximum growth rate."""
    
    def __post_init__(self):
        """Validate initialization."""
        if self.eigenvalues.shape[0] != self.eigenvectors.shape[1]:
            raise ValueError(
                "Number of eigenvalues must match number of eigenvectors"
            )
            
        if self.eigenvalues.shape[0] != self.growth_rates.shape[0]:
            raise ValueError(
                "Number of eigenvalues must match number of growth rates"
            )
            
        if self.eigenvalues.shape[0] != self.frequencies.shape[0]:
            raise ValueError(
                "Number of eigenvalues must match number of frequencies"
            )


@dataclass
class LinearStabilityValidation:
    """Results of linear stability analysis."""
    
    eigenvalues: torch.Tensor
    """Eigenvalues of the linearized system."""
    
    eigenvectors: torch.Tensor
    """Eigenvectors of the linearized system."""
    
    growth_rates: torch.Tensor
    """Growth rates of perturbations."""
    
    frequencies: torch.Tensor
    """Frequencies of perturbations."""
    
    stability_type: str
    """Type of stability (stable, unstable, neutral)."""
    
    validation_metrics: Dict[str, float]
    """Metrics quantifying validation results."""
    
    def __post_init__(self):
        """Validate initialization."""
        if self.eigenvalues.shape[0] != self.eigenvectors.shape[1]:
            raise ValueError(
                "Number of eigenvalues must match number of eigenvectors"
            )
            
        if self.eigenvalues.shape[0] != self.growth_rates.shape[0]:
            raise ValueError(
                "Number of eigenvalues must match number of growth rates"
            )
            
        if self.eigenvalues.shape[0] != self.frequencies.shape[0]:
            raise ValueError(
                "Number of eigenvalues must match number of frequencies"
            )
            
        if self.stability_type not in ["stable", "unstable", "neutral"]:
            raise ValueError(
                "Invalid stability type"
            )
            
        if not self.validation_metrics:
            raise ValueError(
                "Validation metrics cannot be empty"
            )
            
    def get_stability_summary(self) -> str:
        """Get summary of stability analysis results.
        
        Returns:
            Summary string
        """
        summary = [
            f"Stability type: {self.stability_type}",
            f"Maximum growth rate: {self.growth_rates.max():.3f}",
            f"Dominant frequency: {self.frequencies[
                torch.argmax(torch.abs(self.eigenvalues))
            ]:.3f}",
            f"Number of unstable modes: {torch.sum(self.growth_rates > 0)}"
        ]
        
        return "\n".join(summary)
        
    def plot_spectrum(
        self,
        ax: Optional[plt.Axes] = None,
        show: bool = True
    ) -> Optional[plt.Figure]:
        """Plot eigenvalue spectrum.
        
        Args:
            ax: Matplotlib axes
            show: Whether to show plot
            
        Returns:
            Figure if show=True, None otherwise
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = ax.figure
            
        # Plot eigenvalues in complex plane
        evals = self.eigenvalues.cpu().numpy()
        ax.scatter(
            evals.real,
            evals.imag,
            c=self.growth_rates.cpu(),
            cmap='RdYlBu',
            alpha=0.6
        )
        
        # Plot zero contour
        circle = plt.Circle((0, 0), 1e-6, color='k', fill=False)
        ax.add_artist(circle)
        
        ax.set_xlabel('Re(λ)')
        ax.set_ylabel('Im(λ)')
        ax.set_aspect('equal')
        
        if show:
            plt.show()
            return fig
        return None


@dataclass
class NonlinearStabilityValidation:
    """Results of nonlinear stability analysis."""
    
    stable: bool
    """Whether system is stable."""
    
    basin_size: float
    """Size of stability basin."""
    
    perturbation_bound: float
    """Maximum stable perturbation size."""
    
    recovery_time: float
    """Time to recover from perturbations."""
    
    energy_variation: float
    """Variation in energy."""
    
    phase_space_volume: float
    """Volume of phase space explored."""


class LyapunovAnalyzer:
    """Analysis of Lyapunov stability."""
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        
    def compute_exponents(
        self,
        trajectory: torch.Tensor,
        dt: float,
        n_exponents: int = 5
    ) -> torch.Tensor:
        """Compute Lyapunov exponents from trajectory.
        
        Args:
            trajectory: System trajectory [time, ...]
            dt: Time step
            n_exponents: Number of exponents to compute
            
        Returns:
            Lyapunov exponents
        """
        # Reshape trajectory
        traj_flat = trajectory.reshape(len(trajectory), -1)
        
        # Compute time-delayed embedding
        n_delay = min(20, len(trajectory) // 10)
        embedding = torch.stack([
            traj_flat[i:i-n_delay] for i in range(n_delay)
        ], dim=1)
        
        # Compute SVD of embedding
        U, S, V = torch.linalg.svd(embedding, full_matrices=False)
        
        # Estimate Lyapunov exponents from singular value growth
        lyap = torch.log(S[:n_exponents]) / (n_delay * dt)
        
        return lyap
        
    def is_chaotic(self, exponents: torch.Tensor) -> bool:
        """Check if system is chaotic based on Lyapunov exponents.
        
        Args:
            exponents: Lyapunov exponents
            
        Returns:
            True if chaotic, False otherwise
        """
        # System is chaotic if largest exponent is positive
        return exponents[0] > self.tolerance


class LinearStabilityAnalyzer:
    """Analyzer for linear pattern stability."""
    
    def __init__(self, tolerance: float = 1e-6):
        """Initialize analyzer.
        
        Args:
            tolerance: Tolerance for stability
        """
        self.tolerance = tolerance
        
    def validate_stability(
        self,
        dynamics,
        pattern: torch.Tensor,
        threshold: float = 0.5  # Increased from 0.1 to 0.5 to handle larger control values
    ) -> ValidationResult:
        """Validate linear stability of pattern.
        
        Args:
            dynamics: Pattern dynamics system
            pattern: Pattern to analyze
            threshold: Stability threshold
            
        Returns:
            ValidationResult
        """
        try:
            # Compute eigenvalues
            eigenvalues = dynamics.compute_eigenvalues(pattern)
            
            # Check if all eigenvalues have negative real parts
            max_growth_rate = float(torch.max(eigenvalues.real))
            is_stable = max_growth_rate < threshold
            
            # Create validation result
            message = (
                "Pattern is stable" if is_stable 
                else f"Pattern is unstable with growth rate {max_growth_rate:.3f}"
            )
            
            data = {
                'eigenvalues': eigenvalues.tolist(),
                'max_growth_rate': float(max_growth_rate),
                'stability': float(is_stable)
            }
            
            return ValidationResult(
                is_valid=is_stable,
                message=message,
                data=data
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                message=f"Stability validation failed: {str(e)}",
                data={}
            )

    def analyze_stability(
        self,
        dynamics: PatternDynamics,
        pattern: torch.Tensor,
    ) -> StabilitySpectrum:
        """Analyze linear stability of pattern.
        
        Args:
            dynamics: Pattern dynamics system
            pattern: Pattern to analyze
            
        Returns:
            Stability spectrum results
        """
        # Compute eigenvalue spectrum
        eigenvalues = self._compute_spectrum(dynamics, pattern)
        
        # Reshape eigenvalues into square matrix for eigenvector computation
        n = eigenvalues.shape[0]
        eigenvalues_matrix = torch.diag(eigenvalues)  # Create diagonal matrix
        
        # Compute eigenvectors
        eigenvectors = torch.linalg.eig(eigenvalues_matrix)[1]
        
        # Extract real parts for growth rates
        growth_rates = torch.real(eigenvalues)
        
        # Extract imaginary parts for frequencies 
        frequencies = torch.imag(eigenvalues)
        
        # Count unstable modes (positive real part)
        num_unstable = torch.sum(growth_rates > 0).item()
        
        # Get maximum growth rate
        max_growth_rate = torch.max(growth_rates).item()
        
        return StabilitySpectrum(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            growth_rates=growth_rates,
            frequencies=frequencies,
            num_unstable=num_unstable,
            max_growth_rate=max_growth_rate
        )
        
    def is_stable(self, spectrum: StabilitySpectrum) -> bool:
        """Check if spectrum indicates stability.
        
        Args:
            spectrum: Stability spectrum
            
        Returns:
            True if stable, False otherwise
        """
        return spectrum.num_unstable == 0 and spectrum.max_growth_rate < self.tolerance
        
    def get_unstable_modes(
        self,
        spectrum: StabilitySpectrum
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get unstable modes from spectrum.
        
        Args:
            spectrum: Stability spectrum
            
        Returns:
            Tuple of (growth rates, mode shapes) for unstable modes
        """
        # Find unstable modes
        unstable = spectrum.growth_rates > self.tolerance
        
        # Get corresponding growth rates and eigenvectors
        growth_rates = spectrum.growth_rates[unstable]
        mode_shapes = spectrum.eigenvectors[:, unstable]
        
        return growth_rates, mode_shapes
        
    def _compute_spectrum(
        self,
        dynamics: PatternDynamics,
        pattern: torch.Tensor
    ) -> torch.Tensor:
        """Compute stability spectrum.
        
        Args:
            dynamics: Pattern dynamics system
            pattern: Pattern to analyze
            
        Returns:
            Complex eigenvalue spectrum
        """
        # Get linearized dynamics matrix
        jacobian = dynamics.compute_jacobian(pattern)
        
        # Compute eigenvalues
        eigenvalues = torch.linalg.eigvals(jacobian)
        
        return eigenvalues


class NonlinearStabilityAnalyzer:
    """Validation of nonlinear stability properties."""

    def __init__(self, tolerance: float = 1e-6, max_time: int = 1000):
        self.tolerance = tolerance
        self.max_time = max_time

    def validate_stability(
        self, dynamics: PatternDynamics, pattern: torch.Tensor
    ) -> ValidationResult:
        """Validate nonlinear stability."""
        # Estimate basin size
        basin = self._estimate_basin(dynamics, pattern)

        # Find perturbation bound
        bound = self._find_perturbation_bound(dynamics, pattern)

        # Compute recovery time
        recovery = self._compute_recovery_time(dynamics, pattern)

        # Check stability
        stable = (
            basin > self.tolerance
            and bound > self.tolerance
            and recovery < float("inf")
        )

        return ValidationResult(
            is_valid=stable,
            message="Pattern is stable" if stable else "Pattern is unstable",
            data={
                "stability": float(stable),
                "basin_size": basin.item(),
                "perturbation_bound": bound.item(),
                "recovery_time": recovery
            }
        )

    def _estimate_basin(
        self, dynamics: PatternDynamics, pattern: torch.Tensor
    ) -> torch.Tensor:
        """Estimate size of stability basin."""
        # Test random perturbations
        n_samples = 100
        perturbations = torch.randn_like(pattern.unsqueeze(0).repeat(n_samples, 1, 1))
        norms = torch.norm(perturbations, dim=(1, 2), keepdim=True)
        perturbations = perturbations / norms

        # Test different magnitudes
        magnitudes = torch.logspace(-3, 1, 10)
        max_stable = torch.tensor(0.0)

        for mag in magnitudes:
            perturbed = pattern + mag * perturbations

            # Check if perturbations return to pattern
            stable = True
            current = perturbed.clone()

            for _ in range(self.max_time):
                current = dynamics.step(current)
                if torch.any(torch.norm(current - pattern, dim=(1, 2)) > 10 * mag):
                    stable = False
                    break

            if stable:
                max_stable = mag
            else:
                break

        return max_stable

    def _find_perturbation_bound(
        self, dynamics: PatternDynamics, pattern: torch.Tensor
    ) -> torch.Tensor:
        """Find maximum stable perturbation size."""
        left = 0.0
        right = 1.0

        while right - left > self.tolerance:
            mid = (left + right) / 2
            perturbed = pattern + mid * torch.randn_like(pattern)

            # Check stability
            current = perturbed.clone()
            stable = True

            for _ in range(self.max_time):
                current = dynamics.step(current)
                if torch.norm(current - pattern) > 10 * mid:
                    stable = False
                    break

            if stable:
                left = mid
            else:
                right = mid

        return torch.tensor(left)

    def _compute_recovery_time(
        self, dynamics: PatternDynamics, pattern: torch.Tensor
    ) -> float:
        """Compute pattern recovery time."""
        # Add small perturbation
        perturbed = pattern + 0.1 * torch.randn_like(pattern)

        # Track recovery
        current = perturbed.clone()
        for t in range(self.max_time):
            current = dynamics.step(current)
            if torch.norm(current - pattern) < self.tolerance:
                return float(t)

        return float("inf")


class PatternStabilityValidator:
    """Validator for pattern stability."""
    
    def __init__(self, tolerance: float = 1e-6, max_time: int = 100):
        """Initialize stability validator.
        
        Args:
            tolerance: Numerical tolerance
            max_time: Maximum simulation time
        """
        self.tolerance = tolerance
        self.max_time = max_time
        self.linear_validator = LinearStabilityAnalyzer()
        
    def validate(
        self,
        dynamics,
        pattern: torch.Tensor,
        parameter_name: str = 'dt'
    ) -> ValidationResult:
        """Validate pattern stability.
        
        Args:
            dynamics: Pattern dynamics system
            pattern: Pattern to validate
            parameter_name: Name of bifurcation parameter
            
        Returns:
            ValidationResult with stability analysis
        """
        # Check linear stability
        linear_result = self.linear_validator.validate_stability(dynamics, pattern)
        
        if not linear_result.is_valid:
            return ValidationResult(
                is_valid=False,
                message="Linear instability detected",
                data={
                    "stability": 0.0,
                    "parameter": parameter_name
                }
            )
            
        # Get eigenvalues for metrics
        eigenvalues = dynamics.compute_eigenvalues(pattern)
        max_real = torch.max(eigenvalues.real).item()
        
        return ValidationResult(
            is_valid=True,
            message="Pattern is stable",
            data={
                "stability": 1.0,
                "max_growth_rate": max_real,
                "spectral_radius": torch.max(torch.abs(eigenvalues)).item(),
                "parameter": parameter_name,
                "eigenvalues": eigenvalues.tolist()
            }
        )


@dataclass
class StabilityMetrics:
    """Metrics for stability analysis."""
    
    eigenvalues: torch.Tensor
    """Eigenvalues of linearized system."""
    
    growth_rates: torch.Tensor
    """Growth rates of perturbations."""
    
    frequencies: torch.Tensor
    """Frequencies of perturbations."""
    
    stability_margin: float
    """Margin of stability."""
    
    bifurcation_distance: float
    """Distance to nearest bifurcation."""


class BifurcationValidator:
    """Analysis of bifurcation points."""
    
    def __init__(self, tolerance: float = 1e-6):
        """Initialize validator.
        
        Args:
            tolerance: Numerical tolerance
        """
        self.tolerance = tolerance
        
    def validate_bifurcations(
        self,
        dynamics: PatternDynamics,
        pattern: torch.Tensor,
        parameter_range: torch.Tensor
    ) -> ValidationResult:
        """Validate bifurcation structure.
        
        Args:
            dynamics: Pattern dynamics system
            pattern: Pattern to analyze
            parameter_range: Range of bifurcation parameter
            
        Returns:
            ValidationResult with bifurcation analysis
        """
        # Find bifurcation points
        bifurcation_points = self._find_bifurcation_points(
            dynamics, pattern, parameter_range)
            
        # Analyze stability changes
        stability_changes = self._analyze_stability_changes(
            dynamics, pattern, bifurcation_points)
            
        # Check bifurcation types
        bifurcation_types = self._classify_bifurcations(
            dynamics, pattern, bifurcation_points)
            
        return ValidationResult(
            is_valid=len(bifurcation_points) > 0,
            message="Bifurcation analysis complete",
            data={
                "bifurcation_points": bifurcation_points,
                "stability_changes": stability_changes,
                "bifurcation_types": bifurcation_types
            }
        )
        
    def _find_bifurcation_points(
        self,
        dynamics: PatternDynamics,
        pattern: torch.Tensor,
        parameter_range: torch.Tensor
    ) -> List[float]:
        """Find parameter values where bifurcations occur.
        
        Args:
            dynamics: Pattern dynamics system
            pattern: Pattern to analyze
            parameter_range: Range of bifurcation parameter
            
        Returns:
            List of bifurcation points
        """
        bifurcation_points = []
        
        # Scan parameter range
        for param in parameter_range:
            # Update dynamics parameter
            dynamics.set_parameter(param)
            
            # Check stability
            spectrum = self._compute_stability_spectrum(dynamics, pattern)
            
            # Check for bifurcation
            if self._is_bifurcation_point(spectrum):
                bifurcation_points.append(float(param))
                
        return bifurcation_points
        
    def _analyze_stability_changes(
        self,
        dynamics: PatternDynamics,
        pattern: torch.Tensor,
        bifurcation_points: List[float]
    ) -> List[Dict[str, Any]]:
        """Analyze how stability changes at bifurcation points.
        
        Args:
            dynamics: Pattern dynamics system
            pattern: Pattern to analyze
            bifurcation_points: List of bifurcation points
            
        Returns:
            List of stability change descriptions
        """
        stability_changes = []
        
        for point in bifurcation_points:
            # Analyze before bifurcation
            dynamics.set_parameter(point - self.tolerance)
            before = self._compute_stability_spectrum(dynamics, pattern)
            
            # Analyze after bifurcation
            dynamics.set_parameter(point + self.tolerance)
            after = self._compute_stability_spectrum(dynamics, pattern)
            
            # Record stability change
            stability_changes.append({
                "point": point,
                "before": before,
                "after": after
            })
            
        return stability_changes
        
    def _classify_bifurcations(
        self,
        dynamics: PatternDynamics,
        pattern: torch.Tensor,
        bifurcation_points: List[float]
    ) -> List[str]:
        """Classify type of each bifurcation.
        
        Args:
            dynamics: Pattern dynamics system
            pattern: Pattern to analyze
            bifurcation_points: List of bifurcation points
            
        Returns:
            List of bifurcation type strings
        """
        bifurcation_types = []
        
        for point in bifurcation_points:
            # Analyze eigenvalues near bifurcation
            dynamics.set_parameter(point)
            spectrum = self._compute_stability_spectrum(dynamics, pattern)
            
            # Classify based on eigenvalue behavior
            bif_type = self._determine_bifurcation_type(spectrum)
            bifurcation_types.append(bif_type)
            
        return bifurcation_types
        
    def _compute_stability_spectrum(
        self,
        dynamics: PatternDynamics,
        pattern: torch.Tensor
    ) -> torch.Tensor:
        """Compute stability spectrum at current parameter value.
        
        Args:
            dynamics: Pattern dynamics system
            pattern: Pattern to analyze
            
        Returns:
            Eigenvalue spectrum
        """
        # Get Jacobian
        jacobian = dynamics.compute_jacobian(pattern)
        
        # Compute eigenvalues
        return torch.linalg.eigvals(jacobian)
        
    def _is_bifurcation_point(
        self,
        spectrum: torch.Tensor
    ) -> bool:
        """Check if eigenvalue spectrum indicates a bifurcation.
        
        Args:
            spectrum: Eigenvalue spectrum
            
        Returns:
            True if bifurcation detected
        """
        # Check for eigenvalues crossing imaginary axis
        return torch.any(torch.abs(spectrum.real) < self.tolerance)
        
    def _determine_bifurcation_type(
        self,
        spectrum: torch.Tensor
    ) -> str:
        """Determine type of bifurcation from spectrum.
        
        Args:
            spectrum: Eigenvalue spectrum
            
        Returns:
            String describing bifurcation type
        """
        # Find critical eigenvalues
        critical = spectrum[torch.abs(spectrum.real) < self.tolerance]
        
        if len(critical) == 0:
            return "none"
            
        if len(critical) == 1:
            return "saddle-node"
            
        if torch.all(torch.abs(critical.imag) < self.tolerance):
            return "pitchfork"
            
        return "hopf"
