"""Pattern stability validation implementation.

This module validates pattern stability:
- Linear stability analysis
- Nonlinear stability analysis
- Perturbation response
- Lyapunov analysis
- Mode decomposition
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt

import numpy as np
import torch
from scipy import linalg

from ...neural.flow.geometric_flow import GeometricFlow
from ...neural.flow.hamiltonian import HamiltonianSystem
from ...neural.attention.pattern_dynamics import PatternDynamics


@dataclass
class NonlinearStabilityValidation:
    """Results of nonlinear stability validation."""

    stable: bool  # Overall stability
    basin_size: float  # Size of stability basin
    recovery_time: float  # Recovery timescale
    perturbation_bound: float  # Maximum perturbation
    energy_variation: float  # Energy variation under perturbation
    phase_space_volume: float  # Volume of stability basin


@dataclass
class StabilitySpectrum:
    """Linear stability spectrum data."""
    
    eigenvalues: torch.Tensor  # Stability eigenvalues
    eigenvectors: torch.Tensor  # Stability eigenvectors
    growth_rates: torch.Tensor  # Modal growth rates
    frequencies: torch.Tensor  # Modal frequencies
    num_unstable: int = 0  # Number of unstable modes
    max_growth_rate: float = 0.0  # Maximum growth rate
    
    def __post_init__(self):
        """Compute derived quantities."""
        # Count unstable modes
        self.num_unstable = torch.sum(
            torch.real(self.eigenvalues) > 0
        ).item()
        
        # Find maximum growth rate
        self.max_growth_rate = torch.max(
            torch.real(self.eigenvalues)
        ).item()


@dataclass
class LyapunovAnalyzer:
    """Analyzer for Lyapunov stability of patterns."""

    def __init__(
        self,
        time_step: float = 0.01,
        num_steps: int = 1000,
        perturbation_size: float = 1e-4
    ):
        """Initialize Lyapunov analyzer.
        
        Args:
            time_step: Time step for evolution
            num_steps: Number of time steps
            perturbation_size: Size of perturbations
        """
        self.time_step = time_step
        self.num_steps = num_steps
        self.perturbation_size = perturbation_size
        
    def compute_jacobian(
        self,
        flow: GeometricFlow,
        state: torch.Tensor
    ) -> torch.Tensor:
        """Compute Jacobian matrix at given state.
        
        Args:
            flow: Geometric flow
            state: Current state
            
        Returns:
            Jacobian matrix
        """
        # Create perturbation basis
        dim = state.shape[-1]
        eye = torch.eye(dim, device=state.device)
        
        # Compute Jacobian numerically
        jacobian = []
        for i in range(dim):
            # Apply perturbation in i-th direction
            perturbed = state + self.perturbation_size * eye[i]
            
            # Compute flow difference
            diff = (
                flow.evolve(perturbed, self.time_step) -
                flow.evolve(state, self.time_step)
            )
            
            # Approximate derivative
            jacobian.append(diff / self.perturbation_size)
            
        return torch.stack(jacobian, dim=-2)
        
    def compute_lyapunov_spectrum(
        self,
        flow: GeometricFlow,
        initial_state: torch.Tensor
    ) -> torch.Tensor:
        """Compute Lyapunov spectrum.
        
        Args:
            flow: Geometric flow
            initial_state: Initial state
            
        Returns:
            Lyapunov exponents
        """
        # Initialize variables
        dim = initial_state.shape[-1]
        current_state = initial_state
        
        # Initialize orthonormal basis
        basis = torch.eye(dim, device=initial_state.device)
        
        # Initialize Lyapunov sums
        lyap_sums = torch.zeros(dim, device=initial_state.device)
        
        # Evolve system and track separation
        for _ in range(self.num_steps):
            # Compute Jacobian
            jac = self.compute_jacobian(flow, current_state)
            
            # Evolve basis vectors
            basis = torch.matmul(jac, basis)
            
            # Perform QR decomposition
            q, r = torch.linalg.qr(basis)
            
            # Update basis and accumulate Lyapunov sums
            basis = q
            lyap_sums += torch.log(torch.abs(torch.diagonal(r)))
            
            # Evolve state
            current_state = flow.evolve(current_state, self.time_step)
            
        # Compute Lyapunov exponents
        return lyap_sums / (self.num_steps * self.time_step)
        
    def is_stable(
        self,
        lyapunov_exponents: torch.Tensor,
        tolerance: float = 0.0
    ) -> bool:
        """Check if system is stable based on Lyapunov exponents.
        
        Args:
            lyapunov_exponents: Lyapunov exponents
            tolerance: Tolerance for stability
            
        Returns:
            True if system is stable
        """
        # System is stable if all Lyapunov exponents are non-positive
        return torch.all(lyapunov_exponents <= tolerance)
        
    def analyze_stability(
        self,
        flow: GeometricFlow,
        initial_state: torch.Tensor,
        tolerance: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """Analyze stability using Lyapunov analysis.
        
        Args:
            flow: Geometric flow
            initial_state: Initial state
            tolerance: Tolerance for stability
            
        Returns:
            Dictionary with analysis results
        """
        # Compute Lyapunov spectrum
        lyapunov_exponents = self.compute_lyapunov_spectrum(
            flow, initial_state
        )
        
        # Check stability
        stable = self.is_stable(lyapunov_exponents, tolerance)
        
        # Compute additional metrics
        max_exponent = torch.max(lyapunov_exponents)
        sum_exponents = torch.sum(lyapunov_exponents)
        
        return {
            "lyapunov_exponents": lyapunov_exponents,
            "max_exponent": max_exponent,
            "sum_exponents": sum_exponents,
            "is_stable": stable
        }


@dataclass
class LinearStabilityAnalyzer:
    """Analyzer for linear pattern stability."""

    def __init__(
        self,
        tolerance: float = 1e-6,
        max_modes: int = 10
    ):
        """Initialize stability analyzer.
        
        Args:
            tolerance: Numerical tolerance
            max_modes: Maximum number of modes to analyze
        """
        self.tolerance = tolerance
        self.max_modes = max_modes

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
            Stability spectrum data
        """
        # Compute Jacobian at pattern
        jacobian = dynamics.compute_jacobian(pattern)
        
        # Compute stability spectrum
        eigenvals, eigenvecs = self._compute_spectrum(jacobian)
        
        # Extract growth rates and frequencies
        growth_rates = eigenvals.real
        frequencies = eigenvals.imag
        
        # Sort by growth rate magnitude
        sort_idx = torch.argsort(torch.abs(growth_rates), descending=True)
        eigenvals = eigenvals[sort_idx]
        eigenvecs = eigenvecs[:, sort_idx]
        growth_rates = growth_rates[sort_idx]
        frequencies = frequencies[sort_idx]
        
        # Truncate to max modes
        if len(eigenvals) > self.max_modes:
            eigenvals = eigenvals[:self.max_modes]
            eigenvecs = eigenvecs[:, :self.max_modes]
            growth_rates = growth_rates[:self.max_modes]
            frequencies = frequencies[:self.max_modes]
        
        return StabilitySpectrum(
            eigenvalues=eigenvals,
            eigenvectors=eigenvecs,
            growth_rates=growth_rates,
            frequencies=frequencies
        )

    def _compute_spectrum(
        self,
        jacobian: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute stability spectrum.
        
        Args:
            jacobian: Jacobian matrix
            
        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        # Convert to numpy for better numerical stability
        jacobian_np = jacobian.detach().cpu().numpy()
        
        # Compute eigendecomposition
        eigenvals, eigenvecs = torch.linalg.eig(torch.from_numpy(jacobian_np).to(jacobian.device))
        
        return eigenvals, eigenvecs

    def is_stable(
        self,
        spectrum: StabilitySpectrum,
        strict: bool = True
    ) -> bool:
        """Check if pattern is linearly stable.
        
        Args:
            spectrum: Stability spectrum
            strict: If True, require all eigenvalues negative
                   If False, allow zero eigenvalues
                   
        Returns:
            True if pattern is stable
        """
        if strict:
            return torch.all(spectrum.growth_rates < -self.tolerance)
        else:
            return torch.all(spectrum.growth_rates < self.tolerance)

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
        
        if not torch.any(unstable):
            return torch.tensor([]), torch.tensor([])
            
        return (
            spectrum.growth_rates[unstable],
            spectrum.eigenvectors[:, unstable]
        )


class NonlinearStabilityValidator:
    """Validation of nonlinear stability properties."""

    def __init__(self, tolerance: float = 1e-6, max_time: int = 1000):
        self.tolerance = tolerance
        self.max_time = max_time

    def validate_stability(
        self, dynamics: PatternDynamics, pattern: torch.Tensor
    ) -> NonlinearStabilityValidation:
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

        return NonlinearStabilityValidation(
            stable=stable,
            basin_size=basin.item(),
            perturbation_bound=bound.item(),
            recovery_time=recovery,
            energy_variation=0.0,
            phase_space_volume=0.0
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


@dataclass
class BifurcationValidation:
    """Results of bifurcation analysis validation."""
    
    parameter_values: torch.Tensor
    """Parameter values at which bifurcations occur."""
    
    bifurcation_types: List[str]
    """Types of bifurcations detected."""
    
    stability_changes: List[Dict[str, Any]]
    """Changes in stability properties at bifurcation points."""
    
    branch_structure: Dict[str, torch.Tensor]
    """Structure of solution branches."""
    
    validation_metrics: Dict[str, float]
    """Metrics quantifying validation results."""
    
    def __post_init__(self):
        """Validate initialization."""
        if len(self.parameter_values) != len(self.bifurcation_types):
            raise ValueError(
                "Number of parameter values must match number of bifurcation types"
            )
            
        if len(self.parameter_values) != len(self.stability_changes):
            raise ValueError(
                "Number of parameter values must match number of stability changes"
            )
            
        if not self.branch_structure:
            raise ValueError("Branch structure cannot be empty")
            
        if not self.validation_metrics:
            raise ValueError("Validation metrics cannot be empty")
            
    def get_bifurcation_summary(self) -> str:
        """Get summary of bifurcation analysis results.
        
        Returns:
            Summary string
        """
        summary = []
        for i, (param, btype) in enumerate(zip(
            self.parameter_values, self.bifurcation_types
        )):
            summary.append(
                f"Bifurcation {i+1} at {param:.3f}: {btype}"
            )
            
        return "\n".join(summary)
        
    def plot_bifurcation_diagram(
        self,
        ax: Optional[plt.Axes] = None,
        show: bool = True
    ) -> Optional[plt.Figure]:
        """Plot bifurcation diagram.
        
        Args:
            ax: Matplotlib axes
            show: Whether to show plot
            
        Returns:
            Figure if show=True, None otherwise
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure
            
        # Plot branches
        for branch_name, branch_data in self.branch_structure.items():
            ax.plot(
                branch_data[:, 0].cpu(),
                branch_data[:, 1].cpu(),
                label=branch_name
            )
            
        # Plot bifurcation points
        ax.scatter(
            self.parameter_values.cpu(),
            [self.branch_structure[k][
                torch.argmin(torch.abs(
                    self.branch_structure[k][:, 0] - p
                ))
            ][1].cpu() for k, p in zip(
                self.branch_structure.keys(),
                self.parameter_values
            )],
            color='red',
            zorder=5,
            label='Bifurcation Points'
        )
        
        ax.set_xlabel('Parameter')
        ax.set_ylabel('Solution Measure')
        ax.legend()
        
        if show:
            plt.show()
            return fig
        return None

class BifurcationValidator:
    """Validation of bifurcation properties."""

    def __init__(
        self, parameter_range: Tuple[float, float] = (-1.0, 1.0), n_points: int = 100
    ):
        self.parameter_range = parameter_range
        self.n_points = n_points

    def validate_bifurcation(
        self, dynamics: PatternDynamics, pattern: torch.Tensor, parameter_name: str
    ) -> BifurcationValidation:
        """Validate bifurcation structure."""
        # Scan parameter
        parameter_values = torch.linspace(
            self.parameter_range[0], self.parameter_range[1], self.n_points
        )

        # Track pattern changes
        patterns = []
        stabilities = []

        for param in parameter_values:
            # Set parameter
            dynamics.set_parameter(parameter_name, param)

            # Find equilibrium
            current = pattern.clone()
            for _ in range(100):
                current = dynamics.step(current)
            patterns.append(current)

            # Check stability
            jacobian = dynamics.compute_jacobian(current)
            eigenvalues = torch.real(torch.linalg.eigvals(jacobian))
            stabilities.append(torch.all(eigenvalues < 0))

        # Analyze bifurcation
        bifurcation = self._analyze_bifurcation(patterns, stabilities, parameter_values)

        return bifurcation

    def _analyze_bifurcation(
        self,
        patterns: List[torch.Tensor],
        stabilities: List[bool],
        parameters: torch.Tensor,
    ) -> BifurcationValidation:
        """Analyze type of bifurcation."""
        # Find bifurcation point
        stability_changes = [
            i
            for i in range(len(stabilities) - 1)
            if stabilities[i] != stabilities[i + 1]
        ]

        if len(stability_changes) == 0:
            return BifurcationValidation(
                parameter_values=parameters,
                bifurcation_types=["none"],
                stability_changes=[{}],
                branch_structure={"stable": torch.stack(patterns)},
                validation_metrics={"accuracy": 1.0}
            )

        # Get bifurcation point
        bif_idx = stability_changes[0]
        bif_param = parameters[bif_idx].item()

        # Analyze pattern changes
        patterns = torch.stack(patterns)
        pattern_diff = torch.norm(patterns[1:] - patterns[:-1], dim=(1, 2))

        # Classify bifurcation
        if torch.all(pattern_diff < 0.1):
            bif_type = "transcritical"
            normal_form = "x*(r-x)"
            branch = "exchange"
        elif torch.any(pattern_diff > 1.0):
            bif_type = "subcritical"
            normal_form = "rx + x^3"
            branch = "unstable"
        else:
            bif_type = "supercritical"
            normal_form = "rx - x^3"
            branch = "stable"

        return BifurcationValidation(
            parameter_values=parameters,
            bifurcation_types=[bif_type],
            stability_changes=[{"stability": stabilities[bif_idx]}],
            branch_structure={branch: patterns},
            validation_metrics={"accuracy": 1.0}
        )


class ModeValidator:
    """Validation of pattern mode properties."""

    def __init__(self, n_modes: int = 10, coupling_threshold: float = 0.1):
        self.n_modes = n_modes
        self.coupling_threshold = coupling_threshold

    def validate_modes(self, pattern: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Validate pattern mode structure."""
        # Compute spatial modes
        modes, amplitudes = self._compute_modes(pattern)

        # Analyze mode interactions
        interactions = self._analyze_interactions(modes)

        # Determine mode hierarchy
        hierarchy = self._determine_hierarchy(modes, amplitudes, interactions)

        return {
            "modes": modes,
            "amplitudes": amplitudes,
            "interactions": interactions,
            "hierarchy": hierarchy
        }

    def _compute_modes(
        self, pattern: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute pattern modes using SVD."""
        # Reshape pattern
        shape = pattern.shape
        flattened = pattern.reshape(-1, shape[-1])

        # Compute SVD
        U, S, V = torch.linalg.svd(flattened)

        # Get modes and amplitudes
        modes = V[: self.n_modes]
        amplitudes = S[: self.n_modes]

        return modes, amplitudes

    def _analyze_interactions(self, modes: torch.Tensor) -> torch.Tensor:
        """Analyze mode interactions."""
        n = len(modes)
        interactions = torch.zeros((n, n))

        for i in range(n):
            for j in range(n):
                # Compute mode overlap
                overlap = torch.abs(torch.sum(modes[i] * modes[j]))
                interactions[i, j] = overlap

        return interactions

    def _determine_hierarchy(
        self, modes: torch.Tensor, amplitudes: torch.Tensor, interactions: torch.Tensor
    ) -> List[int]:
        """Determine mode hierarchy."""
        n = len(modes)

        # Score modes by amplitude and coupling
        scores = amplitudes.clone()

        for i in range(n):
            # Add interaction terms
            coupling = torch.sum(interactions[i] * amplitudes)
            scores[i] += self.coupling_threshold * coupling

        # Sort by score
        order = torch.argsort(scores, descending=True)
        return order.tolist()


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
class StabilityMetrics:
    """Collection of stability metrics."""
    
    linear_stability: Dict[str, float]
    """Linear stability metrics."""
    
    nonlinear_stability: Dict[str, float]
    """Nonlinear stability metrics."""
    
    bifurcation_metrics: Dict[str, float]
    """Bifurcation analysis metrics."""
    
    mode_metrics: Dict[str, float]
    """Mode decomposition metrics."""
    
    def __post_init__(self):
        """Validate initialization."""
        if not self.linear_stability:
            raise ValueError("Linear stability metrics cannot be empty")
            
        if not self.nonlinear_stability:
            raise ValueError("Nonlinear stability metrics cannot be empty")
            
        if not self.bifurcation_metrics:
            raise ValueError("Bifurcation metrics cannot be empty")
            
        if not self.mode_metrics:
            raise ValueError("Mode metrics cannot be empty")
            
    def get_summary(self) -> str:
        """Get summary of stability metrics.
        
        Returns:
            Summary string
        """
        summary = []
        
        # Linear stability
        summary.append("Linear Stability:")
        for name, value in self.linear_stability.items():
            summary.append(f"  {name}: {value:.3f}")
            
        # Nonlinear stability
        summary.append("\nNonlinear Stability:")
        for name, value in self.nonlinear_stability.items():
            summary.append(f"  {name}: {value:.3f}")
            
        # Bifurcation metrics
        summary.append("\nBifurcation Analysis:")
        for name, value in self.bifurcation_metrics.items():
            summary.append(f"  {name}: {value:.3f}")
            
        # Mode metrics
        summary.append("\nMode Analysis:")
        for name, value in self.mode_metrics.items():
            summary.append(f"  {name}: {value:.3f}")
            
        return "\n".join(summary)
        
    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """Convert metrics to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "linear_stability": self.linear_stability,
            "nonlinear_stability": self.nonlinear_stability,
            "bifurcation_metrics": self.bifurcation_metrics,
            "mode_metrics": self.mode_metrics
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Dict[str, float]]) -> "StabilityMetrics":
        """Create metrics from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            StabilityMetrics instance
        """
        return cls(
            linear_stability=data["linear_stability"],
            nonlinear_stability=data["nonlinear_stability"],
            bifurcation_metrics=data["bifurcation_metrics"],
            mode_metrics=data["mode_metrics"]
        )

class NonlinearStabilityAnalyzer:
    """Analysis of nonlinear stability properties."""

    def __init__(
        self,
        tolerance: float = 1e-6,
        max_time: float = 100.0,
        dt: float = 0.1
    ):
        """Initialize analyzer.
        
        Args:
            tolerance: Tolerance for stability
            max_time: Maximum simulation time
            dt: Time step size
        """
        self.tolerance = tolerance
        self.max_time = max_time
        self.dt = dt
        
    def analyze_stability(
        self,
        pattern: torch.Tensor,
        dynamics: PatternDynamics,
        perturbation: Optional[torch.Tensor] = None
    ) -> NonlinearStabilityValidation:
        """Analyze nonlinear stability of pattern.
        
        Args:
            pattern: Pattern to analyze
            dynamics: Pattern dynamics
            perturbation: Optional perturbation to apply
            
        Returns:
            Stability validation results
        """
        # Initialize perturbation if not provided
        if perturbation is None:
            perturbation = torch.randn_like(pattern) * 0.1
            
        # Add perturbation to pattern
        perturbed = pattern + perturbation
        
        # Simulate dynamics
        time_points = torch.arange(0, self.max_time, self.dt)
        trajectories = []
        
        current = perturbed
        for t in time_points:
            trajectories.append(current.clone())
            current = dynamics.evolve(current, self.dt)
            
        trajectories = torch.stack(trajectories)
        
        # Compute stability metrics
        distances = torch.norm(
            trajectories - pattern.unsqueeze(0),
            dim=list(range(1, pattern.ndim + 1))
        )
        
        growth_rates = torch.diff(torch.log(distances + 1e-10)) / self.dt
        
        # Determine stability type
        if torch.all(distances[-10:] < self.tolerance):
            stability = "stable"
        elif torch.any(torch.isnan(distances)) or torch.any(torch.isinf(distances)):
            stability = "unstable"
        else:
            stability = "bounded"
            
        # Compute Lyapunov exponents
        lyap_exp = self._estimate_lyapunov(trajectories, self.dt)
        
        return NonlinearStabilityValidation(
            trajectories=trajectories,
            distances=distances,
            growth_rates=growth_rates,
            stability_type=stability,
            lyapunov_exponents=lyap_exp,
            validation_metrics={
                "max_distance": float(torch.max(distances)),
                "final_distance": float(distances[-1]),
                "mean_growth": float(torch.mean(growth_rates)),
                "max_lyapunov": float(torch.max(lyap_exp))
            }
        )
        
    def _estimate_lyapunov(
        self,
        trajectories: torch.Tensor,
        dt: float,
        n_exponents: int = 5
    ) -> torch.Tensor:
        """Estimate Lyapunov exponents from trajectories.
        
        Args:
            trajectories: System trajectories [time, ...]
            dt: Time step
            n_exponents: Number of exponents to compute
            
        Returns:
            Lyapunov exponents
        """
        # Reshape trajectories
        traj_flat = trajectories.reshape(len(trajectories), -1)
        
        # Compute time-delayed embedding
        n_delay = min(20, len(trajectories) // 10)
        embedding = torch.stack([
            traj_flat[i:i-n_delay] for i in range(n_delay)
        ], dim=1)
        
        # Compute SVD of embedding
        U, S, V = torch.linalg.svd(embedding, full_matrices=False)
        
        # Estimate Lyapunov exponents from singular value growth
        lyap = torch.log(S[:n_exponents]) / (n_delay * dt)
        
        return lyap

class PatternStabilityValidator:
    """Complete pattern stability validation system."""

    def __init__(
        self,
        tolerance: float = 1e-6,
        max_time: int = 1000,
        parameter_range: Tuple[float, float] = (-1.0, 1.0),
        n_points: int = 100,
        n_modes: int = 10,
        coupling_threshold: float = 0.1,
    ):
        self.linear_validator = LinearStabilityAnalyzer(tolerance)
        self.nonlinear_validator = NonlinearStabilityValidator(tolerance, max_time)
        self.bifurcation_validator = BifurcationValidator(parameter_range, n_points)
        self.mode_validator = ModeValidator(n_modes, coupling_threshold)

    def validate(
        self, dynamics: PatternDynamics, pattern: torch.Tensor, parameter_name: str
    ) -> Tuple[
        LinearStabilityValidation,
        NonlinearStabilityValidation,
        BifurcationValidation,
        Dict[str, torch.Tensor],
    ]:
        """Perform complete pattern stability validation."""
        # Validate linear stability
        linear = self.linear_validator.validate_stability(dynamics, pattern)

        # Validate nonlinear stability
        nonlinear = self.nonlinear_validator.validate_stability(dynamics, pattern)

        # Validate bifurcation structure
        bifurcation = self.bifurcation_validator.validate_bifurcation(
            dynamics, pattern, parameter_name
        )

        # Validate mode structure
        modes = self.mode_validator.validate_modes(pattern)

        return linear, nonlinear, bifurcation, modes

class PerturbationAnalyzer:
    """Analysis of pattern response to perturbations."""
    
    def __init__(
        self,
        n_perturbations: int = 10,
        perturbation_scale: float = 0.1,
        max_time: float = 100.0,
        dt: float = 0.1
    ):
        """Initialize analyzer.
        
        Args:
            n_perturbations: Number of perturbations to analyze
            perturbation_scale: Scale of perturbations
            max_time: Maximum simulation time
            dt: Time step size
        """
        self.n_perturbations = n_perturbations
        self.perturbation_scale = perturbation_scale
        self.max_time = max_time
        self.dt = dt
        
    def analyze_perturbations(
        self,
        pattern: torch.Tensor,
        dynamics: PatternDynamics
    ) -> Dict[str, torch.Tensor]:
        """Analyze pattern response to perturbations.
        
        Args:
            pattern: Pattern to analyze
            dynamics: Pattern dynamics
            
        Returns:
            Dictionary with perturbation analysis results
        """
        # Generate perturbations
        perturbations = []
        responses = []
        
        for _ in range(self.n_perturbations):
            # Generate random perturbation
            perturb = torch.randn_like(pattern) * self.perturbation_scale
            perturbations.append(perturb)
            
            # Simulate response
            perturbed = pattern + perturb
            trajectory = self._simulate_response(perturbed, pattern, dynamics)
            responses.append(trajectory)
            
        perturbations = torch.stack(perturbations)
        responses = torch.stack(responses)
        
        # Analyze stability properties
        stability = self._analyze_stability(responses, pattern)
        
        # Analyze response patterns
        patterns = self._analyze_patterns(responses, perturbations)
        
        return {
            "perturbations": perturbations,
            "responses": responses,
            "stability": stability,
            "patterns": patterns
        }
        
    def _simulate_response(
        self,
        initial: torch.Tensor,
        target: torch.Tensor,
        dynamics: PatternDynamics
    ) -> torch.Tensor:
        """Simulate system response to perturbation.
        
        Args:
            initial: Initial perturbed state
            target: Target pattern
            dynamics: Pattern dynamics
            
        Returns:
            Response trajectory
        """
        time_points = torch.arange(0, self.max_time, self.dt)
        trajectory = []
        
        current = initial
        for t in time_points:
            trajectory.append(current.clone())
            current = dynamics.evolve(current, self.dt)
            
        return torch.stack(trajectory)
        
    def _analyze_stability(
        self,
        responses: torch.Tensor,
        pattern: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Analyze stability of responses.
        
        Args:
            responses: Response trajectories [n_perturb, time, ...]
            pattern: Target pattern
            
        Returns:
            Dictionary with stability metrics
        """
        # Compute distances from target
        distances = torch.norm(
            responses - pattern.unsqueeze(0).unsqueeze(0),
            dim=list(range(2, pattern.ndim + 2))
        )
        
        # Compute convergence rates
        rates = torch.diff(torch.log(distances + 1e-10)) / self.dt
        
        # Compute stability metrics
        metrics = {
            "distances": distances,
            "rates": rates,
            "max_distance": torch.max(distances, dim=1)[0],
            "mean_rate": torch.mean(rates, dim=1),
            "convergence_time": torch.argmin(distances < 0.1, dim=1).float() * self.dt
        }
        
        return metrics
        
    def _analyze_patterns(
        self,
        responses: torch.Tensor,
        perturbations: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Analyze patterns in responses.
        
        Args:
            responses: Response trajectories [n_perturb, time, ...]
            perturbations: Applied perturbations [n_perturb, ...]
            
        Returns:
            Dictionary with pattern metrics
        """
        # Compute response patterns
        patterns = {
            "initial_amplitude": torch.norm(perturbations, dim=list(range(1, perturbations.ndim))),
            "final_amplitude": torch.norm(responses[:, -1], dim=list(range(1, responses.ndim - 1))),
            "max_amplitude": torch.max(
                torch.norm(responses, dim=list(range(2, responses.ndim))),
                dim=1
            )[0]
        }
        
        # Compute pattern correlations
        corr_matrix = torch.zeros(len(responses), len(responses))
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                corr = torch.corrcoef(
                    responses[i].reshape(-1),
                    responses[j].reshape(-1)
                )[0, 1]
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
                
        patterns["correlations"] = corr_matrix
        
        return patterns
