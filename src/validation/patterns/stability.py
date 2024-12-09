"""Pattern Stability Validation Implementation.

This module validates pattern stability properties:
- Linear stability
- Nonlinear stability
- Bifurcation analysis
- Mode decomposition
"""

from dataclasses import dataclass
from typing import List, Tuple

import torch

from ...neural.attention.pattern_dynamics import PatternDynamics


@dataclass
class LinearStabilityValidation:
    """Results of linear stability validation."""

    stable: bool  # Linear stability
    eigenvalues: torch.Tensor  # Stability spectrum
    growth_rates: torch.Tensor  # Modal growth rates
    critical_modes: int  # Number of critical modes


@dataclass
class NonlinearStabilityValidation:
    """Results of nonlinear stability validation."""

    stable: bool  # Nonlinear stability
    basin_size: float  # Stability basin
    perturbation_bound: float  # Maximum perturbation
    recovery_time: float  # Recovery timescale


@dataclass
class BifurcationValidation:
    """Results of bifurcation validation."""

    bifurcation_type: str  # Type of bifurcation
    parameter_value: float  # Bifurcation point
    normal_form: str  # Normal form
    branch_structure: str  # Branch structure


@dataclass
class ModeValidation:
    """Results of mode validation."""

    modes: torch.Tensor  # Mode shapes
    amplitudes: torch.Tensor  # Mode amplitudes
    interactions: torch.Tensor  # Mode coupling
    hierarchy: List[int]  # Mode hierarchy


class LinearStabilityValidator:
    """Validation of linear stability properties."""

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance

    def validate_stability(
        self, dynamics: PatternDynamics, pattern: torch.Tensor
    ) -> LinearStabilityValidation:
        """Validate linear stability."""
        # Compute Jacobian
        jacobian = dynamics.compute_jacobian(pattern)

        # Compute eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eig(jacobian)

        # Get growth rates
        growth_rates = torch.real(eigenvalues)

        # Count critical modes
        critical_modes = torch.sum(torch.abs(growth_rates) < self.tolerance)

        # Check stability
        stable = torch.all(growth_rates < self.tolerance)

        return LinearStabilityValidation(
            stable=stable,
            eigenvalues=eigenvalues,
            growth_rates=growth_rates,
            critical_modes=int(critical_modes.item()),
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
                bifurcation_type="none",
                parameter_value=0.0,
                normal_form="stable",
                branch_structure="single",
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
            bifurcation_type=bif_type,
            parameter_value=bif_param,
            normal_form=normal_form,
            branch_structure=branch,
        )


class ModeValidator:
    """Validation of pattern mode properties."""

    def __init__(self, n_modes: int = 10, coupling_threshold: float = 0.1):
        self.n_modes = n_modes
        self.coupling_threshold = coupling_threshold

    def validate_modes(self, pattern: torch.Tensor) -> ModeValidation:
        """Validate pattern mode structure."""
        # Compute spatial modes
        modes, amplitudes = self._compute_modes(pattern)

        # Analyze mode interactions
        interactions = self._analyze_interactions(modes)

        # Determine mode hierarchy
        hierarchy = self._determine_hierarchy(modes, amplitudes, interactions)

        return ModeValidation(
            modes=modes,
            amplitudes=amplitudes,
            interactions=interactions,
            hierarchy=hierarchy,
        )

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
        self.linear_validator = LinearStabilityValidator(tolerance)
        self.nonlinear_validator = NonlinearStabilityValidator(tolerance, max_time)
        self.bifurcation_validator = BifurcationValidator(parameter_range, n_points)
        self.mode_validator = ModeValidator(n_modes, coupling_threshold)

    def validate(
        self, dynamics: PatternDynamics, pattern: torch.Tensor, parameter_name: str
    ) -> Tuple[
        LinearStabilityValidation,
        NonlinearStabilityValidation,
        BifurcationValidation,
        ModeValidation,
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
