"""Path Integral Implementation for Quantum Attention.

This module implements Feynman path integral formulation for attention patterns:
- Action functional computation
- Path sampling and evaluation
- Quantum propagator calculation
- Stationary phase approximation
"""

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn

from .state_space import HilbertSpace, QuantumState


@dataclass
class Path:
    """Represents a quantum path in configuration space."""

    points: torch.Tensor  # Path points in configuration space
    momenta: torch.Tensor  # Conjugate momenta
    action: torch.Tensor  # Action along path
    weight: torch.Tensor  # Path weight (exp(iS/ħ))


class ActionFunctional:
    """Compute action functional for quantum paths."""

    def __init__(self, hilbert_space: HilbertSpace, potential_rank: int = 4):
        self.hilbert_space = hilbert_space

        # Kinetic and potential terms
        self.kinetic = nn.Parameter(torch.eye(hilbert_space.dim, dtype=torch.complex64))

        # Learnable potential
        self.potential = nn.Sequential(
            nn.Linear(hilbert_space.dim, potential_rank),
            nn.Tanh(),
            nn.Linear(potential_rank, 1),
        )

    def compute_action(self, path: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        """Compute action along a path."""
        # Compute velocities
        velocities = (path[1:] - path[:-1]) / dt

        # Kinetic term
        kinetic = 0.5 * torch.sum(
            velocities * (self.kinetic @ velocities.unsqueeze(-1)).squeeze(-1)
        )

        # Potential term
        potential = torch.sum(self.potential(path))

        return kinetic - potential

    def stationary_points(
        self, boundary_points: Tuple[torch.Tensor, torch.Tensor], num_points: int = 100
    ) -> Path:
        """Find stationary points of the action."""
        start, end = boundary_points

        # Initialize path with linear interpolation
        path = (
            torch.linspace(0, 1, num_points).unsqueeze(-1) @ (end - start).unsqueeze(0)
            + start
        )

        # Make path learnable
        path = nn.Parameter(path)
        optimizer = torch.optim.Adam([path], lr=0.01)

        # Minimize action
        for _ in range(100):
            action = self.compute_action(path)
            (-action).backward()
            optimizer.step()
            optimizer.zero_grad()

        # Compute final quantities
        with torch.no_grad():
            final_action = self.compute_action(path)
            momenta = self.kinetic @ (path[1:] - path[:-1]).unsqueeze(-1)
            weight = torch.exp(1j * final_action)

        return Path(points=path, momenta=momenta, action=final_action, weight=weight)


class PathSampler:
    """Sample and evaluate quantum paths."""

    def __init__(self, action: ActionFunctional, num_paths: int = 1000):
        self.action = action
        self.num_paths = num_paths

        # Path generation network
        self.path_generator = nn.Sequential(
            nn.Linear(self.action.hilbert_space.dim * 2 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action.hilbert_space.dim),
        )

    def sample_paths(
        self, start: torch.Tensor, end: torch.Tensor, num_points: int = 100
    ) -> List[Path]:
        """Sample paths between boundary points."""
        paths = []
        times = torch.linspace(0, 1, num_points)

        for _ in range(self.num_paths):
            # Generate path points
            path_input = torch.cat(
                [
                    start.unsqueeze(0).expand(num_points, -1),
                    end.unsqueeze(0).expand(num_points, -1),
                    times.unsqueeze(-1),
                ],
                dim=-1,
            )

            path = self.path_generator(path_input)

            # Enforce boundary conditions
            path[0] = start
            path[-1] = end

            # Compute path properties
            with torch.no_grad():
                action = self.action.compute_action(path)
                momenta = self.action.kinetic @ (path[1:] - path[:-1]).unsqueeze(-1)
                weight = torch.exp(1j * action)

            paths.append(
                Path(points=path, momenta=momenta, action=action, weight=weight)
            )

        return paths


class Propagator:
    """Quantum propagator using path integrals."""

    def __init__(self, sampler: PathSampler, hilbert_space: HilbertSpace):
        self.sampler = sampler
        self.hilbert_space = hilbert_space

    def propagate(
        self,
        initial_state: QuantumState,
        final_points: torch.Tensor,
        num_points: int = 100,
    ) -> QuantumState:
        """Propagate quantum state using path integral."""
        # Sample paths for each final point
        all_paths = []
        for final in final_points:
            paths = self.sampler.sample_paths(
                initial_state.amplitudes, final, num_points
            )
            all_paths.extend(paths)

        # Compute propagator
        weights = torch.stack([p.weight for p in all_paths])
        propagator = torch.sum(
            weights.unsqueeze(-1) * torch.stack([p.points[-1] for p in all_paths]),
            dim=0,
        )

        # Normalize
        propagator = propagator / (torch.norm(propagator) + 1e-8)

        return QuantumState(
            amplitudes=propagator,
            basis_labels=initial_state.basis_labels,
            phase=initial_state.phase,
        )


class StationaryPhase:
    """Stationary phase approximation for path integrals."""

    def __init__(self, action: ActionFunctional):
        self.action = action

    def find_classical_path(
        self, boundary_points: Tuple[torch.Tensor, torch.Tensor], num_points: int = 100
    ) -> Path:
        """Find classical path using stationary phase."""
        return self.action.stationary_points(boundary_points, num_points)

    def quantum_corrections(self, classical_path: Path, order: int = 2) -> torch.Tensor:
        """Compute quantum corrections around classical path."""
        # Second variation of action
        points = classical_path.points
        points.requires_grad_(True)

        action = self.action.compute_action(points)
        grad = torch.autograd.grad(action, points, create_graph=True)[0]
        hessian = torch.stack(
            [torch.autograd.grad(g, points, retain_graph=True)[0] for g in grad]
        )

        # Van Vleck determinant
        det = torch.det(hessian[1:-1, 1:-1])  # Exclude boundary points

        # Quantum corrections
        correction = torch.sqrt(det) * torch.exp(-order * action)

        return correction
