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

        # Kinetic and potential terms with consistent dtype
        self.kinetic = nn.Parameter(torch.eye(hilbert_space.dim, dtype=torch.float32))

        # Learnable potential with float32 dtype
        self.potential = nn.Sequential(
            nn.Linear(hilbert_space.dim, potential_rank),
            nn.Tanh(),
            nn.Linear(potential_rank, 1),
        )

    def compute_action(self, path: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        """Compute action along a path."""
        # Ensure path is float32
        path = path.to(torch.float32)
        
        # Compute velocities
        velocities = (path[1:] - path[:-1]) / dt

        # Kinetic term
        kinetic = 0.5 * torch.sum(
            velocities * (self.kinetic @ velocities.unsqueeze(-1)).squeeze(-1)
        )

        # Potential term
        potential = torch.sum(self.potential(path))

        # Return real action
        return kinetic - potential

    def stationary_points(
        self, boundary_points: Tuple[torch.Tensor, torch.Tensor], num_points: int = 100
    ) -> Path:
        """Find stationary points of the action."""
        start, end = boundary_points
        start = start.to(torch.float32)
        end = end.to(torch.float32)

        # Initialize path with linear interpolation
        t = torch.linspace(0, 1, num_points, dtype=torch.float32)
        path = start.unsqueeze(0) + t.unsqueeze(-1) * (end - start).unsqueeze(0)
        
        # Make path learnable, but exclude boundary points
        path_interior = nn.Parameter(path[1:-1].clone())
        optimizer = torch.optim.Adam([path_interior], lr=0.01)

        # Minimize action while preserving boundary conditions
        for _ in range(100):
            # Reconstruct full path with fixed boundary points
            full_path = torch.cat([
                start.unsqueeze(0),
                path_interior,
                end.unsqueeze(0)
            ], dim=0)
            
            action = self.compute_action(full_path)
            (-action).backward()
            optimizer.step()
            optimizer.zero_grad()

        # Compute final quantities
        with torch.no_grad():
            # Reconstruct final path
            final_path = torch.cat([
                start.unsqueeze(0),
                path_interior,
                end.unsqueeze(0)
            ], dim=0)
            
            final_action = self.compute_action(final_path)
            momenta = self.kinetic @ (final_path[1:] - final_path[:-1]).unsqueeze(-1)
            # Convert to complex for weight
            weight = torch.exp(1j * final_action.to(torch.complex64))

        return Path(points=final_path, momenta=momenta, action=final_action, weight=weight)


class PathSampler:
    """Sample and evaluate quantum paths."""

    def __init__(self, action: ActionFunctional, num_paths: int = 1000):
        self.action = action
        self.num_paths = num_paths

        # Path generation network with float32 dtype
        self.path_generator = nn.Sequential(
            nn.Linear(self.action.hilbert_space.dim * 2 + 1, 128, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(128, 128, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(128, self.action.hilbert_space.dim, dtype=torch.float32),
        )

    def sample_paths(
        self, start: torch.Tensor, end: torch.Tensor, num_points: int = 100
    ) -> List[Path]:
        """Sample paths between boundary points."""
        # Ensure float32 dtype
        start = start.to(torch.float32)
        end = end.to(torch.float32)
        
        paths = []
        times = torch.linspace(0, 1, num_points, dtype=torch.float32)

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
                # Convert to complex for weight
                weight = torch.exp(1j * action.to(torch.complex64))

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
        # Convert final points to float32
        final_points = final_points.to(torch.float32)
        
        # Sample paths for each final point
        all_paths = []
        for final in final_points:
            paths = self.sampler.sample_paths(
                initial_state.amplitudes.real.to(torch.float32),  # Use real part for path sampling
                final,
                num_points
            )
            all_paths.extend(paths)

        # Compute propagator
        weights = torch.stack([p.weight for p in all_paths])
        propagator = torch.sum(
            weights.unsqueeze(-1) * torch.stack([p.points[-1] for p in all_paths]),
            dim=0,
        )

        # Normalize and convert to complex64
        propagator = propagator / (torch.norm(propagator) + 1e-8)
        propagator = propagator.to(torch.complex64)

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
        points = classical_path.points.to(torch.float32)
        points.requires_grad_(True)

        # Compute action and its gradient
        action = self.action.compute_action(points)
        grad = torch.autograd.grad(action, points, create_graph=True)[0]
        
        # Compute Hessian components for interior points
        interior_points = points[1:-1]  # Exclude boundary points
        n_interior = interior_points.shape[0]
        dim = interior_points.shape[1]
        hessian = torch.zeros(n_interior * dim, n_interior * dim, dtype=torch.float32)
        
        # Compute each component of the Hessian
        for i in range(n_interior):
            for d1 in range(dim):
                row = i * dim + d1
                grad_component = grad[i + 1, d1]  # +1 because we excluded first point
                grad2 = torch.autograd.grad(grad_component, points, retain_graph=True)[0]
                # Only take interior point gradients
                grad2_interior = grad2[1:-1].reshape(-1)
                hessian[row] = grad2_interior

        # Van Vleck determinant
        det = torch.det(hessian)

        # Quantum corrections (convert to complex64 for exponential)
        correction = torch.sqrt(torch.abs(det)) * torch.exp(-order * action.to(torch.complex64))

        return correction
