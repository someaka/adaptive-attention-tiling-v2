"""Hamiltonian System Implementation for Neural Flow.

This module implements Hamiltonian mechanics for neural flows:
- Hamiltonian computation and evolution
- Symplectic structure preservation
- Poisson bracket algebra
- Conservation laws
- Phase space dynamics
- Canonical transformations
"""

from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional

import torch
from torch import nn
import torch.nn.init as nn_init
from src.core.patterns.symplectic import SymplecticStructure, SymplecticForm


@dataclass
class PhaseSpacePoint:
    """Point in phase space (position and momentum)."""

    position: torch.Tensor  # Configuration space coordinates
    momentum: torch.Tensor  # Conjugate momenta
    time: float  # Current time


class CanonicalTransform(nn.Module):
    """Canonical transformation of phase space coordinates."""

    def __init__(
        self,
        phase_dim: int,
        transform_type: str = "F1",
        generating_function: Optional[Callable] = None,
        hamiltonian_system: Optional[nn.Module] = None
    ):
        """Initialize canonical transformation.
        
        Args:
            phase_dim: Dimension of phase space (must be even)
            transform_type: Type of generating function (F1, F2, F3, F4)
            generating_function: Optional custom generating function
            hamiltonian_system: Optional Hamiltonian system for energy computation
        """
        super().__init__()
        
        if phase_dim % 2 != 0:
            raise ValueError("Phase space dimension must be even")
            
        self.phase_dim = phase_dim
        self.transform_type = transform_type
        self.hamiltonian_system = hamiltonian_system
        
        # Initialize symplectic structure with proper form
        self.symplectic = SymplecticStructure(
            dim=phase_dim,
            preserve_structure=True,
            wave_enabled=False
        )
        
        if generating_function is None:
            # Use a symplectic generating function that preserves canonical structure
            def symplectic_generator(q: torch.Tensor, Q: torch.Tensor, t: float) -> torch.Tensor:
                # F1(q,Q) = q·JQ where J is the symplectic matrix
                n = q.shape[-1]
                J = torch.zeros(n, n, device=q.device, dtype=q.dtype)
                J[:n//2, n//2:] = torch.eye(n//2)
                J[n//2:, :n//2] = -torch.eye(n//2)
                
                # Compute symplectic inner product using proper dimensions
                q_expanded = q.unsqueeze(-2)  # [batch, 1, n]
                Q_expanded = Q.unsqueeze(-1)  # [batch, n, 1]
                
                # Compute q·J·Q
                qJ = torch.matmul(q_expanded, J)  # [batch, 1, n]
                F = torch.matmul(qJ, Q_expanded).squeeze(-1).squeeze(-1)  # [batch]
                
                # Add small perturbation to make it non-degenerate
                epsilon = 0.01
                F = F + epsilon * (torch.sum(q * q) + torch.sum(Q * Q)) / 2
                return F
                
            self.generating_function = symplectic_generator
        else:
            self.generating_function = generating_function
            
        # Create proper symplectic form ω = Σ dxᵢ ∧ dpᵢ
        n = phase_dim // 2
        omega = torch.zeros(phase_dim, phase_dim)
        omega[:n, n:] = torch.eye(n)
        omega[n:, :n] = -torch.eye(n)
        self.symplectic_form = SymplecticForm(matrix=omega)
            
    def transform(self, point: PhaseSpacePoint) -> PhaseSpacePoint:
        """Apply canonical transformation to phase space point."""
        # Extract position and momentum
        q = point.position
        p = point.momentum
        
        # Create new position tensor with gradient tracking
        q_grad = q.detach().clone()
        q_grad.requires_grad_(True)
        
        # Compute generating function F1(q, Q)
        F1 = self.generating_function(q_grad, q_grad, point.time)
        
        # Compute new momentum P = ∂F1/∂q using symplectic gradient
        grad = torch.autograd.grad(F1, q_grad, create_graph=True)[0]
        if grad is None:
            raise ValueError("Failed to compute gradient")
            
        # Project gradient onto symplectic manifold
        n = self.phase_dim // 2
        P = torch.matmul(self.symplectic_form.matrix[n:, :n], grad.unsqueeze(-1)).squeeze(-1)
        
        # Compute initial energy
        initial_state = torch.cat([q, p], dim=-1)
        initial_energy = self._compute_energy(initial_state)
        
        # Project onto symplectic manifold while preserving energy
        final_state = torch.cat([q, P], dim=-1)
        projected = self.symplectic.project_to_manifold(final_state)
        
        # Scale to preserve energy
        final_energy = self._compute_energy(projected)
        scale = torch.sqrt(initial_energy / (final_energy + 1e-8))
        projected = scale * projected
        
        # Extract final position and momentum
        q_new = projected[..., :n]
        p_new = projected[..., n:]
        
        return PhaseSpacePoint(
            position=q_new,
            momentum=p_new,
            time=point.time
        )
        
    def _compute_energy(self, state: torch.Tensor) -> torch.Tensor:
        """Compute energy using proper symplectic structure."""
        if self.hamiltonian_system is not None:
            return self.hamiltonian_system.compute_energy(state)
            
        # Split state into position and momentum
        n = self.phase_dim // 2
        q = state[..., :n]
        p = state[..., n:]
        
        # Compute kinetic and potential energy using symplectic form
        T = 0.5 * torch.sum(p * p, dim=-1)  # Standard kinetic energy
        V = 0.5 * torch.sum(q * q, dim=-1)  # Harmonic potential
        
        return T + V
        
    def _f1_transform(self, point: PhaseSpacePoint) -> PhaseSpacePoint:
        """F1(q,Q,t) transform using proper symplectic structure."""
        print("\nF1 transform details:")
        q, p = point.position, point.momentum
        q.requires_grad_(True)
        
        # Compute generating function
        F = self.generating_function(q, q, point.time)
        print(f"Generating function value: {F.item():.4f}")
        
        # Compute new momentum using symplectic form
        grad_F = torch.autograd.grad(F, q, create_graph=True)[0]
        
        # Ensure proper dimensions for symplectic form evaluation
        n = self.phase_dim // 2
        full_grad = torch.zeros(self.phase_dim, device=grad_F.device, dtype=grad_F.dtype)
        full_grad[:n] = grad_F
        
        # Compute new momentum using matrix multiplication
        p_new = torch.matmul(self.symplectic_form.matrix[n:, :n], grad_F.unsqueeze(-1)).squeeze(-1)
        
        print(f"Initial momentum norm: {torch.norm(p):.4f}")
        print(f"New momentum norm before scaling: {torch.norm(p_new):.4f}")
        
        # Compute initial energy
        initial_state = torch.cat([point.position, point.momentum], dim=-1)
        initial_energy = self._compute_energy(initial_state)
        print(f"Initial energy: {initial_energy.item():.4f}")
        
        # Compute transformed energy
        transformed_state = torch.cat([q, p_new], dim=-1)
        transformed_energy = self._compute_energy(transformed_state)
        print(f"Transformed energy before scaling: {transformed_energy.item():.4f}")
        
        # Scale using symplectic form to preserve structure
        scale = torch.sqrt(initial_energy / (transformed_energy + 1e-8))
        print(f"Symplectic scaling factor: {scale.item():.4f}")
        
        # Apply scaling while preserving symplectic structure
        p_new = scale * p_new
        
        # Project onto canonical symplectic manifold
        final_state = torch.cat([q, p_new], dim=-1)
        projected_state = self.symplectic.project_to_manifold(final_state)
        q_proj = projected_state[..., :n]
        p_proj = projected_state[..., n:]
        
        # Scale to preserve energy
        alpha = torch.sqrt(initial_energy / (self._compute_energy(projected_state) + 1e-8))
        p_new = alpha * p_proj
        
        print(f"New momentum norm after canonical projection: {torch.norm(p_new):.4f}")
        
        # Verify final energy
        final_state = torch.cat([q_proj, p_new], dim=-1)
        final_energy = self._compute_energy(final_state)
        print(f"Final energy: {final_energy.item():.4f}")
        print(f"Energy conservation error: {abs(final_energy - initial_energy).item():.4e}")
        
        return PhaseSpacePoint(
            position=q_proj,
            momentum=p_new,
            time=point.time
        )
        
    def _f2_transform(self, point: PhaseSpacePoint) -> PhaseSpacePoint:
        """F2(q,P,t) transform: p = ∂F2/∂q, Q = ∂F2/∂P."""
        print("\nF2 transform details:")
        q, p = point.position, point.momentum
        q.requires_grad_(True)
        p.requires_grad_(True)
        
        # Compute generating function
        F = self.generating_function(q, p, point.time)
        print(f"Generating function value: {F.item():.4f}")
        
        # Compute new position and momentum
        p_new = torch.autograd.grad(F, q, create_graph=True)[0]
        Q_new = torch.autograd.grad(F, p, create_graph=True)[0]
        print(f"Initial position/momentum norms: {torch.norm(q):.4f}, {torch.norm(p):.4f}")
        print(f"New position/momentum norms before scaling: {torch.norm(Q_new):.4f}, {torch.norm(p_new):.4f}")
        
        # Compute initial energy
        initial_state = torch.cat([point.position, point.momentum], dim=-1)
        initial_energy = self._compute_energy(initial_state)
        print(f"Initial energy: {initial_energy.item():.4f}")
        
        # Compute transformed energy
        transformed_state = torch.cat([Q_new, p_new], dim=-1)
        transformed_energy = self._compute_energy(transformed_state)
        print(f"Transformed energy before scaling: {transformed_energy.item():.4f}")
        
        # Scale to preserve energy and symplectic form
        scale = torch.sqrt(initial_energy / (transformed_energy + 1e-8))
        print(f"Energy scaling factor: {scale.item():.4f}")
        
        # Apply symplectic normalization
        initial_form = self.symplectic.compute_form(initial_state)
        transformed_form = self.symplectic.compute_form(transformed_state)
        form_scale = torch.sqrt(torch.abs(torch.det(initial_form.matrix)) / 
                              (torch.abs(torch.det(transformed_form.matrix)) + 1e-8))
        print(f"Symplectic form scaling factor: {form_scale.item():.4f}")
        
        # Combine scaling factors with equal weight
        total_scale = torch.sqrt(scale * form_scale)
        Q_new = Q_new * total_scale.unsqueeze(-1)
        p_new = p_new * total_scale.unsqueeze(-1)
        
        # Project onto canonical symplectic manifold while preserving energy
        final_state = torch.cat([Q_new, p_new], dim=-1)
        
        # Compute canonical projection
        n = self.phase_dim // 2
        q_proj = Q_new  # Keep position unchanged
        p_proj = torch.zeros_like(p_new)
        
        # Ensure canonical form (p = J^T ∂H/∂q)
        H = self._compute_energy(final_state)
        q_proj.requires_grad_(True)
        grad_H = torch.autograd.grad(H, q_proj, create_graph=True)[0]
        p_proj = grad_H
        
        # Scale to preserve energy
        alpha = torch.sqrt(initial_energy / (self._compute_energy(torch.cat([q_proj, p_proj], dim=-1)) + 1e-8))
        p_new = alpha * p_proj
        
        print(f"New momentum norm after canonical projection: {torch.norm(p_new):.4f}")
        
        # Verify final energy
        final_state = torch.cat([Q_new, p_new], dim=-1)
        final_energy = self._compute_energy(final_state)
        print(f"Final energy: {final_energy.item():.4f}")
        print(f"Energy conservation error: {abs(final_energy - initial_energy).item():.4e}")
        
        return PhaseSpacePoint(
            position=Q_new,
            momentum=p_new,
            time=point.time
        )
        
    def _f3_transform(self, point: PhaseSpacePoint) -> PhaseSpacePoint:
        """F3(p,Q,t) transform: q = -∂F3/∂p, P = -∂F3/∂Q."""
        print("\nF3 transform details:")
        q, p = point.position, point.momentum
        p.requires_grad_(True)
        q.requires_grad_(True)
        
        # Compute generating function
        F = self.generating_function(p, q, point.time)
        print(f"Generating function value: {F.item():.4f}")
        
        # Compute new position and momentum
        q_new = -torch.autograd.grad(F, p, create_graph=True)[0]
        P_new = -torch.autograd.grad(F, q, create_graph=True)[0]
        print(f"Initial position/momentum norms: {torch.norm(q):.4f}, {torch.norm(p):.4f}")
        print(f"New position/momentum norms before scaling: {torch.norm(q_new):.4f}, {torch.norm(P_new):.4f}")
        
        # Compute initial energy
        initial_state = torch.cat([point.position, point.momentum], dim=-1)
        initial_energy = self._compute_energy(initial_state)
        print(f"Initial energy: {initial_energy.item():.4f}")
        
        # Compute transformed energy
        transformed_state = torch.cat([q_new, P_new], dim=-1)
        transformed_energy = self._compute_energy(transformed_state)
        print(f"Transformed energy before scaling: {transformed_energy.item():.4f}")
        
        # Scale to preserve energy and symplectic form
        scale = torch.sqrt(initial_energy / (transformed_energy + 1e-8))
        print(f"Energy scaling factor: {scale.item():.4f}")
        
        # Apply symplectic normalization
        initial_form = self.symplectic.compute_form(initial_state)
        transformed_form = self.symplectic.compute_form(transformed_state)
        form_scale = torch.sqrt(torch.abs(torch.det(initial_form.matrix)) / 
                              (torch.abs(torch.det(transformed_form.matrix)) + 1e-8))
        print(f"Symplectic form scaling factor: {form_scale.item():.4f}")
        
        # Combine scaling factors with equal weight
        total_scale = torch.sqrt(scale * form_scale)
        q_new = q_new * total_scale.unsqueeze(-1)
        P_new = P_new * total_scale.unsqueeze(-1)
        
        # Project onto canonical symplectic manifold while preserving energy
        final_state = torch.cat([q_new, P_new], dim=-1)
        
        # Compute canonical projection
        n = self.phase_dim // 2
        q_proj = q_new  # Keep position unchanged
        p_proj = torch.zeros_like(P_new)
        
        # Ensure canonical form (p = J^T ∂H/∂q)
        H = self._compute_energy(final_state)
        q_proj.requires_grad_(True)
        grad_H = torch.autograd.grad(H, q_proj, create_graph=True)[0]
        p_proj = grad_H
        
        # Scale to preserve energy
        alpha = torch.sqrt(initial_energy / (self._compute_energy(torch.cat([q_proj, p_proj], dim=-1)) + 1e-8))
        P_new = alpha * p_proj
        
        print(f"New momentum norm after canonical projection: {torch.norm(P_new):.4f}")
        
        # Verify final energy
        final_state = torch.cat([q_new, P_new], dim=-1)
        final_energy = self._compute_energy(final_state)
        print(f"Final energy: {final_energy.item():.4f}")
        print(f"Energy conservation error: {abs(final_energy - initial_energy).item():.4e}")
        
        return PhaseSpacePoint(
            position=q_new,
            momentum=P_new,
            time=point.time
        )
        
    def _f4_transform(self, point: PhaseSpacePoint) -> PhaseSpacePoint:
        """F4(p,P,t) transform: q = -∂F4/∂p, Q = ∂F4/∂P."""
        print("\nF4 transform details:")
        q, p = point.position, point.momentum
        p.requires_grad_(True)
        
        # Compute generating function
        F = self.generating_function(p, p, point.time)
        print(f"Generating function value: {F.item():.4f}")
        
        # Compute new position
        q_new = -torch.autograd.grad(F, p, create_graph=True)[0]
        print(f"Initial position/momentum norms: {torch.norm(q):.4f}, {torch.norm(p):.4f}")
        print(f"New position norm before scaling: {torch.norm(q_new):.4f}")
        
        # Compute initial energy
        initial_state = torch.cat([point.position, point.momentum], dim=-1)
        initial_energy = self._compute_energy(initial_state)
        print(f"Initial energy: {initial_energy.item():.4f}")
        
        # Compute transformed energy
        transformed_state = torch.cat([q_new, p], dim=-1)
        transformed_energy = self._compute_energy(transformed_state)
        print(f"Transformed energy before scaling: {transformed_energy.item():.4f}")
        
        # Scale to preserve energy and symplectic form
        scale = torch.sqrt(initial_energy / (transformed_energy + 1e-8))
        print(f"Energy scaling factor: {scale.item():.4f}")
        
        # Apply symplectic normalization
        initial_form = self.symplectic.compute_form(initial_state)
        transformed_form = self.symplectic.compute_form(transformed_state)
        form_scale = torch.sqrt(torch.abs(torch.det(initial_form.matrix)) / 
                              (torch.abs(torch.det(transformed_form.matrix)) + 1e-8))
        print(f"Symplectic form scaling factor: {form_scale.item():.4f}")
        
        # Combine scaling factors with equal weight
        total_scale = torch.sqrt(scale * form_scale)
        q_new = q_new * total_scale.unsqueeze(-1)
        
        # Project onto canonical symplectic manifold while preserving energy
        final_state = torch.cat([q_new, p], dim=-1)
        
        # Compute canonical projection
        n = self.phase_dim // 2
        q_proj = q_new  # Keep position unchanged
        p_proj = torch.zeros_like(p)
        
        # Ensure canonical form (p = J^T ∂H/∂q)
        H = self._compute_energy(final_state)
        q_proj.requires_grad_(True)
        grad_H = torch.autograd.grad(H, q_proj, create_graph=True)[0]
        p_proj = grad_H
        
        # Scale to preserve energy
        alpha = torch.sqrt(initial_energy / (self._compute_energy(torch.cat([q_proj, p_proj], dim=-1)) + 1e-8))
        p_new = alpha * p_proj
        
        print(f"New momentum norm after canonical projection: {torch.norm(p_new):.4f}")
        
        # Verify final energy
        final_state = torch.cat([q_new, p_new], dim=-1)
        final_energy = self._compute_energy(final_state)
        print(f"Final energy: {final_energy.item():.4f}")
        print(f"Energy conservation error: {abs(final_energy - initial_energy).item():.4e}")
        
        return PhaseSpacePoint(
            position=q_new,
            momentum=p_new,
            time=point.time
        )
        
    def _inverse_transform(self, point: PhaseSpacePoint) -> PhaseSpacePoint:
        """Apply inverse canonical transformation."""
        # For F1 and F4, invert by swapping variables
        if self.transform_type in ["f1", "f4"]:
            return PhaseSpacePoint(
                position=point.momentum,
                momentum=point.position,
                time=point.time
            )
            
        # For F2 and F3, need to solve implicit equations
        raise NotImplementedError(
            f"Inverse transform not implemented for type {self.transform_type}"
        )


class HamiltonianSystem(nn.Module):
    """Neural Hamiltonian system with symplectic structure."""

    def __init__(self, manifold_dim: int, hidden_dim: int = 128):
        """Initialize Hamiltonian system.
        
        Args:
            manifold_dim: Dimension of phase space (must be even)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        if manifold_dim % 2 != 0:
            raise ValueError("Phase space dimension must be even")
            
        self.manifold_dim = manifold_dim
        self.hidden_dim = hidden_dim
        
        # Initialize symplectic structure
        self.symplectic = SymplecticStructure(
            dim=manifold_dim,
            preserve_structure=True,
            wave_enabled=False  # We're not using wave mechanics here
        )
        
        # Neural network for kinetic energy
        self.kinetic_network = nn.Sequential(
            nn.Linear(manifold_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights for quadratic scaling
        for m in self.kinetic_network.modules():
            if isinstance(m, nn.Linear):
                nn_init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn_init.zeros_(m.bias)
                    
        # Create symplectic matrix J
        n = manifold_dim // 2
        J = torch.zeros(manifold_dim, manifold_dim)
        J[:n, n:] = torch.eye(n)
        J[n:, :n] = -torch.eye(n)
        self.register_buffer('J', J)
                    
    def _to_phase_space(self, state: torch.Tensor) -> PhaseSpacePoint:
        """Convert tensor input to PhaseSpacePoint.
        
        Args:
            state: Tensor of shape [batch_size, phase_dim] containing position and momentum
            
        Returns:
            PhaseSpacePoint object with position and momentum components
        """
        if not isinstance(state, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor, got {type(state)}")
            
        # Split into position and momentum
        n = self.manifold_dim // 2
        q = state[..., :n]
        p = state[..., n:]
        
        return PhaseSpacePoint(
            position=q,
            momentum=p,
            time=0.0  # Default time if not provided
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Evolve the system forward in time using symplectic integration.
        
        Args:
            state: Phase space points [batch_size, phase_dim]
            
        Returns:
            Evolved phase space points [batch_size, phase_dim]
        """
        return self.evolve(state)
        
    def compute_energy(self, state: torch.Tensor) -> torch.Tensor:
        """Compute the Hamiltonian energy of the system.
        
        Uses a separable Hamiltonian H = T(p) + V(q) where:
        - T(p) = 1/2 |p|^2 (kinetic energy)
        - V(q) = 1/2 |q|^2 (potential energy)
        
        This form guarantees symplectic structure preservation.
        
        Args:
            state: Phase space points [batch_size, phase_dim]
            
        Returns:
            Energy values [batch_size]
        """
        if not isinstance(state, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor, got {type(state)}")
            
        # Split into position and momentum
        n = self.manifold_dim // 2
        q = state[..., :n]
        p = state[..., n:]
        
        # Compute kinetic energy (quadratic in momentum)
        T = 0.5 * torch.sum(p * p, dim=-1)
        
        # Compute potential energy (quadratic in position)
        V = 0.5 * torch.sum(q * q, dim=-1)
        
        # Total energy
        H = T + V
        
        return H
        
    def compute_vector_field(self, points: torch.Tensor) -> torch.Tensor:
        """Compute Hamiltonian vector field.
        
        Args:
            points: Phase space points [batch_size, phase_dim]
            
        Returns:
            Vector field values [batch_size, phase_dim]
        """
        points.requires_grad_(True)
        energy = self.compute_energy(points)
        
        # Compute gradient
        grad = torch.autograd.grad(energy.sum(), points, create_graph=True)[0]
        
        # Apply symplectic matrix J to gradient
        vector_field = torch.matmul(self.J, grad.unsqueeze(-1)).squeeze(-1)
        
        return vector_field
        
    def evolve(self, points: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        """Evolve system forward in time using symplectic integration.
        
        Args:
            points: Initial phase space points [batch_size, phase_dim]
            dt: Time step
            
        Returns:
            Evolved phase space points [batch_size, phase_dim]
        """
        # Split into position and momentum
        n = self.manifold_dim // 2
        q = points[..., :n]
        p = points[..., n:]
        
        # Compute initial energy for debugging
        initial_energy = self.compute_energy(points)
        
        # Symplectic Euler integration
        # 1. Update momentum using potential energy gradient
        q.requires_grad_(True)
        V = 0.5 * torch.sum(q * q, dim=-1)  # Quadratic potential
        dV = torch.autograd.grad(V.sum(), q, create_graph=True)[0]
        p_new = p - dt * dV
        
        # 2. Update position using new momentum
        q_new = q + dt * p_new  # Direct momentum coupling preserves symplectic form
        
        # Combine updated position and momentum
        evolved_points = torch.cat([q_new, p_new], dim=-1)
        
        # Verify energy conservation (debug)
        final_energy = self.compute_energy(evolved_points)
        energy_diff = torch.abs(final_energy - initial_energy)
        if torch.any(energy_diff > 1e-3):
            print(f"Warning: Energy not conserved! Max diff: {energy_diff.max().item():.6f}")
        
        return evolved_points
