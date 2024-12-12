"""
Riemannian Framework Implementation.

This module implements the core Riemannian geometric structure for pattern spaces,
including metric tensors, connections, and curvature computations.
"""

from dataclasses import dataclass
from typing import Generic, Optional, Protocol, Tuple, TypeVar

import torch
from torch import nn

T = TypeVar("T")


@dataclass
class ChristoffelSymbols(Generic[T]):
    """Christoffel symbols of the Levi-Civita connection."""

    first_kind: T  # Γijk
    second_kind: T  # Γij^k


@dataclass
class CurvatureTensor:
    """Riemann curvature tensor components."""

    riemann: T
    ricci: T
    scalar: T
    dimension: int


class RiemannianFramework(Protocol[T]):
    """Protocol for Riemannian geometric structure."""

    def metric_tensor(self, point: T, vectors: Tuple[T, T]) -> T:
        """Computes the metric tensor at a point."""
        ...

    def christoffel_symbols(self, chart: T) -> ChristoffelSymbols[T]:
        """Computes Christoffel symbols in a chart."""
        ...

    def covariant_derivative(self, vector_field: T, direction: T) -> T:
        """Computes covariant derivative of a vector field."""
        ...

    def geodesic_flow(self, initial_point: T, initial_velocity: T) -> T:
        """Computes geodesic flow from initial conditions."""
        ...

    def curvature_tensor(self, point: T) -> CurvatureTensor:
        """Computes various curvature tensors at a point."""
        ...


class PatternRiemannianStructure(nn.Module):
    """Concrete implementation of Riemannian structure for pattern spaces."""

    def __init__(
        self, manifold_dim: int, rank: Optional[int] = None, device: torch.device = None
    ):
        super().__init__()
        self.manifold_dim = manifold_dim
        self.rank = rank or manifold_dim
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Initialize metric as identity plus low-rank perturbation
        self.metric_factors = nn.Parameter(
            torch.randn(self.rank, manifold_dim, device=self.device, requires_grad=True) * 0.1
        )

        # Initialize connection coefficients
        self.connection_coeffs = nn.Parameter(
            torch.zeros(manifold_dim, manifold_dim, manifold_dim, device=self.device, requires_grad=True)
        )

    def metric_tensor(
        self,
        point: torch.Tensor,
        vectors: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Computes the metric tensor or its action on vectors."""
        # Compute full metric tensor: g = I + V^T V for stability
        metric = torch.eye(self.manifold_dim, device=self.device).unsqueeze(
            0
        ) + torch.matmul(self.metric_factors.T, self.metric_factors)

        if vectors is None:
            return metric

        # Compute metric action on vectors if provided
        v1, v2 = vectors
        return torch.sum(
            torch.matmul(metric, v1.unsqueeze(-1)) * v2.unsqueeze(-1), dim=-1
        )

    def compute_metric(self, points: torch.Tensor) -> torch.Tensor:
        """Compute metric tensor at given points.
        
        Args:
            points: Points tensor of shape (batch_size, manifold_dim)
            
        Returns:
            Metric tensor of shape (batch_size, manifold_dim, manifold_dim)
        """
        batch_size = points.shape[0]
        
        # Ensure points require gradients
        if not points.requires_grad:
            points = points.detach().requires_grad_(True)
        
        # Compute position-dependent factors
        position_factors = torch.einsum('bi,rj->brij', points, self.metric_factors)
        
        # Compute metric using factors: g = I + V^T V where V are the metric factors
        identity = torch.eye(
            self.manifold_dim, 
            device=points.device
        ).expand(batch_size, -1, -1)
        
        perturbation = torch.einsum(
            'brij,brkj->bik', 
            position_factors, 
            position_factors
        )
        
        return identity + perturbation

    def compute_christoffel(self, points: torch.Tensor) -> torch.Tensor:
        """Compute Christoffel symbols using Levi-Civita connection.
        
        Args:
            points: Points tensor (batch_size x manifold_dim)
            
        Returns:
            Christoffel symbols (batch_size x manifold_dim x manifold_dim x manifold_dim)
        """
        batch_size = points.shape[0]
        
        # Resource guard - check input dimensions
        if points.dim() != 2 or points.shape[1] != self.manifold_dim:
            raise ValueError(f"Expected points shape (batch_size, {self.manifold_dim}), got {points.shape}")
            
        # Memory guard - check if computation is feasible
        expected_memory = batch_size * (self.manifold_dim ** 3) * 4  # 4 bytes per float
        if expected_memory > 1e9:  # 1GB limit
            raise RuntimeError(f"Computation would require {expected_memory/1e9:.2f}GB memory")
        
        # Ensure points require gradients
        if not points.requires_grad:
            points = points.detach().requires_grad_(True)
        
        # Get metric and its inverse
        metric = self.compute_metric(points)
        metric_inv = torch.linalg.inv(metric)
        
        # Compute metric gradients more efficiently
        metric_flat = metric.reshape(batch_size, -1)  # [batch_size, manifold_dim * manifold_dim]
        
        # Guard against too many gradient computations
        max_grad_ops = 1000
        if metric_flat.shape[1] > max_grad_ops:
            raise RuntimeError(f"Too many gradient operations required: {metric_flat.shape[1]} > {max_grad_ops}")
        
        # Compute all gradients at once
        grads = []
        with torch.no_grad():
            for i in range(metric_flat.shape[1]):
                grad = torch.autograd.grad(
                    metric_flat[:, i].sum(),
                    points,
                    create_graph=True,
                    retain_graph=True
                )[0]
                grads.append(grad)
        
        # Stack gradients and reshape
        metric_grad = torch.stack(grads, dim=1)
        metric_grad = metric_grad.reshape(batch_size, self.manifold_dim, self.manifold_dim, self.manifold_dim)
        
        # Compute Christoffel symbols using metric and its derivatives
        christoffel = torch.einsum(
            'bim,bjkm->bijk',
            metric_inv,
            0.5 * (
                metric_grad.transpose(-2, -1) +
                metric_grad.transpose(-2, -1).transpose(-3, -2) -
                metric_grad
            )
        )
        
        return christoffel

    def christoffel_symbols(
        self, chart: torch.Tensor
    ) -> ChristoffelSymbols[torch.Tensor]:
        """Computes Christoffel symbols in a chart."""
        # Get metric and its derivatives
        metric = self.metric_tensor(chart)

        # Compute metric derivatives using autograd
        metric.requires_grad_(True)
        grad_metric = torch.autograd.grad(metric.sum(), chart, create_graph=True)[0]

        # Reshape gradient to get metric derivatives
        metric_derivs = grad_metric.view(
            *chart.shape[:-1], self.manifold_dim, self.manifold_dim, self.manifold_dim
        )

        # Compute Christoffel symbols of first kind
        gamma_first = 0.5 * (
            metric_derivs
            + metric_derivs.transpose(-2, -3)
            - metric_derivs.transpose(-2, -1)
        )

        # Compute inverse metric
        metric_inv = torch.inverse(metric)

        # Compute Christoffel symbols of second kind
        gamma_second = torch.einsum("...ij,...jkl->...ikl", metric_inv, gamma_first)

        return ChristoffelSymbols(
            first_kind=gamma_first,
            second_kind=gamma_second,
        )

    def covariant_derivative(
        self, vector_field: torch.Tensor, direction: torch.Tensor
    ) -> torch.Tensor:
        """Computes covariant derivative of a vector field."""
        # Get Christoffel symbols
        christoffel = self.christoffel_symbols(direction)

        # Compute directional derivative
        dir_deriv = torch.autograd.grad(
            vector_field,
            direction,
            grad_outputs=torch.ones_like(vector_field),
            create_graph=True,
        )[0]

        # Add connection term
        connection_term = torch.einsum(
            "...ijk,...j,...k->...i", christoffel.second_kind, vector_field, direction
        )

        return dir_deriv + connection_term

    def geodesic_flow(
        self,
        initial_point: torch.Tensor,
        initial_velocity: torch.Tensor,
        num_steps: int = 100,
        step_size: float = 0.01,
    ) -> torch.Tensor:
        """Computes geodesic flow using numerical integration."""
        # Initialize trajectory
        trajectory = [initial_point]
        velocity = initial_velocity

        for _ in range(num_steps):
            # Get current point and velocity
            current_point = trajectory[-1]

            # Get Christoffel symbols
            christoffel = self.christoffel_symbols(current_point)

            # Compute acceleration using geodesic equation
            acceleration = -torch.einsum(
                "...ijk,...j,...k->...i", christoffel.second_kind, velocity, velocity
            )

            # Update velocity using acceleration
            velocity = velocity + step_size * acceleration

            # Update position using velocity
            new_point = current_point + step_size * velocity
            trajectory.append(new_point)

        return torch.stack(trajectory, dim=0)

    def compute_riemann(self, points: torch.Tensor) -> torch.Tensor:
        """Compute Riemann curvature tensor.
        
        Args:
            points: Points tensor (batch_size x manifold_dim)
            
        Returns:
            Riemann curvature tensor (batch_size x manifold_dim x manifold_dim x manifold_dim x manifold_dim)
        """
        batch_size = points.shape[0]
        
        # Get Christoffel symbols and their derivatives
        christoffel = self.compute_christoffel(points)
        
        # Enable gradients for computing Christoffel derivatives
        points.requires_grad_(True)
        christoffel_with_grad = self.compute_christoffel(points)
        
        # Initialize storage for Christoffel derivatives
        christoffel_grad = torch.zeros(
            batch_size,
            self.manifold_dim,
            self.manifold_dim,
            self.manifold_dim,
            self.manifold_dim,
            device=points.device
        )
        
        # Compute partial derivatives of Christoffel symbols
        for l in range(self.manifold_dim):
            grad_l = torch.autograd.grad(
                christoffel_with_grad[..., l].sum(),
                points,
                create_graph=True,
                allow_unused=True,
                retain_graph=True
            )[0]
            if grad_l is not None:
                christoffel_grad[..., l] = grad_l
        
        points.requires_grad_(False)
        
        # Compute Riemann tensor
        # R^i_jkl = ∂_k Γ^i_jl - ∂_l Γ^i_jk + Γ^i_mk Γ^m_jl - Γ^i_ml Γ^m_jk
        riemann = (
            christoffel_grad[..., :, None, :] 
            - christoffel_grad[..., None, :, :]
        )
        
        # Add contraction terms
        riemann = riemann + torch.einsum(
            'bimk,bmjl->bijkl',
            christoffel,
            christoffel
        ) - torch.einsum(
            'biml,bmjk->bijkl',
            christoffel,
            christoffel
        )
        
        return riemann

    def curvature_tensor(self, point: torch.Tensor) -> CurvatureTensor:
        """Computes curvature tensors at a point."""
        # Get Christoffel symbols and their derivatives
        christoffel = self.christoffel_symbols(point)
        gamma = christoffel.second_kind

        # Compute Christoffel symbol derivatives
        gamma.requires_grad_(True)
        gamma_grad = torch.autograd.grad(gamma.sum(), point, create_graph=True)[0]

        # Reshape to get Christoffel derivatives
        gamma_derivs = gamma_grad.view(
            *point.shape[:-1],
            self.manifold_dim,
            self.manifold_dim,
            self.manifold_dim,
            self.manifold_dim,
        )

        # Compute Riemann tensor
        riemann = (
            gamma_derivs
            - gamma_derivs.transpose(-2, -3)
            + torch.einsum("...ijm,...klm->...ijkl", gamma, gamma)
            - torch.einsum("...ikm,...jlm->...ijkl", gamma, gamma)
        )

        # Compute Ricci tensor by contracting Riemann
        ricci = torch.einsum("...ijij->...ij", riemann)

        # Compute scalar curvature by contracting Ricci with inverse metric
        metric_inv = torch.inverse(self.metric_tensor(point))
        scalar = torch.einsum("...ij,...ij->...", metric_inv, ricci)

        return CurvatureTensor(
            riemann=riemann, ricci=ricci, scalar=scalar, dimension=self.manifold_dim
        )

    def sectional_curvature(
        self, point: torch.Tensor, plane: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Computes sectional curvature for a 2-plane at a point."""
        v1, v2 = plane

        # Get curvature tensor
        curvature = self.curvature_tensor(point)

        # Compute numerator: <R(X,Y)Y,X>
        numerator = torch.einsum(
            "...ijkl,...i,...j,...k,...l->...", curvature.riemann, v1, v2, v2, v1
        )

        # Compute denominator: |X∧Y|^2
        g11 = self.metric_tensor(point, (v1, v1))
        g12 = self.metric_tensor(point, (v1, v2))
        g22 = self.metric_tensor(point, (v2, v2))
        denominator = g11 * g22 - g12 * g12

        return numerator / (denominator + 1e-8)  # Add small epsilon for stability
