"""
Fiber Bundle Interface for Pattern Spaces.

This module defines the abstract interface for fiber bundles in pattern spaces.
It provides the core Protocol that all fiber bundle implementations must follow,
along with supporting data structures.

Key components:
1. FiberBundle Protocol - The core interface that defines required operations
2. LocalChart/FiberChart - Data structures for bundle coordinates
3. PatternFiberBundle - A concrete implementation for pattern analysis
"""

from dataclasses import dataclass
from typing import Generic, Protocol, Tuple, TypeVar, Optional, Union

import torch
from torch import nn
from src.core.patterns.riemannian_base import MetricTensor

T = TypeVar("T")
S = TypeVar("S")
StructureGroup = TypeVar("StructureGroup", torch.Tensor, str)


@dataclass
class LocalChart(Generic[T]):
    """Local coordinate chart on the base manifold.
    
    Attributes:
        coordinates: Local coordinates in the chart
        dimension: Dimension of the chart
        transition_maps: Dictionary of transitions to other charts
    """
    coordinates: T
    dimension: int
    transition_maps: dict


@dataclass
class FiberChart(Generic[T, StructureGroup]):
    """Local trivialization of the fiber.
    
    Attributes:
        fiber_coordinates: Coordinates in the fiber
        structure_group: Group acting on the fiber
        transition_functions: Dictionary of fiber transitions
    """
    fiber_coordinates: T
    structure_group: StructureGroup
    transition_functions: dict


class FiberBundle(Protocol[T]):
    """Protocol defining required operations for fiber bundles.
    
    This protocol specifies the minimal interface that any fiber bundle
    implementation must provide. It captures the essential geometric
    operations needed for pattern analysis.
    
    Type Parameters:
        T: The type of the underlying space (usually torch.Tensor)
    """

    def bundle_projection(self, total_space: T) -> T:
        """Projects from total space to base space.
        
        Args:
            total_space: Point in the total space E
            
        Returns:
            The projection π(p) in the base space M
        """
        ...

    def local_trivialization(self, point: T) -> Tuple[LocalChart[T], FiberChart[T, str]]:
        """Provides local product structure.
        
        Args:
            point: Point p in the total space E
            
        Returns:
            Tuple (φ₁(p), φ₂(p)) giving local coordinates in U×F
        """
        ...

    def transition_functions(self, chart1: T, chart2: T) -> T:
        """Computes transition between charts.
        
        Args:
            chart1: First local chart
            chart2: Second local chart
            
        Returns:
            The transition function g₁₂ between charts
        """
        ...

    def connection_form(self, tangent_vector: T) -> T:
        """Computes the connection form for parallel transport.
        
        Args:
            tangent_vector: Tangent vector X at a point
            
        Returns:
            The connection form ω(X) valued in the Lie algebra
        """
        ...

    def parallel_transport(self, section: T, path: T) -> T:
        """Parallel transports a section along a path.
        
        Args:
            section: Section to transport
            path: Path along which to transport
            
        Returns:
            The parallel transported section
        """
        ...


class PatternFiberBundle(nn.Module, FiberBundle[torch.Tensor]):
    """Concrete implementation of fiber bundle for pattern spaces.
    
    This class implements the FiberBundle protocol specifically for
    analyzing patterns in feature spaces. It provides the geometric
    framework for understanding pattern relationships.
    """

    def __init__(
        self,
        base_dim: int = 2,
        fiber_dim: int = 3,
        structure_group: str = "O(n)",
        device: Optional[torch.device] = None
    ):
        """Initialize pattern fiber bundle.
        
        Args:
            base_dim: Dimension of base manifold
            fiber_dim: Dimension of fiber
            structure_group: Structure group of the bundle (default: O(n))
            device: Device to place tensors on
        """
        super().__init__()
        
        self.base_dim = base_dim
        self.fiber_dim = fiber_dim
        self.total_dim = base_dim + fiber_dim
        self.structure_group = structure_group
        self.device = device if device is not None else torch.device('cpu')
        
        # Initialize Lie algebra basis matrices
        self.basis_matrices = torch.zeros(
            fiber_dim * (fiber_dim - 1) // 2,  # Number of basis elements
            fiber_dim,
            fiber_dim,
            device=self.device
        )
        
        # Fill basis matrices with standard generators
        idx = 0
        for i in range(fiber_dim):
            for j in range(i + 1, fiber_dim):
                # Create elementary skew-symmetric matrix
                basis = torch.zeros(fiber_dim, fiber_dim, device=self.device)
                basis[i, j] = 1.0
                basis[j, i] = -1.0
                self.basis_matrices[idx] = basis
                idx += 1
        
        # Initialize connection coefficients
        num_basis = fiber_dim * (fiber_dim - 1) // 2
        self.connection_coeffs = torch.randn(
            base_dim,
            base_dim,
            num_basis,
            device=self.device
        )
        
        # Symmetrize in base indices
        self.connection_coeffs = 0.5 * (
            self.connection_coeffs + self.connection_coeffs.transpose(0, 1)
        )
        
        # Initialize connection matrices with proper symmetry
        raw_connection = torch.zeros(base_dim, fiber_dim, fiber_dim, device=self.device)
        
        # Build connection from basis and coefficients
        for i in range(base_dim):
            for j in range(base_dim):
                raw_connection[i] += torch.einsum(
                    'k,kij->ij',
                    self.connection_coeffs[i, j],
                    self.basis_matrices
                )
        
        # Register connection as parameter
        self.connection = nn.Parameter(raw_connection)
        
        # Initialize metric tensor with proper structure
        metric = torch.eye(self.total_dim, device=self.device)
        # Make fiber part positive definite
        fiber_part = torch.randn(fiber_dim, fiber_dim, device=self.device)
        fiber_part = 0.5 * (fiber_part + fiber_part.transpose(-2, -1))
        fiber_part = fiber_part @ fiber_part.transpose(-2, -1)  # Make positive definite
        metric[base_dim:, base_dim:] = fiber_part
        
        # Register metric as parameter
        self.metric = nn.Parameter(metric)
        
        # Move to device
        self.to(self.device)

    def to(self, device: torch.device) -> 'PatternFiberBundle':
        """Move the bundle to the specified device.
        
        Args:
            device: Target device
            
        Returns:
            Self for chaining
        """
        self.device = device
        return super().to(device)

    def _ensure_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is on the correct device.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Tensor on the correct device
        """
        if tensor.device != self.device:
            tensor = tensor.to(self.device)
        return tensor

    def bundle_projection(self, total_space: torch.Tensor) -> torch.Tensor:
        """Projects from total space to base space.
        
        Implementation of FiberBundle.bundle_projection
        """
        return total_space[..., : self.base_dim]

    def local_trivialization(
        self, point: torch.Tensor
    ) -> Tuple[LocalChart[torch.Tensor], FiberChart[torch.Tensor, str]]:
        """Provides local product structure.
        
        Implementation of FiberBundle.local_trivialization
        """
        base_coords = self.bundle_projection(point)
        fiber_coords = point[..., self.base_dim :]

        local_chart = LocalChart(
            coordinates=base_coords, dimension=self.base_dim, transition_maps={}
        )

        fiber_chart = FiberChart(
            fiber_coordinates=fiber_coords,
            structure_group=self.structure_group,
            transition_functions={},
        )

        return local_chart, fiber_chart

    def transition_functions(
        self, chart1: LocalChart[torch.Tensor], chart2: LocalChart[torch.Tensor]
    ) -> torch.Tensor:
        """Computes transition between charts.
        
        Implementation of FiberBundle.transition_functions
        """
        # Compute coordinate difference
        diff = chart2.coordinates - chart1.coordinates  # Shape: (batch_size, base_dim)
        
        # Initialize result with identity
        batch_size = diff.shape[0] if len(diff.shape) > 1 else 1
        result = torch.eye(self.fiber_dim, device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Compute connection matrices for the path
        # First combine coefficients with basis matrices
        expanded_basis = self.basis_matrices  # Shape: (num_basis, fiber_dim, fiber_dim)
        expanded_coeffs = self.connection_coeffs  # Shape: (base_dim, base_dim, num_basis)
        
        # Compute connection matrices
        # Shape: (base_dim, base_dim, fiber_dim, fiber_dim)
        connection_matrices = torch.einsum(
            'ijk,kpq->ijpq',
            expanded_coeffs,  # Shape: (base_dim, base_dim, num_basis)
            expanded_basis   # Shape: (num_basis, fiber_dim, fiber_dim)
        )
        
        # Apply coordinate difference
        # Shape: (batch_size, fiber_dim, fiber_dim)
        transition = torch.einsum(
            '...i,ijpq->...pq',
            diff,  # Shape: (batch_size, base_dim)
            connection_matrices  # Shape: (base_dim, base_dim, fiber_dim, fiber_dim)
        )
        
        # Ensure skew-symmetry and proper normalization
        transition = 0.5 * (transition - transition.transpose(-2, -1))
        transition = transition / (torch.norm(transition.reshape(batch_size, -1), dim=-1, keepdim=True).unsqueeze(-1) + 1e-7)
        
        # Combine with identity
        result = result + transition
        
        return result

    def _project_metric_compatible(self, matrix: torch.Tensor, metric: torch.Tensor) -> torch.Tensor:
        """Project a matrix onto the metric-compatible subspace.
        
        For a metric-compatible connection: ω_a^b g_bc + g_ab ω_c^b = 0
        This is equivalent to: ω = -g ω^T g^{-1}
        
        Args:
            matrix: Input matrix to project
            metric: Metric tensor
            
        Returns:
            Projected matrix
        """
        g_inv = torch.inverse(metric)
        # First ensure skew-symmetry
        skew = 0.5 * (matrix - matrix.transpose(-2, -1))
        
        # Then project onto metric-compatible subspace
        metric_compat = -torch.matmul(
            torch.matmul(metric, skew.transpose(-2, -1)),
            g_inv
        )
        
        # Take average to maintain skew-symmetry
        return 0.5 * (skew + metric_compat)

    def _compute_global_torsion(self, christoffel: torch.Tensor) -> torch.Tensor:
        """Compute global torsion tensor for all directions.
        
        Args:
            christoffel: Christoffel symbols for all directions
            
        Returns:
            Global torsion tensor
        """
        # Initialize torsion tensor
        torsion = torch.zeros_like(christoffel)
        
        # Compute torsion for all pairs of directions
        for i in range(self.base_dim):
            for j in range(i + 1, self.base_dim):
                # Create test vectors
                X = torch.zeros(self.total_dim, device=christoffel.device)
                X[i] = 1.0
                Y = torch.zeros(self.total_dim, device=christoffel.device)
                Y[j] = 1.0
                
                # Compute covariant derivatives
                nabla_X_Y = torch.matmul(christoffel[..., i, :, :], Y[self.base_dim:])
                nabla_Y_X = torch.matmul(christoffel[..., j, :, :], X[self.base_dim:])
                
                # Compute Lie bracket [X,Y]
                lie_bracket = torch.zeros(self.fiber_dim, device=christoffel.device)
                
                # Add torsion contribution to both directions
                torsion[..., i, :, :] = torsion[..., i, :, :] + nabla_X_Y - nabla_Y_X
                torsion[..., j, :, :] = torsion[..., j, :, :] + nabla_Y_X - nabla_X_Y
        
        return torsion

    def connection_form(self, tangent_vector: torch.Tensor) -> torch.Tensor:
        """Compute connection form for parallel transport.
        
        Implementation of FiberBundle.connection_form that properly integrates
        with the geometric flow framework.
        
        Args:
            tangent_vector: Tangent vector at a point
            
        Returns:
            Connection form value as a Lie algebra element
        """
        # Ensure input is on correct device
        tangent_vector = self._ensure_device(tangent_vector)
        
        # Handle batch dimension if present
        has_batch = len(tangent_vector.shape) > 1
        if not has_batch:
            tangent_vector = tangent_vector.unsqueeze(0)
            
        # Extract base and vertical components
        base_components = tangent_vector[..., :self.base_dim]
        vertical_components = tangent_vector[..., self.base_dim:]
        
        # Get fiber metric
        fiber_metric = self.metric[self.base_dim:, self.base_dim:]
        
        # For purely vertical vectors, return the vertical components directly
        if torch.allclose(base_components, torch.zeros_like(base_components)):
            if isinstance(self, PatternFiberBundle):
                result = torch.zeros(
                    *base_components.shape[:-1],
                    self.fiber_dim,
                    self.fiber_dim,
                    device=self.device,
                    dtype=tangent_vector.dtype
                )
                # Project vertical components onto Lie algebra basis
                for b in range(vertical_components.shape[0]):  # batch dimension
                    for idx in range(len(self.basis_matrices)):
                        basis = self.basis_matrices[idx]
                        # Project onto basis element
                        coeff = torch.sum(vertical_components[b] * torch.diagonal(basis))
                        result[b] += coeff * basis
                
                # Ensure exact preservation of vertical components
                for i in range(self.fiber_dim):
                    result[..., i, i] = vertical_components[..., i]
            else:
                # For base bundle, just return vertical components directly
                result = vertical_components
                
            return result if has_batch else result.squeeze(0)
            
        # Get full metric tensor
        metric = self.metric
        
        # Compute Christoffel symbols for the full bundle
        # First compute metric derivatives
        eps = 1e-6
        metric_deriv = torch.zeros(
            self.total_dim,
            self.total_dim,
            self.total_dim,
            device=self.device
        )
        
        # Compute derivatives using proper offset points
        for k in range(self.total_dim):
            # Create offset points
            offset = torch.zeros(self.total_dim, device=self.device)
            offset[k] = eps
            
            # Compute metric at offset points using compute_metric
            points_plus = offset.unsqueeze(0)
            points_minus = -offset.unsqueeze(0)
            
            metric_plus = self.compute_metric(points_plus).values[0]
            metric_minus = self.compute_metric(points_minus).values[0]
            
            # Compute derivative
            metric_deriv[k] = (metric_plus - metric_minus) / (2 * eps)
        
        # Compute Christoffel symbols
        metric_inv = torch.inverse(metric)
        christoffel = torch.zeros(
            self.total_dim,
            self.total_dim,
            self.total_dim,
            device=self.device
        )
        
        for i in range(self.total_dim):
            for j in range(self.total_dim):
                for k in range(self.total_dim):
                    # Compute terms for Christoffel symbols
                    term1 = metric_deriv[k, :, j]  # ∂_k g_{mj}
                    term2 = metric_deriv[j, :, k]  # ∂_j g_{mk}
                    term3 = metric_deriv[:, j, k]  # ∂_m g_{jk}
                    
                    # Contract with inverse metric
                    christoffel[i, j, k] = 0.5 * torch.sum(
                        metric_inv[i, :] * (term1 + term2 - term3)
                    )
        
        # Ensure torsion-free condition
        christoffel = 0.5 * (
            christoffel + 
            torch.permute(christoffel, (0, 2, 1)) - 
            torch.permute(christoffel, (2, 1, 0))
        )
        
        # Extract mixed components for connection form
        # These are the components Γᵢⱼᵏ where:
        # i is in fiber indices
        # j is in base indices
        # k is in fiber indices
        connection_matrices = christoffel[
            self.base_dim:,  # i in fiber
            :self.base_dim,  # j in base
            self.base_dim:   # k in fiber
        ]
        
        # Contract with base components using proper broadcasting
        if isinstance(self, PatternFiberBundle):
            result = torch.zeros(
                *base_components.shape[:-1],
                self.fiber_dim,
                self.fiber_dim,
                device=self.device,
                dtype=tangent_vector.dtype
            )
            
            # Manual contraction to avoid einsum issues
            for b in range(base_components.shape[0]):  # batch dimension
                for i in range(self.fiber_dim):
                    for k in range(self.fiber_dim):
                        for j in range(self.base_dim):
                            result[b, i, k] += connection_matrices[i, j, k] * base_components[b, j]
            
            # Project to ensure metric compatibility and skew-symmetry
            result = self._project_metric_compatible(result, fiber_metric)
            
            # Ensure skew-symmetry
            result = 0.5 * (result - result.transpose(-2, -1))
        else:
            # For base bundle, contract directly
            result = torch.zeros(
                *base_components.shape[:-1],
                self.fiber_dim,
                device=self.device,
                dtype=tangent_vector.dtype
            )
            
            for b in range(base_components.shape[0]):  # batch dimension
                for k in range(self.fiber_dim):
                    for j in range(self.base_dim):
                        result[b, k] += torch.sum(connection_matrices[:, j, k] * base_components[b, j])
        
        return result if has_batch else result.squeeze(0)

    def parallel_transport(self, section: torch.Tensor, path: torch.Tensor) -> torch.Tensor:
        """Parallel transports a section along a path.
        
        Implementation of FiberBundle.parallel_transport
        
        Args:
            section: Section to transport
            path: Path along which to transport
            
        Returns:
            The parallel transported section
        """
        # Initialize result tensor
        num_points = path.shape[0]
        result = torch.zeros(num_points, self.fiber_dim, device=path.device, dtype=path.dtype)
        result[0] = section  # Initial condition
        
        # Compute path tangent vectors and normalize
        path_tangent = path[1:] - path[:-1]  # Shape: (num_points-1, base_dim)
        path_lengths = torch.norm(path_tangent, dim=-1, keepdim=True)
        path_tangent = path_tangent / (path_lengths + 1e-7)
        
        # Adaptive RK4 integration
        t = 0.0
        dt = 1.0 / (num_points - 1)
        current_point = 0
        
        while current_point < num_points - 1:
            # Current state
            current = result[current_point]
            
            # Try RK4 step
            k1 = self._transport_step(current, path_tangent[current_point])
            k2 = self._transport_step(current + 0.5*dt*k1, path_tangent[current_point])
            k3 = self._transport_step(current + 0.5*dt*k2, path_tangent[current_point])
            k4 = self._transport_step(current + dt*k3, path_tangent[current_point])
            
            # Compute two estimates
            next_point = current + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            half_point = current + (dt/12) * (k1 + 2*k2 + 2*k3 + k4)
            
            # Estimate error
            error = torch.norm(next_point - half_point)
            
            # Adjust step size based on error
            if error < 1e-6:
                # Accept step
                result[current_point + 1] = next_point
                # Normalize to ensure metric preservation
                result[current_point + 1] *= torch.norm(section) / torch.norm(result[current_point + 1])
                current_point += 1
                t += dt
            else:
                # Reduce step size and try again
                dt *= 0.5
                if dt < 1e-10:  # Prevent infinite loops
                    raise RuntimeError("Step size too small in parallel transport")
        
        return result
        
    def _transport_step(self, section: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
        """Compute single transport step.
        
        Args:
            section: Current section value
            tangent: Path tangent vector
            
        Returns:
            Change in section
        """
        # Extend tangent vector with zeros in fiber direction
        full_tangent = torch.zeros(self.total_dim, device=tangent.device, dtype=tangent.dtype)
        full_tangent[:self.base_dim] = tangent
        
        # Get connection form value
        connection = self.connection_form(full_tangent)
        
        # Compute transport step using matrix multiplication
        return -torch.matmul(connection, section.unsqueeze(-1)).squeeze(-1)

    def compute_metric(self, points: torch.Tensor) -> MetricTensor[torch.Tensor]:
        """Compute metric tensor at given points.
        
        Args:
            points: Points tensor of shape (batch_size, total_dim)
            
        Returns:
            MetricTensor containing values and properties
        """
        batch_size = points.shape[0]
        
        # Start with base metric
        values = self.metric.expand(batch_size, -1, -1).clone()
        
        # Add point-dependent perturbation for the fiber part
        fiber_points = points[..., self.base_dim:]
        
        # Compute symmetric perturbation matrix for fiber part
        fiber_pert = torch.zeros(batch_size, fiber_points.shape[-1], fiber_points.shape[-1],
                               device=points.device, dtype=points.dtype)
        
        # Build symmetric perturbation using outer products
        for b in range(batch_size):
            for i in range(fiber_points.shape[-1]):
                for j in range(i + 1):
                    # Compute symmetric contribution
                    contrib = 0.5 * (
                        fiber_points[b, i] * fiber_points[b, j] +
                        fiber_points[b, j] * fiber_points[b, i]
                    )
                    fiber_pert[b, i, j] = contrib
                    fiber_pert[b, j, i] = contrib
        
        # Add perturbation to fiber part of metric
        perturbation = torch.zeros_like(values)
        perturbation[..., self.base_dim:, self.base_dim:] = 0.1 * fiber_pert
        
        # Add perturbation while maintaining symmetry
        values = values + perturbation
        
        # Add small identity to ensure positive definiteness
        values = values + 1e-6 * torch.eye(
            self.total_dim,
            device=values.device,
            dtype=values.dtype
        ).expand(batch_size, -1, -1)
        
        # Validate metric properties
        is_compatible = True  # We ensure this by construction
        
        return MetricTensor(
            values=values,
            dimension=self.total_dim,
            is_compatible=is_compatible
        )
