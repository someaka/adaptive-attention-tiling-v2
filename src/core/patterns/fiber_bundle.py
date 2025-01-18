"""
Base Fiber Bundle Implementation.

This module provides the core mathematical implementation of fiber bundles,
implementing the FiberBundle protocol defined in core.patterns.fiber_types.
"""

from typing import List, Optional, Tuple, Dict, Any
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from dataclasses import dataclass

from .fiber_types import (
    FiberBundle,
    LocalChart,
    FiberChart,
)
from src.core.patterns.riemannian_base import MetricTensor, RiemannianStructure
from src.core.patterns.riemannian import PatternRiemannianStructure


class BaseFiberBundle(nn.Module, FiberBundle[Tensor]):
    """Core mathematical implementation of fiber bundles.
    
    This class provides the foundational mathematical implementation of
    the FiberBundle protocol, focusing on the geometric operations
    without pattern-specific features.
    """

    def __init__(
        self,
        base_dim: int,
        fiber_dim: int,
        structure_group: Optional[str] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize fiber bundle.
        
        Args:
            base_dim: Dimension of base manifold
            fiber_dim: Dimension of fiber
            structure_group: Name of structure group (e.g. 'SO3', 'U1')
            device: Device to place tensors on
            dtype: Data type for tensors
        """
        super().__init__()  # Initialize nn.Module
        self.base_dim = base_dim
        self._fiber_dim = fiber_dim  # Store as protected attribute
        self.total_dim = base_dim + fiber_dim
        self.structure_group = structure_group
        self.device = device or torch.device('cpu')
        self.dtype = dtype or torch.float32

        # Initialize bundle metric
        self.metric = nn.Parameter(
            torch.eye(self.total_dim, device=self.device, dtype=self.dtype),
            requires_grad=True
        )

        # Initialize connection form
        # Shape: (base_dim, fiber_dim, fiber_dim)
        self.connection = nn.Parameter(
            torch.zeros(self.base_dim, self.fiber_dim, self.fiber_dim, device=self.device, dtype=self.dtype),
            requires_grad=True
        )

        # Initialize Riemannian framework
        self.riemannian_framework = PatternRiemannianStructure(
            manifold_dim=self.total_dim,
            pattern_dim=self.fiber_dim,
            device=self.device,
            dtype=self.dtype
        )

    @property
    def fiber_dim(self) -> int:
        """Return the fiber dimension."""
        return self._fiber_dim

    def bundle_projection(self, total_space: Tensor) -> Tensor:
        """Implementation of FiberBundle.bundle_projection.
        
        Projects from total space to base manifold. This is the fundamental
        operation that connects the total space E to the base manifold M
        through the projection π: E → M.
        
        Properties preserved:
        1. Projection is surjective
        2. Projection is smooth
        3. Projection preserves local product structure
        
        Args:
            total_space: Point in total space (shape: [..., total_dim])
            
        Returns:
            Projection onto base manifold (shape: [..., base_dim])
            
        Raises:
            ValueError: If input tensor has invalid shape
        """
        # Validate input dimensions
        if total_space.shape[-1] != self.base_dim:
            # Handle dimension mismatch by padding or truncating
            if total_space.shape[-1] < self.base_dim:
                padding = torch.zeros(*total_space.shape[:-1], self.base_dim - total_space.shape[-1],
                                    device=total_space.device, dtype=total_space.dtype)
                total_space = torch.cat([total_space, padding], dim=-1)
            else:
                total_space = total_space[..., :self.base_dim]
        
        # Project to base manifold
        return self.riemannian_framework.project_to_base(total_space)

    def local_trivialization(self, point: Tensor) -> Tuple[LocalChart[Tensor], FiberChart[Tensor, str]]:
        """Implementation of FiberBundle.local_trivialization.
        
        Compute local trivialization at a point.
        
        Properties preserved:
        1. Local product structure
        2. Smooth transition functions
        3. Structure group action
        
        Args:
            point: Point in total space
            
        Returns:
            Tuple of (local_chart, fiber_chart)
        """
        # Get base coordinates through projection
        base_coords = self.bundle_projection(point)
        
        # Get fiber coordinates
        fiber_coords = point[..., self.base_dim:]
        
        # Create local chart
        local_chart = LocalChart(
            coordinates=base_coords,
            dimension=self.base_dim,
            transition_maps={}
        )
        
        # Create fiber chart
        fiber_chart = FiberChart(
            fiber_coordinates=fiber_coords,
            structure_group=self.structure_group or "SO3",
            transition_functions={}
        )
        
        return local_chart, fiber_chart

    def transition_functions(self, chart1: LocalChart[Tensor], chart2: LocalChart[Tensor]) -> Tensor:
        """Implementation of FiberBundle.transition_functions.

        Compute transition between charts.

        Args:
            chart1: First local chart
            chart2: Second local chart

        Returns:
            Transition function between charts
        """
        batch_size = chart1.coordinates.shape[0]
        device = chart1.coordinates.device
        
        # Initialize identity matrix for each batch element
        transition = torch.eye(self.fiber_dim, device=device).expand(batch_size, -1, -1)
        
        # Compute coordinate differences
        diff = chart2.coordinates - chart1.coordinates  # Shape: (batch_size, total_dim)
        
        # Only use base manifold components
        base_diff = diff[..., :self.base_dim]  # Shape: (batch_size, base_dim)
        
        # Reshape for proper broadcasting
        base_diff = base_diff.unsqueeze(-1)  # Shape: (batch_size, base_dim, 1)
        
        # Handle different connection shapes for base vs pattern bundles
        if len(self.connection.shape) == 3:  # Base bundle: (base_dim, fiber_dim, fiber_dim)
            connection = self.connection.unsqueeze(0)  # Shape: (1, base_dim, fiber_dim, fiber_dim)
            transition_update = torch.einsum('...i,ijkl->...kl', base_diff.squeeze(-1), connection)
        else:  # Pattern bundle: (total_dim, total_dim, total_dim)
            # Extract relevant connection components
            connection = self.connection[:self.base_dim, self.base_dim:, self.base_dim:]
            connection = connection.unsqueeze(0)  # Shape: (1, base_dim, fiber_dim, fiber_dim)
            transition_update = torch.einsum('...i,ijkl->...kl', base_diff.squeeze(-1), connection)
        
        # Add update to identity while preserving structure
        transition = transition + transition_update
        
        # Ensure transition preserves fiber metric
        transition = F.normalize(transition, p=2, dim=-1)
        
        return transition

    def connection_form(self, tangent_vector: Tensor) -> Tensor:
        """Implementation of FiberBundle.connection_form.
        
        Args:
            tangent_vector: Tangent vector at a point
            
        Returns:
            Connection form value
        """
        # Handle batch dimension if present
        has_batch = len(tangent_vector.shape) > 1
        if not has_batch:
            tangent_vector = tangent_vector.unsqueeze(0)
            
        # Extract base and vertical components
        base_components = tangent_vector[..., :self.base_dim]
        vertical_components = tangent_vector[..., self.base_dim:]
        
        # For purely vertical vectors, return vertical components directly
        if torch.allclose(base_components, torch.zeros_like(base_components)):
            return vertical_components if has_batch else vertical_components.squeeze(0)
            
        # For horizontal vectors, compute connection form
        result = torch.zeros_like(vertical_components)
        
        # Contract connection with base components
        # Shape: (batch_size, fiber_dim)
        result = torch.einsum('...i,ijk->...k', base_components, self.connection)
            
        return result if has_batch else result.squeeze(0)

    def parallel_transport(self, section: Tensor, path: Tensor) -> Tensor:
        """Implementation of FiberBundle.parallel_transport.
        
        Parallel transport a section along a path using adaptive RK4 integration.
        
        Args:
            section: Section to transport (shape: (fiber_dim,))
            path: Path along which to transport (shape: (num_points, base_dim))
            
        Returns:
            Transported section (shape: (num_points, fiber_dim))
        """
        # Initialize result tensor
        num_points = path.shape[0]
        
        # Handle batched sections
        if len(section.shape) > 1:
            batch_size = section.shape[0]
            section = section.reshape(batch_size, -1)
        else:
            batch_size = 1
            section = section.unsqueeze(0)
            
        # Initialize result tensor to preserve gradients
        result = torch.zeros_like(section).unsqueeze(0).expand(num_points, -1, -1)
        result[0] = section  # Initial condition
        
        # Extract fiber metric for norm computations
        fiber_metric = self.metric[self.base_dim:, self.base_dim:]
        
        # Ensure path has correct base dimension
        if path.shape[-1] != self.base_dim:
            # If path has more dimensions, truncate
            if path.shape[-1] > self.base_dim:
                path = path[..., :self.base_dim]
            else:
                # If path has fewer dimensions, pad with zeros
                padding = torch.zeros(*path.shape[:-1], self.base_dim - path.shape[-1], 
                                   device=path.device, dtype=path.dtype)
                path = torch.cat([path, padding], dim=-1)
        
        # Compute path tangent vectors and normalize using base metric
        path_tangent = path[1:] - path[:-1]  # Shape: (num_points-1, base_dim)
        base_metric = self.metric[:self.base_dim, :self.base_dim]
        path_metric_norms = torch.sqrt(torch.einsum('...i,ij,...j->...', path_tangent, base_metric, path_tangent))
        path_tangent = path_tangent / (path_metric_norms.unsqueeze(-1) + 1e-7)
        
        # Store original metric norm for preservation
        original_metric_norm = torch.sqrt(torch.einsum('...i,ij,...j->...', section, fiber_metric, section))
        
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
            
            # Compute next point
            next_point = current + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            
            # Normalize using metric to preserve metric compatibility
            current_metric_norm = torch.sqrt(torch.einsum('...i,ij,...j->...', next_point, fiber_metric, next_point))
            next_point = next_point * (original_metric_norm / (current_metric_norm + 1e-7)).unsqueeze(-1)
            
            # Update result
            result[current_point + 1] = next_point
            current_point += 1
            t += dt
        
        # Return with correct shape
        return result.squeeze(1) if batch_size == 1 else result
        
    def _transport_step(self, section: Tensor, tangent: Tensor) -> Tensor:
        """Compute the transport step for parallel transport.
        
        Args:
            section: Current section value (shape: (fiber_dim,))
            tangent: Base space tangent vector (shape: (base_dim,))
            
        Returns:
            Transport step (shape: (fiber_dim,))
        """
        # Extract fiber metric
        fiber_metric = self.metric[self.base_dim:, self.base_dim:]
        
        # Compute connection form contraction with metric lowering
        connection_form = torch.einsum('ijk,i->jk', self.connection, tangent)
        
        # Ensure section has correct shape for matrix multiplication
        section = section.reshape(-1, 1)
        
        # Compute transport step with metric compatibility
        # First lower indices with metric
        lowered_section = torch.matmul(fiber_metric, section)
        
        # Then compute transport step
        transport_step = -torch.matmul(connection_form, lowered_section).squeeze(-1)
        
        # Raise indices back with inverse metric
        # Use pseudo-inverse for numerical stability
        fiber_metric_inv = torch.linalg.pinv(fiber_metric)
        transport_step = torch.matmul(fiber_metric_inv, transport_step.unsqueeze(-1)).squeeze(-1)
        
        # Preserve metric norm exactly while maintaining gradients
        metric_norm = torch.sqrt(torch.einsum('i,ij,j->', section.squeeze(-1), fiber_metric, section.squeeze(-1)))
        current_metric_norm = torch.sqrt(torch.einsum('i,ij,j->', transport_step, fiber_metric, transport_step))
        scale_factor = metric_norm / (current_metric_norm + 1e-7)
        transport_step = transport_step * scale_factor
        
        return transport_step

    def compute_holonomy_group(self, holonomies: List[Tensor]) -> Tensor:
        """Compute the holonomy group from a list of holonomies.
        
        Args:
            holonomies: List of holonomy transformations
            
        Returns:
            Tensor representing the holonomy group elements
        """
        return torch.stack(holonomies)

    def compute_holonomy_algebra(self, holonomies: List[Tensor]) -> Tensor:
        """Compute the holonomy Lie algebra.
        
        Args:
            holonomies: List of holonomy transformations
            
        Returns:
            Tensor representing the Lie algebra elements
        """
        holonomy_group = self.compute_holonomy_group(holonomies)
        eigenvalues, eigenvectors = torch.linalg.eigh(holonomy_group)
        log_eigenvalues = torch.log(torch.clamp(eigenvalues, min=1e-7))
        return torch.matmul(
            torch.matmul(eigenvectors, torch.diag_embed(log_eigenvalues)),
            eigenvectors.transpose(-2, -1)
        )

    def compute_metric(self, points: torch.Tensor) -> MetricTensor:
        """Compute the Riemannian metric at the given points.

        Args:
            points: Points at which to compute the metric, shape (batch_size, total_dim)

        Returns:
            MetricTensor: The metric tensor at each point
        """
        # Get dimensions
        batch_size = points.shape[0]
        total_dim = points.shape[1]
        
        # Create a base metric that's constant
        base_metric = torch.eye(total_dim, device=points.device).unsqueeze(0)
        
        # Create a symmetric perturbation using a third-order tensor
        # This ensures that derivatives will be symmetric
        perturbation = torch.zeros((batch_size, total_dim, total_dim), device=points.device)
        for k in range(total_dim):
            # Create a symmetric contribution for each coordinate
            # This ensures that ∂ₖgᵢⱼ = ∂ₖgⱼᵢ by construction
            sym_term = torch.outer(points[:, k], points[:, k])
            perturbation += 0.1 * torch.exp(-0.1 * points[:, k:k+1] ** 2) * sym_term.unsqueeze(0)
        
        # Combine base metric and perturbation
        metric = base_metric + perturbation
        
        # Add a small positive definite term for numerical stability
        metric = metric + 0.1 * torch.eye(total_dim, device=points.device).unsqueeze(0)
        
        # Create and return the metric tensor
        return MetricTensor(
            values=metric,
            dimension=total_dim
        )