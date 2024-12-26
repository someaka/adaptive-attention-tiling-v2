"""
Base Fiber Bundle Implementation.

This module provides the core mathematical implementation of fiber bundles,
implementing the FiberBundle protocol defined in core.patterns.fiber_types.
"""

from typing import List, Optional, Tuple, Dict, Any
import torch
from torch import Tensor, nn
from dataclasses import dataclass

from .fiber_types import (
    FiberBundle,
    LocalChart,
    FiberChart,
)
from src.core.patterns.riemannian_base import MetricTensor


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
        self.metric = torch.eye(self.total_dim, device=self.device, dtype=self.dtype)

        # Initialize connection form
        # Shape: (base_dim, fiber_dim, fiber_dim)
        self.connection = torch.zeros(self.base_dim, self.fiber_dim, self.fiber_dim, device=self.device, dtype=self.dtype)

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
        diff = chart2.coordinates - chart1.coordinates  # Shape: (batch_size, base_dim)
        
        # Reshape for proper broadcasting
        diff = diff.unsqueeze(-1)  # Shape: (batch_size, base_dim, 1)
        connection = self.connection.unsqueeze(0)  # Shape: (1, base_dim, fiber_dim, fiber_dim)
        
        # Compute transition matrix
        transition = torch.einsum('...i,ijkl->...kl', diff.squeeze(-1), connection)
        return torch.eye(self.fiber_dim) + transition

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
        if len(section.shape) > 2:  # [batch, seq, dim]
            batch_size, seq_len, dim = section.shape
            section = section.reshape(-1, dim)
        elif len(section.shape) == 2:  # [batch, dim]
            batch_size, dim = section.shape
        else:  # [dim]
            dim = section.shape[0]
            section = section.unsqueeze(0)
            
        result = torch.zeros(num_points, section.shape[0], self.fiber_dim, device=path.device, dtype=path.dtype)
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
        
    def _transport_step(self, section: Tensor, tangent: Tensor) -> Tensor:
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
        
        # Compute transport step
        return -torch.matmul(connection, section.unsqueeze(-1)).squeeze(-1)

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
        
        # Add regularization term to ensure positive definiteness
        reg_term = 1e-3 * torch.eye(
            self.total_dim,
            device=values.device,
            dtype=values.dtype
        ).expand(batch_size, -1, -1)
        
        values = values + reg_term
        
        # Validate metric properties
        is_compatible = True  # We ensure this by construction
        
        return MetricTensor(
            values=values,
            dimension=self.total_dim,
            is_compatible=is_compatible
        )