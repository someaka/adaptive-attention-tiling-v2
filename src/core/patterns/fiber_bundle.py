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
        ).requires_grad_(True)

        # Initialize connection form
        # Shape: (base_dim, fiber_dim, fiber_dim)
        self.connection = nn.Parameter(
            torch.zeros(self.base_dim, self.fiber_dim, self.fiber_dim, device=self.device, dtype=self.dtype),
            requires_grad=True
        ).requires_grad_(True)

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
        if total_space.shape[-1] != self.total_dim:
            raise ValueError(f"Expected input with last dimension {self.total_dim}, got {total_space.shape[-1]}")
        
        # Project to base manifold by taking first base_dim components
        return total_space[..., :self.base_dim]

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
        original_shape = tangent_vector.shape
        if len(original_shape) == 1:
            tangent_vector = tangent_vector.unsqueeze(0)
            
        # Extract base and vertical components
        base_components = tangent_vector[..., :self.base_dim]  # Shape: (..., base_dim)
        vertical_components = tangent_vector[..., self.base_dim:]  # Shape: (..., fiber_dim)
        
        # For purely vertical vectors, return them unchanged
        if torch.allclose(base_components, torch.zeros_like(base_components)):
            return vertical_components if len(original_shape) > 1 else vertical_components.squeeze(0)
        
        # Get connection matrices and ensure skew-symmetry
        connection = self.connection.clone()  # Shape: (base_dim, fiber_dim, fiber_dim)
        
        # Make each base direction's connection matrix skew-symmetric
        for i in range(self.base_dim):
            # First make skew-symmetric
            connection[i] = 0.5 * (connection[i] - connection[i].transpose(-2, -1))
            # Then ensure diagonal is zero
            connection[i].diagonal().zero_()
            # Scale by 1/2 for proper Lie algebra basis
            connection[i] = 0.5 * connection[i]
        
        # Contract base components with connection matrices using einsum
        # Shape: (..., fiber_dim)
        horizontal_part = torch.einsum('...i,ijk->...k', base_components, connection)
        
        # Add vertical components to maintain linearity
        result = horizontal_part + vertical_components
        
        # Restore original shape if needed
        if len(original_shape) == 1:
            result = result.squeeze(0)
            
        return result

    def parallel_transport(self, section: Tensor, path: Tensor) -> Tensor:
        """Implementation of FiberBundle.parallel_transport.
        
        Parallel transport a section along a path using adaptive RK4 integration.
        
        Args:
            section: Section to transport (shape: (fiber_dim,) or (batch_size, fiber_dim))
            path: Path along which to transport (shape: (num_points, base_dim) or (batch_size, num_points, base_dim))
            
        Returns:
            Transported section (shape: (num_points, fiber_dim) or (batch_size, num_points, fiber_dim))
        """
        # Handle batch dimension if present
        has_batch = len(section.shape) > 1
        if not has_batch:
            section = section.unsqueeze(0)
            path = path.unsqueeze(0)
        
        # Initialize result tensor with proper shape
        batch_size = section.shape[0]
        num_points = path.shape[1]
        result = torch.zeros(batch_size, num_points, self.fiber_dim, 
                           device=section.device, dtype=section.dtype)
        
        # Set initial condition
        result[:, 0] = F.normalize(section, p=2, dim=-1) * torch.norm(section, dim=-1, keepdim=True)
        
        # Transport along path using RK4 integration
        dt = 1.0 / (num_points - 1)
        for i in range(num_points - 1):
            # Get current point and tangent vector
            current = result[:, i]
            tangent = (path[:, i+1] - path[:, i]) / dt
            
            # RK4 integration step
            k1 = self._transport_step(current, tangent)
            k2 = self._transport_step(current + 0.5*dt*k1, tangent)
            k3 = self._transport_step(current + 0.5*dt*k2, tangent)
            k4 = self._transport_step(current + dt*k3, tangent)
            
            # Update with RK4 step
            result[:, i+1] = current + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
            
            # Project to preserve fiber metric by normalizing and rescaling
            result[:, i+1] = F.normalize(result[:, i+1], p=2, dim=-1) * torch.norm(section, dim=-1, keepdim=True)
        
        return result if has_batch else result.squeeze(0)

    def _transport_step(self, current: Tensor, tangent: Tensor) -> Tensor:
        """Compute transport step for parallel transport.
        
        Args:
            current: Current section value (shape: (batch_size, fiber_dim))
            tangent: Tangent vector (shape: (batch_size, base_dim))
            
        Returns:
            Transport step value (shape: (batch_size, fiber_dim))
        """
        # Create full tangent vector with zeros in fiber components
        full_tangent = torch.zeros(
            *tangent.shape[:-1],
            self.total_dim,
            device=tangent.device,
            dtype=tangent.dtype
        )
        full_tangent[..., :self.base_dim] = tangent
        
        # Compute connection form
        omega = self.connection_form(full_tangent)  # Shape: (batch_size, fiber_dim)
        
        # Contract with current section using einsum for linearity
        step = -torch.einsum('bi,bi->bi', omega, current)
            
        return step

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
        
        # Ensure fiber metric part is identity
        metric[..., self.base_dim:, self.base_dim:] = torch.eye(
            self.fiber_dim, device=points.device
        ).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Create and return the metric tensor
        return MetricTensor(
            values=metric,
            dimension=total_dim
        )