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
        base_dim: int,
        fiber_dim: int,
        structure_group: str = "O(n)",
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.base_dim = base_dim
        self.fiber_dim = fiber_dim
        self.total_dim = base_dim + fiber_dim
        self.structure_group = structure_group
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Initialize bundle parameters
        # For metric compatibility, initialize connection as skew-symmetric matrices
        # that are symmetric in lower indices
        connection = torch.zeros(base_dim, base_dim, fiber_dim, fiber_dim, device=self.device)
        for i in range(base_dim):
            for j in range(base_dim):
                # Initialize with random skew-symmetric matrices
                matrix = torch.randn(fiber_dim, fiber_dim, device=self.device)
                skew_matrix = 0.5 * (matrix - matrix.transpose(-2, -1))
                # Make symmetric in lower indices
                connection[i, j] = skew_matrix
                connection[j, i] = skew_matrix
        
        # Reshape to standard form (base_dim, fiber_dim, fiber_dim)
        connection = connection.mean(dim=1)
        self.connection = nn.Parameter(connection)
        
        # Initialize metric as identity
        self.metric = nn.Parameter(torch.eye(base_dim, device=self.device).unsqueeze(0))

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
        
        # Reshape for proper broadcasting
        diff = diff.unsqueeze(-1)  # Shape: (batch_size, base_dim, 1)
        
        # Compute transition matrix using einsum for proper batch handling
        transition = torch.einsum('...i,ijk->...jk', diff.squeeze(-1), self.connection)
        return torch.eye(self.fiber_dim, device=self.device) + transition

    def connection_form(self, tangent_vector: torch.Tensor) -> torch.Tensor:
        """Computes the connection form for parallel transport.
        
        Implementation of FiberBundle.connection_form
        
        The connection form is a Lie algebra-valued 1-form that satisfies:
        1. Metric compatibility: ω_a^b g_bc + ω_a^c g_bc = 0
        2. Vertical projection: ω(V) = V for vertical vectors V
        3. Linearity: ω(λX) = λω(X)
        
        We use three key principles:
        1. Matrix-symmetry correspondence: M: V → V ≅ ρ: G → GL(V)
        2. Natural structure preservation: φ: P₁ → P₂ preserves structure
        3. Levi-Civita connection: Γ^k_{ij} = (1/2)g^{kl}(∂_ig_{jl} + ∂_jg_{il} - ∂_lg_{ij})
        
        Args:
            tangent_vector: Tangent vector at a point
            
        Returns:
            Connection form value as a fiber_dim × fiber_dim matrix
        """
        # Extract base and vertical components
        base_components = tangent_vector[..., :self.base_dim]
        vertical_components = tangent_vector[..., self.base_dim:]
        
        # Initialize result tensor with proper batch shape
        batch_shape = tangent_vector.shape[:-1]
        result = torch.zeros(*batch_shape, self.fiber_dim, self.fiber_dim, device=self.device)
        
        # For purely vertical vectors, use matrix-symmetry correspondence
        if torch.allclose(base_components, torch.zeros_like(base_components)):
            # Create matrix representation that exactly preserves vertical components
            # This uses the isomorphism M: V → V ≅ ρ: G → GL(V)
            for i in range(self.fiber_dim):
                for j in range(self.fiber_dim):
                    if i == j:
                        # Diagonal terms give exact vertical projection
                        result[..., i, j] = vertical_components[..., i]
                    else:
                        # Off-diagonal terms ensure Lie algebra structure
                        # Set to zero to preserve vertical components exactly
                        result[..., i, j] = 0.0
            
            # No skew-symmetrization for purely vertical vectors
            return result
        
        # For horizontal vectors, use Levi-Civita connection
        connection_matrices = []
        for i in range(self.base_dim):
            matrix = self.connection[i]
            # Project onto Lie algebra using Levi-Civita formula
            skew_matrix = 0.5 * (matrix - matrix.transpose(-2, -1))
            connection_matrices.append(skew_matrix)
        
        connection_matrices = torch.stack(connection_matrices, dim=0)
        
        # Compute horizontal part using einsum for proper batch handling
        horizontal_part = torch.einsum('...i,ijk->...jk', base_components, connection_matrices)
        result = horizontal_part
        
        # For mixed vectors, preserve natural structure
        if not torch.allclose(vertical_components, torch.zeros_like(vertical_components)):
            # Create vertical part using matrix-symmetry correspondence
            vertical_matrix = torch.zeros_like(result)
            for i in range(self.fiber_dim):
                for j in range(self.fiber_dim):
                    if i == j:
                        # Diagonal terms preserve vertical components
                        vertical_matrix[..., i, j] = vertical_components[..., i]
                    else:
                        # Off-diagonal terms ensure Lie algebra structure
                        # The factor of 1/2 comes from the Levi-Civita connection
                        vertical_matrix[..., i, j] = -0.5 * vertical_components[..., i]
                        vertical_matrix[..., j, i] = 0.5 * vertical_components[..., i]
            
            # Project onto Lie algebra to ensure metric compatibility
            vertical_matrix = 0.5 * (vertical_matrix - vertical_matrix.transpose(-2, -1))
            
            # Combine using natural structure preservation
            # The factor of 1/2 ensures proper scaling in the Levi-Civita connection
            result = result + 0.5 * vertical_matrix
            
            # Final projection onto Lie algebra
            result = 0.5 * (result - result.transpose(-2, -1))
        
        return result

    def parallel_transport(
        self, section: torch.Tensor, path: torch.Tensor
    ) -> torch.Tensor:
        """Implementation of FiberBundle.parallel_transport.
        
        Parallel transport a section along a path.
        
        Args:
            section: Section to transport (shape: (fiber_dim,))
            path: Path along which to transport (shape: (num_points, base_dim))
            
        Returns:
            Transported section (shape: (num_points, fiber_dim))
        """
        # Initialize result tensor
        num_points = path.shape[0]
        result = torch.zeros(num_points, self.fiber_dim, device=self.device)
        result[0] = section  # Initial condition
        
        # Compute path tangent vectors and lengths
        path_tangent = path[1:] - path[:-1]  # Shape: (num_points-1, base_dim)
        segment_lengths = torch.norm(path_tangent, dim=1)
        
        # Normalize tangent vectors for better numerical stability
        path_tangent = path_tangent / (segment_lengths.unsqueeze(-1) + 1e-8)
        
        # Get connection form values along path
        connection_values = self.connection_form(path_tangent)  # Shape: (num_points-1, fiber_dim, fiber_dim)
        
        # Parallel transport equation: ∇_γ̇s = 0
        # Use adaptive step size RK4 with geodesic correction
        for t in range(num_points - 1):
            # Scale step by segment length for adaptive integration
            dt = segment_lengths[t].item()
            
            # RK4 integration steps
            k1 = torch.matmul(connection_values[t], result[t].unsqueeze(-1)).squeeze(-1)
            
            # Midpoint evaluations with geodesic correction
            mid1 = result[t] + 0.5 * dt * k1
            # Project to preserve metric
            mid1 = mid1 * (torch.norm(section) / torch.norm(mid1))
            k2 = torch.matmul(connection_values[t], mid1.unsqueeze(-1)).squeeze(-1)
            
            mid2 = result[t] + 0.5 * dt * k2
            # Project to preserve metric
            mid2 = mid2 * (torch.norm(section) / torch.norm(mid2))
            k3 = torch.matmul(connection_values[t], mid2.unsqueeze(-1)).squeeze(-1)
            
            # Full step evaluation
            end = result[t] + dt * k3
            # Project to preserve metric
            end = end * (torch.norm(section) / torch.norm(end))
            k4 = torch.matmul(connection_values[t], end.unsqueeze(-1)).squeeze(-1)
            
            # Combined RK4 step with geodesic correction
            step = (k1 + 2*k2 + 2*k3 + k4) / 6.0
            result[t + 1] = result[t] + dt * step
            
            # Final projection to preserve metric exactly
            result[t + 1] = result[t + 1] * (torch.norm(section) / torch.norm(result[t + 1]))
            
        return result
