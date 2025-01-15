"""Pattern Evolution Module.

This module implements pattern evolution dynamics on manifolds with geometric structure.
It integrates Riemannian geometry, symplectic structure, and wave packet evolution
for structure-preserving pattern dynamics.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, Union, overload, cast
from dataclasses import dataclass
from torch import Tensor

from .riemannian import PatternRiemannianStructure
from .symplectic import SymplecticStructure, SymplecticForm
from .operadic_structure import AttentionOperad, EnrichedAttention

@dataclass
class PatternEvolutionMetrics:
    """Metrics for pattern evolution."""
    velocity_norm: float
    pattern_norm: float
    gradient_norm: float
    momentum_norm: float = 0.0
    symplectic_invariant: float = 0.0
    quantum_metric: Optional[Tensor] = None
    geometric_flow: Optional[Tensor] = None
    wave_energy: float = 0.0

class PatternEvolution(nn.Module):
    """Pattern evolution on geometric manifolds with structure preservation.
    
    This class implements pattern evolution that preserves:
    1. Riemannian structure (geodesic flow)
    2. Symplectic structure (Hamiltonian dynamics)
    3. Wave packet structure (quantum behavior)
    4. Geometric invariants (stability)
    """

    def __init__(
        self,
        framework: PatternRiemannianStructure,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        symplectic: Optional[SymplecticStructure] = None,
        preserve_structure: bool = True,
        wave_enabled: bool = True,
        dim: Optional[int] = None
    ):
        """Initialize pattern evolution with geometric structure.
        
        Args:
            framework: Riemannian framework for geometric computations
            learning_rate: Learning rate for gradient updates
            momentum: Momentum coefficient for updates
            symplectic: Optional symplectic structure
            preserve_structure: Whether to preserve geometric structure
            wave_enabled: Whether to enable wave packet evolution
            dim: Optional dimension override (defaults to pattern dimension)
        """
        super().__init__()
        self.framework = framework
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.preserve_structure = preserve_structure
        self.wave_enabled = wave_enabled
        self.register_buffer('velocity', None)
        
        # Initialize geometric structures
        if symplectic is None:
            # Infer dimension from first pattern if not provided
            self._dim = dim if dim is not None else 2  # Default to 2D if not specified
            self.symplectic = SymplecticStructure(
                dim=self._dim,
                preserve_structure=preserve_structure,
                wave_enabled=wave_enabled
            )
        else:
            self.symplectic = symplectic
            self._dim = symplectic.dim
            
        # Initialize enriched structure
        self.enriched = EnrichedAttention()
        self.enriched.wave_enabled = wave_enabled
        
        # Initialize operadic structure
        self.operadic = AttentionOperad(
            base_dim=self._dim,
            preserve_symplectic=preserve_structure,
            preserve_metric=True
        )

        # Get manifold dimension from framework's metric
        dummy_point = torch.zeros(1, self._dim)  # Single point with correct input dimension
        metric = framework.compute_metric(dummy_point)
        manifold_dim = metric.dimension

        # Add projection layers with correct dimensions
        self.pattern_proj = None  # Initialize later when we know the input dimension
        self.pattern_proj_inv = None  # Initialize later when we know the input dimension
        self.manifold_dim = manifold_dim
        
        # Add submodules
        self.add_module('framework', self.framework)
        self.add_module('symplectic', self.symplectic)
        self.add_module('enriched', self.enriched)
        self.add_module('operadic', self.operadic)

    def _ensure_projections(self, pattern: torch.Tensor):
        """Ensure projection layers are initialized with correct dimensions."""
        # Calculate expected input dimension
        if len(pattern.shape) == 2:
            expected_input_dim = pattern.shape[0] * pattern.shape[1]
        else:
            expected_input_dim = pattern.shape[-1]
            
        # Check if we need to reinitialize the projection layers
        if (self.pattern_proj is None or 
            self.pattern_proj.in_features != expected_input_dim):
            
            # Initialize projection layers with correct dimensions
            self.pattern_proj = nn.Linear(expected_input_dim, self.manifold_dim)
            self.pattern_proj_inv = nn.Linear(self.manifold_dim, expected_input_dim)
            
            # Initialize weights to preserve pattern structure
            nn.init.orthogonal_(self.pattern_proj.weight)
            nn.init.orthogonal_(self.pattern_proj_inv.weight)
            nn.init.zeros_(self.pattern_proj.bias)
            nn.init.zeros_(self.pattern_proj_inv.bias)

    def _project_to_manifold(self, pattern: torch.Tensor) -> torch.Tensor:
        """Project pattern to manifold dimension.
        
        Args:
            pattern: Pattern tensor [batch_size, pattern_dim] or [size, size]
            
        Returns:
            Projected tensor [batch_size, manifold_dim]
        """
        # Store original shape for later
        original_shape = pattern.shape
        
        # For 2D patterns, flatten completely
        if len(pattern.shape) == 2:
            pattern = pattern.reshape(1, -1)  # [1, size*size]
            self._is_batched = False
            self._original_size = original_shape  # Store for later use
        elif len(pattern.shape) > 2:
            # For higher dimensional patterns, flatten all but batch dimension
            batch_size = pattern.shape[0]
            pattern = pattern.reshape(batch_size, -1)
            
        # Ensure projections are initialized with correct dimensions
        self._ensure_projections(pattern)

        # Project to manifold dimension
        if self.pattern_proj is None:
            raise RuntimeError("Pattern projection layer not properly initialized")
            
        projected = self.pattern_proj(pattern)
        
        return projected

    def _project_from_manifold(self, pattern: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
        """Project pattern back from manifold dimension.
        
        Args:
            pattern: Pattern tensor [batch_size, manifold_dim]
            original_shape: Original shape to restore
            
        Returns:
            Projected tensor with original shape
        """
        # Project back to original dimension
        if self.pattern_proj_inv is None:
            raise RuntimeError("Inverse pattern projection layer not properly initialized")
            
        output = self.pattern_proj_inv(pattern)
        
        # Get the original size for reshaping
        if not getattr(self, '_is_batched', True):
            # If the original input wasn't batched, use the stored original size
            original_size = getattr(self, '_original_size', original_shape)
            # Remove batch dimension and reshape to original size
            output = output.squeeze(0).reshape(original_size)
        else:
            # For batched inputs, reshape using the original shape
            output = output.reshape(original_shape)
            
        return output

    @overload
    def step(
        self,
        pattern: torch.Tensor,
        gradient: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_metrics: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    @overload
    def step(
        self,
        pattern: torch.Tensor,
        gradient: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_metrics: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, PatternEvolutionMetrics]: ...

    def step(
        self,
        pattern: torch.Tensor,
        gradient: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_metrics: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, PatternEvolutionMetrics]]:
        """Perform one evolution step with structure preservation.
        
        Args:
            pattern: Current pattern state
            gradient: Pattern gradient
            mask: Optional mask for selective updates
            return_metrics: Whether to return evolution metrics
            
        Returns:
            If return_metrics is False:
                Tuple of (updated pattern, velocity)
            If return_metrics is True:
                Tuple of (updated pattern, velocity, metrics)
        """
        # Store original shape
        original_shape = pattern.shape
    
        # Project pattern and gradient to manifold dimension
        pattern_manifold = self._project_to_manifold(pattern)
        gradient_manifold = self._project_to_manifold(gradient)
    
        # Update dimension if needed
        if pattern_manifold.shape[-1] != self._dim:
            self._dim = pattern_manifold.shape[-1]
            self.symplectic = SymplecticStructure(
                dim=self._dim,
                preserve_structure=self.preserve_structure,
                wave_enabled=self.wave_enabled
            )
            self.operadic = AttentionOperad(
                base_dim=self._dim,
                preserve_symplectic=self.preserve_structure,
                preserve_metric=True
            )
    
        # Handle pattern dimensions
        pattern_symplectic = self.symplectic._handle_dimension(pattern_manifold)
        gradient_symplectic = self.symplectic._handle_dimension(gradient_manifold)
    
        if self.velocity is None:
            self.velocity = torch.zeros_like(gradient_symplectic)
    
        # Compute quantum geometric tensor
        Q = self.symplectic.compute_quantum_geometric_tensor(pattern_symplectic)
        g = Q.real  # Metric part
        omega = Q.imag  # Symplectic part
    
        # Ensure dimensions match for metric tensor multiplication
        if g.shape[-1] != gradient_symplectic.shape[-1]:
            # Pad or truncate gradient to match metric dimensions
            target_dim = g.shape[-1]
            if gradient_symplectic.shape[-1] > target_dim:
                gradient_symplectic = gradient_symplectic[..., :target_dim]
            else:
                padding_size = target_dim - gradient_symplectic.shape[-1]
                gradient_symplectic = torch.nn.functional.pad(gradient_symplectic, (0, padding_size))

            # Also adjust velocity dimensions
            if self.velocity is not None:
                if self.velocity.shape[-1] > target_dim:
                    self.velocity = self.velocity[..., :target_dim]
                else:
                    padding_size = target_dim - self.velocity.shape[-1]
                    self.velocity = torch.nn.functional.pad(self.velocity, (0, padding_size))

        if self.velocity is None:
            self.velocity = torch.zeros_like(gradient_symplectic)

        # Update velocity with momentum and metric structure
        self.velocity = self.momentum * self.velocity - self.learning_rate * torch.einsum(
            'bij,bj->bi',  # Use batch-wise einsum
            g,  # Use metric for gradient descent
            gradient_symplectic
        )
    
        # Apply mask if provided
        if mask is not None:
            self.velocity = self.velocity * mask
    
        # Update pattern with velocity
        pattern_updated = pattern_symplectic + self.velocity
    
        # Project back to original shape
        pattern_updated = self._project_from_manifold(pattern_updated, original_shape)
        velocity_original = self._project_from_manifold(self.velocity, original_shape)
    
        if return_metrics:
            metrics = PatternEvolutionMetrics(
                velocity_norm=torch.norm(velocity_original).item(),
                pattern_norm=torch.norm(pattern_updated).item(),
                gradient_norm=torch.norm(gradient).item(),
                momentum_norm=self.momentum,
                symplectic_invariant=torch.norm(omega).item(),
                quantum_metric=g.detach(),
                geometric_flow=pattern_updated - pattern,
                wave_energy=torch.norm(self.velocity).item() if self.wave_enabled else 0.0
            )
            return pattern_updated, velocity_original, metrics
    
        return pattern_updated, velocity_original

    def compute_hamiltonian(
        self,
        pattern: torch.Tensor,
        momentum: torch.Tensor
    ) -> torch.Tensor:
        """Compute Hamiltonian with geometric structure.
        
        Args:
            pattern: Pattern state
            momentum: Conjugate momentum
            
        Returns:
            Hamiltonian value
        """
        # Handle dimensions
        pattern_symplectic = self.symplectic._handle_dimension(pattern)
        momentum_symplectic = self.symplectic._handle_dimension(momentum)
        
        # Compute metric contribution
        Q = self.symplectic.compute_quantum_geometric_tensor(pattern_symplectic)
        g = Q.real
        g_inv = torch.linalg.inv(g)
        kinetic = 0.5 * torch.einsum('...i,...ij,...j->...', momentum_symplectic, g_inv, momentum_symplectic)
        
        # Compute potential (using double-well potential)
        potential = 0.25 * torch.sum(pattern_symplectic**4, dim=-1) - 0.5 * torch.sum(pattern_symplectic**2, dim=-1)
        
        return kinetic + potential

    def reset(self):
        """Reset evolution state."""
        self.velocity = None
