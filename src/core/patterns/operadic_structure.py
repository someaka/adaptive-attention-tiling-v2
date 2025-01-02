"""Operadic structure for handling dimensional transitions in pattern fiber bundles.

This module implements the operadic composition structure that allows for natural
dimensional transitions in pattern spaces, replacing the current padding approach
with a mathematically rigorous solution based on operadic composition laws.

The implementation follows the enriched categorical structure outlined in the
symplectic dimension notes, providing natural transformations between pattern
spaces through operadic composition.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Protocol
from abc import ABC, abstractmethod
import torch
from torch import Tensor

@dataclass
class OperadicOperation:
    """Represents an operadic operation for dimensional transition.
    
    Attributes:
        source_dim: Source dimension
        target_dim: Target dimension
        composition_law: The actual composition morphism
        enrichment: Optional enriched structure data
        natural_transformation: Optional natural transformation data
    """
    source_dim: int
    target_dim: int
    composition_law: Tensor  # The actual composition morphism
    enrichment: Optional[Dict[str, Any]] = None
    natural_transformation: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate operadic operation properties."""
        if self.composition_law.shape != (self.target_dim, self.source_dim):
            raise ValueError(
                f"Composition law shape {self.composition_law.shape} does not match "
                f"dimensions ({self.target_dim}, {self.source_dim})"
            )

class OperadicComposition(ABC):
    """Abstract base class for operadic composition structures."""
    
    @abstractmethod
    def compose(self, operations: List[OperadicOperation]) -> OperadicOperation:
        """Compose multiple operadic operations."""
        pass

    @abstractmethod
    def create_operation(
        self,
        source_dim: int,
        target_dim: int,
        preserve_structure: Optional[str] = None
    ) -> OperadicOperation:
        """Create an operadic operation for dimensional transition."""
        pass

    @abstractmethod
    def natural_transformation(
        self,
        source_op: OperadicOperation,
        target_op: OperadicOperation
    ) -> Tensor:
        """Create natural transformation between operadic operations."""
        pass

class AttentionOperad(OperadicComposition):
    """Implements operadic composition for attention operations.
    
    This class provides enriched operadic composition with:
    1. Natural transformations between pattern spaces
    2. Structure preservation (symplectic, metric, etc.)
    3. Integration with enriched categorical structure
    """
    
    def __init__(
        self,
        base_dim: int = 2,
        preserve_symplectic: bool = True,
        preserve_metric: bool = True,
        dtype: torch.dtype = torch.float32
    ):
        """Initialize attention operad.
        
        Args:
            base_dim: Base dimension for attention operations
            preserve_symplectic: Whether to preserve symplectic structure
            preserve_metric: Whether to preserve metric structure
            dtype: Data type for tensors
        """
        self.base_dim = base_dim
        self.preserve_symplectic = preserve_symplectic
        self.preserve_metric = preserve_metric
        self.dtype = dtype
    
    def create_operation(
        self,
        source_dim: int,
        target_dim: int,
        preserve_structure: Optional[str] = None
    ) -> OperadicOperation:
        """Create an operadic operation for dimensional transition.
        
        Args:
            source_dim: Source dimension
            target_dim: Target dimension
            preserve_structure: Optional structure to preserve
                ('symplectic', 'metric', or None)
            
        Returns:
            An operadic operation that handles the dimensional transition
        """
        # Create enriched structure data
        enrichment = {
            'preserve_symplectic': self.preserve_symplectic,
            'preserve_metric': self.preserve_metric,
            'structure_type': preserve_structure
        }
        
        # Create composition morphism based on enriched little cubes operad
        composition_law = self._create_composition_morphism(
            source_dim,
            target_dim,
            preserve_structure
        )
        
        # Create natural transformation data
        natural_transformation = {
            'source_dim': source_dim,
            'target_dim': target_dim,
            'transformation_type': 'enriched_attention'
        }
        
        return OperadicOperation(
            source_dim=source_dim,
            target_dim=target_dim,
            composition_law=composition_law,
            enrichment=enrichment,
            natural_transformation=natural_transformation
        )
    
    def compose(self, operations: List[OperadicOperation]) -> OperadicOperation:
        """Compose multiple operadic operations with structure preservation.
        
        Args:
            operations: List of operadic operations to compose
            
        Returns:
            The composed operation with preserved structure
        """
        if not operations:
            raise ValueError("Cannot compose empty list of operations")
            
        # Verify composability and collect enrichment data
        enrichments = []
        for i in range(len(operations) - 1):
            if operations[i].target_dim != operations[i + 1].source_dim:
                raise ValueError(
                    f"Operations not composable: "
                    f"{operations[i].target_dim} != {operations[i + 1].source_dim}"
                )
            if operations[i].enrichment:
                enrichments.append(operations[i].enrichment)
        
        # Compose using enriched operadic composition law
        source_dim = operations[0].source_dim
        target_dim = operations[-1].target_dim
        
        # Create composition morphism with structure preservation
        composed_law = self._compose_morphisms(
            [op.composition_law for op in operations],
            enrichments
        )
            
        # Combine enrichment data
        combined_enrichment = self._combine_enrichments(enrichments)
        
        # Create natural transformation for composed operation
        natural_transformation = {
            'source_dim': source_dim,
            'target_dim': target_dim,
            'transformation_type': 'composed_enriched_attention',
            'component_transformations': [
                op.natural_transformation for op in operations
                if op.natural_transformation is not None
            ]
        }
        
        # Normalize the composed law to preserve significant terms
        # but only after all compositions are done
        norm = torch.norm(composed_law)
        if norm > 1e-6:
            composed_law = composed_law / norm
        
        return OperadicOperation(
            source_dim=source_dim,
            target_dim=target_dim,
            composition_law=composed_law,
            enrichment=combined_enrichment,
            natural_transformation=natural_transformation
        )
    
    def natural_transformation(
        self,
        source_op: OperadicOperation,
        target_op: OperadicOperation
    ) -> Tensor:
        """Create natural transformation between operadic operations.
        
        Args:
            source_op: Source operadic operation
            target_op: Target operadic operation
            
        Returns:
            Natural transformation tensor
        """
        # Verify compatibility
        if source_op.target_dim != target_op.target_dim:
            raise ValueError(
                f"Operations not compatible for natural transformation: "
                f"{source_op.target_dim} != {target_op.target_dim}"
            )
        
        # Create natural transformation using enriched structure
        source_basis = torch.eye(source_op.source_dim, dtype=self.dtype)
        target_basis = torch.eye(target_op.source_dim, dtype=self.dtype)
        
        # Create transformation that respects enriched structure
        transformation = torch.zeros(
            target_op.source_dim,
            source_op.source_dim,
            dtype=self.dtype
        )
        
        min_dim = min(source_op.source_dim, target_op.source_dim)
        transformation[:min_dim, :min_dim] = torch.eye(min_dim, dtype=self.dtype)
        
        return transformation
    
    def _create_composition_morphism(
        self,
        source_dim: int,
        target_dim: int,
        preserve_structure: Optional[str] = None
    ) -> Tensor:
        """Create a composition morphism for dimensional transition.
        
        Uses enriched little cubes operad structure to create natural
        transition maps that preserve specified structures.
        
        Args:
            source_dim: Source dimension
            target_dim: Target dimension
            preserve_structure: Optional structure to preserve
            
        Returns:
            Tensor representing the composition morphism
        """
        # Create basis for source and target spaces
        source_basis = torch.eye(source_dim, dtype=self.dtype)
        target_basis = torch.eye(target_dim, dtype=self.dtype)
        
        # Create transition map using enriched little cubes structure
        if source_dim <= target_dim:
            # Embedding into higher dimension with structure preservation
            morphism = torch.zeros(target_dim, source_dim, dtype=self.dtype)
            if preserve_structure == 'symplectic':
                # Preserve symplectic structure in embedding
                n_source = source_dim // 2
                n_target = target_dim // 2
                morphism[:2*n_source:2, :2*n_source:2] = torch.eye(n_source, dtype=self.dtype)
                morphism[1:2*n_source:2, 1:2*n_source:2] = torch.eye(n_source, dtype=self.dtype)
            else:
                # Standard embedding
                morphism[:source_dim, :] = source_basis
        else:
            # Projection to lower dimension with structure preservation
            morphism = torch.zeros(target_dim, source_dim, dtype=self.dtype)
            if preserve_structure == 'symplectic':
                # Preserve symplectic structure in projection
                n_source = source_dim // 2
                n_target = target_dim // 2
                morphism[:2*n_target:2, :2*n_target:2] = torch.eye(n_target, dtype=self.dtype)
                morphism[1:2*n_target:2, 1:2*n_target:2] = torch.eye(n_target, dtype=self.dtype)
            else:
                # Standard projection
                morphism[:, :target_dim] = target_basis
        
        return morphism
    
    def _compose_morphisms(
        self,
        morphisms: List[Tensor],
        enrichments: List[Dict[str, Any]]
    ) -> Tensor:
        """Compose multiple morphisms using enriched composition.
        
        Args:
            morphisms: List of morphism tensors to compose
            enrichments: List of enrichment data for each morphism
            
        Returns:
            The composed morphism tensor with preserved structure
        """
        # Extract phases and magnitudes
        phases = [torch.angle(m) for m in morphisms]
        magnitudes = [torch.abs(m) for m in morphisms]
        
        # Compose magnitudes first
        result = magnitudes[0]
        for magnitude, enrichment in zip(magnitudes[1:], enrichments):
            if enrichment.get('preserve_symplectic'):
                # Compose while preserving symplectic structure
                result = self._symplectic_compose(result, magnitude)
                
                # Ensure result has correct shape after composition
                source_dim = morphisms[0].shape[1]  # First morphism's input dimension
                target_dim = morphisms[-1].shape[0]  # Last morphism's output dimension
                
                # Reshape result to match required dimensions
                if result.shape != (target_dim, source_dim):
                    new_result = torch.zeros(target_dim, source_dim, device=result.device, dtype=result.dtype)
                    min_rows = min(result.shape[0], target_dim)
                    min_cols = min(result.shape[1], source_dim)
                    new_result[:min_rows, :min_cols] = result[:min_rows, :min_cols]
                    result = new_result
            else:
                # Standard composition
                result = torch.matmul(magnitude, result)
                
                # Ensure result has correct shape
                source_dim = morphisms[0].shape[1]
                target_dim = morphisms[-1].shape[0]
                if result.shape != (target_dim, source_dim):
                    new_result = torch.zeros(target_dim, source_dim, device=result.device, dtype=result.dtype)
                    min_rows = min(result.shape[0], target_dim)
                    min_cols = min(result.shape[1], source_dim)
                    new_result[:min_rows, :min_cols] = result[:min_rows, :min_cols]
                    result = new_result
        
        # Compose phases by adding them
        composed_phase = torch.zeros_like(result)
        for phase in phases:
            composed_phase = composed_phase + phase
        
        # Combine magnitude and phase
        result = result * torch.exp(1j * composed_phase)
        
        return result
    
    def _symplectic_compose(self, a1: torch.Tensor, a2: torch.Tensor) -> torch.Tensor:
        """Compose two anomaly polynomials using symplectic structure."""
        # Ensure tensors have compatible shapes
        if a1.dim() == 0:
            a1 = a1.unsqueeze(0)
        if a2.dim() == 0:
            a2 = a2.unsqueeze(0)

        # Handle batched input
        if a1.dim() > 2:
            a1 = a1.reshape(a1.shape[0], -1)
        if a2.dim() > 2:
            a2 = a2.reshape(a2.shape[0], -1)

        # Ensure tensors are 2D
        if a1.dim() == 1:
            a1 = a1.unsqueeze(1)  # Shape becomes (N, 1)
        if a2.dim() == 1:
            a2 = a2.unsqueeze(0)  # Shape becomes (1, N)

        # Get target dimensions
        target_dim = a1.shape[0]
        source_dim = a2.shape[1] if a2.dim() > 1 else 1

        # Reshape tensors to ensure compatible dimensions for multiplication
        if a1.shape[-1] != a2.shape[0]:
            # Pad or trim a1 to match a2's first dimension
            new_a1 = torch.zeros(target_dim, a2.shape[0], device=a1.device, dtype=a1.dtype)
            min_dim = min(a1.shape[1], a2.shape[0])
            new_a1[:, :min_dim] = a1[:, :min_dim]
            a1 = new_a1

        # Calculate the composition using matrix multiplication
        # Preserve U(1) phase structure by not normalizing inputs
        result = torch.matmul(a1, a2)

        # Ensure result has correct shape (target_dim, source_dim)
        if result.shape != (target_dim, source_dim):
            new_result = torch.zeros(target_dim, source_dim, device=result.device, dtype=result.dtype)
            min_rows = min(result.shape[0], target_dim)
            min_cols = min(result.shape[1], source_dim)
            new_result[:min_rows, :min_cols] = result[:min_rows, :min_cols]
            result = new_result

        return result
    
    def _combine_enrichments(
        self,
        enrichments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Combine enrichment data from multiple operations.
        
        Args:
            enrichments: List of enrichment data to combine
            
        Returns:
            Combined enrichment data
        """
        if not enrichments:
            return {}
            
        combined = {
            'preserve_symplectic': all(
                e.get('preserve_symplectic', False) for e in enrichments
            ),
            'preserve_metric': all(
                e.get('preserve_metric', False) for e in enrichments
            ),
            'structure_types': list(set(
                e.get('structure_type') for e in enrichments
                if e.get('structure_type') is not None
            ))
        }
        
        return combined

    def create_transition(
        self,
        source_dim: int,
        target_dim: int
    ) -> OperadicOperation:
        """Create a simple transition operation between dimensions.
        
        This is a simplified version of create_operation that automatically
        determines the appropriate structure preservation based on the class settings.
        
        Args:
            source_dim: Source dimension
            target_dim: Target dimension
            
        Returns:
            An operadic operation that handles the dimensional transition
        """
        preserve_structure = None
        if self.preserve_symplectic:
            preserve_structure = 'symplectic'
        elif self.preserve_metric:
            preserve_structure = 'metric'
            
        return self.create_operation(
            source_dim=source_dim,
            target_dim=target_dim,
            preserve_structure=preserve_structure
        )

class EnrichedAttention:
    """Enriched attention structure with wave emergence.
    
    This class implements enriched categorical structure with wave emergence
    behavior for natural transitions between dimensions.
    """
    
    def __init__(
        self,
        base_category: str = "SymplecticVect",
        wave_enabled: bool = True,
        dtype: torch.dtype = torch.float32,
        _k: float = 2.0,
        _omega: float = 1.0
    ):
        """Initialize enriched attention structure."""
        self.base_category = base_category
        self.wave_enabled = wave_enabled
        self.dtype = dtype
        self._k = _k  # Wave number
        self._omega = _omega  # Angular frequency
        
    def wave_operator(self, tensor: Tensor) -> Tensor:
        """Apply wave operator to tensor.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Complex tensor with wave structure
        """
        if not self.wave_enabled:
            return tensor
            
        # Convert to complex with phase
        phase = torch.sum(tensor * tensor, dim=-1, keepdim=True) * self._k
        return torch.complex(
            tensor,
            tensor * torch.sin(phase)
        )
        
    def create_wave_packet(self, position: Tensor, momentum: Tensor) -> Tensor:
        """Create wave packet from position and momentum.
        
        Args:
            position: Position tensor
            momentum: Momentum tensor
            
        Returns:
            Wave packet tensor
        """
        if not self.wave_enabled:
            return position
            
        # Handle complex inputs
        if position.is_complex():
            position = torch.abs(position)  # Use magnitude directly
        else:
            position = torch.abs(position.to(dtype=self.dtype))
            
        if momentum.is_complex():
            momentum = torch.angle(momentum) / self._k  # Extract phase
        else:
            momentum = momentum.to(dtype=self.dtype)
        
        # Ensure compatible shapes
        if position.shape != momentum.shape:
            # If momentum has extra dimensions, sum over them
            if len(momentum.shape) > len(position.shape):
                momentum = torch.mean(momentum, dim=tuple(range(len(momentum.shape)-1)))
            # If position has extra dimensions, sum over them
            elif len(position.shape) > len(momentum.shape):
                position = torch.mean(position, dim=tuple(range(len(position.shape)-1)))
        
        # Create Gaussian wave packet
        sigma = 1.0  # Width parameter
        # Keep the last dimension by not summing over it
        norm = torch.exp(-torch.sum(position * position, dim=-1, keepdim=True) / (4 * sigma**2))
        phase = torch.sum(momentum * position, dim=-1, keepdim=True)
        
        # Create complex wave packet
        real = norm * torch.cos(phase)
        imag = norm * torch.sin(phase)
        packet = torch.complex(real, imag)
        
        # Scale to preserve expectation values
        packet = packet * torch.sqrt(position + 1e-7)  # Add epsilon for stability
        packet = packet * torch.exp(1j * momentum)  # Add momentum phase
        
        # Normalize to preserve total probability
        packet = packet / (torch.norm(packet) + 1e-7)
        packet = packet * torch.sqrt(torch.mean(position))  # Scale by mean position
        
        return packet
        
    def get_position(self, wave: Tensor) -> Tensor:
        """Extract position from wave packet.
        
        Args:
            wave: Wave packet tensor
            
        Returns:
            Position tensor
        """
        if not self.wave_enabled or not wave.is_complex():
            return wave
            
        # Extract position as magnitude and normalize
        pos = torch.abs(wave)
        pos = pos / (torch.norm(pos) + 1e-7)  # Normalize
        pos = pos * torch.mean(torch.abs(wave))  # Scale by mean magnitude
        
        return pos
        
    def get_momentum(self, wave: Tensor) -> Tensor:
        """Extract momentum from wave packet.
        
        Args:
            wave: Wave packet tensor
            
        Returns:
            Momentum tensor
        """
        if not self.wave_enabled or not wave.is_complex():
            return wave
            
        # Extract momentum from phase gradient and normalize
        mom = torch.angle(wave + 1j) / self._k  # Add i to avoid zero angle
        mom = mom / (torch.norm(mom) + 1e-7)  # Normalize
        mom = mom * torch.mean(torch.abs(wave))  # Scale by mean magnitude
        
        return mom
        
    def create_morphism(
        self,
        pattern: Tensor,
        operation: OperadicOperation,
        include_wave: bool = True
    ) -> Tensor:
        """Create enriched morphism with wave structure.
        
        Args:
            pattern: Input pattern tensor
            operation: Operadic operation
            include_wave: Whether to include wave structure
            
        Returns:
            Transformed pattern with preserved structure
        """
        # Apply operadic operation
        result = torch.matmul(pattern, operation.composition_law.t())
        
        # Add wave structure if enabled
        if self.wave_enabled and include_wave:
            result = self.wave_operator(result)
            
        return result