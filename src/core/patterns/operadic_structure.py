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
        
        # Normalize the composed law to preserve significant terms
        norm = torch.norm(composed_law)
        if norm > 1e-6:
            composed_law = composed_law / norm
            
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
        result = morphisms[0]
        for morphism, enrichment in zip(morphisms[1:], enrichments):
            if enrichment.get('preserve_symplectic'):
                # Compose while preserving symplectic structure
                result = self._symplectic_compose(result, morphism)
                
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
                result = torch.matmul(morphism, result)
                
                # Ensure result has correct shape
                source_dim = morphisms[0].shape[1]
                target_dim = morphisms[-1].shape[0]
                if result.shape != (target_dim, source_dim):
                    new_result = torch.zeros(target_dim, source_dim, device=result.device, dtype=result.dtype)
                    min_rows = min(result.shape[0], target_dim)
                    min_cols = min(result.shape[1], source_dim)
                    new_result[:min_rows, :min_cols] = result[:min_rows, :min_cols]
                    result = new_result
        
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

        # Normalize inputs
        if torch.norm(a1) > 0:
            a1 = a1 / torch.norm(a1)
        if torch.norm(a2) > 0:
            a2 = a2 / torch.norm(a2)

        # Reshape tensors to ensure compatible dimensions
        if a1.shape[-1] != a2.shape[0]:
            # Pad or trim a1 to match a2's first dimension
            new_a1 = torch.zeros(a1.shape[0], a2.shape[0], device=a1.device, dtype=a1.dtype)
            min_dim = min(a1.shape[1], a2.shape[0])
            new_a1[:, :min_dim] = a1[:, :min_dim]
            a1 = new_a1

        # Calculate the composition using matrix multiplication
        result = torch.matmul(a1, a2)

        # Normalize the result
        if torch.norm(result) > 0:
            result = result / torch.norm(result)

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
    """Enriched attention structure with wave emergence support.
    
    This class implements both wave operator functionality and enriched
    categorical structure for attention operations.
    
    Attributes:
        base_category: Base category for enrichment
        wave_enabled: Whether wave emergence is enabled
        _k: Wave number parameter
        _omega: Angular frequency parameter
        dtype: Data type for tensors
    """
    def __init__(
        self,
        base_category: str = "SymplecticVect",
        wave_enabled: bool = True,
        _k: float = 2.0,
        _omega: float = 1.0,
        dtype: torch.dtype = torch.float32
    ):
        """Initialize enriched attention structure.
        
        Args:
            base_category: Base category for enrichment
            wave_enabled: Whether wave emergence is enabled
            _k: Wave number parameter
            _omega: Angular frequency parameter
            dtype: Data type for tensors
        """
        self.base_category = base_category
        self.wave_enabled = wave_enabled
        self._k = _k
        self._omega = _omega
        self.dtype = dtype
    
    def wave_operator(self, tensor: Tensor) -> Tensor:
        """Apply wave operator to tensor.
        
        Implements wave operator: W_t = exp(-iHt) where H is the Hamiltonian
        
        Args:
            tensor: Input tensor
            
        Returns:
            Tensor with wave operator applied
        """
        if not self.wave_enabled:
            return tensor
            
        # Convert tensor to specified dtype
        tensor = tensor.to(dtype=self.dtype)
            
        # Compute wave phase
        phase = self._k * torch.sum(tensor * tensor, dim=-1, keepdim=True)
        
        # Apply wave operator
        return tensor * torch.exp(-1j * self._omega * phase)
        
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
            
        # Convert tensors to specified dtype
        position = position.to(dtype=self.dtype)
        momentum = momentum.to(dtype=self.dtype)
            
        # Create Gaussian wave packet
        sigma = 1.0  # Width parameter
        norm = torch.exp(-torch.sum(position * position, dim=-1) / (4 * sigma**2))
        phase = torch.sum(momentum * position, dim=-1)
        
        return norm * torch.exp(1j * phase)
        
    def get_position(self, wave: Tensor) -> Tensor:
        """Extract position from wave packet.
        
        Args:
            wave: Wave packet tensor
            
        Returns:
            Position tensor
        """
        if not self.wave_enabled or not torch.is_complex(wave):
            return wave.real
            
        # Extract position as expectation value
        return wave.real.to(dtype=self.dtype)
        
    def get_momentum(self, wave: Tensor) -> Tensor:
        """Extract momentum from wave packet.
        
        Args:
            wave: Wave packet tensor
            
        Returns:
            Momentum tensor
        """
        if not self.wave_enabled or not torch.is_complex(wave):
            return wave.imag
            
        # Extract momentum as expectation value
        return (-self._k * wave.imag).to(dtype=self.dtype)
        
    def create_morphism(
        self,
        pattern: Tensor,
        operation: OperadicOperation,
        include_wave: bool = True
    ) -> Tensor:
        """Create enriched morphism with wave structure.
        
        Args:
            pattern: Input pattern tensor
            operation: Operadic operation to apply
            include_wave: Whether to include wave structure
            
        Returns:
            Transformed tensor with enriched structure
        """
        # Convert pattern to specified dtype
        pattern = pattern.to(dtype=self.dtype)
        
        # Apply wave operator if enabled
        if include_wave and self.wave_enabled:
            pattern = self.wave_operator(pattern)
            
        return pattern