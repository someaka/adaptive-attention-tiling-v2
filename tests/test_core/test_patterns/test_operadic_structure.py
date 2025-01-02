"""Tests for operadic and enriched structures."""

import pytest
import torch
import numpy as np
from typing import List

from src.core.patterns.operadic_structure import (
    AttentionOperad,
    OperadicOperation,
    OperadicComposition
)
from src.core.patterns.enriched_structure import (
    EnrichedMorphism,
    PatternTransition,
    WaveEmergence
)

@pytest.fixture
def operad():
    """Create an attention operad instance."""
    return AttentionOperad(base_dim=2)

@pytest.fixture
def wave_emergence():
    """Create a wave emergence instance."""
    return WaveEmergence(dt=0.1, num_steps=10)

@pytest.fixture
def pattern_transition(wave_emergence):
    """Create a pattern transition instance."""
    return PatternTransition(wave_emergence=wave_emergence)

class TestOperadicStructure:
    """Test operadic structure implementation."""
    
    def test_create_operation(self, operad):
        """Test creation of operadic operations."""
        # Test upward dimension change
        operation = operad.create_operation(2, 3)
        assert operation.source_dim == 2
        assert operation.target_dim == 3
        assert operation.composition_law.shape == (3, 2)
        
        # Test downward dimension change
        operation = operad.create_operation(3, 2)
        assert operation.source_dim == 3
        assert operation.target_dim == 2
        assert operation.composition_law.shape == (2, 3)
        
        # Test same dimension
        operation = operad.create_operation(2, 2)
        assert operation.source_dim == 2
        assert operation.target_dim == 2
        assert torch.allclose(operation.composition_law, torch.eye(2))
    
    def test_compose_operations(self, operad):
        """Test composition of operadic operations."""
        # Create sequence of operations with compatible dimensions
        op1 = operad.create_operation(2, 2)  # 2->2
        op2 = operad.create_operation(2, 2)  # 2->2
        op3 = operad.create_operation(2, 2)  # 2->2
        
        # Test composition
        composed = operad.compose([op1, op2, op3])
        assert composed.source_dim == 2
        assert composed.target_dim == 2
        
        # Test composition law properties
        assert composed.composition_law.shape == (2, 2)
        
        # Test error on non-composable operations
        op4 = operad.create_operation(2, 3)  # 2->3
        op5 = operad.create_operation(2, 2)  # 2->2
        with pytest.raises(ValueError, match="Operations not composable"):
            operad.compose([op4, op5])  # Should fail because 3 != 2
    
    def test_little_cubes_structure(self, operad):
        """Test little cubes operad structure."""
        # Test embedding
        op_up = operad.create_operation(2, 3)
        x = torch.randn(2)
        y = torch.matmul(op_up.composition_law, x)
        assert y.shape == (3,)
        assert torch.allclose(y[:2], x)  # Original coordinates preserved
        
        # Test projection
        op_down = operad.create_operation(3, 2)
        z = torch.randn(3)
        w = torch.matmul(op_down.composition_law, z)
        assert w.shape == (2,)
        assert torch.allclose(w, z[:2])  # First coordinates preserved

class TestEnrichedStructure:
    """Test enriched categorical structure implementation."""
    
    def test_wave_emergence(self, wave_emergence):
        """Test wave equation based emergence."""
        # Create test pattern
        pattern = torch.randn(3, 3)
        direction = torch.eye(3)
        
        # Evolve pattern
        evolved = wave_emergence.evolve_structure(pattern, direction)
        assert evolved.shape == pattern.shape
        assert torch.allclose(
            torch.norm(evolved, dim=-1),
            torch.ones(evolved.shape[:-1]),
            rtol=1e-5
        )
    
    def test_pattern_transition(self, pattern_transition):
        """Test pattern space transitions."""
        # Create source and target patterns
        source = torch.randn(2, 3)
        target = torch.randn(2, 3)
        
        # Create morphism
        morphism = pattern_transition.create_morphism(source, target)
        assert isinstance(morphism, EnrichedMorphism)
        assert torch.allclose(morphism.source_space, source)
        assert torch.allclose(morphism.target_space, target)
        
        # Test morphism composition
        mid = torch.randn(2, 3)
        first = pattern_transition.create_morphism(source, mid)
        second = pattern_transition.create_morphism(mid, target)
        composed = pattern_transition.compose(first, second)
        
        assert torch.allclose(composed.source_space, source)
        assert torch.allclose(composed.target_space, target)
        
        # Test error on non-composable morphisms
        with pytest.raises(ValueError):
            bad_mid = torch.randn(2, 4)  # Different dimension
            bad_morphism = pattern_transition.create_morphism(source, bad_mid)
            pattern_transition.compose(bad_morphism, second)
    
    def test_direction_computation(self, pattern_transition):
        """Test optimal direction computation."""
        # Create orthogonal patterns
        source = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        target = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
        
        # Compute direction
        direction = pattern_transition._compute_direction(source, target)
        
        # Verify direction properties
        assert direction.shape == (2, 2)
        assert torch.allclose(
            torch.norm(direction, dim=-1),
            torch.ones(direction.shape[:-1]),
            rtol=1e-5
        ) 