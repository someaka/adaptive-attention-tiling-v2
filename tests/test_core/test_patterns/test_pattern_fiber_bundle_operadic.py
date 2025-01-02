"""Tests for pattern fiber bundle with operadic and enriched structures."""

import pytest
import torch
import numpy as np
from typing import Tuple

from src.core.tiling.patterns.pattern_fiber_bundle import PatternFiberBundle
from src.core.patterns.operadic_structure import AttentionOperad
from src.core.patterns.enriched_structure import PatternTransition, WaveEmergence

@pytest.fixture
def bundle():
    """Create a pattern fiber bundle instance."""
    return PatternFiberBundle(
        base_dim=2,
        fiber_dim=3,
        structure_group="SO(3)"
    )

@pytest.fixture
def test_data(bundle):
    """Create test data for bundle operations."""
    # Create test point in total space
    point = torch.randn(bundle.total_dim)
    
    # Create test tangent vector
    tangent = torch.randn(bundle.total_dim)
    
    # Create test path
    t = torch.linspace(0, 2*np.pi, 10)
    path = torch.stack([torch.cos(t), torch.sin(t)], dim=1)
    
    return point, tangent, path

class TestPatternFiberBundleOperadic:
    """Test pattern fiber bundle with operadic structures."""
    
    def test_dimension_handling(self, bundle, test_data):
        """Test operadic dimension handling."""
        point, _, _ = test_data
        
        # Test dimension handling for different target dimensions
        for target_dim in [2, 3, 4]:
            transformed = bundle._handle_dimension(point, target_dim)
            assert transformed.shape[-1] == target_dim
            
            if target_dim <= point.shape[-1]:
                # Check preservation of original coordinates
                assert torch.allclose(
                    transformed[..., :point.shape[-1]],
                    point[..., :target_dim],
                    rtol=1e-5
                )
    
    def test_connection_form(self, bundle, test_data):
        """Test connection form with operadic structure."""
        _, tangent, _ = test_data
        
        # Compute connection form
        connection = bundle.connection_form(tangent)
        
        # Verify shape
        assert connection.shape[-1] == bundle.fiber_dim
        
        # Test skew-symmetry for horizontal vectors
        horizontal = torch.zeros_like(tangent)
        horizontal[..., :bundle.base_dim] = tangent[..., :bundle.base_dim]
        
        horizontal_connection = bundle.connection_form(horizontal)
        skew_check = horizontal_connection + horizontal_connection.transpose(-2, -1)
        assert torch.allclose(
            skew_check,
            torch.zeros_like(skew_check),
            rtol=1e-5
        )
    
    def test_local_trivialization(self, bundle, test_data):
        """Test local trivialization with enriched structure."""
        point, _, _ = test_data
        
        # Get local trivialization
        local_chart, fiber_chart = bundle.local_trivialization(point)
        
        # Verify chart dimensions
        assert local_chart.dimension == bundle.base_dim
        assert fiber_chart.fiber_coordinates.shape[-1] == bundle.fiber_dim
        
        # Verify transition maps
        assert 'geometric_flow' in local_chart.transition_maps
        assert 'symplectic_form' in local_chart.transition_maps
        assert 'pattern_dynamics' in local_chart.transition_maps
        
        # Verify fiber chart structure
        assert fiber_chart.structure_group == "SO(3)"
        assert 'evolution' in fiber_chart.transition_functions
        assert 'dynamics' in fiber_chart.transition_functions
        assert 'symplectic' in fiber_chart.transition_functions
    
    def test_parallel_transport(self, bundle, test_data):
        """Test parallel transport with operadic structure."""
        _, _, path = test_data
        
        # Create test section
        section = torch.randn(bundle.fiber_dim)
        
        # Transport section
        transported = bundle.parallel_transport(section, path)
        
        # Verify shape
        assert transported.shape[0] == len(path)
        assert transported.shape[1] == bundle.fiber_dim
        
        # Verify parallel transport properties
        for i in range(len(path) - 1):
            # Compute difference between consecutive points
            diff = transported[i+1] - transported[i]
            
            # Project onto vertical space
            vertical_proj = diff[bundle.base_dim:]
            
            # Should be approximately horizontal
            assert torch.allclose(
                vertical_proj,
                torch.zeros_like(vertical_proj),
                rtol=1e-4,
                atol=1e-4
            )
    
    def test_symplectic_structure(self, bundle, test_data):
        """Test symplectic structure with operadic transitions."""
        point, _, _ = test_data
        
        # Get fiber coordinates
        fiber_coords = point[bundle.base_dim:bundle.base_dim + bundle.fiber_dim]
        
        # Compute symplectic form
        symplectic_form = bundle.symplectic.compute_form(fiber_coords)
        
        # Verify symplectic properties
        # 1. Anti-symmetry
        assert torch.allclose(
            symplectic_form.matrix,
            -symplectic_form.transpose().matrix,
            rtol=1e-5
        )
        
        # 2. Non-degeneracy (only check even-dimensional subspace)
        eigenvals = torch.linalg.eigvals(symplectic_form.matrix)
        even_dim = (bundle.fiber_dim // 2) * 2  # Get largest even dimension
        significant_eigenvals = eigenvals[:even_dim]  # Only check even subspace
        assert torch.all(torch.abs(significant_eigenvals) > 1e-5)
        
        # 3. Verify dimension is padded to next even number if needed
        expected_dim = bundle.fiber_dim if bundle.fiber_dim % 2 == 0 else bundle.fiber_dim + 1
        assert symplectic_form.matrix.shape[0] == expected_dim
        assert symplectic_form.matrix.shape[1] == expected_dim
    
    def test_structure_group_action(self, bundle, test_data):
        """Test structure group action with operadic structure."""
        point, tangent, _ = test_data
        
        # Create SO(3) element
        theta = torch.randn(1) * np.pi
        c, s = torch.cos(theta), torch.sin(theta)
        g = torch.tensor([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1.]
        ])
        
        # Apply group action
        transformed_point = torch.clone(point)
        transformed_point[bundle.base_dim:] = torch.matmul(
            g,
            point[bundle.base_dim:].unsqueeze(-1)
        ).squeeze(-1)
        
        transformed_tangent = torch.clone(tangent)
        transformed_tangent[bundle.base_dim:] = torch.matmul(
            g,
            tangent[bundle.base_dim:].unsqueeze(-1)
        ).squeeze(-1)
        
        # Compare connection forms
        original = bundle.connection_form(tangent)
        transformed = bundle.connection_form(transformed_tangent)
        
        # Should be related by conjugation
        expected = torch.matmul(
            torch.matmul(g, original),
            g.transpose(-2, -1)
        )
        
        assert torch.allclose(transformed, expected, rtol=1e-5) 