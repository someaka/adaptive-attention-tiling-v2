"""Debug tests for pattern flow validation.

This module contains focused unit tests to debug specific issues in:
1. Stability validation unpacking
2. Energy conservation validation
3. Flow properties construction
"""

import pytest
import torch
from src.validation.patterns.stability import (
    PatternStabilityValidator,
    LinearStabilityAnalyzer,
    ValidationResult,
    StabilitySpectrum
)
from src.validation.geometric.flow import (
    GeometricFlowValidator,
    FlowProperties,
    EnergyMetrics
)
from src.neural.attention.pattern.dynamics import PatternDynamics

class TestStabilityValidation:
    """Test suite for debugging stability validation."""
    
    @pytest.fixture
    def setup_stability(self):
        """Set up stability validation components."""
        return {
            'validator': LinearStabilityAnalyzer(tolerance=1e-6),
            'pattern': torch.randn(2, 2, 32, 32) * 0.1,
            'dynamics': PatternDynamics(grid_size=32, space_dim=2, dt=0.1)
        }

    def test_validate_stability_return_type(self, setup_stability):
        """Test that validate_stability returns correct type."""
        validator = setup_stability['validator']
        pattern = setup_stability['pattern']
        dynamics = setup_stability['dynamics']
        
        result = validator.validate_stability(dynamics, pattern)
        
        assert isinstance(result, ValidationResult), "Should return ValidationResult"
        assert hasattr(result, 'is_valid'), "Should have is_valid attribute"
        assert hasattr(result, 'message'), "Should have message attribute"
        assert hasattr(result, 'data'), "Should have data attribute"
        
    def test_stability_spectrum_computation(self, setup_stability):
        """Test stability spectrum computation."""
        validator = setup_stability['validator']
        pattern = setup_stability['pattern']
        dynamics = setup_stability['dynamics']
        
        spectrum = validator.analyze_stability(dynamics, pattern)
        
        assert isinstance(spectrum, StabilitySpectrum), "Should return StabilitySpectrum"
        assert spectrum.eigenvalues.shape[0] == spectrum.eigenvectors.shape[1], \
            "Number of eigenvalues should match eigenvectors"
        assert not torch.isnan(spectrum.eigenvalues).any(), "Eigenvalues should not contain NaN"
        assert not torch.isinf(spectrum.eigenvalues).any(), "Eigenvalues should not contain Inf"

class TestEnergyValidation:
    """Test suite for debugging energy conservation validation."""
    
    @pytest.fixture
    def setup_energy(self):
        """Set up energy validation components."""
        return {
            'validator': GeometricFlowValidator(tolerance=1e-5),
            'flow': torch.randn(100, 2, 2, 32, 32) * 0.1  # [time, batch, channels, height, width]
        }
        
    def test_energy_computation(self, setup_energy):
        """Test energy computation is numerically stable."""
        validator = setup_energy['validator']
        flow = setup_energy['flow']
        
        energy = validator.compute_energy(flow)
        
        assert not torch.isnan(energy).any(), "Energy should not contain NaN"
        assert not torch.isinf(energy).any(), "Energy should not contain Inf"
        assert (energy >= 0).all(), "Energy should be non-negative"
        
    def test_energy_conservation_validation(self, setup_energy):
        """Test energy conservation validation result."""
        validator = setup_energy['validator']
        flow = setup_energy['flow']
        
        result = validator.validate_energy_conservation(flow)
        
        assert isinstance(result, ValidationResult), "Should return ValidationResult"
        print(f"Energy validation result: {result.message}")
        print(f"Energy validation data: {result.data}")
        
        if not result.is_valid:
            # Print detailed diagnostics
            energy = validator.compute_energy(flow)
            print(f"Energy statistics:")
            print(f"Mean: {torch.mean(energy)}")
            print(f"Std: {torch.std(energy)}")
            print(f"Min: {torch.min(energy)}")
            print(f"Max: {torch.max(energy)}")

class TestFlowProperties:
    """Test suite for debugging flow properties construction."""
    
    @pytest.fixture
    def setup_flow(self):
        """Set up flow validation components."""
        return {
            'validator': GeometricFlowValidator(tolerance=1e-5),
            'flow': torch.randn(100, 2, 2, 32, 32) * 0.1
        }
        
    def test_flow_properties_construction(self, setup_flow):
        """Test flow properties object construction."""
        validator = setup_flow['validator']
        flow = setup_flow['flow']
        
        try:
            properties = validator.compute_flow_properties(flow)
            
            assert isinstance(properties, FlowProperties), "Should return FlowProperties"
            assert hasattr(properties, 'total_energy'), "Should have total_energy attribute"
            assert hasattr(properties, 'energy_variation'), "Should have energy_variation attribute"
            
            print(f"Flow properties:")
            print(f"Is stable: {properties.is_stable}")
            print(f"Is conservative: {properties.is_conservative}")
            print(f"Total energy: {properties.total_energy}")
            print(f"Energy variation: {properties.energy_variation}")
            
        except Exception as e:
            pytest.fail(f"Flow properties construction failed: {str(e)}")
            
    def test_flow_properties_with_energy(self, setup_flow):
        """Test flow properties with explicit energy metrics."""
        flow = setup_flow['flow']
        
        # Create properties with energy metrics
        properties = FlowProperties(
            is_stable=True,
            is_conservative=True,
            total_energy=100.0,
            energy_variation=0.01
        )
        
        assert properties.total_energy == 100.0, "Should store total_energy"
        assert properties.energy_variation == 0.01, "Should store energy_variation"
