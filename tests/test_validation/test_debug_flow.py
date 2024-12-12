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

# Test parameters for faster execution
TEST_GRID_SIZE = 16
TEST_TIME_STEPS = 20
TEST_BATCH_SIZE = 2
TEST_SPACE_DIM = 2

class TestStabilityValidation:
    """Test suite for debugging stability validation."""
    
    @pytest.fixture
    def setup_stability(self):
        """Set up stability validation components."""
        return {
            'validator': LinearStabilityAnalyzer(tolerance=1e-6),
            'pattern': torch.randn(TEST_BATCH_SIZE, TEST_SPACE_DIM, 
                                 TEST_GRID_SIZE, TEST_GRID_SIZE) * 0.1,
            'dynamics': PatternDynamics(grid_size=TEST_GRID_SIZE, 
                                      space_dim=TEST_SPACE_DIM, dt=0.1)
        }

    @pytest.mark.parametrize("attr", ['is_valid', 'message', 'data'])
    def test_validate_stability_return_type(self, setup_stability, attr):
        """Test that validate_stability returns correct type."""
        validator = setup_stability['validator']
        pattern = setup_stability['pattern']
        dynamics = setup_stability['dynamics']
        
        result = validator.validate_stability(dynamics, pattern)
        
        assert isinstance(result, ValidationResult), "Should return ValidationResult"
        assert hasattr(result, attr), f"Should have {attr} attribute"
        
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
            'flow': torch.randn(TEST_TIME_STEPS, TEST_BATCH_SIZE, TEST_SPACE_DIM, 
                              TEST_GRID_SIZE, TEST_GRID_SIZE) * 0.1
        }
        
    @pytest.mark.parametrize("check", [
        (torch.isnan, "NaN"),
        (torch.isinf, "Inf"),
        (lambda x: x < 0, "negative values")
    ])
    def test_energy_computation(self, setup_energy, check):
        """Test energy computation is numerically stable."""
        validator = setup_energy['validator']
        flow = setup_energy['flow']
        check_fn, desc = check
        
        energy = validator.compute_energy(flow)
        assert not check_fn(energy).any(), f"Energy should not contain {desc}"
        
    def test_energy_conservation_validation(self, setup_energy):
        """Test energy conservation validation result."""
        validator = setup_energy['validator']
        flow = setup_energy['flow']
        
        result = validator.validate_energy_conservation(flow)
        
        assert isinstance(result, ValidationResult), "Should return ValidationResult"
        assert 'total_energy' in result.data, "Should have total_energy in data"
        assert 'energy_variation' in result.data, "Should have energy_variation in data"
        
class TestFlowProperties:
    """Test suite for debugging flow properties construction."""
    
    @pytest.fixture
    def setup_flow(self):
        """Set up flow validation components."""
        return {
            'validator': GeometricFlowValidator(tolerance=1e-5),
            'flow': torch.randn(TEST_TIME_STEPS, TEST_BATCH_SIZE, TEST_SPACE_DIM, 
                              TEST_GRID_SIZE, TEST_GRID_SIZE) * 0.1
        }
        
    def test_flow_properties_construction(self, setup_flow):
        """Test flow properties object construction."""
        validator = setup_flow['validator']
        flow = setup_flow['flow']
        
        properties = validator.compute_flow_properties(flow)
        
        assert isinstance(properties, FlowProperties), "Should return FlowProperties"
        assert hasattr(properties, 'is_stable'), "Should have stability flag"
        assert hasattr(properties, 'is_convergent'), "Should have convergence flag"
        
    def test_flow_properties_with_energy(self, setup_flow):
        """Test flow properties with explicit energy metrics."""
        validator = setup_flow['validator']
        flow = setup_flow['flow']
        
        properties = validator.compute_flow_properties(flow)
        
        assert hasattr(properties, 'total_energy'), "Should have total energy"
        assert hasattr(properties, 'energy_variation'), "Should have energy variation"
