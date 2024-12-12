"""Tests for pattern-flow integration validation.

This module tests the integration between pattern dynamics and geometric flow validation,
ensuring that patterns evolve correctly under the flow and maintain stability.
"""

import torch
import pytest
import numpy as np
from src.validation.patterns.stability import PatternStabilityValidator
from src.validation.patterns.formation import PatternFormationValidator
from src.validation.geometric.flow import GeometricFlowValidator
from src.neural.attention.pattern.dynamics import PatternDynamics

class TestPatternFlowIntegration:
    """Test suite for pattern-flow integration validation."""
    
    @pytest.fixture
    def setup_test_parameters(self):
        """Set up test parameters."""
        return {
            'batch_size': 1,
            'grid_size': 16,
            'space_dim': 2,
            'time_steps': 20,
            'tolerance': 1e-4,
            'energy_threshold': 1e-6,
            'dt': 0.1
        }
        
    @pytest.fixture
    def pattern_validator(self, setup_test_parameters):
        """Create pattern stability validator."""
        return PatternStabilityValidator(
            tolerance=setup_test_parameters['tolerance'],
            max_time=setup_test_parameters['time_steps']
        )
        
    @pytest.fixture
    def formation_validator(self, setup_test_parameters):
        """Create pattern formation validator."""
        return PatternFormationValidator(
            tolerance=setup_test_parameters['tolerance']
        )
        
    @pytest.fixture
    def flow_validator(self, setup_test_parameters):
        """Create geometric flow validator."""
        return GeometricFlowValidator(
            tolerance=setup_test_parameters['tolerance']
        )
        
    @pytest.fixture
    def pattern_dynamics(self, setup_test_parameters):
        """Create pattern dynamics."""
        return PatternDynamics(
            grid_size=setup_test_parameters['grid_size'],
            space_dim=setup_test_parameters['space_dim'],
            dt=setup_test_parameters['dt']
        )
        
    def test_pattern_flow_stability(self, setup_test_parameters, pattern_validator, 
                                  formation_validator, flow_validator, pattern_dynamics):
        """Test that pattern-induced flow maintains stability."""
        params = setup_test_parameters
        
        # Create initial pattern
        pattern = torch.randn(
            params['batch_size'],
            params['space_dim'],
            params['grid_size'],
            params['grid_size'],
            dtype=torch.float32
        ) * 0.1
        
        # Validate initial pattern formation
        formation_result = formation_validator.validate(
            pattern_dynamics, 
            pattern,
            time_steps=params['time_steps']
        )
        assert formation_result.is_valid, "Initial pattern formation should be valid"
        
        # Evolve pattern and get flow
        evolution = pattern_dynamics.evolve_pattern(
            pattern,
            diffusion_coefficient=0.1,
            steps=params['time_steps']
        )
        flow = torch.stack(evolution)
        
        # Validate pattern stability
        stability_result = pattern_validator.validate(
            pattern_dynamics,
            pattern,
            parameter_name='dt'
        )
        assert stability_result.is_valid, "Initial pattern should be stable"
        assert stability_result.error_message == '', "No error message should be present"
        
        # Get stability metrics
        growth_rates, mode_shapes = pattern_validator.linear_validator.get_unstable_modes(
            pattern_validator.linear_validator.analyze_stability(pattern_dynamics, pattern)
        )
        assert len(growth_rates) == 0, "No unstable modes should be present"
        
        # Validate flow existence
        flow_result = flow_validator.validate_long_time_existence(flow)
        assert flow_result.is_valid, "Flow should exist for long time"
        
        # Check flow properties
        properties = flow_validator.compute_flow_properties(flow)
        assert properties.is_stable, "Flow should be stable"
        assert properties.is_convergent, "Flow should converge"
        
        # Validate final pattern
        final_pattern = flow[-1]
        final_result = pattern_validator.validate(
            pattern_dynamics,
            final_pattern,
            parameter_name='dt'
        )
        assert final_result.is_valid, "Final pattern should be stable"
        assert final_result.error_message == '', "No error message should be present"
        
    def test_pattern_flow_energy(self, setup_test_parameters, pattern_validator,
                                formation_validator, flow_validator, pattern_dynamics):
        """Test energy conservation in pattern-induced flow."""
        params = setup_test_parameters
        
        # Create initial pattern
        pattern = torch.randn(
            params['batch_size'],
            params['space_dim'],
            params['grid_size'],
            params['grid_size'],
            dtype=torch.float32
        ) * 0.1
        
        # Evolve pattern and get flow
        evolution = pattern_dynamics.evolve_pattern(
            pattern,
            diffusion_coefficient=0.1,
            steps=params['time_steps']
        )
        flow = torch.stack(evolution)
        
        # Validate energy conservation
        energy_result = flow_validator.validate_energy_conservation(flow)
        assert energy_result.is_valid, "Flow should conserve energy"
        
        # Check energy metrics
        assert energy_result.data['total_energy'] > 0, "Total energy should be positive"
        assert energy_result.data['energy_variation'] < params['energy_threshold'], \
            "Energy variation should be small"
        
    def test_pattern_flow_bifurcation(self, setup_test_parameters, pattern_validator,
                                     formation_validator, flow_validator, pattern_dynamics):
        """Test pattern-flow behavior near bifurcation points."""
        params = setup_test_parameters
        
        # Generate patterns with different control parameters
        control_values = torch.linspace(0.1, 1.0, 2)  
        patterns = []
        flows = []
        
        # Create base pattern
        base_pattern = torch.randn(
            params['batch_size'],
            params['space_dim'],
            params['grid_size'],
            params['grid_size'],
            dtype=torch.float32
        ) * 0.1
        
        # Reduce time steps for bifurcation test
        test_steps = params['time_steps'] // 2
        
        for control in control_values:
            # Scale pattern by control parameter
            pattern = base_pattern * control
            patterns.append(pattern)
            
            # Evolve pattern
            evolution = pattern_dynamics.evolve_pattern(
                pattern,
                diffusion_coefficient=0.1,
                steps=test_steps
            )
            flow = torch.stack(evolution)
            flows.append(flow)
            
        # Check stability across control parameter range
        for pattern, flow in zip(patterns, flows):
            # Validate pattern stability
            stability_result = pattern_validator.validate(
                pattern_dynamics,
                pattern,
                parameter_name='dt'
            )
            assert stability_result.is_valid, "Pattern should be stable"
            assert stability_result.error_message == '', "No error message should be present"
            
            # Validate flow existence
            flow_result = flow_validator.validate_long_time_existence(flow)
            assert flow_result.is_valid, "Flow should exist for long time"
            
            # Check flow properties
            properties = flow_validator.compute_flow_properties(flow)
            assert properties.is_stable, "Flow should be stable"
