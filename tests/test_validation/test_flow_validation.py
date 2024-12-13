"""Unit tests for flow validation.

Tests cover:
1. Energy conservation
2. Flow monotonicity
3. Long-time existence
4. Singularity detection
"""

import pytest
import torch

from src.validation.geometric.flow import (
    FlowValidator,
    FlowProperties,
    FlowValidationResult
)


class TestFlowValidation:
    """Test flow validation utilities."""

    def setup_method(self):
        """Setup test parameters."""
        self.batch_size = 2
        self.dim = 3
        self.validator = FlowValidator(
            energy_threshold=1e-6,
            monotonicity_threshold=1e-4,
            singularity_threshold=1.0,
            max_iterations=1000,
            tolerance=1e-6
        )
        self.t = torch.linspace(0, 10, 100)

    def test_energy_conservation(self):
        """Test energy conservation validation."""
        def generate_flow(t):
            # Create a flow that conserves energy (exponential decay)
            t = t.reshape(-1, 1)  # Add batch dimension
            flow = torch.exp(-0.1 * t) * torch.ones((t.shape[0], self.dim))
            return flow.unsqueeze(0).repeat(self.batch_size, 1, 1)  # Add batch dimension
            
        # Generate test flow
        flow = generate_flow(self.t)
        
        # Validate energy conservation
        result = self.validator.validate(flow)
        
        # Check results
        assert isinstance(result, FlowValidationResult)
        assert result.data is not None
        assert 'energy_metrics' in result.data
        assert result.is_valid  # Energy should be conserved for exponential decay

    def test_flow_monotonicity(self):
        """Test flow monotonicity validation."""
        def generate_monotonic_flow(t):
            # Create a monotonic flow (exponential decay)
            t = t.reshape(-1, 1)  # Add batch dimension
            flow = torch.exp(-0.1 * t) * torch.ones((t.shape[0], self.dim))
            return flow.unsqueeze(0).repeat(self.batch_size, 1, 1)  # Add batch dimension
            
        # Generate test flow
        flow = generate_monotonic_flow(self.t)
        
        # Validate monotonicity
        result = self.validator.validate(flow)
        
        # Check results
        assert isinstance(result, FlowValidationResult)
        assert result.data is not None
        assert 'monotonicity_metrics' in result.data
        assert result.is_valid  # Flow should be monotonic

    def test_long_time_existence(self):
        """Test long-time existence validation."""
        def generate_stable_flow(t):
            # Create a stable flow (tanh)
            t = t.reshape(-1, 1)  # Add batch dimension
            flow = torch.tanh(0.1 * t) * torch.ones((t.shape[0], self.dim))
            return flow.unsqueeze(0).repeat(self.batch_size, 1, 1)  # Add batch dimension
            
        # Generate test flow with longer time horizon
        t = torch.linspace(0, 100, 100)
        flow = generate_stable_flow(t)
        
        # Validate long-time existence
        result = self.validator.validate(flow)
        
        # Check results
        assert isinstance(result, FlowValidationResult)
        assert result.data is not None
        assert 'existence_metrics' in result.data
        assert result.is_valid  # Flow should exist for long time

    def test_singularity_detection(self):
        """Test singularity detection."""
        def generate_singular_flow(t):
            # Create a flow with singularity at t=0
            t = t.reshape(-1, 1)  # Add batch dimension
            x = t * torch.ones((t.shape[0], self.dim))
            flow = 1 / (x + 1e-6)  # Add small epsilon to avoid division by zero
            return flow.unsqueeze(0).repeat(self.batch_size, 1, 1)  # Add batch dimension
            
        # Generate test flow near singularity
        t = torch.linspace(-5, 5, 1000)
        flow = generate_singular_flow(t)
        
        # Detect singularities
        result = self.validator.validate(flow)
        
        # Check results
        assert isinstance(result, FlowValidationResult)
        assert result.data is not None
        assert 'singularity_metrics' in result.data
        assert not result.is_valid  # Should detect singularity

    def test_flow_properties(self):
        """Test flow properties computation."""
        # Generate random flow
        t = self.t.reshape(-1, 1)  # Add batch dimension
        flow = torch.exp(-0.1 * t) * torch.randn((t.shape[0], self.dim))
        flow = flow.unsqueeze(0).repeat(self.batch_size, 1, 1)  # Add batch dimension
        
        # Compute properties
        result = self.validator.validate(flow)
        
        # Check results
        assert isinstance(result, FlowValidationResult)
        assert result.data is not None
        assert 'flow_properties' in result.data
        assert isinstance(result.data['flow_properties'], FlowProperties)

    def test_flow_decomposition(self):
        """Test flow decomposition validation."""
        # Generate random flow
        t = self.t.reshape(-1, 1)  # Add batch dimension
        flow = torch.exp(-0.1 * t) * torch.randn((t.shape[0], self.dim))
        flow = flow.unsqueeze(0).repeat(self.batch_size, 1, 1)  # Add batch dimension
        
        # Validate decomposition
        result = self.validator.validate(flow)
        
        # Check results
        assert isinstance(result, FlowValidationResult)
        assert result.data is not None
        assert 'decomposition_metrics' in result.data
        assert result.is_valid  # Decomposition should always be valid

    def test_validation_integration(self):
        """Test integration of all validation methods."""
        # Generate random flow
        t = self.t.reshape(-1, 1)  # Add batch dimension
        flow = torch.exp(-0.1 * t) * torch.randn((t.shape[0], self.dim))
        flow = flow.unsqueeze(0).repeat(self.batch_size, 1, 1)  # Add batch dimension
        
        # Run all validations
        result = self.validator.validate(flow)
        
        # Check results
        assert isinstance(result, FlowValidationResult)
        assert result.data is not None
        assert 'energy_metrics' in result.data
        assert 'monotonicity_metrics' in result.data
        assert 'existence_metrics' in result.data
        assert 'singularity_metrics' in result.data
