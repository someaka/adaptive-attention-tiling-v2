"""Tests for framework validation with motivic components."""

import pytest
import torch
from typing import Dict, Any

from src.validation.framework import FrameworkValidationResult
from src.validation.geometric.motivic import (
    MotivicValidation,
    MotivicValidator,
    MotivicRiemannianValidator
)
from src.core.patterns.motivic_riemannian import MotivicRiemannianStructure


@pytest.fixture
def motivic_validation() -> MotivicValidation:
    """Create a sample motivic validation result."""
    return MotivicValidation(
        is_valid=True,
        height_valid=True,
        dynamics_valid=True,
        cohomology_valid=True,
        message="All motivic validations passed",
        data={"test": "data"}
    )


@pytest.fixture
def framework_result(motivic_validation) -> FrameworkValidationResult:
    """Create a framework validation result with motivic component."""
    return FrameworkValidationResult(
        is_valid=True,
        message="Framework validation passed",
        motivic_result=motivic_validation
    )


class TestFrameworkMotivicIntegration:
    """Test suite for framework integration with motivic validation."""

    def test_framework_with_motivic(self, framework_result):
        """Test framework result with motivic component."""
        assert framework_result.is_valid
        assert framework_result.motivic_result is not None
        assert framework_result.motivic_result.is_valid
        assert framework_result.motivic_result.height_valid
        assert framework_result.motivic_result.dynamics_valid
        assert framework_result.motivic_result.cohomology_valid

    def test_framework_merge(self, framework_result, motivic_validation):
        """Test merging framework results with motivic components."""
        other_result = FrameworkValidationResult(
            is_valid=True,
            message="Other validation",
            motivic_result=MotivicValidation(
                is_valid=True,
                height_valid=True,
                dynamics_valid=True,
                cohomology_valid=True,
                message="Other motivic validation",
                data={"other": "data"}
            )
        )
        
        merged = framework_result.merge(other_result)
        assert merged.is_valid
        assert merged.motivic_result is not None
        assert merged.motivic_result.is_valid
        assert "Other motivic validation" in str(merged)

    def test_framework_to_dict(self, framework_result):
        """Test dictionary conversion with motivic component."""
        result_dict = framework_result.to_dict()
        assert "motivic" in result_dict
        motivic_dict = result_dict["motivic"]
        assert "is_valid" in motivic_dict
        assert "height_valid" in motivic_dict
        assert "dynamics_valid" in motivic_dict
        assert "cohomology_valid" in motivic_dict
        assert "message" in motivic_dict

    def test_framework_from_dict(self):
        """Test creating framework result from dictionary with motivic data."""
        data = {
            "is_valid": True,
            "message": "Test validation",
            "motivic": {
                "is_valid": True,
                "height_valid": True,
                "dynamics_valid": True,
                "cohomology_valid": True,
                "message": "Motivic validation passed"
            }
        }
        
        result = FrameworkValidationResult.from_dict(data)
        assert result.is_valid
        assert result.motivic_result is not None
        assert result.motivic_result.is_valid
        assert result.motivic_result.height_valid

    def test_framework_str_representation(self, framework_result):
        """Test string representation with motivic component."""
        str_result = str(framework_result)
        assert "Motivic:" in str_result
        assert "valid=True" in str_result

    def test_invalid_motivic_result(self):
        """Test framework with invalid motivic result."""
        invalid_motivic = MotivicValidation(
            is_valid=False,
            height_valid=False,
            dynamics_valid=True,
            cohomology_valid=True,
            message="Height validation failed",
            data={}
        )
        
        result = FrameworkValidationResult(
            is_valid=True,
            message="Framework validation",
            motivic_result=invalid_motivic
        )
        
        assert result.motivic_result is not None and not result.motivic_result.height_valid
        assert result.motivic_result is not None and "Height validation failed" in result.motivic_result.message

    def test_framework_with_invalid_motivic(self):
        """Test framework result with invalid motivic component."""
        invalid_motivic = MotivicValidation(
            is_valid=False,
            height_valid=False,
            dynamics_valid=True,
            cohomology_valid=True,
            message="Height validation failed",
            data={}
        )
        
        result = FrameworkValidationResult(
            is_valid=True,
            message="Framework validation",
            motivic_result=invalid_motivic
        )
        
        assert result.motivic_result is not None
        assert not result.motivic_result.height_valid
        assert "Height validation failed" in result.motivic_result.message

    def test_framework_without_motivic(self):
        """Test framework result without motivic component."""
        result = FrameworkValidationResult(
            is_valid=True,
            message="No motivic validation"
        )
        
        assert result.motivic_result is None
        assert "Motivic:" not in str(result)

    @pytest.mark.parametrize("height_valid,dynamics_valid,cohomology_valid", [
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (False, False, False)
    ])
    def test_partial_motivic_validation(
        self,
        height_valid: bool,
        dynamics_valid: bool,
        cohomology_valid: bool
    ):
        """Test framework with partially valid motivic results."""
        motivic_result = MotivicValidation(
            is_valid=all([height_valid, dynamics_valid, cohomology_valid]),
            height_valid=height_valid,
            dynamics_valid=dynamics_valid,
            cohomology_valid=cohomology_valid,
            message="Partial validation",
            data={}
        )
        
        result = FrameworkValidationResult(
            is_valid=True,
            message="Framework validation",
            motivic_result=motivic_result
        )
        
        assert result.motivic_result is not None
        assert result.motivic_result.height_valid == height_valid
        assert result.motivic_result.dynamics_valid == dynamics_valid
        assert result.motivic_result.cohomology_valid == cohomology_valid
  