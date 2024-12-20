"""Tests for fiber type management system."""

import pytest
import torch
from src.core.patterns.fiber_types import FiberType, FiberTypeManager


@pytest.fixture
def fiber_type_manager():
    """Create a fiber type manager instance."""
    return FiberTypeManager()


@pytest.fixture
def test_section():
    """Create a test section tensor."""
    return torch.randn(10, 3)  # Batch size 10, dimension 3


class TestFiberTypeManager:
    """Test fiber type management functionality."""

    def test_initialization(self, fiber_type_manager):
        """Test that manager initializes with standard types."""
        assert "Vector" in fiber_type_manager.list_fiber_types()
        assert "Principal" in fiber_type_manager.list_fiber_types()
        assert "Complex" in fiber_type_manager.list_fiber_types()
        
        assert "SO3" in fiber_type_manager.list_structure_groups()
        assert "U1" in fiber_type_manager.list_structure_groups()

    def test_register_fiber_type(self, fiber_type_manager):
        """Test fiber type registration."""
        # Try registering a valid type
        new_type = FiberType(
            name="TestType",
            dimension=3,
            structure_group="SO3",
            is_complex=False
        )
        fiber_type_manager.register_fiber_type(new_type)
        assert "TestType" in fiber_type_manager.list_fiber_types()
        
        # Try registering an invalid type
        invalid_type = FiberType(
            name="InvalidType",
            dimension=2,
            structure_group="InvalidGroup",
            is_complex=False
        )
        with pytest.raises(ValueError):
            fiber_type_manager.register_fiber_type(invalid_type)

    def test_register_conversion(self, fiber_type_manager):
        """Test conversion registration."""
        def test_conversion(x):
            return x

        # Register valid conversion
        fiber_type_manager.register_conversion(
            "Vector",
            "Principal",
            test_conversion
        )
        
        # Try registering invalid conversion
        with pytest.raises(ValueError):
            fiber_type_manager.register_conversion(
                "InvalidType",
                "Vector",
                test_conversion
            )

    def test_validate_fiber_type(self, fiber_type_manager, test_section):
        """Test fiber type validation."""
        # Test vector type validation
        assert fiber_type_manager.validate_fiber_type(
            test_section,
            "Vector",
            3
        )
        
        # Test principal type validation
        principal_section = torch.eye(3).expand(10, -1, -1)  # Batch of identity matrices
        assert fiber_type_manager.validate_fiber_type(
            principal_section,
            "Principal",
            3
        )
        
        # Test complex type validation
        complex_section = torch.randn(10, 1, dtype=torch.complex64)
        assert fiber_type_manager.validate_fiber_type(
            complex_section,
            "Complex",
            1
        )
        
        # Test invalid type
        with pytest.raises(ValueError):
            fiber_type_manager.validate_fiber_type(
                test_section,
                "InvalidType",
                3
            )

    def test_convert_fiber_type(self, fiber_type_manager, test_section):
        """Test fiber type conversion."""
        # Test vector to principal conversion
        principal = fiber_type_manager.convert_fiber_type(
            test_section,
            "Vector",
            "Principal",
            3
        )
        assert principal.shape[-2:] == (3, 3)
        assert fiber_type_manager.validate_fiber_type(
            principal,
            "Principal",
            3
        )
        
        # Test principal to vector conversion
        vector = fiber_type_manager.convert_fiber_type(
            principal,
            "Principal",
            "Vector",
            3
        )
        assert vector.shape[-1] == 3
        assert fiber_type_manager.validate_fiber_type(
            vector,
            "Vector",
            3
        )
        
        # Test invalid conversion
        with pytest.raises(ValueError):
            fiber_type_manager.convert_fiber_type(
                test_section,
                "Vector",
                "Complex",
                3
            )

    def test_check_compatibility(self, fiber_type_manager):
        """Test structure group compatibility checking."""
        # Test valid compatibilities
        assert fiber_type_manager.check_compatibility("Vector", "SO3")
        assert fiber_type_manager.check_compatibility("Principal", "SO3")
        assert fiber_type_manager.check_compatibility("Complex", "U1")
        
        # Test invalid compatibilities
        assert not fiber_type_manager.check_compatibility("Vector", "U1")
        assert not fiber_type_manager.check_compatibility("Complex", "SO3")
        assert not fiber_type_manager.check_compatibility("InvalidType", "SO3")

    def test_get_fiber_type(self, fiber_type_manager):
        """Test fiber type retrieval."""
        vector_type = fiber_type_manager.get_fiber_type("Vector")
        assert vector_type is not None
        assert vector_type.name == "Vector"
        assert vector_type.dimension == 3
        assert vector_type.structure_group == "SO3"
        
        assert fiber_type_manager.get_fiber_type("InvalidType") is None

    def test_get_structure_group(self, fiber_type_manager):
        """Test structure group retrieval."""
        so3_group = fiber_type_manager.get_structure_group("SO3")
        assert so3_group is not None
        assert so3_group["dimension"] == 3
        assert so3_group["is_compact"]
        assert so3_group["is_connected"]
        assert "Vector" in so3_group["compatible_types"]
        assert "Principal" in so3_group["compatible_types"]
        
        assert fiber_type_manager.get_structure_group("InvalidGroup") is None

    def test_group_element_validation(self, fiber_type_manager):
        """Test group element validation."""
        # Test SO(3) element
        so3_element = torch.eye(3)  # Identity in SO(3)
        assert fiber_type_manager._validate_group_element(so3_element, "SO3")
        
        # Test invalid SO(3) element
        invalid_so3 = torch.randn(3, 3)  # Random matrix, not in SO(3)
        assert not fiber_type_manager._validate_group_element(invalid_so3, "SO3")
        
        # Test U(1) element
        u1_element = torch.exp(1j * torch.tensor([0.5]))  # Complex phase
        assert fiber_type_manager._validate_group_element(u1_element, "U1")
        
        # Test invalid U(1) element
        invalid_u1 = torch.tensor([2.0], dtype=torch.complex64)  # Not unit modulus
        assert not fiber_type_manager._validate_group_element(invalid_u1, "U1")

    def test_batch_operations(self, fiber_type_manager):
        """Test operations with batched tensors."""
        batch_size = 10
        
        # Test batched vector validation
        vector_batch = torch.randn(batch_size, 3)
        assert fiber_type_manager.validate_fiber_type(
            vector_batch,
            "Vector",
            3
        )
        
        # Test batched principal validation
        principal_batch = torch.eye(3).expand(batch_size, -1, -1)
        assert fiber_type_manager.validate_fiber_type(
            principal_batch,
            "Principal",
            3
        )
        
        # Test batched conversion
        converted = fiber_type_manager.convert_fiber_type(
            vector_batch,
            "Vector",
            "Principal",
            3
        )
        assert converted.shape == (batch_size, 3, 3)
        assert fiber_type_manager.validate_fiber_type(
            converted,
            "Principal",
            3
        )
