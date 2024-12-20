"""Tests for symplectic structure validation."""

import pytest
import torch
from typing import Tuple

from src.core.patterns.symplectic import SymplecticStructure
from src.validation.geometric.symplectic import (
    SymplecticStructureValidator,
    WavePacketValidator,
    OperadicValidator,
    QuantumGeometricValidator
)

@pytest.fixture
def structure() -> SymplecticStructure:
    """Create a test symplectic structure."""
    return SymplecticStructure(
        dim=4,
        preserve_structure=True,
        wave_enabled=True
    )

@pytest.fixture
def point() -> torch.Tensor:
    """Create a test point."""
    return torch.randn(4)

@pytest.fixture
def wave_packet(structure: SymplecticStructure) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a test wave packet with position and momentum."""
    position = torch.randn(2)
    momentum = torch.randn(2)
    packet = structure.enriched.create_wave_packet(position, momentum)
    return packet, position, momentum

class TestWavePacketValidator:
    """Tests for wave packet validation."""

    def test_validate_wave_packet(self, structure: SymplecticStructure, wave_packet: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        """Test wave packet validation."""
        validator = WavePacketValidator()
        packet, position, momentum = wave_packet
        
        result = validator.validate_wave_packet(
            structure,
            packet,
            position=position,
            momentum=momentum
        )
        
        assert result.is_valid
        assert 'is_normalized' in result.data
        assert 'position_valid' in result.data
        assert 'momentum_valid' in result.data

    def test_invalid_wave_packet(self, structure: SymplecticStructure):
        """Test validation of invalid wave packet."""
        validator = WavePacketValidator()
        invalid_packet = torch.randn(4) * 100  # Not normalized
        
        result = validator.validate_wave_packet(structure, invalid_packet)
        
        assert not result.is_valid
        assert not result.data['is_normalized']

class TestOperadicValidator:
    """Tests for operadic structure validation."""

    def test_validate_operadic_transition(self, structure: SymplecticStructure, point: torch.Tensor):
        """Test operadic transition validation."""
        validator = OperadicValidator()
        target = torch.randn(6)  # Different dimension
        
        result = validator.validate_operadic_transition(
            structure,
            point,
            target
        )
        
        assert 'dimensions_match' in result.data
        assert 'form_preserved' in result.data

    def test_invalid_operadic_transition(self, structure: SymplecticStructure):
        """Test validation of invalid operadic transition."""
        validator = OperadicValidator()
        source = torch.randn(2)  # Too small dimension
        target = torch.randn(4)
        
        result = validator.validate_operadic_transition(
            structure,
            source,
            target
        )
        
        assert not result.is_valid

class TestQuantumGeometricValidator:
    """Tests for quantum geometric tensor validation."""

    def test_validate_quantum_geometric(self, structure: SymplecticStructure, point: torch.Tensor):
        """Test quantum geometric tensor validation."""
        validator = QuantumGeometricValidator()
        
        result = validator.validate_quantum_geometric(structure, point)
        
        assert result.is_valid
        assert result.data['metric_symmetric']
        assert result.data['metric_positive']
        assert result.data['form_antisymmetric']
        assert result.data['compatible']

    def test_invalid_quantum_geometric(self, structure: SymplecticStructure):
        """Test validation of invalid quantum geometric tensor."""
        validator = QuantumGeometricValidator()
        invalid_point = torch.zeros(4)  # Degenerate point
        
        result = validator.validate_quantum_geometric(structure, invalid_point)
        
        assert not result.is_valid

class TestSymplecticStructureValidator:
    """Tests for complete symplectic structure validation."""

    def test_validate_all(self, structure: SymplecticStructure, point: torch.Tensor, wave_packet: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        """Test complete validation."""
        validator = SymplecticStructureValidator()
        packet, _, _ = wave_packet
        target = torch.randn(6)
        
        result = validator.validate_all(
            structure,
            point,
            wave_packet=packet,
            target_point=target
        )
        
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.message, str)
        assert isinstance(result.data, dict)

    def test_validate_partial(self, structure: SymplecticStructure, point: torch.Tensor):
        """Test validation with only required components."""
        validator = SymplecticStructureValidator()
        
        result = validator.validate_all(structure, point)
        
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.message, str)
        assert isinstance(result.data, dict) 