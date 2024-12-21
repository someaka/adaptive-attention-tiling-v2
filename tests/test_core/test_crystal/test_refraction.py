"""Tests for crystal refraction functionality."""

import numpy as np
import pytest
import torch

from src.core.crystal.refraction import RefractionSystem, BravaisLattice, BandStructure


def test_refraction_system():
    """Test the complete refraction system."""
    # Initialize system
    dim = 2
    system = RefractionSystem(dim=dim)
    
    # Create test pattern
    pattern = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ])
    
    # Test complete analysis
    lattice, symmetries, band_structure = system.analyze_pattern(pattern)
    
    # Check lattice
    assert isinstance(lattice, BravaisLattice)
    assert lattice.dim == dim
    
    # Check symmetries
    assert len(symmetries) > 0
    for sym in symmetries:
        assert sym.matrix.shape == (dim, dim)
        assert sym.translation.shape == (dim,)
        assert isinstance(sym.order, int)
        assert isinstance(sym.type, str)
    
    # Check band structure
    assert isinstance(band_structure, BandStructure)
    assert band_structure.energies.shape[1] == 4  # default num_bands
    assert len(band_structure.states) == len(band_structure.k_points)


def test_lattice_detection():
    """Test lattice detection functionality."""
    detector = RefractionSystem(dim=2).lattice_detector
    
    # Test with square lattice
    points = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ])
    
    lattice = detector.detect_lattice(points)
    assert lattice.dim == 2
    
    # Test lattice type determination
    lattice_type = detector._determine_lattice_type(torch.eye(2))
    assert lattice_type in ["cubic", "hexagonal", "triclinic"]
    
    # Test with invalid input
    with pytest.raises(ValueError):
        detector.detect_lattice(torch.randn(1, 3))  # Wrong dimension
