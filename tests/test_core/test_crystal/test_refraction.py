"""
Unit tests for the crystal refraction pattern analysis system.

Tests cover:
1. Symmetry group detection and analysis
2. Lattice structure identification
3. Brillouin zone computation
4. Band structure analysis
5. Phonon mode computation
6. Defect structure analysis
7. Optical properties
8. Thermal transport properties
9. Surface states
10. Strain effects
"""

import pytest
import torch
import numpy as np
from typing import List, Tuple
from itertools import product

from src.core.crystal.refraction import RefractionSystem
from src.utils.test_helpers import assert_crystal_properties

class TestRefractionSystem:
    @pytest.fixture
    def space_dim(self):
        """Dimension of physical space."""
        return 3

    @pytest.fixture
    def lattice_points(self):
        """Number of lattice points per dimension."""
        return 8

    @pytest.fixture
    def refraction_system(self, space_dim, lattice_points):
        """Create a test refraction system."""
        return RefractionSystem(
            dim=space_dim,
            size=lattice_points,
            symmetry_tolerance=1e-5
        )

    def test_symmetry_computation(self, refraction_system, space_dim):
        """Test symmetry group detection and properties."""
        # Create test pattern with cubic symmetry
        points = torch.tensor(list(product([-1, 1], repeat=space_dim)))
        pattern = torch.zeros((2,)*space_dim)
        for p in points:
            pattern[tuple(p)] = 1.0
            
        # Compute symmetries
        symmetry_group = refraction_system.compute_symmetries(pattern)
        
        # Test group properties
        assert len(symmetry_group.generators) == 3, "Cubic group should have 3 generators"
        
        # Test closure
        def test_closure(g1, g2):
            """Test if composition stays in group."""
            return symmetry_group.contains(g1 @ g2)
            
        generators = symmetry_group.generators
        assert all(
            test_closure(g1, g2)
            for g1, g2 in product(generators, repeat=2)
        ), "Group should be closed under composition"
        
        # Test pattern invariance
        for g in generators:
            transformed = refraction_system.apply_symmetry(pattern, g)
            assert torch.allclose(
                pattern, transformed, rtol=1e-5
            ), "Pattern should be invariant under symmetry"

    def test_lattice_detection(self, refraction_system, space_dim):
        """Test Bravais lattice identification."""
        # Create simple cubic lattice
        basis_vectors = torch.eye(space_dim)
        points = torch.tensor(list(product(range(2), repeat=space_dim)))
        
        # Detect lattice
        lattice = refraction_system.detect_lattice(points)
        
        # Test basis vector properties
        assert len(lattice.basis) == space_dim, "Should have correct number of basis vectors"
        
        # Test orthogonality (for cubic)
        basis = torch.stack(lattice.basis)
        products = basis @ basis.T
        assert torch.allclose(
            products, torch.eye(space_dim), rtol=1e-4
        ), "Cubic lattice vectors should be orthogonal"
        
        # Test primitive cell volume
        volume = lattice.primitive_cell_volume()
        assert torch.allclose(
            volume, torch.tensor(1.0), rtol=1e-4
        ), "Primitive cell volume should be 1 for simple cubic"

    def test_brillouin_zones(self, refraction_system, space_dim):
        """Test Brillouin zone computation."""
        # Create reciprocal lattice vectors
        recip_vectors = 2 * np.pi * torch.eye(space_dim)
        
        # Compute Brillouin zones
        zones = refraction_system.brillouin_zones(recip_vectors)
        
        # Test first Brillouin zone
        first_zone = zones[0]
        
        # Test volume
        volume = first_zone.volume()
        expected_volume = (2 * np.pi)**space_dim
        assert torch.allclose(
            volume, torch.tensor(expected_volume), rtol=1e-4
        ), "First Brillouin zone should have correct volume"
        
        # Test point classification
        def is_in_first_zone(k_point):
            """Test if k-point is in first Brillouin zone."""
            return first_zone.contains(k_point)
            
        # Origin should be in first zone
        assert is_in_first_zone(torch.zeros(space_dim)), "Origin should be in first zone"
        
        # Points beyond π should not be in first zone
        far_point = 2 * np.pi * torch.ones(space_dim)
        assert not is_in_first_zone(far_point), "Far points should not be in first zone"

    def test_band_structure(self, refraction_system, space_dim):
        """Test band structure computation."""
        # Create test potential (simple harmonic)
        def potential(r):
            return torch.sum(torch.sin(2 * np.pi * r)**2)
            
        # Compute band structure
        bands = refraction_system.band_structure(potential)
        
        # Test basic band properties
        assert len(bands) > 0, "Should have at least one band"
        
        # Test energy ordering
        energies = [band.energy_at_gamma() for band in bands]
        assert all(
            e1 <= e2 for e1, e2 in zip(energies[:-1], energies[1:])
        ), "Bands should be ordered by energy"
        
        # Test time reversal symmetry
        k_point = torch.randn(space_dim)
        for band in bands:
            energy_k = band.energy(k_point)
            energy_minus_k = band.energy(-k_point)
            assert torch.allclose(
                energy_k, energy_minus_k, rtol=1e-4
            ), "Bands should respect time reversal symmetry"

    def test_phonon_modes(self, refraction_system, space_dim):
        """Test phonon spectrum computation."""
        # Create force constant matrix (simple spring model)
        force_constants = torch.eye(space_dim)
        
        # Compute phonon spectrum
        phonons = refraction_system.phonon_modes(force_constants)
        
        # Test acoustic modes
        acoustic_modes = phonons.acoustic_modes()
        assert len(acoustic_modes) == space_dim, "Should have correct number of acoustic modes"
        
        # Test acoustic mode properties
        k_small = 0.01 * torch.ones(space_dim)
        for mode in acoustic_modes:
            omega = mode.frequency(k_small)
            # Should be approximately linear for small k
            assert torch.allclose(
                omega, torch.norm(k_small), rtol=1e-2
            ), "Acoustic modes should be linear at small k"
            
        # Test polarization vectors
        polarizations = phonons.polarization_vectors(k_small)
        # Should be orthonormal
        products = polarizations @ polarizations.T
        assert torch.allclose(
            products, torch.eye(len(polarizations)), rtol=1e-4
        ), "Polarization vectors should be orthonormal"
        
        # Test Debye model at low k
        def test_debye_scaling(k):
            """Test if frequencies scale correctly at low k."""
            return all(
                torch.allclose(
                    mode.frequency(k),
                    mode.sound_velocity() * torch.norm(k),
                    rtol=1e-2
                )
                for mode in acoustic_modes
            )
            
        assert test_debye_scaling(k_small), "Should follow Debye model at low k"

    def test_defect_structures(self, refraction_system, space_dim, lattice_points):
        """Test defect structure analysis."""
        # Create perfect lattice
        perfect_lattice = torch.ones((lattice_points,) * space_dim)
        
        # Create point defect
        point_defect = perfect_lattice.clone()
        defect_pos = (lattice_points//2,) * space_dim
        point_defect[defect_pos] = 0.0
        
        # Create dislocation
        dislocation = perfect_lattice.clone()
        burgers_vector = torch.tensor([1, 0, 0])
        dislocation = refraction_system.create_edge_dislocation(
            dislocation, burgers_vector
        )
        
        # Test defect detection
        defects = refraction_system.detect_defects(point_defect)
        assert len(defects.point_defects) == 1, "Should detect single point defect"
        
        # Test strain field around dislocation
        strain = refraction_system.compute_strain_field(dislocation)
        r = torch.tensor([1.0, 0.0, 0.0])  # Position relative to dislocation line
        theoretical_strain = burgers_vector.outer(r) / (2 * np.pi * torch.norm(r)**2)
        assert torch.allclose(
            strain(r), theoretical_strain, rtol=1e-2
        ), "Dislocation strain field should match theory"
        
        # Test defect energetics
        formation_energy = refraction_system.defect_formation_energy(point_defect)
        assert formation_energy > 0, "Defect formation should cost energy"
        
        # Test defect diffusion
        activation_energy = refraction_system.compute_migration_barrier(
            point_defect, defect_pos
        )
        assert activation_energy > 0, "Diffusion should have energy barrier"

    def test_optical_properties(self, refraction_system, space_dim):
        """Test optical and dielectric properties."""
        # Create dielectric tensor
        epsilon = torch.eye(space_dim) * 2.0  # Simple isotropic case
        
        # Test dispersion relation
        def test_dispersion(k_vector):
            """Test dispersion relation for light."""
            omega = refraction_system.photon_frequency(k_vector, epsilon)
            return torch.allclose(
                omega**2,
                torch.dot(k_vector, k_vector) / epsilon[0,0],
                rtol=1e-4
            )
        
        k_test = torch.ones(space_dim)
        assert test_dispersion(k_test), "Should satisfy dispersion relation"
        
        # Test birefringence
        crystal_axis = torch.tensor([0., 0., 1.])
        birefringent_epsilon = epsilon.clone()
        birefringent_epsilon[2,2] = 2.5  # Different along c-axis
        
        # Get ordinary and extraordinary rays
        k_vector = torch.tensor([1., 0., 1.]) / np.sqrt(2)
        rays = refraction_system.compute_optical_rays(k_vector, birefringent_epsilon)
        
        assert len(rays) == 2, "Should have ordinary and extraordinary rays"
        assert torch.abs(
            rays[0].polarization.dot(rays[1].polarization)
        ) < 1e-4, "Rays should be orthogonally polarized"

    def test_thermal_transport(self, refraction_system, space_dim, lattice_points):
        """Test thermal transport properties."""
        # Create temperature gradient
        T_hot = 300.0
        T_cold = 290.0
        temp_gradient = torch.linspace(T_hot, T_cold, lattice_points)
        temp_field = temp_gradient.repeat(lattice_points, lattice_points, 1)
        
        # Compute thermal conductivity tensor
        kappa = refraction_system.thermal_conductivity(temp_field)
        
        # Test positive definiteness
        eigenvals = torch.linalg.eigvalsh(kappa)
        assert torch.all(eigenvals > 0), "Thermal conductivity should be positive definite"
        
        # Test Fourier's law
        heat_flux = -kappa @ torch.tensor([0., 0., (T_hot - T_cold)/lattice_points])
        assert heat_flux[2] < 0, "Heat should flow from hot to cold"
        
        # Test Wiedemann-Franz law (if electronic contribution)
        if hasattr(refraction_system, 'electrical_conductivity'):
            sigma = refraction_system.electrical_conductivity(temp_field)
            L = kappa / (sigma * temp_gradient.mean())
            L_theoretical = 2.44e-8  # Lorenz number
            assert torch.allclose(
                L, torch.tensor(L_theoretical), rtol=1e-2
            ), "Should satisfy Wiedemann-Franz law"

    def test_surface_states(self, refraction_system, space_dim):
        """Test surface and interface phenomena."""
        # Create surface by terminating bulk
        bulk = torch.ones((lattice_points,) * space_dim)
        surface = bulk.clone()
        surface[:,:,lattice_points//2:] = 0.0
        
        # Compute surface states
        states = refraction_system.surface_states(surface)
        
        # Test state localization
        for state in states:
            # Check exponential decay into bulk
            z_profile = state.density_profile(axis=2)
            log_density = torch.log(z_profile + 1e-10)
            # Should be roughly linear (exponential decay)
            fit = np.polyfit(range(len(log_density)), log_density.numpy(), 1)
            assert fit[0] < 0, "Surface states should decay into bulk"
        
        # Test surface band structure
        surface_bands = refraction_system.surface_band_structure(surface)
        
        # Test topological protection if applicable
        if hasattr(refraction_system, 'topological_invariant'):
            bulk_invariant = refraction_system.topological_invariant(bulk)
            assert len(surface_bands) >= abs(bulk_invariant), \
                "Should satisfy bulk-boundary correspondence"

    def test_strain_effects(self, refraction_system, space_dim):
        """Test strain and elasticity properties."""
        # Create strain tensor
        strain = torch.eye(space_dim) * 0.01  # 1% uniform strain
        
        # Test elastic response
        stress = refraction_system.compute_stress(strain)
        
        # Test Hooke's law
        C = refraction_system.elastic_constants()
        stress_hooke = torch.einsum('ijkl,kl->ij', C, strain)
        assert torch.allclose(
            stress, stress_hooke, rtol=1e-4
        ), "Should satisfy Hooke's law"
        
        # Test elastic energy
        energy = refraction_system.elastic_energy(strain)
        assert energy > 0, "Elastic energy should be positive"
        
        # Test elastic stability
        eigenvals = torch.linalg.eigvalsh(
            C.reshape(space_dim**2, space_dim**2)
        )
        assert torch.all(eigenvals > 0), "System should be elastically stable"
        
        # Test phonon frequency shifts
        unstrained_phonons = refraction_system.phonon_modes(torch.eye(space_dim))
        strained_phonons = refraction_system.phonon_modes_with_strain(
            torch.eye(space_dim), strain
        )
        
        # Grüneisen parameter
        gamma = -(
            strained_phonons.frequencies - unstrained_phonons.frequencies
        ) / (space_dim * strain[0,0] * unstrained_phonons.frequencies)
        
        assert torch.all(gamma > 0), "Grüneisen parameter should be positive"
