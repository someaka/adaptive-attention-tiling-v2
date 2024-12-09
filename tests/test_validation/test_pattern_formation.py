"""
Unit tests for pattern formation validation.

Tests cover:
1. Pattern emergence
2. Spatial organization
3. Temporal evolution
4. Bifurcation analysis
5. Mode decomposition
6. Reaction-diffusion pattern formation
7. Symmetry breaking in pattern formation
8. Pattern stability analysis
"""

import pytest
import torch
import numpy as np
from typing import List, Tuple, Dict
from scipy.integrate import solve_ivp

from src.validation.patterns.formation import (
    PatternFormationValidator,
    EmergenceMetrics,
    SpatialMetrics,
    TemporalMetrics,
    BifurcationAnalyzer,
    ModeDecomposer
)

class TestPatternFormation:
    @pytest.fixture
    def batch_size(self) -> int:
        return 16

    @pytest.fixture
    def spatial_dim(self) -> int:
        return 32

    @pytest.fixture
    def time_steps(self) -> int:
        return 100

    @pytest.fixture
    def validator(self) -> PatternFormationValidator:
        return PatternFormationValidator(
            emergence_threshold=0.1,
            spatial_coherence_threshold=0.8,
            temporal_stability_threshold=0.9
        )

    def test_pattern_emergence(self, validator: PatternFormationValidator,
                             batch_size: int, spatial_dim: int):
        """Test pattern emergence detection and validation."""
        # Create test pattern evolution
        time_series = torch.randn(batch_size, spatial_dim, requires_grad=True)
        noise = 0.1 * torch.randn_like(time_series)
        
        # Test emergence detection
        emergence = validator.detect_emergence(time_series + noise)
        assert isinstance(emergence, EmergenceMetrics)
        assert hasattr(emergence, 'emergence_time')
        assert hasattr(emergence, 'emergence_strength')
        
        # Test emergence validation
        result = validator.validate_emergence(emergence)
        assert isinstance(result, bool)
        assert hasattr(result, 'metrics')
        assert 'emergence_score' in result.metrics
        
        # Test gradual pattern formation
        def generate_growing_pattern(t: float) -> torch.Tensor:
            """Generate pattern with increasing amplitude."""
            base_pattern = torch.sin(torch.linspace(0, 4*np.pi, spatial_dim))
            return (1 - np.exp(-t)) * base_pattern
            
        times = torch.linspace(0, 10, 100)
        patterns = torch.stack([generate_growing_pattern(t) for t in times])
        emergence_gradual = validator.detect_emergence(patterns)
        assert emergence_gradual.emergence_time > 0
        assert emergence_gradual.emergence_strength > 0

    def test_spatial_organization(self, validator: PatternFormationValidator,
                                batch_size: int, spatial_dim: int):
        """Test spatial organization analysis."""
        # Create test spatial patterns
        patterns = []
        for _ in range(batch_size):
            # Generate pattern with specific wavelength
            k = torch.randint(1, 5, (1,)).item()
            x = torch.linspace(0, 2*np.pi, spatial_dim)
            pattern = torch.sin(k * x) + 0.1 * torch.randn(spatial_dim)
            patterns.append(pattern)
        patterns = torch.stack(patterns)
        
        # Test spatial metrics computation
        spatial_metrics = validator.analyze_spatial_organization(patterns)
        assert isinstance(spatial_metrics, SpatialMetrics)
        assert hasattr(spatial_metrics, 'wavelength')
        assert hasattr(spatial_metrics, 'amplitude')
        assert hasattr(spatial_metrics, 'coherence')
        
        # Test wavelength detection
        wavelengths = spatial_metrics.wavelength
        assert wavelengths.shape[0] == batch_size
        assert torch.all(wavelengths > 0)
        
        # Test spatial coherence
        coherence = spatial_metrics.coherence
        assert coherence.shape[0] == batch_size
        assert torch.all(coherence >= 0) and torch.all(coherence <= 1)
        
        # Test pattern symmetry
        symmetry = validator.analyze_symmetry(patterns)
        assert 'translation' in symmetry
        assert 'rotation' in symmetry
        assert all(0 <= v <= 1 for v in symmetry.values())

    def test_temporal_evolution(self, validator: PatternFormationValidator,
                              batch_size: int, spatial_dim: int,
                              time_steps: int):
        """Test temporal evolution analysis."""
        # Create test temporal evolution
        time_series = []
        for _ in range(batch_size):
            # Generate pattern with temporal dynamics
            t = torch.linspace(0, 10, time_steps)
            x = torch.linspace(0, 2*np.pi, spatial_dim)
            X, T = torch.meshgrid(x, t, indexing='ij')
            pattern = torch.sin(X - 0.5*T) + 0.1 * torch.randn_like(X)
            time_series.append(pattern)
        time_series = torch.stack(time_series)
        
        # Test temporal metrics computation
        temporal_metrics = validator.analyze_temporal_evolution(time_series)
        assert isinstance(temporal_metrics, TemporalMetrics)
        assert hasattr(temporal_metrics, 'frequency')
        assert hasattr(temporal_metrics, 'phase_velocity')
        assert hasattr(temporal_metrics, 'stability')
        
        # Test frequency detection
        frequencies = temporal_metrics.frequency
        assert frequencies.shape[0] == batch_size
        assert torch.all(frequencies >= 0)
        
        # Test phase velocity
        velocities = temporal_metrics.phase_velocity
        assert velocities.shape[0] == batch_size
        
        # Test temporal stability
        stability = temporal_metrics.stability
        assert stability.shape[0] == batch_size
        assert torch.all(stability >= 0) and torch.all(stability <= 1)

    def test_bifurcation_analysis(self, validator: PatternFormationValidator,
                                spatial_dim: int):
        """Test bifurcation analysis."""
        # Create bifurcation analyzer
        analyzer = BifurcationAnalyzer()
        
        # Generate pattern evolution with parameter variation
        def pattern_evolution(param: float) -> torch.Tensor:
            """Generate pattern evolution for given parameter."""
            x = torch.linspace(0, 2*np.pi, spatial_dim)
            if param < 1.0:
                return torch.zeros_like(x)
            elif param < 2.0:
                return torch.sin(x)
            else:
                return torch.sin(x) + 0.5 * torch.sin(2*x)
                
        # Test bifurcation detection
        params = torch.linspace(0, 3, 30)
        patterns = torch.stack([pattern_evolution(p) for p in params])
        bifurcations = analyzer.detect_bifurcations(patterns, params)
        
        assert len(bifurcations) > 0
        for bif in bifurcations:
            assert hasattr(bif, 'parameter_value')
            assert hasattr(bif, 'type')
            assert hasattr(bif, 'stability_change')
            
        # Test critical parameter estimation
        critical_params = analyzer.estimate_critical_parameters(patterns, params)
        assert len(critical_params) > 0
        assert all(0 <= p <= 3 for p in critical_params)

    def test_mode_decomposition(self, validator: PatternFormationValidator,
                              batch_size: int, spatial_dim: int):
        """Test pattern mode decomposition."""
        # Create mode decomposer
        decomposer = ModeDecomposer(n_modes=5)
        
        # Generate test patterns with known modes
        patterns = []
        for _ in range(batch_size):
            x = torch.linspace(0, 2*np.pi, spatial_dim)
            # Combine multiple modes with different amplitudes
            amplitudes = torch.rand(5)
            pattern = sum(a * torch.sin(k*x) 
                        for k, a in enumerate(amplitudes, start=1))
            patterns.append(pattern)
        patterns = torch.stack(patterns)
        
        # Test mode decomposition
        modes, coefficients = decomposer.decompose(patterns)
        assert modes.shape == (batch_size, 5, spatial_dim)
        assert coefficients.shape == (batch_size, 5)
        assert torch.all(coefficients >= 0)
        
        # Test mode reconstruction
        reconstructed = decomposer.reconstruct(modes, coefficients)
        assert reconstructed.shape == patterns.shape
        assert torch.allclose(reconstructed, patterns, rtol=1e-4)
        
        # Test dominant mode extraction
        dominant_modes = decomposer.extract_dominant_modes(patterns, n_modes=3)
        assert len(dominant_modes) == 3
        assert all(0 <= mode_idx < 5 for mode_idx in dominant_modes)

    def test_validation_integration(self, validator: PatternFormationValidator,
                                  batch_size: int, spatial_dim: int,
                                  time_steps: int):
        """Test integrated pattern formation validation."""
        # Generate test pattern evolution
        time_series = []
        for _ in range(batch_size):
            t = torch.linspace(0, 10, time_steps)
            x = torch.linspace(0, 2*np.pi, spatial_dim)
            X, T = torch.meshgrid(x, t, indexing='ij')
            # Generate pattern with emergence and spatial-temporal structure
            amplitude = 1 - torch.exp(-0.5*T)
            pattern = amplitude * torch.sin(X - 0.3*T)
            pattern = pattern + 0.1 * torch.randn_like(pattern)
            time_series.append(pattern)
        time_series = torch.stack(time_series)
        
        # Run full validation
        result = validator.validate_pattern_formation(time_series)
        assert isinstance(result, Dict)
        assert 'emergence' in result
        assert 'spatial' in result
        assert 'temporal' in result
        assert 'bifurcation' in result
        assert 'modes' in result
        
        # Check validation scores
        assert all(0 <= score <= 1 for score in result.values())
        assert 'overall_score' in result
        
        # Test validation with parameters
        params = torch.linspace(0, 2, time_steps)
        param_result = validator.validate_pattern_formation(
            time_series, parameters=params
        )
        assert 'bifurcation' in param_result
        assert param_result['bifurcation']['critical_params'] is not None

    def test_reaction_diffusion(self, validator: PatternFormationValidator,
                             batch_size: int, spatial_dim: int):
        """Test reaction-diffusion pattern formation."""
        # Test diffusion terms
        def test_diffusion():
            """Test diffusion operator properties."""
            # Get diffusion operator
            D = validator.get_diffusion_operator()
            assert validator.validate_diffusion_operator(D)
            
            # Test positivity
            assert validator.validate_positivity(D)
            
            # Test conservation
            assert validator.validate_conservation(D)
            
            return D
            
        D = test_diffusion()
        
        # Test reaction terms
        def test_reaction():
            """Test reaction term properties."""
            # Get reaction term
            R = validator.get_reaction_term()
            assert validator.validate_reaction_term(R)
            
            # Test mass conservation
            assert validator.validate_mass_conservation(R)
            
            # Test detailed balance
            if validator.has_detailed_balance():
                assert validator.validate_detailed_balance(R)
                
            return R
            
        R = test_reaction()
        
        # Test pattern formation
        def test_pattern():
            """Test pattern formation dynamics."""
            # Get pattern
            pattern = validator.get_test_pattern()
            
            # Test stability
            assert validator.validate_pattern_stability(pattern)
            
            # Test regularity
            assert validator.validate_pattern_regularity(pattern)
            
            # Test wavelength selection
            if validator.has_wavelength_selection():
                assert validator.validate_wavelength(pattern)
                
            return pattern
            
        pattern = test_pattern()

    def test_symmetry_breaking(self, validator: PatternFormationValidator,
                             batch_size: int, spatial_dim: int):
        """Test symmetry breaking in pattern formation."""
        # Test symmetry group
        def test_symmetry():
            """Test symmetry group properties."""
            # Get symmetry group
            G = validator.get_symmetry_group()
            assert validator.validate_symmetry_group(G)
            
            # Test representation
            rho = validator.get_representation(G)
            assert validator.validate_representation(rho)
            
            return G, rho
            
        G, rho = test_symmetry()
        
        # Test bifurcation
        def test_bifurcation():
            """Test bifurcation properties."""
            # Get bifurcation
            bif = validator.get_bifurcation()
            assert validator.validate_bifurcation(bif)
            
            # Test criticality
            assert validator.validate_criticality(bif)
            
            # Test equivariance
            assert validator.validate_equivariance(bif, G)
            
            return bif
            
        bif = test_bifurcation()
        
        # Test mode selection
        def test_mode_selection():
            """Test mode selection in symmetry breaking."""
            # Get critical modes
            modes = validator.get_critical_modes()
            assert validator.validate_critical_modes(modes)
            
            # Test isotropy
            H = validator.get_isotropy_subgroup(modes)
            assert validator.validate_isotropy(H, G)
            
            # Test branching
            if validator.has_branching():
                assert validator.validate_branching(modes, H)
                
            return modes, H
            
        modes, H = test_mode_selection()

    def test_pattern_stability(self, validator: PatternFormationValidator,
                             batch_size: int, spatial_dim: int):
        """Test pattern stability analysis."""
        # Test linear stability
        def test_linear_stability():
            """Test linear stability analysis."""
            # Get linearization
            L = validator.get_linearization()
            assert validator.validate_linearization(L)
            
            # Test spectrum
            spec = validator.compute_spectrum(L)
            assert validator.validate_spectrum(spec)
            
            # Test eigenfunctions
            if validator.has_eigenfunctions():
                eig = validator.get_eigenfunctions(L)
                assert validator.validate_eigenfunctions(eig)
                
            return L, spec
            
        L, spec = test_linear_stability()
        
        # Test nonlinear stability
        def test_nonlinear_stability():
            """Test nonlinear stability analysis."""
            # Get Lyapunov function
            V = validator.get_lyapunov_function()
            assert validator.validate_lyapunov_function(V)
            
            # Test gradient structure
            if validator.is_gradient():
                assert validator.validate_gradient_structure(V)
                
            # Test basin of attraction
            basin = validator.get_basin_of_attraction(V)
            assert validator.validate_basin(basin)
            
            return V, basin
            
        V, basin = test_nonlinear_stability()
        
        # Test structural stability
        def test_structural_stability():
            """Test structural stability properties."""
            # Get perturbation
            pert = validator.get_perturbation()
            assert validator.validate_perturbation(pert)
            
            # Test persistence
            assert validator.validate_persistence(pert)
            
            # Test normal hyperbolicity
            if validator.is_normally_hyperbolic():
                assert validator.validate_normal_hyperbolicity(pert)
                
            return pert
            
        pert = test_structural_stability()
