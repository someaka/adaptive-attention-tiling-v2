"""
Unit tests for the path integral framework implementation.

Tests cover:
1. Action functional computation
2. Propagator evaluation
3. Partition function calculation
4. Correlation functions
5. Effective action
6. Renormalization group flow
7. Saddle point approximation
8. Ward identities
9. Schwinger-Dyson equations
10. Instanton configurations
"""

import pytest
import torch
import numpy as np
from typing import Tuple, Callable
from functools import partial

from src.core.quantum.path_integral import PathIntegralFramework
from src.utils.test_helpers import assert_quantum_properties, numerical_gradient

class TestPathIntegralFramework:
    @pytest.fixture
    def spacetime_dim(self):
        """Dimension of spacetime for path integrals."""
        return 4

    @pytest.fixture
    def lattice_size(self):
        """Size of discretized lattice."""
        return 16

    @pytest.fixture
    def path_integral(self, spacetime_dim, lattice_size):
        """Create a test path integral framework."""
        return PathIntegralFramework(
            dim=spacetime_dim,
            size=lattice_size,
            integration_method="metropolis"
        )

    def test_action_functional(self, path_integral, lattice_size):
        """Test action functional properties."""
        # Create test field configuration
        field = torch.randn(lattice_size, lattice_size, dtype=torch.complex64)
        
        # Compute action
        action = path_integral.action_functional(field)
        
        # Test reality
        assert torch.allclose(
            action.imag, torch.tensor(0.0), rtol=1e-5
        ), "Action must be real for physical configurations"
        
        # Test locality
        def local_variation(field: torch.Tensor, point: Tuple[int, int]) -> torch.Tensor:
            """Apply local variation to field."""
            varied = field.clone()
            varied[point] += 0.1
            return varied
        
        # Test action variation is local
        for x in range(1, lattice_size-1):
            for y in range(1, lattice_size-1):
                point = (x, y)
                varied_field = local_variation(field, point)
                action_diff = path_integral.action_functional(varied_field) - action
                
                # Check that action difference only depends on nearby points
                neighborhood = field[x-1:x+2, y-1:y+2]
                assert torch.allclose(
                    action_diff,
                    path_integral.compute_local_action(neighborhood),
                    rtol=1e-4
                ), "Action should be local"

    def test_propagator(self, path_integral, lattice_size):
        """Test quantum propagator computation."""
        # Create initial and final states
        initial_state = torch.randn(lattice_size, dtype=torch.complex64)
        final_state = torch.randn(lattice_size, dtype=torch.complex64)
        time_interval = 1.0
        
        # Compute propagator
        propagator = path_integral.propagator(initial_state, final_state, time_interval)
        
        # Test unitarity
        assert torch.allclose(
            propagator @ propagator.conj().T,
            torch.eye(lattice_size, dtype=torch.complex64),
            rtol=1e-4
        ), "Propagator should be unitary"
        
        # Test composition property
        half_time = time_interval / 2
        mid_state = torch.randn(lattice_size, dtype=torch.complex64)
        
        prop1 = path_integral.propagator(initial_state, mid_state, half_time)
        prop2 = path_integral.propagator(mid_state, final_state, half_time)
        
        composed = prop2 @ prop1
        full = path_integral.propagator(initial_state, final_state, time_interval)
        
        assert torch.allclose(
            composed, full, rtol=1e-3
        ), "Propagator should satisfy composition law"

    def test_partition_function(self, path_integral):
        """Test partition function properties."""
        # Test temperatures
        temperatures = torch.tensor([0.5, 1.0, 2.0])
        
        # Compute partition functions
        Z = torch.stack([
            path_integral.partition_function(T) for T in temperatures
        ])
        
        # Test positivity
        assert torch.all(Z > 0), "Partition function must be positive"
        
        # Test high temperature behavior
        high_temp = path_integral.partition_function(torch.tensor(100.0))
        assert torch.allclose(
            high_temp.log(),
            torch.tensor(0.0),
            rtol=1e-2
        ), "Partition function should approach 1 at high temperature"
        
        # Test KMS condition
        def test_KMS(op1: Callable, op2: Callable, beta: float) -> bool:
            """Test Kubo-Martin-Schwinger condition."""
            corr1 = path_integral.thermal_correlator(op1, op2, beta)
            corr2 = path_integral.thermal_correlator(op2, op1, beta)
            return torch.allclose(corr1, corr2 * torch.exp(-beta), rtol=1e-3)
        
        # Simple test operators
        op1 = lambda x: x
        op2 = lambda x: x**2
        
        assert test_KMS(op1, op2, 1.0), "KMS condition should hold"

    def test_correlation_function(self, path_integral, lattice_size):
        """Test correlation function properties."""
        # Create test operators
        operators = [
            lambda x: x,
            lambda x: x**2,
            lambda x: torch.sin(x)
        ]
        
        # Compute correlation function
        correlation = path_integral.correlation_function(operators)
        
        # Test symmetry properties
        assert torch.allclose(
            correlation,
            correlation.conj().transpose(-1, -2),
            rtol=1e-4
        ), "Correlation matrix should be Hermitian"
        
        # Test clustering property
        def test_clustering(x1: int, x2: int, separation: int) -> bool:
            """Test cluster decomposition principle."""
            op1 = lambda x: x[x1]
            op2 = lambda x: x[x2]
            
            corr = path_integral.correlation_function([op1, op2])
            if separation > 5:  # Some reasonable correlation length
                return torch.allclose(
                    corr,
                    path_integral.expectation(op1) * path_integral.expectation(op2),
                    rtol=1e-2
                )
            return True
        
        # Test for well-separated points
        assert test_clustering(0, lattice_size-1, lattice_size-1), \
            "Cluster decomposition should hold"

    def test_effective_action(self, path_integral, lattice_size):
        """Test effective action computation."""
        # Create background field
        field = torch.randn(lattice_size, lattice_size, dtype=torch.complex64)
        
        # Compute effective action
        effective_action = path_integral.effective_action(field)
        
        # Test gauge invariance if applicable
        def gauge_transform(field: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
            """Apply gauge transformation."""
            return field * torch.exp(1j * param)
        
        gauge_param = torch.randn(lattice_size, lattice_size)
        transformed_field = gauge_transform(field, gauge_param)
        transformed_action = path_integral.effective_action(transformed_field)
        
        assert torch.allclose(
            effective_action, transformed_action, rtol=1e-4
        ), "Effective action should be gauge invariant"
        
        # Test convexity
        field1 = torch.randn(lattice_size, lattice_size, dtype=torch.complex64)
        field2 = torch.randn(lattice_size, lattice_size, dtype=torch.complex64)
        t = torch.tensor(0.3)  # Interpolation parameter
        
        interpolated = t * field1 + (1-t) * field2
        action1 = path_integral.effective_action(field1)
        action2 = path_integral.effective_action(field2)
        action_interp = path_integral.effective_action(interpolated)
        
        assert action_interp <= t * action1 + (1-t) * action2, \
            "Effective action should be convex"

    def test_renormalization_group(self, path_integral, lattice_size):
        """Test renormalization group flow properties."""
        # Create initial coupling constants
        initial_couplings = {
            'mass': torch.tensor(1.0),
            'interaction': torch.tensor(0.1)
        }
        
        # Define RG flow steps
        scale_factors = torch.tensor([2.0, 4.0, 8.0])
        
        # Compute RG flow
        evolved_couplings = [
            path_integral.rg_flow(initial_couplings, scale)
            for scale in scale_factors
        ]
        
        # Test Wilson's RG equation
        def check_rg_equation(couplings1, couplings2, scale1, scale2):
            """Verify composition property of RG transformations."""
            direct = path_integral.rg_flow(couplings1, scale2/scale1)
            composed = path_integral.rg_flow(
                path_integral.rg_flow(couplings1, scale1),
                scale2/scale1
            )
            return all(
                torch.allclose(direct[k], composed[k], rtol=1e-4)
                for k in direct.keys()
            )
        
        assert check_rg_equation(
            initial_couplings, evolved_couplings[0],
            scale_factors[0], scale_factors[1]
        ), "RG flow should satisfy composition law"
        
        # Test relevant/irrelevant operator scaling
        mass_scaling = torch.tensor([
            c['mass'] / initial_couplings['mass']
            for c in evolved_couplings
        ])
        interaction_scaling = torch.tensor([
            c['interaction'] / initial_couplings['interaction']
            for c in evolved_couplings
        ])
        
        # Mass is relevant, should grow
        assert torch.all(mass_scaling[1:] > mass_scaling[:-1]), \
            "Mass term should be relevant"
        
        # Interaction is irrelevant in 4D, should decrease
        assert torch.all(interaction_scaling[1:] < interaction_scaling[:-1]), \
            "Interaction should be irrelevant in 4D"

    def test_saddle_point(self, path_integral, lattice_size):
        """Test saddle point approximation."""
        # Create test action
        def quadratic_action(field):
            """Simple quadratic action for testing."""
            return torch.sum(field**2) / 2
        
        # Find saddle point
        initial_field = torch.randn(lattice_size, lattice_size, dtype=torch.complex64)
        saddle_point = path_integral.find_saddle_point(quadratic_action, initial_field)
        
        # Test that it's actually a saddle point
        gradient = numerical_gradient(quadratic_action, saddle_point)
        assert torch.allclose(
            gradient, torch.zeros_like(gradient), atol=1e-4
        ), "Saddle point should have zero gradient"
        
        # Test Gaussian fluctuations
        hessian = path_integral.compute_hessian(quadratic_action, saddle_point)
        eigenvalues = torch.linalg.eigvalsh(hessian)
        assert torch.all(eigenvalues > 0), \
            "Quadratic action should have positive definite Hessian"

    def test_ward_identities(self, path_integral, lattice_size):
        """Test Ward identities from symmetries."""
        # Create conserved current
        def create_current(field):
            """Create U(1) Noether current."""
            return torch.gradient(
                field * field.conj(),
                dim=0
            )[0]
        
        # Test current conservation
        field = torch.randn(lattice_size, lattice_size, dtype=torch.complex64)
        current = create_current(field)
        divergence = torch.sum(torch.gradient(current, dim=0)[0])
        
        assert torch.allclose(
            divergence, torch.tensor(0.0), atol=1e-4
        ), "Current should be conserved"
        
        # Test Ward identity for correlation functions
        def verify_ward_identity(field1, field2):
            """Verify Ward identity for two-point function."""
            correlation = path_integral.correlation_function(
                [lambda x: x[field1], lambda x: x[field2]]
            )
            ward_term = path_integral.ward_variation(correlation, field1, field2)
            return torch.allclose(ward_term, torch.tensor(0.0), atol=1e-4)
        
        assert verify_ward_identity(
            (0, 0), (lattice_size-1, lattice_size-1)
        ), "Ward identity should hold"

    def test_schwinger_dyson(self, path_integral, lattice_size):
        """Test Schwinger-Dyson equations."""
        # Create test operator
        def test_operator(field):
            """Simple local operator for testing."""
            return field**2
        
        # Test SD equation
        field = torch.randn(lattice_size, lattice_size, dtype=torch.complex64)
        
        lhs = path_integral.functional_derivative(
            lambda f: test_operator(f) * torch.exp(-path_integral.action_functional(f)),
            field
        )
        
        rhs = test_operator(field) * path_integral.functional_derivative(
            lambda f: torch.exp(-path_integral.action_functional(f)),
            field
        )
        
        assert torch.allclose(lhs, rhs, rtol=1e-4), \
            "Schwinger-Dyson equation should hold"
        
        # Test for connected correlation functions
        def verify_sd_hierarchy(n_point):
            """Verify SD equations for n-point functions."""
            ops = [test_operator for _ in range(n_point)]
            sd_relation = path_integral.schwinger_dyson_relation(ops)
            return torch.allclose(sd_relation, torch.tensor(0.0), atol=1e-4)
        
        assert verify_sd_hierarchy(2), "SD equations should hold for 2-point function"
        assert verify_sd_hierarchy(3), "SD equations should hold for 3-point function"

    def test_instantons(self, path_integral, lattice_size):
        """Test instanton configurations."""
        # Create instanton solution
        def create_instanton(position, size):
            """Create BPST instanton configuration."""
            x = torch.arange(lattice_size)
            y = torch.arange(lattice_size)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            r = torch.sqrt((X - position[0])**2 + (Y - position[1])**2)
            return size**2 / (r**2 + size**2)
        
        # Test instanton properties
        position = (lattice_size//2, lattice_size//2)
        size = torch.tensor(3.0)
        instanton = create_instanton(position, size)
        
        # Test topological charge
        charge = path_integral.topological_charge(instanton)
        assert torch.allclose(
            charge, torch.tensor(1.0), rtol=1e-4
        ), "Instanton should have unit topological charge"
        
        # Test action bound
        action = path_integral.action_functional(instanton)
        assert action >= 8 * np.pi**2, \
            "Instanton action should satisfy BPS bound"
        
        # Test zero modes
        zero_modes = path_integral.compute_zero_modes(instanton)
        assert len(zero_modes) == 4, \
            "BPST instanton should have 4 zero modes"
        
        # Test moduli space metric
        moduli_metric = path_integral.moduli_space_metric(instanton)
        assert torch.allclose(
            moduli_metric,
            moduli_metric.transpose(-1, -2),
            rtol=1e-4
        ), "Moduli space metric should be symmetric"
