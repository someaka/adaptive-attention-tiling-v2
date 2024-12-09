"""
Unit tests for the cohomology theory implementation with arithmetic dynamics.

Tests cover:
1. Differential forms and operations
2. Cohomology classes and cup products
3. Arithmetic height functions
4. Information flow metrics
5. Pattern stability measures
6. De Rham cohomology
7. Spectral sequences
8. Morse theory
"""

import pytest
import torch
import numpy as np
from typing import List, Tuple, Optional
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from src.core.patterns.cohomology import (
    CohomologyStructure,
    DifferentialForm,
    CohomologyClass,
    SpectralSequence
)
from src.utils.test_helpers import assert_form_properties

class TestCohomologyStructure:
    @pytest.fixture
    def manifold_dim(self):
        """Dimension of test manifold."""
        return 4

    @pytest.fixture
    def max_degree(self):
        """Maximum degree of differential forms to test."""
        return 3

    @pytest.fixture
    def cohomology(self, manifold_dim):
        """Create a test cohomology structure."""
        return CohomologyStructure(dim=manifold_dim)

    def test_differential_forms(self, cohomology, manifold_dim, max_degree):
        """Test differential form properties."""
        for degree in range(max_degree + 1):
            forms = cohomology.differential_forms(degree)
            
            # Test shape
            assert forms.shape[-1] == pytest.approx(
                np.math.comb(manifold_dim, degree)
            ), f"{degree}-forms should have correct dimension"
            
            # Test antisymmetry
            if degree > 1:
                permuted = forms.transpose(-1, -2)
                assert torch.allclose(
                    permuted, -forms, rtol=1e-5
                ), f"{degree}-forms should be antisymmetric"

    def test_exterior_derivative(self, cohomology, max_degree):
        """Test exterior derivative properties."""
        for degree in range(max_degree):
            form = cohomology.differential_forms(degree)[0]  # Take first form
            d_form = cohomology.exterior_derivative(form)
            
            # Test degree
            assert d_form.degree == form.degree + 1, "d should increase degree by 1"
            
            # Test d² = 0
            d2_form = cohomology.exterior_derivative(d_form)
            assert torch.allclose(
                d2_form, torch.zeros_like(d2_form), rtol=1e-5
            ), "d² should vanish"

    def test_cohomology_classes(self, cohomology, max_degree):
        """Test cohomology class computations."""
        for degree in range(max_degree + 1):
            classes = cohomology.cohomology_classes(degree)
            
            # Test that classes are closed
            for cls in classes:
                d_cls = cohomology.exterior_derivative(cls)
                assert torch.allclose(
                    d_cls, torch.zeros_like(d_cls), rtol=1e-5
                ), "Cohomology classes should be closed"

    def test_cup_product(self, cohomology):
        """Test cup product properties."""
        class1 = cohomology.cohomology_classes(1)[0]  # 1-class
        class2 = cohomology.cohomology_classes(1)[0]  # Another 1-class
        
        cup = cohomology.cup_product(class1, class2)
        
        # Test degree
        assert cup.degree == class1.degree + class2.degree, \
            "Cup product should add degrees"
            
        # Test anticommutativity
        cup_reverse = cohomology.cup_product(class2, class1)
        assert torch.allclose(
            cup_reverse, (-1)**(class1.degree * class2.degree) * cup, rtol=1e-5
        ), "Cup product should have correct anticommutativity"

    def test_arithmetic_height(self, cohomology):
        """Test arithmetic height function properties."""
        point = torch.randn(4)  # Test point
        height = cohomology.arithmetic_height(point)
        
        # Test positivity
        assert height >= 0, "Height should be non-negative"
        
        # Test scaling
        scalar = 2.0
        scaled_height = cohomology.arithmetic_height(scalar * point)
        assert torch.allclose(
            scaled_height,
            height * torch.log(torch.tensor(scalar)),
            rtol=1e-4
        ), "Height should have logarithmic scaling"

    def test_information_flow_metrics(self, cohomology):
        """Test information flow metric properties."""
        pattern = torch.randn(4, 8)  # Test pattern
        metrics = cohomology.information_flow_metrics(pattern)
        
        # Test metric properties
        assert metrics.entropy >= 0, "Entropy should be non-negative"
        assert metrics.complexity >= 0, "Complexity should be non-negative"
        assert metrics.flow_rate is not None, "Flow rate should be computed"

    def test_pattern_stability_measures(self, cohomology):
        """Test pattern stability measure properties."""
        pattern = torch.randn(4, 8)  # Test pattern
        measures = cohomology.pattern_stability_measures(pattern)
        
        # Test measure properties
        assert measures.linear_stability is not None, "Linear stability should be computed"
        assert measures.nonlinear_stability is not None, "Nonlinear stability should be computed"
        assert measures.bifurcation_distance >= 0, "Bifurcation distance should be non-negative"

    def test_de_rham_cohomology(self, cohomology, manifold_dim, max_degree):
        """Test de Rham cohomology computations."""
        # Compute de Rham cohomology groups
        for k in range(max_degree + 1):
            H_k = cohomology.compute_de_rham_cohomology(k)
            
            # Test Betti numbers
            betti = H_k.dimension()
            assert isinstance(betti, int)
            assert betti >= 0
            
            # Test Poincaré duality
            if hasattr(cohomology, 'is_oriented') and cohomology.is_oriented:
                H_dual = cohomology.compute_de_rham_cohomology(manifold_dim - k)
                assert H_k.dimension() == H_dual.dimension()
        
        # Test long exact sequence
        if hasattr(cohomology, 'submanifold'):
            # Get inclusion map
            i = cohomology.inclusion_map()
            # Get connecting homomorphism
            delta = cohomology.connecting_homomorphism()
            
            # Test exactness
            im_i = i.image()
            ker_delta = delta.kernel()
            assert all(x in ker_delta for x in im_i)

    def test_spectral_sequence(self, cohomology):
        """Test spectral sequence computations."""
        # Initialize spectral sequence
        E = SpectralSequence(cohomology)
        
        # Test convergence
        E2 = E.compute_page(2)  # E2 page
        E_inf = E.compute_limit()  # E∞ page
        
        # Test differential properties
        d2 = E.differential(2)  # d2 differential
        assert torch.allclose(
            d2 @ d2,
            torch.zeros_like(d2),
            rtol=1e-5
        ), "Differential should square to zero"
        
        # Test filtration
        F = E.filtration()
        assert all(F[i].is_subspace_of(F[i+1]) for i in range(len(F)-1))
        
        # Test spectral sequence convergence
        if hasattr(E, 'converges_at'):
            page = E.converges_at()
            assert isinstance(page, int)
            assert page >= 2

    def test_morse_theory(self, cohomology):
        """Test Morse theory applications."""
        # Generate Morse function
        def morse_function(x: torch.Tensor) -> torch.Tensor:
            return torch.sum(x**2, dim=-1)
        
        # Compute critical points
        critical_points = cohomology.find_critical_points(morse_function)
        
        # Test Morse index
        for point in critical_points:
            index = cohomology.morse_index(morse_function, point)
            assert isinstance(index, int)
            assert 0 <= index <= manifold_dim
        
        # Test Morse inequalities
        betti = [cohomology.compute_de_rham_cohomology(k).dimension() 
                for k in range(manifold_dim + 1)]
        morse = [sum(1 for p in critical_points if cohomology.morse_index(morse_function, p) == k)
                for k in range(manifold_dim + 1)]
        
        # Weak Morse inequalities
        assert all(betti[k] <= morse[k] for k in range(manifold_dim + 1))
        
        # Strong Morse inequalities
        alt_sum = sum((-1)**k * (morse[k] - betti[k]) for k in range(manifold_dim + 1))
        assert alt_sum >= 0

    def test_hodge_theory(self, cohomology, manifold_dim):
        """Test Hodge theory computations."""
        # Compute Hodge star operator
        for k in range(manifold_dim + 1):
            omega = cohomology.differential_forms(k)[0]
            star_omega = cohomology.hodge_star(omega)
            
            # Test degree
            assert star_omega.degree == manifold_dim - omega.degree
            
            # Test double application
            star_star_omega = cohomology.hodge_star(star_omega)
            assert torch.allclose(
                star_star_omega,
                (-1)**(k*(manifold_dim-k)) * omega,
                rtol=1e-5
            )
        
        # Test codifferential
        delta = cohomology.codifferential()
        d = cohomology.exterior_derivative
        
        # Test adjoint property
        form1 = cohomology.differential_forms(k)[0]
        form2 = cohomology.differential_forms(k+1)[0]
        lhs = cohomology.inner_product(delta(form2), form1)
        rhs = cohomology.inner_product(form2, d(form1))
        assert torch.allclose(lhs, rhs, rtol=1e-5)
        
        # Test Laplacian
        laplacian = cohomology.laplacian()
        assert torch.allclose(
            laplacian,
            d @ delta + delta @ d,
            rtol=1e-5
        )

    def test_characteristic_classes(self, cohomology):
        """Test characteristic class computations."""
        # Compute Chern classes
        if hasattr(cohomology, 'compute_chern_classes'):
            chern = cohomology.compute_chern_classes()
            assert len(chern) > 0
            assert all(isinstance(c, CohomologyClass) for c in chern)
            
            # Test Whitney sum formula
            if len(chern) >= 2:
                total_chern = cohomology.total_chern_class()
                assert torch.allclose(
                    total_chern,
                    sum(c for c in chern),
                    rtol=1e-5
                )
        
        # Compute Pontryagin classes
        if hasattr(cohomology, 'compute_pontryagin_classes'):
            pont = cohomology.compute_pontryagin_classes()
            assert len(pont) > 0
            assert all(isinstance(p, CohomologyClass) for p in pont)
            
            # Test signature formula
            if hasattr(cohomology, 'signature'):
                sig = cohomology.signature()
                L = cohomology.L_genus()
                assert torch.allclose(
                    sig,
                    L.evaluate(),
                    rtol=1e-5
                )

    def test_differential_forms(
        self, cohomology, manifold_dim
    ):
        """Test differential form properties."""
        # Test exterior algebra
        def test_exterior_algebra():
            """Test exterior algebra operations."""
            # Get differential forms
            alpha = cohomology.random_k_form(1)
            beta = cohomology.random_k_form(1)
            
            # Test anticommutativity
            wedge = cohomology.wedge_product(alpha, beta)
            assert torch.allclose(
                wedge,
                -cohomology.wedge_product(beta, alpha)
            ), "Wedge should be anticommutative"
            
            # Test associativity
            gamma = cohomology.random_k_form(1)
            assert torch.allclose(
                cohomology.wedge_product(
                    alpha,
                    cohomology.wedge_product(beta, gamma)
                ),
                cohomology.wedge_product(
                    cohomology.wedge_product(alpha, beta),
                    gamma
                )
            ), "Wedge should be associative"
            
            return alpha, beta, gamma
            
        alpha, beta, gamma = test_exterior_algebra()
        
        # Test exterior derivative
        def test_exterior_derivative():
            """Test exterior derivative properties."""
            # Get k-form
            omega = cohomology.random_k_form(2)
            
            # Test d^2 = 0
            d_omega = cohomology.exterior_derivative(omega)
            assert torch.allclose(
                cohomology.exterior_derivative(d_omega),
                torch.zeros_like(d_omega)
            ), "Should have d^2 = 0"
            
            # Test Leibniz rule
            alpha = cohomology.random_k_form(1)
            beta = cohomology.random_k_form(1)
            
            d_wedge = cohomology.exterior_derivative(
                cohomology.wedge_product(alpha, beta)
            )
            leibniz = (
                cohomology.wedge_product(
                    cohomology.exterior_derivative(alpha),
                    beta
                ) +
                (-1)**alpha.degree * cohomology.wedge_product(
                    alpha,
                    cohomology.exterior_derivative(beta)
                )
            )
            assert torch.allclose(
                d_wedge, leibniz
            ), "Should satisfy Leibniz rule"
            
            return omega, d_omega
            
        omega, d_omega = test_exterior_derivative()
        
        # Test integration
        def test_integration():
            """Test integration of forms."""
            # Get top form
            top_form = cohomology.random_k_form(manifold_dim)
            
            # Test Stokes' theorem
            chain = cohomology.random_chain(manifold_dim)
            boundary = cohomology.boundary_operator(chain)
            
            stokes_lhs = cohomology.integrate(
                cohomology.exterior_derivative(top_form),
                chain
            )
            stokes_rhs = cohomology.integrate(
                top_form,
                boundary
            )
            assert torch.allclose(
                stokes_lhs, stokes_rhs
            ), "Should satisfy Stokes' theorem"
            
            return top_form
            
        top_form = test_integration()

    def test_cohomology_groups(
        self, cohomology, manifold_dim
    ):
        """Test cohomology group structure."""
        # Test de Rham cohomology
        def test_de_rham():
            """Test de Rham cohomology groups."""
            # Compute Betti numbers
            betti = []
            for k in range(manifold_dim + 1):
                # Get k-forms
                forms = cohomology.get_k_forms(k)
                
                # Get closed and exact forms
                closed = cohomology.get_closed_forms(k)
                exact = cohomology.get_exact_forms(k)
                
                # Compute dimension
                betti.append(
                    len(closed) - len(exact)
                )
                
            # Test Euler characteristic
            euler = sum((-1)**k * b for k, b in enumerate(betti))
            assert euler == cohomology.euler_characteristic(), \
                "Should match Euler characteristic"
                
            return betti
            
        betti = test_de_rham()
        
        # Test Mayer-Vietoris
        def test_mayer_vietoris():
            """Test Mayer-Vietoris sequence."""
            # Get open covers
            U = cohomology.get_open_set()
            V = cohomology.get_open_set()
            
            # Get restriction maps
            i_U = cohomology.get_restriction_map(U)
            i_V = cohomology.get_restriction_map(V)
            
            # Test exactness
            sequence = cohomology.get_mayer_vietoris_sequence(
                U, V, i_U, i_V
            )
            assert cohomology.is_exact_sequence(
                sequence
            ), "Mayer-Vietoris should be exact"
            
            return sequence
            
        mv_sequence = test_mayer_vietoris()
        
        # Test Poincaré duality
        def test_poincare_duality():
            """Test Poincaré duality."""
            if cohomology.is_oriented():
                # Test duality pairing
                for k in range(manifold_dim + 1):
                    H_k = cohomology.get_cohomology_group(k)
                    H_n_k = cohomology.get_cohomology_group(
                        manifold_dim - k
                    )
                    
                    # Get bases
                    alpha_basis = cohomology.get_harmonic_forms(k)
                    beta_basis = cohomology.get_harmonic_forms(
                        manifold_dim - k
                    )
                    
                    # Test non-degeneracy
                    pairing_matrix = torch.stack([
                        [
                            cohomology.integrate(
                                cohomology.wedge_product(
                                    alpha, beta
                                )
                            )
                            for beta in beta_basis
                        ]
                        for alpha in alpha_basis
                    ])
                    
                    assert torch.matrix_rank(
                        pairing_matrix
                    ) == len(alpha_basis), \
                        "Poincaré pairing should be non-degenerate"
                        
            return pairing_matrix
            
        poincare_pairing = test_poincare_duality()

    def test_cup_products(
        self, cohomology, manifold_dim
    ):
        """Test cup product structure."""
        # Test cup product properties
        def test_cup_properties():
            """Test algebraic properties of cup product."""
            # Get cohomology classes
            alpha = cohomology.random_cohomology_class(1)
            beta = cohomology.random_cohomology_class(1)
            gamma = cohomology.random_cohomology_class(1)
            
            # Test associativity
            assert torch.allclose(
                cohomology.cup_product(
                    alpha,
                    cohomology.cup_product(beta, gamma)
                ),
                cohomology.cup_product(
                    cohomology.cup_product(alpha, beta),
                    gamma
                )
            ), "Cup product should be associative"
            
            # Test graded commutativity
            assert torch.allclose(
                cohomology.cup_product(alpha, beta),
                (-1)**(
                    alpha.degree * beta.degree
                ) * cohomology.cup_product(beta, alpha)
            ), "Cup product should be graded commutative"
            
            return alpha, beta, gamma
            
        alpha, beta, gamma = test_cup_properties()
        
        # Test ring structure
        def test_ring_structure():
            """Test cohomology ring structure."""
            # Get generators
            generators = cohomology.get_ring_generators()
            
            # Test relations
            relations = cohomology.get_ring_relations()
            for rel in relations:
                assert torch.allclose(
                    cohomology.evaluate_relation(rel),
                    torch.zeros_like(rel[0])
                ), "Relations should evaluate to zero"
                
            # Test Poincaré polynomial
            polynomial = cohomology.poincare_polynomial()
            assert len(polynomial) == manifold_dim + 1, \
                "Should have correct degree"
                
            return generators, relations
            
        generators, relations = test_ring_structure()
        
        # Test spectral sequence
        def test_spectral_sequence():
            """Test spectral sequence computation."""
            if cohomology.has_filtration():
                # Get spectral sequence
                E = cohomology.get_spectral_sequence()
                
                # Test differentials
                for r in range(2, len(E)):
                    d_r = cohomology.get_differential(r)
                    assert torch.allclose(
                        cohomology.compose(d_r, d_r),
                        torch.zeros_like(d_r)
                    ), f"d_{r} should square to zero"
                    
                # Test convergence
                E_infty = E[-1]
                gr = cohomology.associated_graded()
                assert torch.allclose(
                    E_infty, gr
                ), "Should converge to associated graded"
                
            return E if cohomology.has_filtration() else None
            
        spectral_sequence = test_spectral_sequence()
