"""
Unit tests for the crystal scale cohomology system.

Tests cover:
1. Scale connection properties
2. Renormalization group flow
3. Fixed point analysis
4. Anomaly detection
5. Scale invariants
6. Callan-Symanzik equations
7. Operator product expansion
8. Conformal symmetry
9. Holographic scaling
10. Entanglement scaling
"""

import numpy as np
import pytest
import torch

from src.core.crystal.scale import ScaleCohomology


class TestScaleCohomology:
    @pytest.fixture
    def space_dim(self):
        """Dimension of base space."""
        return 4

    @pytest.fixture
    def scale_system(self, space_dim):
        """Create scale system fixture."""
        return ScaleCohomology(
            dim=space_dim,
            num_scales=4
        )

    def test_scale_connection(self, scale_system, space_dim):
        """Test scale connection properties."""
        # Create test scales
        scale1, scale2 = torch.tensor(1.0), torch.tensor(2.0)

        # Compute connection
        connection = scale_system.scale_connection(scale1, scale2)

        # Test connection properties
        assert connection.connection_map.shape == (
            space_dim,
            space_dim,
        ), "Connection should be matrix-valued"

        # Test compatibility condition
        def test_compatibility(s1, s2, s3):
            """Test connection compatibility between scales."""
            c12 = scale_system.scale_connection(s1, s2)
            c23 = scale_system.scale_connection(s2, s3)
            c13 = scale_system.scale_connection(s1, s3)
            
            # Debug prints
            print("\nConnection c12:")
            print(c12.connection_map)
            print("\nConnection c23:")
            print(c23.connection_map)
            print("\nConnection c13:")
            print(c13.connection_map)
            print("\nComposed c23 @ c12:")
            print(c23.connection_map @ c12.connection_map)
            
            return torch.allclose(
                c13.connection_map,
                c23.connection_map @ c12.connection_map,
                rtol=1e-4
            )

        scale3 = torch.tensor(4.0)
        assert test_compatibility(
            scale1, scale2, scale3
        ), "Connection should satisfy compatibility"

        # Test infinitesimal generator
        def test_infinitesimal(scale, epsilon):
            """Test infinitesimal connection properties."""
            connection = scale_system.scale_connection(scale, scale + epsilon)
            generator = scale_system.connection_generator(scale)
            expected = torch.eye(space_dim) + epsilon * generator
            
            # Debug prints
            print("\nInfinitesimal test:")
            print("Connection map:")
            print(connection.connection_map)
            print("\nGenerator:")
            print(generator)
            print("\nExpected (I + eps*generator):")
            print(expected)
            
            return torch.allclose(connection.connection_map, expected, rtol=1e-3)

        assert test_infinitesimal(
            scale1, 1e-3
        ), "Connection should have correct infinitesimal form"

    def test_renormalization_flow(self, scale_system):
        """Test renormalization group flow properties."""

        # Create test observable
        def test_observable(x: torch.Tensor) -> torch.Tensor:
            return torch.sum(x**2)

        # Compute RG flow
        flow = scale_system.renormalization_flow(test_observable)

        # Test flow properties
        scales = flow.scale_points()
        assert len(scales) > 0, "Flow should have scale points"

        # Test basic evolution
        def test_basic_evolution(t):
            """Test basic evolution for time t."""
            flow_t = flow.evolve(t)
            assert flow_t.observable is not None, "Evolution should preserve observable"
            assert not torch.isnan(flow_t.observable).any(), "Evolution produced NaN values"
            return flow_t

        # Test composition of small steps
        def test_step_composition(dt, n_steps):
            """Test if evolution by dt n times equals evolution by dt*n."""
            flow_small = flow
            print("\nTesting step composition:")
            print(f"Initial observable: {flow_small.observable}")
            
            # Evolve in small steps
            for i in range(n_steps):
                flow_small = flow_small.evolve(dt)
                print(f"After step {i+1}: {flow_small.observable}")
                if torch.isnan(flow_small.observable).any():
                    print(f"NaN detected at step {i+1}")
                    return False
            
            # Evolve in one large step
            flow_large = flow.evolve(dt * n_steps)
            print(f"Large step result: {flow_large.observable}")
            
            # Compare results
            are_close = torch.allclose(
                flow_small.observable, flow_large.observable, rtol=1e-4
            )
            if not are_close:
                print("Step composition test failed:")
                print(f"Small steps result: {flow_small.observable}")
                print(f"Large step result: {flow_large.observable}")
                print(f"Difference: {torch.abs(flow_small.observable - flow_large.observable).max().item()}")
            return are_close

        # Test metric consistency
        def test_metric_consistency(t):
            """Test if metric evolution is consistent."""
            flow_t = flow.evolve(t)
            if flow_t._metric is not None:
                eigenvals = torch.linalg.eigvals(flow_t._metric).real
                return torch.all(eigenvals > 0)
            return True

        # Test beta function consistency
        def test_beta_consistency(t):
            """Test if beta function evolution is consistent."""
            flow_t = flow.evolve(t)
            obs = flow_t.observable
            beta = flow_t.beta_function(obs)
            return not torch.isnan(beta).any()

        # Run diagnostic tests
        assert test_basic_evolution(0.1), "Basic evolution failed"
        assert test_step_composition(0.1, 5), "Step composition failed"
        assert test_metric_consistency(0.5), "Metric consistency failed"
        assert test_beta_consistency(0.5), "Beta function consistency failed"

        # Test semigroup property
        def test_semigroup(t1, t2):
            """Test semigroup property of RG flow."""
            flow_t1 = flow.evolve(t1)
            flow_t2 = flow.evolve(t2)
            flow_sum = flow.evolve(t1 + t2)

            # Test intermediate results
            assert not torch.isnan(flow_t1.observable).any(), "t1 evolution produced NaN"
            assert not torch.isnan(flow_t2.observable).any(), "t2 evolution produced NaN"
            assert not torch.isnan(flow_sum.observable).any(), "Combined evolution produced NaN"

            # Test application
            applied = flow_t2.apply(flow_t1.observable)
            assert not torch.isnan(applied).any(), "Application produced NaN"

            # Compare results
            are_close = torch.allclose(
                flow_sum.observable, applied, rtol=1e-4
            )
            if not are_close:
                print(f"Semigroup test failed:")
                print(f"flow_sum.observable: {flow_sum.observable}")
                print(f"flow_t2.apply(flow_t1.observable): {applied}")
                print(f"Difference: {torch.abs(flow_sum.observable - applied).max().item()}")
            return are_close

        assert test_semigroup(0.5, 0.5), "Flow should satisfy semigroup property"

        # Test scaling dimension
        scaling_dim = flow.scaling_dimension()
        assert scaling_dim is not None, "Should compute scaling dimension"

        # Test correlation length
        corr_length = flow.correlation_length()
        assert corr_length > 0, "Correlation length should be positive"

    def test_fixed_points(self, scale_system):
        """Test fixed point analysis."""

        # Create test flow
        def beta_function(x):
            """Simple β-function with known fixed point."""
            return x * (1 - x)

        # Find fixed points
        fixed_points = scale_system.fixed_points(beta_function)

        # Test fixed point properties
        assert len(fixed_points) > 0, "Should find fixed points"

        # Test stability
        for fp in fixed_points:
            stability = scale_system.fixed_point_stability(fp, beta_function)
            assert stability in [
                "stable",
                "unstable",
                "marginal",
            ], "Should classify fixed point stability"

        # Test critical exponents
        critical_exps = scale_system.critical_exponents(fixed_points[0], beta_function)
        assert len(critical_exps) > 0, "Should compute critical exponents"

        # Test universality
        def perturbed_beta(x):
            """Slightly perturbed β-function."""
            return beta_function(x) + 0.1 * x**3

        perturbed_fps = scale_system.fixed_points(perturbed_beta)
        perturbed_exps = scale_system.critical_exponents(
            perturbed_fps[0], perturbed_beta
        )

        assert torch.allclose(
            torch.tensor(critical_exps), torch.tensor(perturbed_exps), rtol=1e-2
        ), "Critical exponents should be universal"

    def test_anomaly_polynomial(self, scale_system):
        """Test anomaly polynomial computation."""

        # Create test symmetry
        def symmetry_action(x):
            """Simple U(1) symmetry."""
            return torch.exp(1j * x)

        # Compute anomaly
        anomaly = scale_system.anomaly_polynomial(symmetry_action)

        # Test anomaly properties
        assert anomaly is not None, "Should compute anomaly"

        # Test Wess-Zumino consistency
        def test_consistency(g1, g2):
            """Test Wess-Zumino consistency condition."""
            a1 = scale_system.anomaly_polynomial(g1)
            a2 = scale_system.anomaly_polynomial(g2)
            composed = scale_system.anomaly_polynomial(lambda x: g1(g2(x)))
            
            # Compare coefficients of each anomaly polynomial
            for i, (c, a1p, a2p) in enumerate(zip(composed, a1, a2)):
                print(f"Polynomial {i}:")
                print(f"  Composed: {c.coefficients}")
                print(f"  A1: {a1p.coefficients}")
                print(f"  A2: {a2p.coefficients}")
                print(f"  A1 + A2: {a1p.coefficients + a2p.coefficients}")
                print(f"  Difference: {torch.norm(c.coefficients - (a1p.coefficients + a2p.coefficients))}")
                if not torch.allclose(c.coefficients, a1p.coefficients + a2p.coefficients, rtol=1e-4):
                    return False
            return True

        def g1(x):
            return torch.exp(1j * x)

        def g2(x):
            return torch.exp(2j * x)

        assert test_consistency(g1, g2), "Anomaly should satisfy consistency"

    def test_scale_invariants(self, scale_system):
        """Test scale invariant quantity computation."""
        # Create test structure
        structure = torch.randn(10, 10)

        # Compute invariants
        invariants = scale_system.scale_invariants(structure)

        # Test invariant properties
        assert len(invariants) > 0, "Should find scale invariants"

        # Test invariance
        def test_invariance(invariant, scale_factor):
            """Test scale invariance of quantity."""
            scaled_structure = structure * scale_factor
            original_value = invariant(structure)
            scaled_value = invariant(scaled_structure)
            return torch.allclose(original_value, scaled_value, rtol=1e-4)

        for inv in invariants:
            assert test_invariance(inv, 2.0), "Quantity should be scale invariant"

        # Test algebraic properties
        if len(invariants) >= 2:
            # Test product is also invariant
            def product_inv(x):
                return invariants[0](x) * invariants[1](x)

            assert test_invariance(
                product_inv, 2.0
            ), "Product of invariants should be invariant"

        # Test completeness
        def test_completeness(structure):
            """Test if invariants form complete set."""
            values = torch.tensor([inv(structure) for inv in invariants])
            return len(values) >= torch.tensor(scale_system.minimal_invariant_number())

        assert test_completeness(structure), "Should find complete set of invariants"

    def test_callan_symanzik(self, scale_system, space_dim):
        """Test Callan-Symanzik equation properties."""

        # Create test correlation function
        def correlation(x1, x2, coupling):
            """Simple two-point correlation function."""
            return torch.exp(-coupling * torch.norm(x1 - x2))

        # Define beta function
        def beta(g):
            """Simple beta function."""
            return -(g**2)

        # Define anomalous dimension
        def gamma(g):
            """Anomalous dimension."""
            return g**2 / (4 * np.pi) ** 2

        # Test points
        x1 = torch.zeros(space_dim)
        x2 = torch.ones(space_dim)
        g = torch.tensor(0.1)  # coupling

        # Compute CS equation terms
        cs_operator = scale_system.callan_symanzik_operator(beta, gamma)
        cs_result = cs_operator(correlation, x1, x2, g)

        assert torch.allclose(
            cs_result, torch.tensor(0.0, dtype=cs_result.dtype), atol=1e-4
        ), "Correlation should satisfy CS equation"

        # Test scaling behavior
        def test_scaling(lambda_):
            """Test scaling behavior of correlation."""
            scaled_corr = correlation(lambda_ * x1, lambda_ * x2, g)
            dim = scale_system.scaling_dimension(correlation)
            return torch.allclose(
                scaled_corr, correlation(x1, x2, g) * lambda_ ** (-dim), rtol=1e-3
            )

        assert test_scaling(
            torch.tensor(2.0)
        ), "Correlation should have correct scaling"

    def test_operator_expansion(self, scale_system):
        """Test operator product expansion."""

        # Create test operators
        def op1(x):
            """First operator."""
            return torch.sin(x)

        def op2(x):
            """Second operator."""
            return torch.exp(-(x**2))

        # Compute OPE
        ope = scale_system.operator_product_expansion(op1, op2)

        # Test convergence
        x_near = torch.tensor(0.1)
        direct = op1(x_near) * op2(torch.tensor(0.0))
        expanded = sum(c * o(x_near) for c, o in ope)
        expanded = torch.as_tensor(expanded)
        assert torch.allclose(
            direct, expanded, rtol=1e-2
        ), "OPE should converge for nearby points"

        # Test associativity
        def test_associativity(op_a, op_b, op_c):
            """Test OPE associativity."""
            ope1 = scale_system.operator_product_expansion(
                lambda x: scale_system.operator_product_expansion(op_a, op_b)(x), op_c
            )
            ope2 = scale_system.operator_product_expansion(
                op_a, lambda x: scale_system.operator_product_expansion(op_b, op_c)(x)
            )
            return all(
                torch.allclose(c1, c2, rtol=1e-3)
                for (c1, _), (c2, _) in zip(ope1, ope2)
            )

        def op3(x):
            return x**2

        assert test_associativity(op1, op2, op3), "OPE should be associative"

    def test_conformal_symmetry(self, scale_system, space_dim):
        """Test conformal symmetry properties."""
        # Create test field and state
        field = torch.randn(10, 10)
        state = torch.randn(10, space_dim)  # Define state for mutual information tests

        # Test special conformal transformations
        def test_special_conformal(b_vector):
            """Test special conformal transformation."""
            x = torch.randn(space_dim)
            scale_system.special_conformal_transform(x, b_vector)
            # Should preserve angles
            v1 = torch.randn(space_dim)
            v2 = torch.randn(space_dim)
            angle1 = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))
            transformed_v1 = scale_system.transform_vector(v1, x, b_vector)
            transformed_v2 = scale_system.transform_vector(v2, x, b_vector)
            angle2 = torch.dot(transformed_v1, transformed_v2) / (
                torch.norm(transformed_v1) * torch.norm(transformed_v2)
            )
            return torch.allclose(angle1, angle2, rtol=1e-3)

        assert test_special_conformal(
            torch.ones(space_dim)
        ), "Special conformal transformation should preserve angles"

        # Test primary fields
        def test_primary_scaling(field, dimension):
            """Test primary field scaling."""
            lambda_ = torch.tensor(2.0)
            transformed = scale_system.transform_primary(field, lambda_, dimension=dimension)
            return torch.allclose(transformed, field * lambda_**dimension, rtol=1e-3)

        assert test_primary_scaling(field, dimension=torch.tensor(1.0)), "Primary fields should transform correctly"

        # Test conformal blocks
        blocks = scale_system.conformal_blocks(field, dimension=space_dim)
        assert len(blocks) > 0, "Should decompose into conformal blocks"

        # Test mutual information
        def test_mutual_info_monogamy(regions):
            """Test strong subadditivity via mutual information."""
            I12 = scale_system.mutual_information(state, regions[0], regions[1])
            I13 = scale_system.mutual_information(state, regions[0], regions[2])
            I23 = scale_system.mutual_information(state, regions[1], regions[2])
            return torch.all(I12 + I13 + I23 >= torch.tensor(0.0))

        # Create test regions
        test_regions = [
            torch.tensor([[0, 0], [1, 1]]),
            torch.tensor([[1, 1], [2, 2]]),
            torch.tensor([[2, 2], [3, 3]])
        ]
        assert test_mutual_info_monogamy(test_regions), "Mutual information should satisfy monogamy"

    def test_holographic_scaling(self, scale_system, space_dim):
        """Test holographic scaling relations."""
        # Create bulk and boundary data
        boundary_field = torch.randn(10, 10)
        radial_coordinate = torch.linspace(0.1, 10.0, 50)

        # Test radial evolution
        bulk_field = scale_system.holographic_lift(boundary_field, radial_coordinate)
        assert bulk_field.shape[0] == radial_coordinate.shape[0], "Bulk field should extend along radial direction"

        # Test UV/IR connection
        def test_uv_ir_connection(field):
            """Test UV/IR connection in holographic scaling."""
            uv_data = scale_system.extract_uv_data(field)
            ir_data = scale_system.extract_ir_data(field)
            reconstructed = scale_system.reconstruct_from_ir(ir_data)
            return torch.allclose(uv_data, reconstructed, atol=0.0, rtol=1e-2)

        assert test_uv_ir_connection(bulk_field), "Should satisfy UV/IR connection"

        # Test holographic c-theorem
        c_function = scale_system.compute_c_function(bulk_field, radial_coordinate)
        assert torch.all(
            c_function[1:] - c_function[:-1] <= 0
        ), "C-function should decrease monotonically"

        # Test holographic entanglement
        subsystem = torch.tensor([[0, 0], [1, 1]])  # Define subsystem region
        entanglement = scale_system.holographic_entanglement(
            bulk_field, subsystem, radial_coordinate
        )
        assert entanglement > 0, "Entanglement entropy should be positive"

    def test_entanglement_scaling(self, scale_system, space_dim):
        """Test entanglement entropy scaling."""
        # Create test state
        state = torch.randn(32, 32)  # Lattice state

        # Test area law
        def test_area_law(region_sizes):
            """Test area law scaling of entanglement."""
            entropies = []
            areas = []
            for size in region_sizes:
                region = torch.ones(size, size)
                entropy = scale_system.entanglement_entropy(state, region)
                area = 4 * size  # Perimeter of square region
                entropies.append(entropy)
                areas.append(area)

            # Fit to area law S = α A + β
            coeffs = np.polyfit(areas, entropies, 1)
            return coeffs[0] > 0  # α should be positive

        sizes = [2, 4, 6, 8]
        assert test_area_law(sizes), "Should satisfy area law scaling"

        # Test mutual information
        def test_mutual_info_monogamy(regions):
            """Test strong subadditivity via mutual information."""
            I12 = scale_system.mutual_information(state, regions[0], regions[1])
            I13 = scale_system.mutual_information(state, regions[0], regions[2])
            I23 = scale_system.mutual_information(state, regions[1], regions[2])
            return torch.all(I12 + I13 + I23 >= torch.tensor(0.0))

        test_regions = [torch.ones(4, 4), torch.ones(4, 4), torch.ones(4, 4)]
        assert test_mutual_info_monogamy(
            test_regions
        ), "Should satisfy mutual information monogamy"

        # Test critical scaling
        if hasattr(scale_system, "is_critical"):
            critical_state = scale_system.prepare_critical_state(32)
            log_terms = scale_system.logarithmic_corrections(
                critical_state,
                dimension=space_dim
            )
            assert len(log_terms) > 0, "Critical state should have log corrections"

    def test_beta_function_consistency(self, scale_system):
        """Test if beta function is consistent under composition."""
        # Create test observable
        def test_observable(x: torch.Tensor) -> torch.Tensor:
            return torch.sum(x**2)
        
        # Compute RG flow
        flow = scale_system.renormalization_flow(test_observable)
        
        # Test beta function at different points
        x = torch.ones(4) * 2.0
        beta1 = flow.beta_function(x)
        
        # Evolve x and compute beta
        x_evolved = x + 0.1 * beta1
        beta2 = flow.beta_function(x_evolved)
        
        # The beta function should change smoothly
        diff = torch.norm(beta2 - beta1)
        print(f"\nBeta function test:")
        print(f"Initial beta: {beta1}")
        print(f"Evolved beta: {beta2}")
        print(f"Difference: {diff}")
        assert diff < 1.0, "Beta function changed too abruptly"

    def test_metric_evolution(self, scale_system):
        """Test if metric evolution is consistent."""
        # Create test observable
        def test_observable(x: torch.Tensor) -> torch.Tensor:
            return torch.sum(x**2)
        
        # Compute RG flow
        flow = scale_system.renormalization_flow(test_observable)
        
        # Get initial metric
        x = torch.ones(4) * 2.0
        metric1 = flow._compute_metric(x)
        
        # Evolve and get new metric
        beta = flow.beta_function(x)
        x_evolved = x + 0.1 * beta
        metric2 = flow._compute_metric(x_evolved)
        
        # Check metric properties
        print(f"\nMetric evolution test:")
        print(f"Initial metric eigenvalues: {torch.linalg.eigvals(metric1).real}")
        print(f"Evolved metric eigenvalues: {torch.linalg.eigvals(metric2).real}")
        
        # Metrics should be positive definite
        assert torch.all(torch.linalg.eigvals(metric1).real > 0), "Initial metric not positive definite"
        assert torch.all(torch.linalg.eigvals(metric2).real > 0), "Evolved metric not positive definite"
        
        # Metric should change smoothly
        diff = torch.norm(metric2 - metric1)
        print(f"Metric difference: {diff}")
        assert diff < 1.0, "Metric changed too abruptly"

    def test_scale_factor_composition(self, scale_system):
        """Test if scale factors compose properly."""
        # Create test observable
        def test_observable(x: torch.Tensor) -> torch.Tensor:
            return torch.sum(x**2)
        
        # Compute RG flow
        flow = scale_system.renormalization_flow(test_observable)
        
        # Test scale factor composition
        x = torch.ones(4) * 2.0
        
        # Evolve in two small steps
        x_evolved, scale1 = flow._integrate_beta(x, 0.1)
        x_evolved2, scale2 = flow._integrate_beta(x_evolved, 0.1)
        combined_small = scale1 * scale2
        
        # Evolve in one large step
        _, scale_large = flow._integrate_beta(x, 0.2)
        
        # Compare results
        print(f"\nScale factor composition test:")
        print(f"Small step 1: {scale1}")
        print(f"Small step 2: {scale2}")
        print(f"Combined small steps: {combined_small}")
        print(f"Large step: {scale_large}")
        print(f"Difference: {abs(combined_small - scale_large)}")
        assert abs(combined_small - scale_large) < 0.1, "Scale factors don't compose properly"

    def test_state_evolution_linearity(self, scale_system):
        """Test if state evolution respects linearity."""
        # Create test observable
        def test_observable(x: torch.Tensor) -> torch.Tensor:
            return torch.sum(x**2)
        
        # Compute RG flow
        flow = scale_system.renormalization_flow(test_observable)
        
        # Test two different states
        x1 = torch.ones(4) * 2.0
        x2 = torch.ones(4) * 3.0
        
        # Evolve states separately
        evolved_x1, _ = flow._integrate_beta(x1, 0.1)
        evolved_x2, _ = flow._integrate_beta(x2, 0.1)
        
        # Evolve their sum
        sum_evolved, _ = flow._integrate_beta(x1 + x2, 0.1)
        
        # Compare results
        print(f"\nLinearity test:")
        print(f"Evolved x1: {evolved_x1}")
        print(f"Evolved x2: {evolved_x2}")
        print(f"Sum of evolved: {evolved_x1 + evolved_x2}")
        print(f"Evolved sum: {sum_evolved}")
        print(f"Difference: {torch.norm(evolved_x1 + evolved_x2 - sum_evolved)}")
        assert torch.allclose(evolved_x1 + evolved_x2, sum_evolved, rtol=1e-4), "Evolution doesn't respect linearity"
