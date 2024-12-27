import torch
from src.core.crystal.scale import ScaleCohomology
from src.core.quantum.u1_utils import compute_winding_number

class TestScaleCohomology:
    """Tests for scale cohomology computations."""
    
    def test_anomaly_polynomial(self):
        """Test that anomaly polynomial satisfies consistency."""
        # Create symmetry actions
        def g1(x):
            return torch.roll(x, shifts=1, dims=-1)
            
        def g2(x):
            return x.flip(-1)
        
        # Initialize cohomology computer
        cohomology = ScaleCohomology(dim=8)
        
        # Compute anomaly polynomials
        A1 = cohomology.anomaly_polynomial(g1)
        print("\nSymmetry actions on test vector:")
        print(f"g1(x): {g1(torch.tensor([1,2,3,4]))}")
        print(f"g2(x): {g2(torch.tensor([1,2,3,4]))}")
        print(f"g1(g2(x)): {g1(g2(torch.tensor([1,2,3,4])))}")
        
        # Validate information Ricci flow
        metric = A1[0].coefficients.reshape(-1, cohomology.dim, cohomology.dim)
        assert torch.all(torch.linalg.eigvals(metric).real > 0), "Metric should remain positive definite"
        print("\nInformation flow validation:")
        print(f"Metric eigenvalues: {torch.linalg.eigvals(metric).real}")
        
        # Validate pattern heat flow
        pattern = A1[0].coefficients
        pattern_evolved = pattern.reshape(-1, cohomology.dim)
        assert torch.isfinite(pattern_evolved).all(), "Pattern evolution should remain finite"
        print("\nPattern heat flow validation:")
        print(f"Pattern norm: {torch.norm(pattern_evolved)}")
        print(f"Pattern mean: {torch.mean(pattern_evolved)}")
        
        # Validate wave emergence
        phase_pattern = torch.exp(1j * torch.angle(A1[0].coefficients))
        assert torch.isfinite(phase_pattern).all(), "Phase pattern should remain finite"
        print("\nWave emergence validation:")
        print(f"Phase pattern norm: {torch.norm(phase_pattern)}")
        
        # Track winding numbers
        test_x = torch.linspace(0, 2*torch.pi, cohomology.dim, dtype=torch.float32)
        g1_winding = compute_winding_number(g1(torch.exp(1j * test_x)))
        g2_winding = compute_winding_number(g2(torch.exp(1j * test_x)))
        composed_winding = compute_winding_number(g1(g2(torch.exp(1j * test_x))))
        
        print("\nWinding numbers:")
        print(f"g1 winding: {g1_winding/torch.pi:.3f}π")
        print(f"g2 winding: {g2_winding/torch.pi:.3f}π")
        print(f"g1∘g2 winding: {composed_winding/torch.pi:.3f}π")
        
        # Validate phase analysis
        A2 = cohomology.anomaly_polynomial(g2)
        composed = cohomology.anomaly_polynomial(lambda x: g1(g2(x)))
        
        print("\nPhase analysis (degree 0):")
        print(f"A1 phase: {torch.angle(A1[0].coefficients[0])/torch.pi:.3f}π")
        print(f"A2 phase: {torch.angle(A2[0].coefficients[0])/torch.pi:.3f}π")
        print(f"Composed phase: {torch.angle(composed[0].coefficients[0])/torch.pi:.3f}π")
        print(f"Sum phase: {torch.angle(A1[0].coefficients[0] + A2[0].coefficients[0])/torch.pi:.3f}π")
        
        # Validate coefficient comparison
        print("\nCoefficient comparison (polynomial 0):")
        print(f"Composed coefficients: {composed[0].coefficients}")
        print(f"A1: {A1[0].coefficients}")
        print(f"A2: {A2[0].coefficients}")
        print(f"Sum: {A1[0].coefficients + A2[0].coefficients}")
        
        abs_diff = torch.abs(composed[0].coefficients - (A1[0].coefficients + A2[0].coefficients))
        rel_diff = abs_diff / (torch.abs(composed[0].coefficients) + 1e-6)
        
        print(f"\nAbsolute difference: {torch.max(abs_diff):.4f}")
        print(f"Maximum relative difference: {torch.max(rel_diff):.4f}")
        
        # Final consistency check
        assert torch.allclose(
            composed[0].coefficients,
            A1[0].coefficients + A2[0].coefficients,
            rtol=1e-2,
            atol=1e-2
        ), "Anomaly should satisfy consistency" 