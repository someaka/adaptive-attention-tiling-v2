"""
Unit tests for the crystal scale cohomology system.

Tests cover:
1. Scale connection properties (CRITICAL)
2. Callan-Symanzik equations (HIGHLY CRITICAL)
3. Anomaly detection (VERY IMPORTANT)
4. Scale invariants (IMPORTANT)
5. Holographic scaling (IMPORTANT)
6. Conformal symmetry (MODERATE)
7. Entanglement scaling (MODERATE)
8. Operator product expansion
9. Renormalization group flow
10. Fixed point analysis
11. Beta function consistency
12. Metric evolution
13. Scale factor composition
14. State evolution linearity
"""

import numpy as np
import pytest
import torch
import os
import yaml
from contextlib import contextmanager
import psutil
import time

from src.core.crystal.scale import ScaleCohomology
from src.core.quantum.u1_utils import compute_winding_number
from src.utils.memory_management import optimize_memory

@contextmanager
def memory_efficient_test():
    """Context manager for memory-efficient test execution."""
    with optimize_memory("test"):
        yield

@pytest.fixture
def test_config():
    """Load test configuration from debug regimen."""
    config_path = os.path.join('configs', 'test_regimens', 'debug.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return {
        "manifold_dim": 4,  # Reduce from config value to minimum
        "num_layers": 2,  # Minimum number of layers
        "hidden_dim": 8,  # Reduce from config value to minimum
        "num_heads": 2,  # Reduce from config value to minimum
        "batch_size": 1,  # Reduce from config value to minimum
        "dtype": config['geometric_tests']['dtype'],
        "dt": 0.1,
        "epsilon": 1e-6
    }

@pytest.fixture
def space_dim(test_config):
    """Dimension of base space."""
    return test_config["manifold_dim"]

@pytest.fixture
def dtype():
    """Data type for quantum computations."""
    return torch.complex64

@pytest.fixture
def scale_system(space_dim, test_config, dtype):
    """Create scale system fixture."""
    return ScaleCohomology(
        dim=space_dim,
        num_scales=test_config["num_layers"],
        dtype=dtype
    )

class TestScaleCohomology:
    def test_anomaly_polynomial(self, scale_system, space_dim, dtype):
        """Test anomaly polynomial computation with memory optimization."""
        from src.core.performance.cpu.memory_management import MemoryManager
        
        # Initialize memory manager
        memory_manager = MemoryManager()
        
        def log_memory():
            stats = memory_manager.get_memory_stats()
            return stats['allocated_memory'] / 1024 / 1024  # Convert to MB
        
        def log_step(step, start_time, start_mem):
            current_mem = log_memory()
            elapsed = time.time() - start_time
            print(f"\n{step}:")
            print(f"Memory: {current_mem:.1f}MB (Change: {current_mem - start_mem:.1f}MB)")
            print(f"Time elapsed: {elapsed:.1f}s")
            cache_stats = memory_manager.get_cache_stats()
            print(f"Cache hits/misses: {cache_stats[0]}/{cache_stats[1]}")
        
        print("\nStarting anomaly polynomial test...")
        start_time = time.time()
        start_mem = log_memory()
        
        with memory_efficient_test():
            # Create test vector with minimal dimension
            log_step("Creating test vector", start_time, start_mem)
            x = memory_manager.allocate_tensor((space_dim,), dtype=dtype)
            with torch.no_grad():
                x.copy_(torch.randn(space_dim, dtype=dtype))
            
            # Define symmetry actions
            log_step("Defining symmetry actions", start_time, start_mem)
            def g1(x):
                return x * torch.exp(1j * torch.tensor(0.5 * np.pi))
                
            def g2(x):
                return x * torch.exp(1j * torch.tensor(np.pi))
            
            # Track original state
            x_orig = memory_manager.allocate_tensor((space_dim,), dtype=dtype)
            with torch.no_grad():
                x_orig.copy_(x)
            
            # Apply symmetries with memory tracking
            log_step("Applying symmetries", start_time, start_mem)
            g1_x = memory_manager.allocate_tensor((space_dim,), dtype=dtype)
            g2_x = memory_manager.allocate_tensor((space_dim,), dtype=dtype)
            g1g2_x = memory_manager.allocate_tensor((space_dim,), dtype=dtype)
            with torch.no_grad():
                g1_x.copy_(g1(x))
                g2_x.copy_(g2(x))
                g1g2_x.copy_(g1(g2(x)))
            
            # Compute winding numbers with memory tracking
            log_step("Computing winding numbers", start_time, start_mem)
            g1_winding = compute_winding_number(g1_x)
            g2_winding = compute_winding_number(g2_x)
            g1g2_winding = compute_winding_number(g1g2_x)
            
            print("\nWinding numbers:")
            print(f"g1 winding: {g1_winding/np.pi:.3f}π")
            print(f"g2 winding: {g2_winding/np.pi:.3f}π")
            print(f"g1∘g2 winding: {g1g2_winding/np.pi:.3f}π")
            
            # Compute anomaly polynomials with memory tracking
            log_step("Computing A1", start_time, start_mem)
            A1 = scale_system.anomaly_polynomial(g1)
            
            log_step("Computing A2", start_time, start_mem)
            A2 = scale_system.anomaly_polynomial(g2)
            
            log_step("Computing composed anomaly", start_time, start_mem)
            composed = scale_system.anomaly_polynomial(lambda x: g1(g2(x)))
            
            # Analyze degree 0 phase factors
            print("\nPhase analysis (degree 0):")
            print(f"A1 phase: {torch.angle(A1[0])/np.pi:.3f}π")
            print(f"A2 phase: {torch.angle(A2[0])/np.pi:.3f}π")
            print(f"Composed phase: {torch.angle(composed[0])/np.pi:.3f}π")
            print(f"Sum phase: {torch.angle(A1[0] + A2[0])/np.pi:.3f}π")
            
            # Compare coefficients
            print("\nCoefficient comparison (polynomial 0):")
            print(f"Composed coefficients: {composed[0]}")
            print(f"A1: {A1[0]}")
            print(f"A2: {A2[0]}")
            print(f"Sum: {A1[0] + A2[0]}")
            
            # Compute differences with memory tracking
            abs_diff = torch.abs(composed[0] - (A1[0] + A2[0]))
            rel_diff = abs_diff / (torch.abs(A1[0] + A2[0]) + 1e-8)
            
            print(f"\nAbsolute difference: {abs_diff}")
            print(f"Maximum relative difference: {torch.max(rel_diff)}")
            
            print("\nChecking consistency...")
            # Assert consistency
            assert torch.allclose(composed[0], A1[0] + A2[0], rtol=1e-3), "Anomaly should satisfy consistency"
            
            print("\nVerifying state preservation...")
            # Verify state is unchanged
            assert torch.allclose(x, x_orig), "Original state should be preserved"
            
            print("\nTest completed successfully.")
            
            # Clean up memory
            memory_manager.defragment_memory() 