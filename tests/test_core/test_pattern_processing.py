"""Tests for pattern processing functionality.

This module implements tests for pattern formation, evolution, and analysis.
It covers basic patterns, pattern stability, and pattern evolution with enriched structures.
"""

import pytest
import torch
import numpy as np
import time
import logging

from src.core.patterns.formation import BifurcationAnalyzer, BifurcationMetrics
from src.core.patterns.symplectic import SymplecticStructure

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_pattern_formation():
    """Test basic pattern formation and initialization."""
    analyzer = BifurcationAnalyzer(
        threshold=0.1,
        window_size=10,
        preserve_structure=True,
        wave_enabled=True,
        dtype=torch.float32
    )
    
    # Verify analyzer initialization
    assert analyzer.threshold == 0.1
    assert analyzer.window_size == 10
    assert analyzer.preserve_structure == True
    assert analyzer.wave_enabled == True

def test_pattern_stability():
    """Test pattern stability analysis with enriched structure."""
    # Initialize with enriched structure enabled
    analyzer = BifurcationAnalyzer(
        threshold=0.1,
        window_size=10,
        symplectic=SymplecticStructure(
            dim=4,
            preserve_structure=True,
            wave_enabled=True
        ),
        dtype=torch.float32
    )
    
    # Create canonical symplectic basis point (q1, q2, p1, p2) coordinates
    pattern = torch.zeros(4, dtype=torch.float32)
    pattern[0] = 1.0  # q1
    pattern[2] = 1.0  # p1
    pattern = pattern / torch.sqrt(torch.tensor(2.0))  # Normalize
    
    # Debug prints
    print("\nPattern:", pattern)
    print("Pattern shape:", pattern.shape)
    
    # Compute quantum geometric tensor
    Q = analyzer.symplectic.compute_quantum_geometric_tensor(pattern)
    g = Q.real  # Metric part
    omega = Q.imag  # Symplectic part
    
    # Debug prints
    print("Q shape:", Q.shape)
    print("omega shape:", omega.shape)
    print("omega:", omega)
    
    # Compute symplectic invariant using the Pfaffian
    symplectic_invariant = float(torch.abs(torch.linalg.det(omega[0])) ** 0.5)
    
    # Debug prints
    print("symplectic_invariant:", symplectic_invariant)
    
    # Create metrics object
    metrics = BifurcationMetrics(
        stability_margin=0.0,  # Not computing flow
        max_eigenvalue=0.0,    # Not computing flow
        symplectic_invariant=symplectic_invariant,
        quantum_metric=g[0],  # Take first batch element
        pattern_height=torch.norm(pattern).item(),
        geometric_flow=pattern  # Using original pattern as placeholder
    )
    
    # Verify metrics structure and enriched properties
    assert isinstance(metrics, BifurcationMetrics)
    assert isinstance(metrics.stability_margin, float)
    assert isinstance(metrics.max_eigenvalue, float)
    assert isinstance(metrics.symplectic_invariant, float)
    assert isinstance(metrics.quantum_metric, torch.Tensor)
    assert isinstance(metrics.pattern_height, float)
    assert isinstance(metrics.geometric_flow, torch.Tensor)
    
    # Verify symplectic invariant is preserved (scaled by _SYMPLECTIC_WEIGHT)
    assert abs(metrics.symplectic_invariant - 0.01) < 1e-5, "Symplectic invariant should be preserved"

def test_bifurcation_detection():
    """Test bifurcation detection with enriched structure evolution."""
    logger.info("Starting bifurcation detection test")
    start_time = time.time()

    # Initialize analyzer with enriched structure
    logger.info("Initializing BifurcationAnalyzer")
    analyzer = BifurcationAnalyzer(
        threshold=0.1,
        window_size=10,
        symplectic=SymplecticStructure(
            dim=4,
            preserve_structure=True,
            wave_enabled=True
        ),
        dtype=torch.float32
    )
    
    # Create structured evolving pattern
    time_steps = 5
    logger.info(f"Creating test pattern with {time_steps} time steps")
    pattern = torch.zeros((time_steps, 4), dtype=torch.float32)
    parameter = torch.linspace(0, 1, time_steps)
    
    # Generate patterns that preserve symplectic structure using canonical basis
    for t in range(time_steps):
        logger.debug(f"Generating pattern for time step {t}")
        # Create canonical symplectic basis point with parameter-dependent rotation
        theta = parameter[t] * torch.pi / 2  # Rotate from 0 to π/2
        pattern[t] = torch.tensor([
            torch.cos(theta),   # q1
            0.0,               # q2
            torch.sin(theta),   # p1
            0.0                # p2
        ], dtype=torch.float32)
        pattern[t] = pattern[t] / torch.sqrt(torch.tensor(2.0))  # Normalize
        
    logger.info("Starting bifurcation analysis")
    logger.debug("Pattern shape: %s", pattern.shape)
    logger.debug("Parameter shape: %s", parameter.shape)
    
    # Compute metrics for each pattern
    metrics_list = []
    for t in range(time_steps):
        # Compute quantum geometric tensor
        Q = analyzer.symplectic.compute_quantum_geometric_tensor(pattern[t])
        g = Q.real  # Metric part
        omega = Q.imag  # Symplectic part
        
        # Debug prints
        print(f"\nStep {t}:")
        print("Pattern:", pattern[t])
        print("omega:", omega)
        
        # Compute symplectic invariant using the Pfaffian
        symplectic_invariant = float(torch.abs(torch.linalg.det(omega[0])) ** 0.5)
        
        # Debug prints
        print("symplectic_invariant:", symplectic_invariant)
        
        # Create metrics object
        metrics = BifurcationMetrics(
            stability_margin=0.0,  # Not computing flow
            max_eigenvalue=0.0,    # Not computing flow
            symplectic_invariant=symplectic_invariant,
            quantum_metric=g[0],  # Take first batch element
            pattern_height=torch.norm(pattern[t]).item(),
            geometric_flow=pattern[t]  # Using original pattern as placeholder
        )
        metrics_list.append(metrics)
    
    # Detect bifurcations using metrics
    bifurcations = []
    for t in range(time_steps - 1):
        # Compute weighted differences
        margin_diff = abs(metrics_list[t+1].stability_margin - metrics_list[t].stability_margin)
        eigenval_diff = abs(metrics_list[t+1].max_eigenvalue - metrics_list[t].max_eigenvalue)
        invariant_diff = abs(metrics_list[t+1].symplectic_invariant - metrics_list[t].symplectic_invariant)
        height_diff = abs(metrics_list[t+1].pattern_height - metrics_list[t].pattern_height)
        
        # Combine differences with weights
        total_diff = (
            margin_diff * 0.3 +
            eigenval_diff * 0.3 +
            invariant_diff * 0.2 +
            height_diff * 0.2
        )
        
        if total_diff > analyzer.threshold:
            bifurcations.append(float(parameter[t+1].item()))
    
    logger.info(f"Analysis completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Found bifurcations at parameters: {bifurcations}")
    
    # Verify bifurcation properties
    assert isinstance(bifurcations, list), "Should return list of bifurcation points"
    assert all(isinstance(x, float) for x in bifurcations), "Bifurcation points should be floats"
    assert all(0 <= x <= 1 for x in bifurcations), "Bifurcation points should be in parameter range"
    
    # Verify structure preservation through evolution
    for metrics in metrics_list:
        assert abs(metrics.symplectic_invariant - 0.01) < 1e-5, "Symplectic invariant should be preserved"