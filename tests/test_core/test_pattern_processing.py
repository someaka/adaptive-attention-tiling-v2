"""Tests for pattern processing functionality.

This module implements tests for pattern formation, evolution, and analysis.
It covers basic patterns, pattern stability, and pattern evolution.
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
    """Test pattern stability analysis."""
    analyzer = BifurcationAnalyzer(
        threshold=0.1,
        window_size=10,
        symplectic=SymplecticStructure(dim=4),
        dtype=torch.float32
    )
    
    # Create test pattern (as a vector)
    pattern = torch.randn(4, dtype=torch.float32)
    
    # Analyze stability
    metrics = analyzer._compute_stability_metrics(pattern)
    
    # Verify metrics structure
    assert isinstance(metrics, BifurcationMetrics)
    assert isinstance(metrics.stability_margin, float)
    assert isinstance(metrics.max_eigenvalue, float)
    assert isinstance(metrics.symplectic_invariant, float)
    assert isinstance(metrics.quantum_metric, torch.Tensor)
    assert isinstance(metrics.pattern_height, float)
    assert isinstance(metrics.geometric_flow, torch.Tensor)

def test_bifurcation_detection():
    """Test bifurcation detection in pattern evolution."""
    logger.info("Starting bifurcation detection test")
    start_time = time.time()

    # Initialize analyzer
    logger.info("Initializing BifurcationAnalyzer")
    analyzer = BifurcationAnalyzer(
        threshold=0.1,
        window_size=10,
        symplectic=SymplecticStructure(dim=4),
        dtype=torch.float32
    )
    
    # Create evolving pattern (as vectors)
    time_steps = 5  # Reduced for debugging
    logger.info(f"Creating test pattern with {time_steps} time steps")
    pattern = torch.zeros((time_steps, 4), dtype=torch.float32)
    parameter = torch.linspace(0, 1, time_steps)
    
    for t in range(time_steps):
        logger.debug(f"Generating pattern for time step {t}")
        pattern[t] = torch.randn(4) * (1 + parameter[t])  # Increasing amplitude
        
    logger.info("Starting bifurcation analysis")
    logger.debug("Pattern shape: %s", pattern.shape)
    logger.debug("Parameter shape: %s", parameter.shape)
    
    t0 = time.time()
    bifurcations = analyzer.detect_bifurcations(pattern, parameter)
    t1 = time.time()
    
    logger.info(f"Analysis completed in {t1 - t0:.2f} seconds")
    logger.info(f"Found bifurcations at parameters: {bifurcations}")
    
    # Basic assertions
    assert isinstance(bifurcations, list), "Should return list of bifurcation points"
    assert all(isinstance(x, float) for x in bifurcations), "Bifurcation points should be floats"
    assert all(0 <= x <= 1 for x in bifurcations), "Bifurcation points should be in parameter range"