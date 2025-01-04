"""
Sympy analysis scratchpad for phase transformations in the OPE.
This is NOT production code - just mathematical analysis.

Key formulas for OPE phase transformations:
1. Single operator phase transformation: O(x) -> e^(iθ) O(x)
2. Combined OPE phase: e^(iθ₁)^Δ₁ * e^(iθ₂)^Δ₂
3. Phase normalization: phase / |phase|
"""

import sympy as sp
from sympy import I, pi, exp, Symbol, symbols, Abs
import numpy as np

# Define symbolic variables
theta1, theta2, delta1, delta2 = symbols('theta1 theta2 delta1 delta2', real=True)

def analyze_phase_transformation():
    """Analyze phase transformation for a single operator."""
    # Phase transformation for single operator
    phase = exp(I*theta1)
    transformed = phase**delta1
    
    # Test with concrete values
    test_vals = {theta1: pi/3, delta1: 1.0}
    result = transformed.subs(test_vals)
    
    print("\nSingle Operator Phase Analysis:")
    print(f"Input phase: {float(test_vals[theta1]):.4f} rad")
    print(f"Transformed phase: {sp.re(result):.4f} + {sp.im(result):.4f}i")
    print(f"Phase angle: {float(sp.arg(result)):.4f} rad")

def analyze_ope_phases():
    """Analyze phase combination in OPE."""
    # Individual phases
    phase1 = exp(I*theta1)
    phase2 = exp(I*theta2)
    
    # Combined phase with conformal weights
    combined = (phase1**delta1 * phase2**delta2)
    # Normalize phase
    normalized = combined / Abs(combined)
    
    # Test cases
    test_cases = [
        {theta1: pi/4, theta2: pi/3, delta1: 1.0, delta2: 1.0},
        {theta1: pi/2, theta2: pi/4, delta1: 1.0, delta2: 1.0},
        {theta1: pi/3, theta2: pi/6, delta1: 1.0, delta2: 1.0}
    ]
    
    print("\nOPE Phase Combination Analysis:")
    for case in test_cases:
        result = normalized.subs(case)
        angle = float(sp.arg(result))
        
        print(f"\nTest case:")
        print(f"θ₁ = {float(case[theta1]):.4f} rad")
        print(f"θ₂ = {float(case[theta2]):.4f} rad")
        print(f"Combined phase: {sp.re(result):.4f} + {sp.im(result):.4f}i")
        print(f"Phase angle: {angle:.4f} rad")
        print(f"Norm: {abs(complex(sp.re(result), sp.im(result))):.6f}")  # Should be 1.0

def verify_composition_law():
    """Verify phase behavior in composition law."""
    # Mode number and dimension
    k, N = symbols('k N', integer=True, positive=True)
    
    # Phase factor (no root scaling)
    phase = exp(I * 2*pi * k/N)
    
    # Test with concrete values
    test_vals = {k: 1, N: 4}
    result = phase.subs(test_vals)
    
    print("\nComposition Law Phase Analysis:")
    print(f"Phase factor: {sp.re(result):.4f} + {sp.im(result):.4f}i")
    print(f"Phase angle: {float(sp.arg(result)):.4f} rad")
    print(f"Norm: {abs(complex(sp.re(result), sp.im(result))):.6f}")  # Should be 1.0

if __name__ == "__main__":
    analyze_phase_transformation()
    analyze_ope_phases()
    verify_composition_law() 