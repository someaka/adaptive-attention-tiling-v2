# Anomaly Polynomial Test Fix Plan

## Problem Analysis
The anomaly polynomial test is failing due to inconsistencies in U(1) symmetry preservation and phase tracking. The test shows:
- Incorrect winding number composition
- Phase inconsistencies in polynomial coefficients
- Large relative differences in anomaly coefficients

## Action Items

### 1. Fix Phase Preservation
- [ ] Modify normalization to preserve complex phases
- [ ] Implement proper U(1)-covariant normalization
- [ ] Add phase tracking through RG flow
- [ ] Test phase consistency with simple U(1) transformations

### 2. Improve Polynomial Basis
- [ ] Replace Hermite polynomials with U(1)-covariant basis
- [ ] Implement proper inner product for complex vector space
- [ ] Add phase-aware coefficient computation
- [ ] Test basis orthogonality and completeness

### 3. Fix Flow Observable
- [ ] Modify flow observable to respect U(1) symmetry
- [ ] Implement proper phase tracking in RG flow
- [ ] Add consistency checks for flow evolution
- [ ] Test observable transformation properties

### 4. Enhance Wess-Zumino Consistency
- [ ] Implement proper composition law for anomalies
- [ ] Add explicit checks for cocycle condition
- [ ] Track phase factors in composition
- [ ] Test consistency with known U(1) anomalies

### 5. Add Debug Tooling
- [ ] Add detailed phase tracking logs
- [ ] Implement visualization for flow trajectories
- [ ] Add numerical stability diagnostics
- [ ] Create test helper functions

## Implementation Order

1. **Phase I: Foundation**
   - Fix normalization and phase tracking
   - Implement U(1)-covariant basis
   - Add basic debug tooling

2. **Phase II: Core Functionality**
   - Modify flow observable
   - Implement proper anomaly composition
   - Add consistency checks

3. **Phase III: Testing & Validation**
   - Add comprehensive test cases
   - Implement visualization tools
   - Document edge cases

4. **Phase IV: Optimization**
   - Improve numerical stability
   - Optimize performance
   - Add error handling

## Success Criteria

1. **Winding Numbers**
   - g1∘g2 winding should equal sum of individual windings (mod 2π)
   - Phase factors should compose properly
   - Winding numbers should be stable under small perturbations

2. **Anomaly Coefficients**
   - Relative difference should be < 1e-3
   - Phases should compose properly
   - Coefficients should transform covariantly

3. **Numerical Stability**
   - Results should be stable under different normalizations
   - Phase tracking should be robust
   - Flow evolution should be smooth

## Testing Strategy

1. **Unit Tests**
   - Test each component independently
   - Verify phase preservation
   - Check basis properties

2. **Integration Tests**
   - Test full anomaly computation
   - Verify Wess-Zumino consistency
   - Check flow properties

3. **Edge Cases**
   - Test with degenerate configurations
   - Verify behavior at singular points
   - Check numerical stability limits

## Notes
- Focus on mathematical correctness first
- Maintain numerical stability throughout
- Document all assumptions and choices
- Keep track of physical constraints 