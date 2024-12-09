# 🏗️ Adaptive Attention Tiling Refactor Plan

> ⚠️ **Critical Backend Harmonization**: Before proceeding with this refactor, see [BACKEND_HARMONIZATION.md](./BACKEND_HARMONIZATION.md) for the detailed plan on unifying CPU (v2) and Vulkan implementations. The backend harmonization must be completed first as it informs the core architecture decisions in this refactor.

## Current State Analysis

The project has evolved into several key components that need unification:

1. **Core CPU Implementation (v2)** - Theoretically sound implementation in `src/core/tiling/v2/`
2. **Vulkan Backend Stub** - Initial implementation in `src/core/backends/vulkan/`
3. **Benchmark Suite** - Comprehensive but disconnected from validation pipeline
4. **Validation Pipeline** - Separate from main benchmarking infrastructure

## 🎯 Refactor Goals

1. **Unified Architecture**: Consolidate implementations under a single, coherent architecture (see [BACKEND_HARMONIZATION.md](./BACKEND_HARMONIZATION.md))
2. **Clean Backend Separation**: Properly abstract CPU and Vulkan implementations
3. **Integrated Validation**: Merge validation pipeline with benchmark suite
4. **Standardized Metrics**: Unify metrics collection across all components

## 📁 Proposed Directory Structure

```
src/
├── core/                   # Core implementation
│   ├── attention/         # Attention mechanisms (after backend harmonization)
│   │   ├── base.py       # Abstract attention interface
│   │   ├── geometric.py  # Geometric attention implementation
│   │   └── quantum.py    # Quantum attention implementation
│   │
│   ├── backends/         # Backend-specific code (see BACKEND_HARMONIZATION.md)
│   │   ├── cpu/         # CPU implementation
│   │   └── vulkan/      # Fresh Vulkan v2 implementation
│   │
│   ├── tiling/            # Tiling strategies
│   │   ├── base/         # Base tiling abstractions
│   │   ├── strategies/   # Different tiling strategies
│   │   └── optimization/ # Tiling optimizations
│   │
│   └── validation/       # Validation framework
│       ├── metrics/     # Validation metrics
│       └── pipeline/    # Validation pipeline
│
├── benchmark/            # Unified benchmark suite
│   ├── suite/           # Benchmark implementations
│   ├── metrics/         # Benchmark metrics
│   └── reporting/       # Results & visualization
│
└── utils/               # Shared utilities
    ├── profiling/      # Performance profiling
    ├── logging/        # Logging infrastructure
    └── visualization/  # Visualization tools
```

## 🔄 Migration Steps

### Phase 1: Core Restructuring
1. Move current `v2` implementation to new structure under `core/attention/`
2. Refactor tiling strategies into dedicated modules
3. Create proper abstractions for backend implementations
4. Update imports and dependencies

### Phase 2: Backend Integration
1. Complete Vulkan implementation following new architecture
2. Implement backend selection mechanism
3. Add backend-specific optimizations
4. Create comprehensive backend tests

### Phase 3: Validation Integration
1. Merge validation pipeline with benchmark suite
2. Standardize metric collection
3. Implement unified reporting
4. Add end-to-end validation tests

### Phase 4: Benchmark Enhancement
1. Update benchmark suite to use new architecture
2. Add support for backend-specific benchmarks
3. Implement MLflow integration for all components
4. Create comprehensive visualization tools

## 📊 Key Components to Merge

### Current Files to Migrate:
```
src/core/tiling/v2/
├── quantum_geometric_attention.py  → core/attention/quantum.py
├── geometric_flow.py              → core/attention/geometric.py
├── arithmetic_dynamics.py         → core/tiling/strategies/
└── state_manager.py              → core/tiling/base/

src/core/backends/vulkan/
└── [All files]                   → core/backends/vulkan/

src/core/benchmarks/
└── [All files]                   → benchmark/
```

## 🔍 Detailed Component Changes

### 1. Core Attention Layer
- Abstract base attention class
- Separate geometric and quantum implementations
- Clear interface for backend selection

### 2. Tiling Framework
- Abstract tiling strategy interface
- Pluggable optimization strategies
- Backend-aware tiling decisions

### 3. Backend System
- Abstract backend interface
- Unified memory management
- Backend-specific optimizations
- Clear fallback mechanisms

### 4. Benchmark & Validation
- Unified metric collection
- Integrated validation checks
- Comprehensive reporting
- MLflow integration throughout

## 📝 Documentation Updates

1. Update main README with new architecture
2. Create detailed backend documentation
3. Add benchmark & validation guides
4. Update API documentation
5. Add migration guide for existing users

## 🧪 Testing Strategy

1. Unit tests for each component
2. Integration tests for backend interactions
3. End-to-end validation tests
4. Performance regression tests
5. Cross-backend compatibility tests

## 🚀 Future Considerations

1. Additional backend support (CUDA, ROCm)
2. Distributed training support
3. Dynamic backend selection
4. Advanced profiling tools
5. Cloud deployment options

## Timeline Dependencies

Note: This refactor timeline must be adjusted based on the backend harmonization schedule:

1. Backend Harmonization (11-16 weeks) - See [BACKEND_HARMONIZATION.md](./BACKEND_HARMONIZATION.md)
2. Core Refactor (begins after backend harmonization):
   - Phase 1: 2-3 weeks
   - Phase 2: 3-4 weeks
   - Phase 3: 2-3 weeks
   - Phase 4: 2-3 weeks

Total estimated time: 20-29 weeks

## 🎯 Success Metrics

1. All tests passing
2. No performance regression
3. Reduced code complexity
4. Improved documentation coverage
5. Successful end-to-end validation
6. Clean separation of concerns
7. Easier onboarding for new contributors
