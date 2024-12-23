"""Test configuration and shared fixtures."""

import pytest
import os
from typing import Dict, Set

def get_test_level(item, items) -> int:
    """Determine test level based on dependencies and location."""
    path = str(item.fspath)
    name = item.name.lower()

    # Level 0: Core functionality tests
    if any(pattern in path for pattern in [
        "tests/core/",
        "tests/unit/",
        "/test_helpers.py",
        "/test_basic",
        "_basic_test"
    ]) or any(pattern in name for pattern in [
        "basic",
        "helper",
        "init",
        "setup"
    ]):
        return 0

    # Level 1.0: Basic geometric operations
    if any(pattern in path for pattern in [
        "/test_geometric.py",
        "geometric/test_",
        "_geometric_",
        "/test_metric",
        "/test_tensor"
    ]) or any(pattern in name for pattern in [
        "metric",
        "tensor",
        "distance",
        "projection"
    ]):
        return 10

    # Level 1.1: Quantum operations
    if any(pattern in path for pattern in [
        "/test_quantum",
        "quantum/test_",
        "_quantum_",
        "/test_state"
    ]) or any(pattern in name for pattern in [
        "quantum",
        "state",
        "measurement",
        "entangle"
    ]):
        return 11

    # Level 1.2: Pattern operations
    if any(pattern in path for pattern in [
        "/test_pattern",
        "pattern/test_",
        "_pattern_",
        "/test_flow"
    ]) or any(pattern in name for pattern in [
        "pattern",
        "flow",
        "diffusion",
        "reaction"
    ]):
        return 12

    # Level 1.3: Neural network operations
    if any(pattern in path for pattern in [
        "/test_neural",
        "neural/test_",
        "_neural_",
        "/test_attention"
    ]) or any(pattern in name for pattern in [
        "neural",
        "attention",
        "network",
        "layer"
    ]):
        return 13

    # Level 1.4: Validation operations
    if any(pattern in path for pattern in [
        "/test_validation",
        "validation/test_",
        "_validation_",
        "/test_verify"
    ]) or any(pattern in name for pattern in [
        "validation",
        "verify",
        "check",
        "assert"
    ]):
        return 14

    # Level 2: Integration and complex tests
    if any(pattern in path for pattern in [
        "tests/test_integration/",
        "tests/performance/",
        "/test_end_to_end",
        "_integration",
        "_advanced"
    ]) or any(pattern in name for pattern in [
        "integration",
        "end_to_end",
        "performance",
        "advanced"
    ]):
        return 20

    # Default to level 1.0 if no clear pattern match
    return 10

def get_dependencies(item, items) -> Set[str]:
    """Get dependencies for a test based on its level and module."""
    level = get_test_level(item, items)
    if level == 0:
        return set()
        
    # Get all tests from previous level in the same module or its dependencies
    module_path = os.path.dirname(str(item.fspath))
    deps = set()
    
    for other in items:
        other_level = get_test_level(other, items)
        other_path = os.path.dirname(str(other.fspath))
        
        # Add dependency if:
        # 1. Test is from previous level or same major level but lower minor level
        # 2. Test is in same module or a core module
        if ((other_level < level // 10 * 10) or  # Previous major level
            (other_level // 10 == level // 10 and other_level < level)) and (  # Same major level, lower minor
            other_path == module_path or
            "tests/core/" in str(other.fspath) or
            "tests/unit/" in str(other.fspath)
        ):
            if other.cls:
                deps.add(f"{other.cls.__name__}::{other.name}")
            else:
                deps.add(other.name)
                
    return deps

def pytest_configure(config):
    """Configure custom markers."""
    # Category markers
    config.addinivalue_line(
        "markers",
        "core: mark test as part of core functionality"
    )
    config.addinivalue_line(
        "markers",
        "geometric: mark test as part of geometric operations"
    )
    config.addinivalue_line(
        "markers",
        "attention: mark test as part of attention mechanism"
    )
    config.addinivalue_line(
        "markers",
        "tiling: mark test as part of tiling operations"
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers",
        "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers",
        "memory: mark test as memory test"
    )

    # Level markers
    config.addinivalue_line(
        "markers",
        "level0: mark test as level 0 (fundamental)"
    )
    config.addinivalue_line(
        "markers",
        "level10: mark test as level 1.0 (basic geometric operations)"
    )
    config.addinivalue_line(
        "markers",
        "level11: mark test as level 1.1 (quantum operations)"
    )
    config.addinivalue_line(
        "markers",
        "level12: mark test as level 1.2 (pattern operations)"
    )
    config.addinivalue_line(
        "markers",
        "level13: mark test as level 1.3 (neural network operations)"
    )
    config.addinivalue_line(
        "markers",
        "level14: mark test as level 1.4 (validation operations)"
    )
    config.addinivalue_line(
        "markers",
        "level20: mark test as level 2.0 (integration and complex tests)"
    )

def pytest_collection_modifyitems(items):
    """Add markers based on test location and dependencies."""
    # First pass: Add basic markers and determine levels
    for item in items:
        # Add markers based on directory structure
        if "tests/core/" in str(item.fspath):
            item.add_marker(pytest.mark.core)
            
        if "tests/core/attention/" in str(item.fspath):
            item.add_marker(pytest.mark.geometric)
            item.add_marker(pytest.mark.attention)
            
        if "tests/core/tiling/" in str(item.fspath):
            item.add_marker(pytest.mark.tiling)
            
        if "tests/test_integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
            
        if "tests/performance/" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            
        if "tests/test_memory/" in str(item.fspath):
            item.add_marker(pytest.mark.memory)

        # Add markers based on test names
        if "geometric" in item.name:
            item.add_marker(pytest.mark.geometric)
            
        if "attention" in item.name:
            item.add_marker(pytest.mark.attention)
            
        if "memory" in item.name:
            item.add_marker(pytest.mark.memory)
            
        if "performance" in item.name:
            item.add_marker(pytest.mark.performance)

        # Determine and add level marker
        level = get_test_level(item, items)
        item.add_marker(getattr(pytest.mark, f"level{level}"))

    # Second pass: Add dependencies based on levels
    for item in items:
        # Get test name with class if it's a method
        if item.cls:
            test_name = f"{item.cls.__name__}::{item.name}"
        else:
            test_name = item.name
            
        # Add dependency marker with automatically determined dependencies
        deps = get_dependencies(item, items)
        if deps:
            item.add_marker(pytest.mark.dependency(name=test_name, depends=list(deps)))
        else:
            item.add_marker(pytest.mark.dependency(name=test_name))

@pytest.fixture(autouse=True)
def run_order():
    """Define test run order based on dependencies."""
    # Tests will run in order based on their level markers and dependencies
    pass 