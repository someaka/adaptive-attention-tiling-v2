"""Test configuration and shared fixtures."""

import pytest

def pytest_configure(config):
    """Configure custom markers."""
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

def pytest_collection_modifyitems(items):
    """Add markers based on test location and dependencies."""
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

        # Add dependencies based on test names
        if "geometric" in item.name:
            item.add_marker(pytest.mark.geometric)
            
        if "attention" in item.name:
            item.add_marker(pytest.mark.attention)
            
        if "memory" in item.name:
            item.add_marker(pytest.mark.memory)
            
        if "performance" in item.name:
            item.add_marker(pytest.mark.performance)

@pytest.fixture(autouse=True)
def run_order():
    """Define test run order based on dependencies."""
    # Tests should run in this order:
    # 1. Core geometric operations
    # 2. Core attention mechanisms
    # 3. Tiling operations
    # 4. Integration tests
    # 5. Performance tests
    # 6. Memory tests
    pass 