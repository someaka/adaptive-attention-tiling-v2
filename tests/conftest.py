"""Test configuration and shared fixtures."""

import pytest
import os
from typing import Dict, Set, List
import re
import torch
from src.core.flow.neural import NeuralGeometricFlow
import yaml

def get_test_dependencies(item, items) -> List[str]:
    """Determine test dependencies based on file location and content."""
    path = str(item.fspath)
    name = item.name.lower()
    
    # Core dependencies - these are always required
    deps = set()
    
    # Add dependencies based on test content
    with open(path, 'r') as f:
        content = f.read()
        
        # Find all test functions that this test might depend on
        test_refs = re.findall(r'test_\w+', content)
        for ref in test_refs:
            if ref != item.name:  # Don't add self as dependency
                deps.add(ref)
                
        # Find all test classes that this test might depend on
        class_refs = re.findall(r'Test\w+', content)
        for ref in class_refs:
            if item.cls and ref != item.cls.__name__:  # Don't add self as dependency
                deps.add(ref)
    
    # Add dependencies based on directory structure
    if "tests/core/attention/" in path:
        deps.add("test_minkowski_inner_product")  # Base geometric operation
        
    if "tests/core/tiling/" in path:
        deps.add("test_tile_dimensions")  # Base tiling operation
        
    if "test_geometric" in path or "geometric" in name:
        deps.add("test_minkowski_inner_product")
        
    if "test_quantum" in path or "quantum" in name:
        deps.add("test_geometric_distance")
        
    if "test_pattern" in path or "pattern" in name:
        deps.add("test_quantum_state")
        
    if "test_validation" in path:
        deps.add("test_basic_tensor_shapes")
        
    # Remove any non-existent dependencies
    valid_deps = set()
    for dep in deps:
        for test_item in items:
            if dep in str(test_item.nodeid):
                valid_deps.add(dep)
                break
                
    return list(valid_deps)

def pytest_collection_modifyitems(session, config, items):
    """Automatically handle test dependencies."""
    # Build dependency graph
    dependency_graph = {}
    
    # First pass: Collect all test names
    test_names = {
        item.nodeid: item for item in items
    }
    
    # Second pass: Build dependency graph
    for item in items:
        deps = get_test_dependencies(item, items)
        if deps:
            dependency_graph[item.nodeid] = [
                dep_id for dep_id in test_names
                if any(dep in dep_id for dep in deps)
            ]
    
    # Third pass: Add dependency markers
    for item in items:
        if item.nodeid in dependency_graph:
            item.add_marker(
                pytest.mark.dependency(
                    depends=dependency_graph[item.nodeid]
                )
            )
        else:
            item.add_marker(pytest.mark.dependency())
            
    # Fourth pass: Order tests based on dependencies
    ordered_items = []
    visited = set()
    
    def visit(item):
        if item.nodeid in visited:
            return
        visited.add(item.nodeid)
        for dep_id in dependency_graph.get(item.nodeid, []):
            if dep_id not in visited and dep_id in test_names:
                visit(test_names[dep_id])
        ordered_items.append(item)
    
    for item in items:
        visit(item)
        
    items[:] = ordered_items

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test."""
    yield
    # Add any cleanup code here if needed

@pytest.fixture
def manifold_dim():
    """Manifold dimension for tests."""
    return 4  # Increased from 2 to match the test requirements

@pytest.fixture
def hidden_dim(manifold_dim):
    """Hidden dimension for tests."""
    return manifold_dim * 2  # Double the manifold dimension for proper scaling

@pytest.fixture
def flow(manifold_dim, hidden_dim, test_config):
    """Create flow system fixture."""
    return NeuralGeometricFlow(
        manifold_dim=manifold_dim,
        hidden_dim=hidden_dim,
        dt=0.1,
        stability_threshold=1e-6,
        fisher_rao_weight=1.0,
        quantum_weight=1.0,
        num_heads=8,
        dropout=0.1,
        test_config=test_config
    )

@pytest.fixture
def points(batch_size, manifold_dim):
    """Create random points in position space."""
    return torch.randn(batch_size, manifold_dim, requires_grad=True)

@pytest.fixture
def batch_size():
    """Batch size for tests."""
    return 10  # Increased from 4 to test with larger batches

@pytest.fixture
def test_config():
    """Load test configuration based on environment."""
    config_name = os.environ.get("TEST_REGIME", "debug")
    config_path = f"configs/test_regimens/{config_name}.yaml"
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config