#!/usr/bin/env python3
"""
Analyze project dependencies to create a proper test execution order.
"""

import os
import re
from collections import defaultdict
import json

def find_imports(file_path):
    """Find all project imports in a file."""
    imports = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Find from-imports
                if line.startswith('from '):
                    match = re.match(r'from\s+(src\.[^\s]+)\s+import', line)
                    if match:
                        imports.append(match.group(1))
                # Find direct imports
                elif line.startswith('import '):
                    match = re.match(r'import\s+(src\.[^\s\n]+)', line)
                    if match:
                        imports.append(match.group(1))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return imports

def get_module_path(file_path, root_dir):
    """Convert file path to module path."""
    rel_path = os.path.relpath(file_path, root_dir)
    if rel_path.startswith('..'):
        return None
    module_path = os.path.splitext(rel_path)[0].replace('/', '.')
    # Only include src and tests modules
    if not (module_path.startswith('src.') or module_path.startswith('tests.')):
        return None
    return module_path

def analyze_dependencies(root_dir):
    """Analyze dependencies in the project."""
    dependency_graph = defaultdict(set)
    reverse_deps = defaultdict(set)
    modules = set()
    
    # Walk through all Python files in src and tests
    for directory in ['src', 'tests']:
        dir_path = os.path.join(root_dir, directory)
        if not os.path.exists(dir_path):
            continue
            
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.py'):
                    full_path = os.path.join(root, file)
                    module_path = get_module_path(full_path, root_dir)
                    
                    if module_path:
                        modules.add(module_path)
                        imports = find_imports(full_path)
                        
                        if imports:
                            print(f"Found imports in {module_path}:")
                            for imp in imports:
                                print(f"  {imp}")
                                dependency_graph[module_path].add(imp)
                                reverse_deps[imp].add(module_path)
    
    return dependency_graph, reverse_deps, modules

def calculate_levels(dependency_graph, reverse_deps, modules):
    """Calculate dependency levels for modules."""
    levels = {}
    visited = set()
    
    def get_level(module):
        if module in visited:
            return levels.get(module, 0)
        
        visited.add(module)
        if module not in dependency_graph or not dependency_graph[module]:
            levels[module] = 0
            return 0
        
        max_dep_level = 0
        for dep in dependency_graph[module]:
            if dep in modules:  # Only consider project modules
                dep_level = get_level(dep)
                max_dep_level = max(max_dep_level, dep_level + 1)
        
        levels[module] = max_dep_level
        return max_dep_level
    
    # Calculate levels for all modules
    for module in modules:
        if module not in levels:
            get_level(module)
    
    return levels

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Analyzing project dependencies...")
    dep_graph, reverse_deps, modules = analyze_dependencies(root_dir)
    
    print("\nCalculating dependency levels...")
    levels = calculate_levels(dep_graph, reverse_deps, modules)
    
    # Group modules by level
    level_groups = defaultdict(list)
    for module, level in levels.items():
        level_groups[level].append(module)
    
    # Output results
    print("\nDependency Levels:")
    print("=================")
    
    for level in sorted(level_groups.keys()):
        print(f"\nLevel {level}:")
        modules_at_level = sorted(level_groups[level])
        
        # Group by type (src vs tests)
        src_modules = [m for m in modules_at_level if m.startswith('src.')]
        test_modules = [m for m in modules_at_level if m.startswith('tests.')]
        
        if src_modules:
            print("\nSource modules:")
            for module in src_modules:
                print(f"  {module}")
                if module in dep_graph and dep_graph[module]:
                    print("    Depends on:", ", ".join(sorted(dep_graph[module])))
        
        if test_modules:
            print("\nTest modules:")
            for module in test_modules:
                print(f"  {module}")
                if module in dep_graph and dep_graph[module]:
                    print("    Depends on:", ", ".join(sorted(dep_graph[module])))
    
    # Save results to file
    output = {
        'dependency_levels': {str(k): sorted(v) for k, v in level_groups.items()},
        'dependency_graph': {k: sorted(v) for k, v in dep_graph.items()},
        'reverse_dependencies': {k: sorted(v) for k, v in reverse_deps.items()}
    }
    
    with open('dependency_analysis.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nAnalysis saved to dependency_analysis.json")

if __name__ == '__main__':
    main()
