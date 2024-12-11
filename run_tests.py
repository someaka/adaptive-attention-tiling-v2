#!/usr/bin/env python3
"""
Advanced test runner script that executes tests in dependency order with granular control.
"""

import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
import re

def collect_test_info():
    """Collect information about available tests."""
    root_dir = Path(__file__).parent
    
    # Run pytest --collect-only to get all test items
    result = subprocess.run([
        "pytest",
        "--collect-only",
        "-q",
        "tests/"
    ], cwd=root_dir, capture_output=True, text=True)
    
    # Parse output to count tests per module
    test_counts = defaultdict(int)
    for line in result.stdout.split('\n'):
        if 'tests/' in line:
            module = line.split('tests/')[1].split('.py')[0]
            test_counts[module] += 1
    
    return test_counts

def print_test_summary(test_counts):
    """Print summary of available tests."""
    print("\nTest Suite Summary")
    print("="*80)
    
    categories = {
        'core': "Core Components",
        'neural': "Neural Network Components",
        'validation': "Validation Tests",
        'integration': "Integration Tests",
        'performance': "Performance Tests",
        'infrastructure': "Infrastructure Tests"
    }
    
    category_counts = defaultdict(int)
    for module, count in test_counts.items():
        for cat in categories:
            if cat in module:
                category_counts[cat] += count
                break
        else:
            category_counts['other'] += count
    
    total_tests = sum(test_counts.values())
    print(f"Total number of tests: {total_tests}\n")
    
    for cat, desc in categories.items():
        if category_counts[cat] > 0:
            print(f"{desc}: {category_counts[cat]} tests")
    
    if category_counts['other'] > 0:
        print(f"Other Tests: {category_counts['other']} tests")

def run_tests_by_level(args):
    """Run tests in dependency order."""
    root_dir = Path(__file__).parent
    
    # Define test levels and their descriptions
    levels = [
        (0, "Base Components", ["core.common", "core.interfaces", "core.utils", "core.metrics.base"]),
        (1, "Core Implementations", ["core.backends", "core.metrics", "core.patterns.base", "core.quantum.base"]),
        (2, "Higher Components", ["core.attention", "core.crystal", "core.patterns", "core.quantum", "core.tiling.base"]),
        (3, "Neural Components", ["neural.flow", "core.tiling"]),
        (4, "Integration Tests", ["test_integration"])
    ]
    
    # Filter levels based on arguments
    if args.level is not None:
        levels = [l for l in levels if l[0] == args.level]
    
    failed_levels = []
    test_counts = collect_test_info()
    print_test_summary(test_counts)
    
    for level, description, modules in levels:
        print(f"\n{'='*80}")
        print(f"Running Level {level} Tests: {description}")
        print(f"Modules: {', '.join(modules)}")
        print('='*80)
        
        # Build pytest command
        cmd = ["pytest", "-v"]
        
        # Add parallel execution if requested
        if args.parallel:
            cmd.extend(["-n", "auto"])
        
        # Add coverage if requested
        if args.coverage:
            cmd.extend(["--cov=src", "--cov-report=term-missing"])
        
        # Add level marker
        cmd.extend(["-m", f"level{level}"])
        
        # Add test selection
        if args.module:
            cmd.extend([f"tests/**/*{args.module}*.py"])
        else:
            cmd.append("tests/")
        
        # Run pytest
        result = subprocess.run(cmd, cwd=root_dir)
        
        if result.returncode != 0:
            failed_levels.append(level)
            print(f"\nLevel {level} tests failed!")
            
            if not args.continue_on_fail and level < 4:
                response = input("\nTests failed. Continue to next level? [y/N]: ")
                if response.lower() != 'y':
                    break
    
    # Print summary
    print("\n" + "="*80)
    print("Test Execution Summary")
    print("="*80)
    
    if not failed_levels:
        print("All test levels passed successfully!")
    else:
        print("Failed levels:", failed_levels)
        print("\nFailed test levels need to be fixed before proceeding with dependent tests.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run tests in dependency order")
    parser.add_argument("--level", type=int, choices=[0,1,2,3,4],
                      help="Run tests for a specific level only")
    parser.add_argument("--module", type=str,
                      help="Run tests for a specific module (e.g., 'geometric' or 'quantum')")
    parser.add_argument("--parallel", action="store_true",
                      help="Run tests in parallel using pytest-xdist")
    parser.add_argument("--coverage", action="store_true",
                      help="Generate coverage report")
    parser.add_argument("--continue-on-fail", action="store_true",
                      help="Continue running tests even if a level fails")
    
    args = parser.parse_args()
    run_tests_by_level(args)

if __name__ == "__main__":
    main()
