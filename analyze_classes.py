import os
import ast
import sys
import logging
from datetime import datetime
from typing import Dict, List, Set, Optional
from collections import defaultdict

class ClassAnalyzer:
    def __init__(self, log_file: str = "class_analysis.log"):
        self.classes = defaultdict(dict)
        self.inheritance = defaultdict(list)
        self.imports = defaultdict(set)
        
        # Setup logging
        self.logger = logging.getLogger('ClassAnalyzer')
        self.logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def get_name_from_node(self, node: ast.expr) -> Optional[str]:
        """Extract name from an AST node safely."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self.get_name_from_node(node.value)}.{node.attr}"
        return None
        
    def visit_file(self, filepath: str) -> None:
        self.logger.debug(f"Analyzing file: {filepath}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                try:
                    tree = ast.parse(f.read(), filename=filepath)
                except SyntaxError as e:
                    self.logger.error(f"Syntax error in {filepath}: {e}")
                    return
                except Exception as e:
                    self.logger.error(f"Error parsing {filepath}: {e}")
                    return
                
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    module = os.path.relpath(filepath)
                    self.logger.debug(f"Found class: {node.name} in {module}")
                    
                    # Safely extract base class names
                    bases = []
                    for base in node.bases:
                        base_name = self.get_name_from_node(base)
                        if base_name:
                            bases.append(base_name)
                    
                    self.classes[module][node.name] = {
                        'bases': bases,
                        'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                        'is_protocol': any('Protocol' in base_name for base_name in bases)
                    }
                    
                    # Update inheritance
                    for base_name in bases:
                        self.inheritance[node.name].append(base_name)
                        
        except Exception as e:
            self.logger.error(f"Error processing {filepath}: {e}")

    def analyze_directory(self, directory: str) -> None:
        self.logger.info(f"Starting analysis of directory: {directory}")
        if not os.path.exists(directory):
            self.logger.error(f"Directory not found: {directory}")
            return
            
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    self.visit_file(filepath)
    
    def print_analysis(self) -> None:
        if not self.classes:
            self.logger.warning("No classes found in analysis!")
            return
            
        self.logger.info("\n=== Class Analysis Report ===\n")
        self.logger.info(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Print summary statistics
        total_classes = sum(len(classes) for classes in self.classes.values())
        total_protocols = sum(1 for classes in self.classes.values() 
                            for info in classes.values() if info['is_protocol'])
        
        self.logger.info(f"Total Classes: {total_classes}")
        self.logger.info(f"Total Protocols: {total_protocols}")
        self.logger.info(f"Total Implementation Classes: {total_classes - total_protocols}\n")
        
        # Print Protocols
        self.logger.info("Protocols:")
        for module, classes in self.classes.items():
            for class_name, info in classes.items():
                if info['is_protocol']:
                    self.logger.info(f"  {class_name} ({module})")
                    if info['bases']:
                        self.logger.info(f"    Bases: {', '.join(info['bases'])}")
                    self.logger.info(f"    Methods: {', '.join(info['methods'])}\n")
        
        # Print Implementation Classes
        self.logger.info("\nImplementation Classes:")
        for module, classes in self.classes.items():
            for class_name, info in classes.items():
                if not info['is_protocol']:
                    self.logger.info(f"  {class_name} ({module})")
                    if info['bases']:
                        self.logger.info(f"    Bases: {', '.join(info['bases'])}")
                    self.logger.info(f"    Methods: {', '.join(info['methods'])}\n")
        
        # Print Inheritance Hierarchy
        self.logger.info("\nInheritance Hierarchy:")
        visited = set()
        def print_hierarchy(class_name: str, level: int = 0):
            if class_name in visited:
                return
            visited.add(class_name)
            self.logger.info("  " * level + f"- {class_name}")
            for child, bases in self.inheritance.items():
                if class_name in bases:
                    print_hierarchy(child, level + 1)
        
        roots = {base for bases in self.inheritance.values() for base in bases} - set(self.inheritance.keys())
        for root in roots:
            print_hierarchy(root)

if __name__ == '__main__':
    analyzer = ClassAnalyzer()
    analyzer.logger.info(f"Current working directory: {os.getcwd()}")
    analyzer.analyze_directory('src')
    analyzer.print_analysis() 