#!/usr/bin/env python3
"""Script to check for circular dependencies in the src directory."""

import ast
import os
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Set


def extract_imports(file_path: Path) -> List[str]:
    """Extract import statements from a Python file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        tree = ast.parse(content)
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("src."):
                        imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("src."):
                    imports.append(node.module)
        
        return imports
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []


def build_dependency_graph(src_dir: Path) -> Dict[str, Set[str]]:
    """Build a dependency graph from Python files in src directory."""
    graph = defaultdict(set)
    
    # Find all Python files
    for py_file in src_dir.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
            
        # Convert file path to module name
        relative_path = py_file.relative_to(src_dir.parent)
        module_name = str(relative_path.with_suffix("")).replace("/", ".")
        
        # Extract imports
        imports = extract_imports(py_file)
        for imp in imports:
            graph[module_name].add(imp)
    
    return graph


def find_cycles(graph: Dict[str, Set[str]]) -> List[List[str]]:
    """Find all cycles in the dependency graph using DFS."""
    cycles = []
    visited = set()
    rec_stack = set()
    path = []
    
    def dfs(node: str) -> bool:
        if node in rec_stack:
            # Found a cycle
            cycle_start = path.index(node)
            cycle = path[cycle_start:] + [node]
            cycles.append(cycle)
            return True
        
        if node in visited:
            return False
        
        visited.add(node)
        rec_stack.add(node)
        path.append(node)
        
        for neighbor in graph.get(node, set()):
            if dfs(neighbor):
                return True
        
        rec_stack.remove(node)
        path.pop()
        return False
    
    for node in graph:
        if node not in visited:
            dfs(node)
    
    return cycles


def main():
    """Main function to check circular dependencies."""
    src_dir = Path("src")
    if not src_dir.exists():
        print("src directory not found")
        sys.exit(1)
    
    print("Building dependency graph...")
    graph = build_dependency_graph(src_dir)
    
    print("Checking for circular dependencies...")
    cycles = find_cycles(graph)
    
    if cycles:
        print(f"\n🚨 Found {len(cycles)} circular dependencies:")
        for i, cycle in enumerate(cycles, 1):
            print(f"\nCycle {i}:")
            for j, module in enumerate(cycle):
                if j == len(cycle) - 1:
                    print(f"  {module} -> {cycle[0]}")
                else:
                    print(f"  {module} -> {cycle[j + 1]}")
    else:
        print("\n✅ No circular dependencies found!")
    
    # Show problematic modules
    problem_modules = set()
    for cycle in cycles:
        problem_modules.update(cycle)
    
    if problem_modules:
        print(f"\n📝 Modules involved in circular dependencies ({len(problem_modules)}):")
        for module in sorted(problem_modules):
            print(f"  - {module}")


if __name__ == "__main__":
    main()