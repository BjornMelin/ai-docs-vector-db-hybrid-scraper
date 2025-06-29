#!/usr/bin/env python3
"""
Architecture Validation Script for Portfolio ULTRATHINK Transformation

Provides automated validation of core architectural improvements including:
- Circular dependency analysis
- ClientManager size verification  
- Service coupling assessment
- Import complexity measurement
"""

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
import subprocess
import re

def analyze_circular_dependencies() -> Tuple[int, List[str]]:
    """Analyze circular dependencies using import analysis."""
    try:
        # Use pydeps if available, otherwise manual analysis
        result = subprocess.run(['pydeps', 'src', '--show-deps'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            # Parse pydeps output for cycles
            cycles = re.findall(r'Cycle detected: (.+)', result.stdout)
            return len(cycles), cycles
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # Manual import analysis fallback
    src_path = Path('src')
    if not src_path.exists():
        return 0, []
    
    imports = {}
    for py_file in src_path.rglob('*.py'):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            
            module_name = str(py_file.relative_to(src_path)).replace('/', '.').replace('.py', '')
            imports[module_name] = []
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        if node.module.startswith('src.'):
                            imports[module_name].append(node.module)
        except Exception:
            continue
    
    # Simple cycle detection
    cycles = []
    for module, deps in imports.items():
        for dep in deps:
            if dep in imports and module in imports[dep]:
                cycle = f"{module} ↔ {dep}"
                if cycle not in cycles:
                    cycles.append(cycle)
    
    return len(cycles), cycles

def analyze_client_manager_size() -> Dict[str, int]:
    """Analyze ClientManager size and complexity."""
    client_manager_path = Path('src/infrastructure/client_manager.py')
    
    if not client_manager_path.exists():
        # Check old location
        client_manager_path = Path('src/config/client_manager.py')
    
    if not client_manager_path.exists():
        return {"lines": 0, "methods": 0, "services_managed": 0}
    
    try:
        with open(client_manager_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = len([line for line in content.split('\n') if line.strip()])
        
        tree = ast.parse(content)
        methods = 0
        services_managed = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and 'ClientManager' in node.name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods += 1
                    elif isinstance(item, ast.Assign):
                        # Count service assignments
                        for target in item.targets:
                            if isinstance(target, ast.Attribute) and '_service' in str(target.attr):
                                services_managed += 1
        
        return {"lines": lines, "methods": methods, "services_managed": services_managed}
    
    except Exception as e:
        return {"lines": 0, "methods": 0, "services_managed": 0, "error": str(e)}

def analyze_service_coupling() -> Dict[str, any]:
    """Analyze service coupling and dependency patterns."""
    src_path = Path('src')
    if not src_path.exists():
        return {"coupling_score": 0, "services_analyzed": 0}
    
    service_files = list(src_path.rglob('**/services/**/*.py'))
    if not service_files:
        return {"coupling_score": 0, "services_analyzed": 0}
    
    total_imports = 0
    internal_imports = 0
    
    for service_file in service_files:
        try:
            with open(service_file, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    total_imports += 1
                    if isinstance(node, ast.ImportFrom) and node.module:
                        if 'src.' in node.module or node.module.startswith('.'):
                            internal_imports += 1
        except Exception:
            continue
    
    coupling_score = (internal_imports / total_imports * 100) if total_imports > 0 else 0
    
    return {
        "coupling_score": round(coupling_score, 2),
        "services_analyzed": len(service_files),
        "total_imports": total_imports,
        "internal_imports": internal_imports
    }

def analyze_import_complexity() -> Dict[str, int]:
    """Analyze import complexity and patterns."""
    src_path = Path('src')
    if not src_path.exists():
        return {"total_files": 0, "complex_imports": 0, "max_imports_per_file": 0}
    
    total_files = 0
    complex_imports = 0
    max_imports = 0
    
    for py_file in src_path.rglob('*.py'):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            
            total_files += 1
            file_imports = 0
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    file_imports += 1
            
            max_imports = max(max_imports, file_imports)
            if file_imports > 20:  # Threshold for complex imports
                complex_imports += 1
                
        except Exception:
            continue
    
    return {
        "total_files": total_files,
        "complex_imports": complex_imports,
        "max_imports_per_file": max_imports,
        "complexity_percentage": round((complex_imports / total_files * 100) if total_files > 0 else 0, 2)
    }

def validate_group_a() -> Dict[str, any]:
    """Validate Group A completion criteria."""
    print("🔍 Analyzing Group A Architecture Metrics...")
    
    # Circular dependencies
    cycle_count, cycles = analyze_circular_dependencies()
    print(f"   Circular Dependencies: {cycle_count}")
    
    # ClientManager analysis
    cm_stats = analyze_client_manager_size()
    print(f"   ClientManager Size: {cm_stats['lines']} lines")
    
    # Service coupling
    coupling = analyze_service_coupling()
    print(f"   Service Coupling: {coupling['coupling_score']}%")
    
    # Import complexity
    imports = analyze_import_complexity()
    print(f"   Import Complexity: {imports['complexity_percentage']}% files complex")
    
    return {
        "circular_dependencies": {
            "count": cycle_count,
            "cycles": cycles,
            "target": 0,
            "status": "✅ PASS" if cycle_count == 0 else "❌ FAIL"
        },
        "client_manager": {
            "lines": cm_stats['lines'],
            "target": 300,
            "status": "✅ PASS" if cm_stats['lines'] <= 300 else "❌ FAIL"
        },
        "service_coupling": {
            "score": coupling['coupling_score'],
            "target": "< 50%",
            "status": "✅ PASS" if coupling['coupling_score'] < 50 else "❌ FAIL"
        },
        "import_complexity": {
            "percentage": imports['complexity_percentage'],
            "target": "< 20%",
            "status": "✅ PASS" if imports['complexity_percentage'] < 20 else "❌ FAIL"
        }
    }

def main():
    parser = argparse.ArgumentParser(description='Validate Portfolio ULTRATHINK architecture')
    parser.add_argument('--group', choices=['A', 'B'], default='A', 
                       help='Validation group (A=architecture, B=quality)')
    parser.add_argument('--json', action='store_true', 
                       help='Output results in JSON format')
    
    args = parser.parse_args()
    
    print("🎯 Portfolio ULTRATHINK Architecture Validation")
    print("=" * 50)
    
    if args.group == 'A':
        results = validate_group_a()
        
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print("\n📊 Group A Validation Results:")
            print("-" * 30)
            
            for metric, data in results.items():
                print(f"{metric.replace('_', ' ').title()}: {data['status']}")
                if 'lines' in data:
                    print(f"  Current: {data['lines']} lines (Target: ≤{data['target']})")
                elif 'count' in data:
                    print(f"  Current: {data['count']} cycles (Target: {data['target']})")
                elif 'score' in data:
                    print(f"  Current: {data['score']}% (Target: {data['target']})")
                elif 'percentage' in data:
                    print(f"  Current: {data['percentage']}% (Target: {data['target']})")
            
            # Overall assessment
            passed = sum(1 for data in results.values() if "✅" in data['status'])
            total = len(results)
            
            print(f"\n🏆 Overall Group A Status: {passed}/{total} metrics passed")
            
            if passed == total:
                print("✅ Group A architecture goals ACHIEVED!")
                print("Ready to proceed to Group B validation.")
            else:
                print("❌ Group A requires additional work before proceeding.")
                
        return 0 if passed == total else 1
    
    else:
        print("Group B validation not yet implemented.")
        return 1

if __name__ == "__main__":
    sys.exit(main())