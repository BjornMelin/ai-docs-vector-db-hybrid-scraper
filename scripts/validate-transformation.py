#!/usr/bin/env python3
"""
Comprehensive Portfolio ULTRATHINK Transformation Validation

Complete validation of all transformation objectives including:
- Architecture metrics (Group A)
- Code quality metrics (Group B)  
- Performance benchmarks
- Security posture
- Test coverage
- Overall transformation success
"""

import argparse
import ast
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import re

def run_command(cmd: List[str], timeout: int = 30) -> tuple[int, str, str]:
    """Run a command safely with timeout."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", f"Command timed out after {timeout}s"
    except FileNotFoundError:
        return 1, "", f"Command not found: {' '.join(cmd)}"

def validate_architecture() -> Dict[str, Any]:
    """Validate core architecture metrics from Group A."""
    print("🏗️  Validating Architecture Foundation...")
    
    # Run the architecture validation script
    returncode, stdout, stderr = run_command(['python', 'scripts/validate-architecture.py', '--group=A', '--json'])
    
    if returncode == 0 and stdout:
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            pass
    
    # Fallback basic validation
    return {
        "circular_dependencies": {"count": 0, "status": "⚠️  UNKNOWN"},
        "client_manager": {"lines": 0, "status": "⚠️  UNKNOWN"},
        "service_coupling": {"score": 0, "status": "⚠️  UNKNOWN"},
        "import_complexity": {"percentage": 0, "status": "⚠️  UNKNOWN"}
    }

def validate_test_coverage() -> Dict[str, Any]:
    """Validate test coverage metrics."""
    print("🧪 Validating Test Coverage...")
    
    # Try to run pytest with coverage
    returncode, stdout, stderr = run_command(['uv', 'run', 'pytest', '--cov=src', '--cov-report=json'])
    
    coverage_data = {"percentage": 0, "status": "❌ FAIL", "error": None}
    
    if returncode == 0:
        # Try to read coverage.json
        coverage_file = Path('coverage.json')
        if coverage_file.exists():
            try:
                with open(coverage_file, 'r') as f:
                    cov_data = json.load(f)
                    percentage = cov_data.get('totals', {}).get('percent_covered', 0)
                    coverage_data = {
                        "percentage": round(percentage, 2),
                        "status": "✅ PASS" if percentage >= 80 else "❌ FAIL",
                        "target": 80
                    }
            except Exception as e:
                coverage_data["error"] = str(e)
    else:
        coverage_data["error"] = stderr
    
    return {"test_coverage": coverage_data}

def validate_function_complexity() -> Dict[str, Any]:
    """Validate function complexity metrics."""
    print("📐 Validating Function Complexity...")
    
    src_path = Path('src')
    if not src_path.exists():
        return {"function_complexity": {"average": 0, "status": "❌ FAIL", "error": "src directory not found"}}
    
    complexities = []
    high_complexity_functions = []
    
    for py_file in src_path.rglob('*.py'):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Simple complexity estimation based on control flow nodes
                    complexity = 1  # Base complexity
                    for child in ast.walk(node):
                        if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler,
                                            ast.With, ast.AsyncWith, ast.Try)):
                            complexity += 1
                        elif isinstance(child, (ast.BoolOp, ast.Compare)):
                            complexity += 1
                    
                    complexities.append(complexity)
                    if complexity > 10:
                        high_complexity_functions.append({
                            "file": str(py_file),
                            "function": node.name,
                            "complexity": complexity
                        })
        
        except Exception:
            continue
    
    if not complexities:
        return {"function_complexity": {"average": 0, "status": "❌ FAIL", "error": "No functions analyzed"}}
    
    average_complexity = sum(complexities) / len(complexities)
    max_complexity = max(complexities)
    
    return {
        "function_complexity": {
            "average": round(average_complexity, 2),
            "max": max_complexity,
            "total_functions": len(complexities),
            "high_complexity_count": len(high_complexity_functions),
            "high_complexity_functions": high_complexity_functions[:5],  # Top 5
            "target": 10,
            "status": "✅ PASS" if average_complexity < 10 else "❌ FAIL"
        }
    }

def validate_security() -> Dict[str, Any]:
    """Validate security posture."""
    print("🔒 Validating Security Posture...")
    
    security_score = 0
    issues = []
    
    # Check for common security issues
    src_path = Path('src')
    if src_path.exists():
        for py_file in src_path.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for hardcoded secrets
                if re.search(r'(api_key|password|secret)\s*=\s*[\'"][^\'"]+[\'"]', content, re.IGNORECASE):
                    issues.append(f"Potential hardcoded secret in {py_file}")
                
                # Check for CORS wildcard
                if 'allow_origins=["*"]' in content:
                    issues.append(f"CORS wildcard found in {py_file}")
                
            except Exception:
                continue
    
    # Check Docker files
    docker_files = list(Path('.').glob('**/Dockerfile*')) + list(Path('.').glob('**/docker-compose*.yml'))
    for docker_file in docker_files:
        try:
            with open(docker_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if ':latest' in content:
                issues.append(f"Unpinned Docker image in {docker_file}")
                
        except Exception:
            continue
    
    # Calculate security score
    max_issues = 10  # Expected maximum issues
    security_score = max(0, (max_issues - len(issues)) / max_issues * 10)
    
    return {
        "security": {
            "score": round(security_score, 1),
            "issues_found": len(issues),
            "issues": issues[:5],  # Top 5 issues
            "target": 9.5,
            "status": "✅ PASS" if security_score >= 9.5 else "❌ FAIL"
        }
    }

def validate_performance() -> Dict[str, Any]:
    """Validate performance metrics."""
    print("⚡ Validating Performance Metrics...")
    
    # Check for parallel processing patterns
    src_path = Path('src')
    parallel_patterns = 0
    async_patterns = 0
    
    if src_path.exists():
        for py_file in src_path.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for parallel processing patterns
                if 'asyncio.gather' in content:
                    parallel_patterns += 1
                if 'async def' in content:
                    async_patterns += 1
                    
            except Exception:
                continue
    
    # Check for caching implementations
    cache_implementations = 0
    for py_file in src_path.rglob('**/cache/**/*.py'):
        if py_file.exists():
            cache_implementations += 1
    
    return {
        "performance": {
            "parallel_patterns": parallel_patterns,
            "async_patterns": async_patterns,
            "cache_implementations": cache_implementations,
            "status": "✅ PASS" if parallel_patterns > 0 and cache_implementations > 0 else "❌ FAIL"
        }
    }

def validate_code_quality() -> Dict[str, Any]:
    """Validate overall code quality."""
    print("✨ Validating Code Quality...")
    
    # Run ruff check
    returncode, stdout, stderr = run_command(['uv', 'run', 'ruff', 'check', 'src'])
    ruff_issues = len(stdout.split('\n')) if stdout else 0
    
    # Run ruff format check
    returncode2, stdout2, stderr2 = run_command(['uv', 'run', 'ruff', 'format', '--check', 'src'])
    format_issues = returncode2 != 0
    
    return {
        "code_quality": {
            "ruff_issues": ruff_issues,
            "format_issues": format_issues,
            "status": "✅ PASS" if ruff_issues == 0 and not format_issues else "❌ FAIL"
        }
    }

def run_comprehensive_validation() -> Dict[str, Any]:
    """Run complete transformation validation."""
    print("🎯 Portfolio ULTRATHINK Comprehensive Transformation Validation")
    print("=" * 60)
    
    results = {}
    
    # Architecture validation
    arch_results = validate_architecture()
    results.update(arch_results)
    
    # Test coverage
    test_results = validate_test_coverage()
    results.update(test_results)
    
    # Function complexity
    complexity_results = validate_function_complexity()
    results.update(complexity_results)
    
    # Security validation
    security_results = validate_security()
    results.update(security_results)
    
    # Performance validation
    performance_results = validate_performance()
    results.update(performance_results)
    
    # Code quality
    quality_results = validate_code_quality()
    results.update(quality_results)
    
    return results

def calculate_overall_score(results: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate overall transformation success score."""
    passed_metrics = 0
    total_metrics = 0
    
    critical_failures = []
    
    for metric_name, metric_data in results.items():
        if isinstance(metric_data, dict) and 'status' in metric_data:
            total_metrics += 1
            if '✅' in metric_data['status']:
                passed_metrics += 1
            elif metric_name in ['circular_dependencies', 'client_manager', 'test_coverage']:
                critical_failures.append(metric_name)
    
    success_rate = (passed_metrics / total_metrics * 100) if total_metrics > 0 else 0
    
    overall_status = "✅ TRANSFORMATION COMPLETE"
    if critical_failures:
        overall_status = "❌ CRITICAL FAILURES"
    elif success_rate < 80:
        overall_status = "⚠️  NEEDS IMPROVEMENT"
    elif success_rate < 95:
        overall_status = "🔄 NEARLY COMPLETE"
    
    return {
        "overall_assessment": {
            "success_rate": round(success_rate, 1),
            "passed_metrics": passed_metrics,
            "total_metrics": total_metrics,
            "critical_failures": critical_failures,
            "status": overall_status
        }
    }

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Portfolio ULTRATHINK transformation validation')
    parser.add_argument('--complete', action='store_true', 
                       help='Run complete validation suite')
    parser.add_argument('--json', action='store_true', 
                       help='Output results in JSON format')
    
    args = parser.parse_args()
    
    if not args.complete:
        print("Use --complete flag for comprehensive validation")
        return 1
    
    start_time = time.time()
    results = run_comprehensive_validation()
    overall = calculate_overall_score(results)
    results.update(overall)
    
    execution_time = time.time() - start_time
    results["validation_metadata"] = {
        "execution_time": round(execution_time, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print("\n" + "=" * 60)
        print("📊 TRANSFORMATION VALIDATION SUMMARY")
        print("=" * 60)
        
        # Print key metrics
        for metric_name, metric_data in results.items():
            if isinstance(metric_data, dict) and 'status' in metric_data:
                print(f"{metric_name.replace('_', ' ').title()}: {metric_data['status']}")
        
        print("\n" + "-" * 60)
        overall_data = results["overall_assessment"]
        print(f"🏆 OVERALL STATUS: {overall_data['status']}")
        print(f"📈 Success Rate: {overall_data['success_rate']}%")
        print(f"✅ Passed: {overall_data['passed_metrics']}/{overall_data['total_metrics']} metrics")
        
        if overall_data['critical_failures']:
            print(f"⚠️  Critical Failures: {', '.join(overall_data['critical_failures'])}")
        
        print(f"⏱️  Validation completed in {results['validation_metadata']['execution_time']}s")
    
    # Return exit code based on overall success
    return 0 if overall_data['success_rate'] >= 95 else 1

if __name__ == "__main__":
    sys.exit(main())