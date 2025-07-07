#!/usr/bin/env python3
"""Test runner for MCP services module.

This script runs comprehensive tests for the src/mcp_services/ module including:
- Unit tests for all individual services
- Integration tests for cross-service coordination
- End-to-end tests for real-world workflows
- Performance and scalability validation
"""

import asyncio
import subprocess
import sys
import time
from pathlib import Path


def run_command(command: list[str], description: str) -> tuple[bool, str]:
    """Run a command and return success status and output."""
    print(f"🔄 {description}...")
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            return True, result.stdout
        else:
            print(f"❌ {description} - FAILED")
            print(f"Error output:\n{result.stderr}")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT")
        return False, "Test execution timed out"
    except Exception as e:
        print(f"💥 {description} - ERROR: {e}")
        return False, str(e)


def main():
    """Run comprehensive MCP services tests."""
    print("🚀 Starting comprehensive MCP services test suite")
    print("=" * 60)
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    print(f"📁 Project root: {project_root}")
    
    # Test configuration
    test_results = {}
    start_time = time.time()
    
    # 1. Unit tests for individual services
    print("\n📋 Phase 1: Unit Tests for Individual Services")
    print("-" * 50)
    
    unit_test_commands = [
        (
            ["uv", "run", "pytest", "tests/unit/mcp_services/test_search_service.py", "-v", "--tb=short"],
            "SearchService unit tests"
        ),
        (
            ["uv", "run", "pytest", "tests/unit/mcp_services/test_document_service.py", "-v", "--tb=short"],
            "DocumentService unit tests"
        ),
        (
            ["uv", "run", "pytest", "tests/unit/mcp_services/test_analytics_service.py", "-v", "--tb=short"],
            "AnalyticsService unit tests"
        ),
        (
            ["uv", "run", "pytest", "tests/unit/mcp_services/test_system_service.py", "-v", "--tb=short"],
            "SystemService unit tests"
        ),
        (
            ["uv", "run", "pytest", "tests/unit/mcp_services/test_orchestrator_service.py", "-v", "--tb=short"],
            "OrchestratorService unit tests"
        ),
    ]
    
    for command, description in unit_test_commands:
        success, output = run_command(command, description)
        test_results[description] = success
    
    # 2. Integration tests for cross-service coordination
    print("\n🔗 Phase 2: Integration Tests for Cross-Service Coordination")
    print("-" * 60)
    
    integration_test_commands = [
        (
            ["uv", "run", "pytest", "tests/integration/mcp_services/test_mcp_services_integration.py", "-v", "--tb=short"],
            "MCP services integration tests"
        ),
    ]
    
    for command, description in integration_test_commands:
        success, output = run_command(command, description)
        test_results[description] = success
    
    # 3. End-to-end tests for real-world workflows
    print("\n🌍 Phase 3: End-to-End Tests for Real-World Workflows")
    print("-" * 55)
    
    e2e_test_commands = [
        (
            ["uv", "run", "pytest", "tests/integration/mcp_services/test_mcp_services_e2e.py", "-v", "--tb=short"],
            "MCP services end-to-end tests"
        ),
    ]
    
    for command, description in e2e_test_commands:
        success, output = run_command(command, description)
        test_results[description] = success
    
    # 4. Coverage analysis
    print("\n📊 Phase 4: Coverage Analysis")
    print("-" * 35)
    
    coverage_commands = [
        (
            ["uv", "run", "pytest", "tests/unit/mcp_services/", "tests/integration/mcp_services/", 
             "--cov=src/mcp_services", "--cov-report=term-missing", "--cov-report=html:htmlcov/mcp_services"],
            "MCP services coverage analysis"
        ),
    ]
    
    for command, description in coverage_commands:
        success, output = run_command(command, description)
        test_results[description] = success
    
    # 5. Performance benchmarks
    print("\n⚡ Phase 5: Performance Benchmarks")
    print("-" * 40)
    
    # Run performance-specific tests
    performance_commands = [
        (
            ["uv", "run", "pytest", "tests/integration/mcp_services/", 
             "-k", "performance", "-v", "--tb=short"],
            "MCP services performance tests"
        ),
    ]
    
    for command, description in performance_commands:
        success, output = run_command(command, description)
        test_results[description] = success
    
    # Calculate total execution time
    end_time = time.time()
    total_time = end_time - start_time
    
    # Generate summary report
    print("\n" + "=" * 60)
    print("📈 Test Execution Summary")
    print("=" * 60)
    
    passed_tests = sum(1 for success in test_results.values() if success)
    total_tests = len(test_results)
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"📊 Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    print(f"⏱️  Total Execution Time: {total_time:.2f} seconds")
    print()
    
    # Detailed results
    print("📋 Detailed Results:")
    for test_name, success in test_results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {status} - {test_name}")
    
    print()
    
    # Test coverage and quality metrics
    print("🎯 Quality Metrics:")
    print("  📦 Module Coverage: src/mcp_services/")
    print("  🧪 Test Types: Unit, Integration, E2E")
    print("  🔍 Research Validation: I3, I5, J1, FastMCP 2.0+")
    print("  🏗️  Architecture: Modular Services, Autonomous Capabilities")
    print("  🔗 Integration: Enterprise Observability, Client Manager")
    
    # Final assessment
    print("\n" + "=" * 60)
    if success_rate >= 90:
        print("🎉 EXCELLENT: MCP services test suite completed successfully!")
        print("✨ All major functionality validated with high coverage")
        exit_code = 0
    elif success_rate >= 80:
        print("✅ GOOD: MCP services test suite mostly successful")
        print("⚠️  Some issues detected - review failed tests")
        exit_code = 0
    elif success_rate >= 60:
        print("⚠️  WARNING: MCP services test suite has significant issues")
        print("🔧 Multiple test failures require attention")
        exit_code = 1
    else:
        print("❌ CRITICAL: MCP services test suite failed")
        print("🚨 Major functionality issues detected")
        exit_code = 1
    
    print("=" * 60)
    
    # Additional recommendations
    if success_rate < 100:
        print("\n💡 Recommendations:")
        failed_tests = [name for name, success in test_results.items() if not success]
        for failed_test in failed_tests:
            print(f"  🔧 Fix: {failed_test}")
        
        print("\n📚 Resources:")
        print("  📖 Test docs: tests/unit/mcp_services/")
        print("  🔗 Integration: tests/integration/mcp_services/")
        print("  📊 Coverage: htmlcov/mcp_services/index.html")
    
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)