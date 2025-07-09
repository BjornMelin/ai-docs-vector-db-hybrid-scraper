#!/usr/bin/env python3
"""Verify all POA imports are working correctly."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    # Test imports from POA modules
    print("Testing imports...")
    
    # API routes
    from src.api.routes.optimization import router
    print("✓ API routes imported successfully")
    
    # Performance modules
    from src.services.performance.poa_service import PerformanceOptimizationAgent
    print("✓ POA service imported successfully")
    
    from src.services.performance.api_optimizer import APIResponseOptimizer
    print("✓ API optimizer imported successfully")
    
    from src.services.performance.async_optimizer import AsyncOptimizer
    print("✓ Async optimizer imported successfully")
    
    from src.services.performance.benchmarks import BenchmarkRunner
    print("✓ Benchmarks imported successfully")
    
    from src.services.performance.database_optimizer import DatabaseOptimizer
    print("✓ Database optimizer imported successfully")
    
    from src.services.performance.memory_optimizer import MemoryOptimizer
    print("✓ Memory optimizer imported successfully")
    
    from src.services.performance.performance_optimizer import PerformanceOptimizer
    print("✓ Performance optimizer imported successfully")
    
    print("\n✅ All imports successful! No import errors found.")
    
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    sys.exit(1)