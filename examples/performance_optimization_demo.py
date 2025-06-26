#!/usr/bin/env python3
"""Performance Optimization Demo - Portfolio Showcase

This script demonstrates the advanced performance optimization capabilities
implemented in the AI docs vector database hybrid scraper system.

Features demonstrated:
- Multi-tier adaptive caching with ML optimization
- Intelligent async concurrency management
- Comprehensive performance benchmarking
- Statistical analysis of performance improvements
- Real-time optimization and monitoring

Run with: uv run python examples/performance_optimization_demo.py
"""

import asyncio
import logging
import time
from pathlib import Path

# Setup logging for demo
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_performance_demo():
    """Run the complete performance optimization demonstration."""
    
    print("🚀 AI Docs Vector DB - Performance Optimization Showcase")
    print("=" * 80)
    print()
    
    try:
        # Import performance modules
        from src.services.performance import run_performance_optimization_showcase
        from src.services.cache.adaptive_cache import EnhancedCacheManager
        from src.services.performance.async_optimization import initialize_async_optimizer, TaskPriority
        
        print("📋 Demo Components:")
        print("   ✓ Multi-tier Adaptive Caching")
        print("   ✓ ML-driven Cache Optimization")
        print("   ✓ Intelligent Async Patterns")
        print("   ✓ Adaptive Concurrency Management")
        print("   ✓ Comprehensive Benchmarking")
        print("   ✓ Statistical Performance Analysis")
        print()
        
        # Set up output directory
        output_dir = Path("demo_results") / f"performance_showcase_{int(time.time())}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Results will be saved to: {output_dir}")
        print()
        
        # Run the complete showcase
        print("🎯 Starting Performance Optimization Showcase...")
        print("   This may take 2-3 minutes to complete comprehensive analysis")
        print()
        
        showcase_start = time.time()
        
        # Execute the showcase
        results = await run_performance_optimization_showcase(output_dir)
        
        showcase_duration = time.time() - showcase_start
        
        print()
        print("✅ Performance Optimization Showcase Completed!")
        print("=" * 80)
        print()
        
        # Display key results
        print("📊 KEY PERFORMANCE IMPROVEMENTS:")
        print()
        
        # Calculate overall improvement metrics
        all_improvements = []
        for category, metrics in results.performance_improvements.items():
            print(f"   {category.upper().replace('_', ' ')}:")
            for metric, data in metrics.items():
                improvement = data['improvement_percent']
                if improvement > 0:
                    all_improvements.append(improvement)
                    direction = "🔽" if metric.endswith(('latency_ms', 'error_rate', 'cpu_percent', 'memory_mb')) else "🔼"
                    print(f"   {direction} {metric}: {improvement:.1f}% improvement")
                    print(f"      {data['baseline_value']:.3f} → {data['optimized_value']:.3f}")
            print()
        
        if all_improvements:
            avg_improvement = sum(all_improvements) / len(all_improvements)
            max_improvement = max(all_improvements)
            print(f"📈 OVERALL PERFORMANCE GAINS:")
            print(f"   • Average Improvement: {avg_improvement:.1f}%")
            print(f"   • Maximum Improvement: {max_improvement:.1f}%")
            print(f"   • Metrics Improved: {len(all_improvements)}")
            print()
        
        # Display portfolio scores
        print("🏆 PORTFOLIO SHOWCASE SCORES:")
        print(f"   • Technical Complexity: {results.technical_complexity_score:.2f}/1.0")
        print(f"   • Business Impact: {results.business_impact_score:.2f}/1.0") 
        print(f"   • Innovation Score: {results.innovation_score:.2f}/1.0")
        print()
        
        # Display business impact
        print("💼 BUSINESS IMPACT ESTIMATES:")
        cost_savings = results.cost_savings_estimate
        scalability = results.scalability_improvements
        print(f"   • Infrastructure Cost Reduction: {cost_savings.get('infrastructure_cost_reduction_percent', 0):.1f}%")
        print(f"   • Operational Efficiency Gain: {cost_savings.get('operational_efficiency_gain_percent', 0):.1f}%")
        print(f"   • Capacity Increase: {scalability.get('capacity_increase_percent', 0):.1f}%")
        print(f"   • Est. Annual Savings: ${cost_savings.get('estimated_annual_savings_usd', 0):,}")
        print()
        
        # Display optimization components
        print("⚙️  OPTIMIZATION COMPONENTS APPLIED:")
        for i, optimization in enumerate(results.optimizations_applied, 1):
            print(f"   {i}. {optimization}")
        print()
        
        # Display recommendations
        print("🎯 OPTIMIZATION RECOMMENDATIONS:")
        for i, recommendation in enumerate(results.optimization_recommendations, 1):
            print(f"   {i}. {recommendation}")
        print()
        
        # Display file outputs
        print("📁 GENERATED OUTPUTS:")
        print(f"   • Complete Results: {output_dir}/performance_optimization_results.json")
        print(f"   • Summary Metrics: {output_dir}/performance_summary.json")
        print(f"   • Portfolio Documentation: {output_dir}/PERFORMANCE_OPTIMIZATION_SHOWCASE.md")
        print()
        
        print(f"⏱️  Total Demo Duration: {showcase_duration:.1f} seconds")
        print()
        print("🎉 Demo completed successfully!")
        print("   Check the generated documentation for detailed technical analysis.")
        
        return results
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("   Make sure you're running from the project root with:")
        print("   uv run python examples/performance_optimization_demo.py")
        return None
        
    except Exception as e:
        print(f"❌ Demo Error: {e}")
        logger.exception("Demo failed with exception")
        return None


async def run_quick_demo():
    """Run a quick demonstration of key components."""
    
    print("⚡ Quick Performance Demo - Key Components")
    print("=" * 60)
    print()
    
    try:
        from src.services.performance.async_optimization import (
            initialize_async_optimizer, 
            TaskPriority,
            execute_optimized
        )
        from src.services.cache.adaptive_cache import EnhancedCacheManager
        
        # Initialize components
        print("🔧 Initializing optimization components...")
        
        # Setup async optimizer
        async_optimizer = await initialize_async_optimizer(
            initial_concurrency=5,
            enable_adaptive_limiting=True,
            enable_intelligent_scheduling=True
        )
        
        # Setup enhanced cache
        cache_manager = EnhancedCacheManager(
            enable_local_cache=True,
            enable_distributed_cache=False,  # Skip Redis for quick demo
            enable_adaptive_optimization=True
        )
        
        print("✅ Components initialized")
        print()
        
        # Demonstrate async optimization
        print("⚡ Testing Adaptive Async Optimization...")
        
        async def sample_workload(intensity: float = 0.01):
            """Sample workload with variable intensity."""
            await asyncio.sleep(intensity)
            return f"Work completed in {intensity}s"
        
        # Execute tasks with different priorities
        start_time = time.time()
        
        high_priority_tasks = [
            execute_optimized(
                sample_workload(0.001), 
                priority=TaskPriority.HIGH
            ) for _ in range(10)
        ]
        
        normal_priority_tasks = [
            execute_optimized(
                sample_workload(0.01),
                priority=TaskPriority.NORMAL
            ) for _ in range(20)
        ]
        
        # Execute all tasks
        results = await asyncio.gather(
            *high_priority_tasks,
            *normal_priority_tasks
        )
        
        execution_time = time.time() - start_time
        
        print(f"   ✓ Executed {len(results)} tasks in {execution_time:.2f}s")
        
        # Get performance metrics
        performance_report = async_optimizer.get_performance_report()
        concurrency_metrics = performance_report.get('concurrency_metrics', {})
        
        print(f"   ✓ Current concurrency limit: {concurrency_metrics.get('current_limit', 'N/A')}")
        print(f"   ✓ Tasks per second: {len(results) / execution_time:.1f}")
        print()
        
        # Demonstrate cache optimization
        print("🧠 Testing Adaptive Cache Optimization...")
        
        cache_start = time.time()
        
        # Simulate cache access patterns
        for i in range(50):
            key = f"demo_key_{i % 10}"  # Create hot keys
            
            # Try to get from cache
            result = await cache_manager.get(key)
            
            if result is None:
                # Cache miss - store new data
                data = {"demo": True, "iteration": i, "timestamp": time.time()}
                await cache_manager.set(key, data)
            
            await asyncio.sleep(0.001)  # Small delay
        
        cache_time = time.time() - cache_start
        
        # Get cache statistics
        cache_stats = await cache_manager.get_stats()
        
        print(f"   ✓ Processed 50 cache operations in {cache_time:.2f}s")
        print(f"   ✓ Cache layers active: {cache_stats.get('manager', {}).get('enabled_layers', [])}")
        
        if 'local' in cache_stats:
            local_stats = cache_stats['local']
            print(f"   ✓ Local cache size: {local_stats.get('size', 0)} items")
            print(f"   ✓ Memory usage: {local_stats.get('memory_usage', 0):.1f} MB")
        
        print()
        
        # Cleanup
        await cache_manager.close()
        
        print("✅ Quick demo completed successfully!")
        print("   Run the full showcase for comprehensive analysis:")
        print("   uv run python examples/performance_optimization_demo.py --full")
        
    except Exception as e:
        print(f"❌ Quick demo error: {e}")
        logger.exception("Quick demo failed")


async def main():
    """Main demo function."""
    import sys
    
    if "--quick" in sys.argv:
        await run_quick_demo()
    else:
        await run_performance_demo()


if __name__ == "__main__":
    asyncio.run(main())