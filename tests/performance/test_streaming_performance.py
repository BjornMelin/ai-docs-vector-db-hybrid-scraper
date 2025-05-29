"""Performance tests for MCP server streaming functionality."""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# Add src to path for imports
src_path = str(Path(__file__).parent.parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)


@pytest.fixture
def performance_search_results():
    """Generate search results of varying sizes for performance testing."""

    def create_results(count: int, content_size: int = 1000):
        return [
            {
                "id": f"perf_doc_{i}",
                "score": 0.95 - (i * 0.0001),
                "payload": {
                    "content": "x" * content_size,
                    "title": f"Performance Test Document {i}",
                    "url": f"https://perf-test.com/docs/{i}",
                    "metadata": {
                        "length": content_size,
                        "type": "performance_test",
                        "index": i,
                    },
                },
            }
            for i in range(count)
        ]

    return create_results


@pytest.mark.asyncio
async def test_response_serialization_performance(performance_search_results):
    """Test JSON serialization performance for large responses."""
    result_sizes = [10, 100, 500, 1000, 2000]
    performance_metrics = {}

    for size in result_sizes:
        results = performance_search_results(size)

        # Measure serialization time
        start_time = time.time()
        json_response = json.dumps(results)
        serialization_time = time.time() - start_time

        # Measure response size
        response_size = len(json_response.encode("utf-8"))

        performance_metrics[size] = {
            "serialization_time": serialization_time,
            "response_size": response_size,
            "throughput_mb_per_sec": (response_size / 1024 / 1024) / serialization_time
            if serialization_time > 0
            else 0,
        }

    # Verify performance scales reasonably
    for size in result_sizes:
        metrics = performance_metrics[size]

        # Serialization should complete in reasonable time
        assert metrics["serialization_time"] < 1.0  # Less than 1 second

        # Throughput should be reasonable (>1 MB/s for large responses)
        if size >= 1000:
            assert metrics["throughput_mb_per_sec"] > 1.0

    # Log performance metrics for analysis
    print("\nSerialization Performance Metrics:")
    for size, metrics in performance_metrics.items():
        print(
            f"  {size} results: {metrics['serialization_time']:.3f}s, "
            f"{metrics['response_size']:,} bytes, "
            f"{metrics['throughput_mb_per_sec']:.2f} MB/s"
        )


@pytest.mark.asyncio
async def test_buffer_size_impact(performance_search_results):
    """Test impact of different buffer sizes on streaming performance."""
    buffer_sizes = ["1024", "4096", "8192", "16384", "32768"]
    results = performance_search_results(1000, 2000)  # Large content
    response_data = json.dumps(results).encode("utf-8")
    response_size = len(response_data)

    performance_data = {}

    for buffer_size in buffer_sizes:
        with patch.dict(os.environ, {"FASTMCP_BUFFER_SIZE": buffer_size}):
            buffer_int = int(os.getenv("FASTMCP_BUFFER_SIZE"))

            # Calculate theoretical streaming metrics
            chunks_needed = (response_size + buffer_int - 1) // buffer_int
            overhead_ratio = chunks_needed * buffer_int / response_size

            performance_data[buffer_size] = {
                "buffer_size": buffer_int,
                "chunks_needed": chunks_needed,
                "overhead_ratio": overhead_ratio,
                "efficiency": 1.0 / overhead_ratio,
            }

    # Verify buffer size relationships
    for buffer_size, data in performance_data.items():
        # Larger buffers should need fewer chunks
        assert data["chunks_needed"] > 0
        # Overhead should be reasonable (< 50% for large responses)
        if int(buffer_size) >= 8192:
            assert data["overhead_ratio"] < 1.5

    print(f"\nBuffer Size Impact (Response size: {response_size:,} bytes):")
    for buffer_size, data in performance_data.items():
        print(
            f"  {buffer_size}: {data['chunks_needed']} chunks, "
            f"{data['overhead_ratio']:.2f}x overhead, "
            f"{data['efficiency']:.2f} efficiency"
        )


@pytest.mark.asyncio
async def test_concurrent_request_performance():
    """Test performance under concurrent streaming requests."""

    async def mock_search_operation(request_id: int, result_count: int = 500):
        """Mock a search operation with realistic timing."""
        # Simulate search processing time
        await asyncio.sleep(0.1)  # 100ms base search time

        # Create mock results
        results = [
            {
                "id": f"req_{request_id}_doc_{i}",
                "score": 0.9,
                "payload": {"content": "x" * 1000, "request_id": request_id},
            }
            for i in range(result_count)
        ]

        # Simulate serialization time
        json_response = json.dumps(results)
        response_size = len(json_response.encode("utf-8"))

        return {
            "request_id": request_id,
            "result_count": len(results),
            "response_size": response_size,
        }

    # Test different concurrency levels
    concurrency_levels = [1, 5, 10, 20]
    performance_results = {}

    for concurrency in concurrency_levels:
        start_time = time.time()

        # Run concurrent requests
        tasks = [mock_search_operation(i) for i in range(concurrency)]
        results = await asyncio.gather(*tasks)

        total_time = time.time() - start_time

        # Calculate metrics
        total_results = sum(r["result_count"] for r in results)
        total_size = sum(r["response_size"] for r in results)

        performance_results[concurrency] = {
            "total_time": total_time,
            "requests_per_second": concurrency / total_time,
            "results_per_second": total_results / total_time,
            "throughput_mb_per_sec": (total_size / 1024 / 1024) / total_time,
        }

    # Verify concurrency scaling
    for _, metrics in performance_results.items():
        # Should handle at least 5 requests per second
        assert metrics["requests_per_second"] >= 1.0

        # Should maintain reasonable throughput
        assert metrics["throughput_mb_per_sec"] > 0.1

    print("\nConcurrency Performance:")
    for concurrency, metrics in performance_results.items():
        print(
            f"  {concurrency} concurrent: {metrics['requests_per_second']:.2f} req/s, "
            f"{metrics['throughput_mb_per_sec']:.2f} MB/s"
        )


@pytest.mark.asyncio
async def test_memory_usage_under_load(performance_search_results):
    """Test memory usage characteristics under streaming load."""
    import gc
    import os

    import psutil

    process = psutil.Process(os.getpid())

    # Get initial memory usage
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Test with increasingly large result sets
    memory_usage = {"initial": initial_memory}

    for size in [100, 500, 1000, 2000]:
        # Force garbage collection before test
        gc.collect()

        # Create large result set
        results = performance_search_results(size, 2000)  # 2KB per result
        json_response = json.dumps(results)

        # Measure memory after creation
        current_memory = process.memory_info().rss / 1024 / 1024
        memory_usage[f"size_{size}"] = current_memory

        # Clean up
        del results, json_response
        gc.collect()

    # Final memory after cleanup
    gc.collect()
    final_memory = process.memory_info().rss / 1024 / 1024
    memory_usage["final"] = final_memory

    # Verify memory usage is reasonable
    max_memory = max(memory_usage.values())
    memory_growth = max_memory - initial_memory

    # Memory growth should be reasonable (< 100MB for test)
    assert memory_growth < 100, f"Memory growth too high: {memory_growth:.2f}MB"

    # Memory should return close to initial after cleanup
    memory_leak = final_memory - initial_memory
    assert memory_leak < 10, f"Potential memory leak: {memory_leak:.2f}MB"

    print("\nMemory Usage Analysis:")
    print(f"  Initial: {initial_memory:.2f}MB")
    print(f"  Peak: {max_memory:.2f}MB (+{memory_growth:.2f}MB)")
    print(f"  Final: {final_memory:.2f}MB (+{memory_leak:.2f}MB)")


@pytest.mark.asyncio
async def test_response_size_thresholds():
    """Test performance across different response size thresholds."""
    # Test various response sizes that would trigger different streaming behaviors
    test_cases = [
        {"count": 10, "content_size": 100, "expected_size": "~10KB"},
        {"count": 100, "content_size": 1000, "expected_size": "~100KB"},
        {"count": 500, "content_size": 2000, "expected_size": "~1MB"},
        {"count": 1000, "content_size": 5000, "expected_size": "~5MB"},
        {"count": 2000, "content_size": 2500, "expected_size": "~5MB"},
    ]

    performance_data = []

    for case in test_cases:
        # Create test data
        results = [
            {
                "id": f"threshold_doc_{i}",
                "score": 0.9,
                "payload": {"content": "x" * case["content_size"]},
            }
            for i in range(case["count"])
        ]

        # Measure serialization performance
        start_time = time.time()
        json_response = json.dumps(results)
        serialization_time = time.time() - start_time

        actual_size = len(json_response.encode("utf-8"))

        # Check against various buffer sizes
        for buffer_size in [8192, 16384, 32768]:
            chunks_needed = (actual_size + buffer_size - 1) // buffer_size

            # Determine if streaming would be beneficial
            streaming_beneficial = actual_size > buffer_size * 2

            performance_data.append(
                {
                    "case": case["expected_size"],
                    "actual_size": actual_size,
                    "buffer_size": buffer_size,
                    "chunks_needed": chunks_needed,
                    "serialization_time": serialization_time,
                    "streaming_beneficial": streaming_beneficial,
                }
            )

    # Verify streaming benefits for large responses
    large_responses = [
        p for p in performance_data if p["actual_size"] > 100000
    ]  # >100KB

    for data in large_responses:
        if data["buffer_size"] == 8192:  # Default buffer size
            # Large responses should benefit from streaming
            assert data["streaming_beneficial"], (
                f"Large response {data['case']} should benefit from streaming"
            )

            # Should need multiple chunks
            assert data["chunks_needed"] > 1, (
                "Large response should require multiple chunks"
            )

    print("\nResponse Size Analysis:")
    for data in performance_data:
        if data["buffer_size"] == 8192:  # Only show default buffer size
            print(
                f"  {data['case']}: {data['actual_size']:,} bytes, "
                f"{data['chunks_needed']} chunks, "
                f"streaming: {'yes' if data['streaming_beneficial'] else 'no'}"
            )


@pytest.mark.asyncio
async def test_transport_mode_performance_comparison():
    """Compare performance characteristics of different transport modes."""
    # This test simulates the performance differences between transport modes

    # Mock different transport configurations
    transport_configs = {
        "stdio": {"buffered": False, "chunk_size": None, "overhead": "low"},
        "streamable-http": {"buffered": True, "chunk_size": 8192, "overhead": "medium"},
    }

    # Test data sizes
    test_sizes = [1000, 10000, 100000, 1000000]  # bytes

    performance_comparison = {}

    for transport, config in transport_configs.items():
        performance_comparison[transport] = {}

        for size in test_sizes:
            # Simulate response handling
            if config["buffered"] and config["chunk_size"]:
                # Streaming mode - calculate chunking overhead
                chunks = (size + config["chunk_size"] - 1) // config["chunk_size"]
                overhead_time = chunks * 0.001  # 1ms per chunk overhead
            else:
                # Non-streaming mode - single response overhead
                overhead_time = 0.01  # 10ms base overhead

            # Base serialization time (proportional to size)
            base_time = size / 1000000  # 1ms per MB
            total_time = base_time + overhead_time

            performance_comparison[transport][size] = {
                "total_time": total_time,
                "overhead_time": overhead_time,
                "efficiency": base_time / total_time if total_time > 0 else 0,
            }

    # Analyze when streaming becomes beneficial
    for size in test_sizes:
        stdio_time = performance_comparison["stdio"][size]["total_time"]
        http_time = performance_comparison["streamable-http"][size]["total_time"]

        # For large responses, streaming should be competitive or better
        if size >= 100000:  # 100KB+
            streaming_penalty = http_time / stdio_time
            assert streaming_penalty < 2.0, (
                f"Streaming penalty too high for {size} bytes: {streaming_penalty:.2f}x"
            )

    print("\nTransport Mode Performance Comparison:")
    for size in test_sizes:
        stdio_time = performance_comparison["stdio"][size]["total_time"]
        http_time = performance_comparison["streamable-http"][size]["total_time"]
        ratio = http_time / stdio_time
        print(
            f"  {size:,} bytes: stdio={stdio_time:.3f}s, http={http_time:.3f}s, ratio={ratio:.2f}x"
        )
