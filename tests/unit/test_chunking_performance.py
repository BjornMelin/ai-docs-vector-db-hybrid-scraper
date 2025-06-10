"""Performance tests for chunking system.

Tests to validate performance targets from issue #74:
- Chunking: >1000 documents/second
- AST parsing: <100ms per source file
"""

import time

import pytest
from src.chunking import EnhancedChunker
from src.config.enums import ChunkingStrategy
from src.config.models import ChunkingConfig


class TestChunkingPerformance:
    """Test chunking performance targets."""

    @pytest.fixture
    def sample_documents(self):
        """Generate sample documents for performance testing."""
        docs = []

        # Small documents (< 1KB)
        for i in range(100):
            docs.append(
                {
                    "content": f"# Document {i}\n\nThis is a test document with some content.\n"
                    * 10,
                    "title": f"Doc {i}",
                    "url": f"https://example.com/doc{i}.md",
                }
            )

        # Medium documents (1-5KB)
        for i in range(50):
            docs.append(
                {
                    "content": f"# Large Document {i}\n\n" + ("Content line.\n" * 100),
                    "title": f"Large Doc {i}",
                    "url": f"https://example.com/large{i}.md",
                }
            )

        # Code documents
        for i in range(50):
            docs.append(
                {
                    "content": f"""
def function_{i}():
    '''Function {i} documentation.'''
    x = {i}
    y = x * 2
    return y

class Class_{i}:
    '''Class {i} documentation.'''

    def __init__(self):
        self.value = {i}

    def method_{i}(self):
        return self.value * 2
""",
                    "title": f"Code {i}",
                    "url": f"https://example.com/code{i}.py",
                }
            )

        return docs

    def test_chunking_throughput_basic_strategy(self, sample_documents):
        """Test chunking throughput with basic strategy (target: >1000 docs/sec)."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.BASIC, chunk_size=800, chunk_overlap=100
        )
        chunker = EnhancedChunker(config)

        # Warm up
        chunker.chunk_content(sample_documents[0]["content"])

        start_time = time.time()
        processed_count = 0

        for doc in sample_documents:
            chunks = chunker.chunk_content(
                doc["content"], title=doc["title"], url=doc["url"]
            )
            processed_count += 1
            assert len(chunks) > 0  # Ensure documents are actually processed

        end_time = time.time()
        elapsed = end_time - start_time
        throughput = processed_count / elapsed

        print("\nBasic chunking performance:")
        print(f"Processed {processed_count} documents in {elapsed:.2f}s")
        print(f"Throughput: {throughput:.0f} documents/second")

        # Performance target: >1000 documents/second
        assert throughput > 1000, (
            f"Chunking throughput {throughput:.0f} docs/sec is below target of 1000 docs/sec"
        )

    def test_chunking_throughput_enhanced_strategy(self, sample_documents):
        """Test chunking throughput with enhanced strategy."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.ENHANCED,
            chunk_size=800,
            chunk_overlap=100,
            preserve_code_blocks=True,
        )
        chunker = EnhancedChunker(config)

        # Warm up
        chunker.chunk_content(sample_documents[0]["content"])

        start_time = time.time()
        processed_count = 0

        for doc in sample_documents:
            chunks = chunker.chunk_content(
                doc["content"], title=doc["title"], url=doc["url"]
            )
            processed_count += 1
            assert len(chunks) > 0

        end_time = time.time()
        elapsed = end_time - start_time
        throughput = processed_count / elapsed

        print("\nEnhanced chunking performance:")
        print(f"Processed {processed_count} documents in {elapsed:.2f}s")
        print(f"Throughput: {throughput:.0f} documents/second")

        # Enhanced strategy may be slower but should still meet target
        assert throughput > 500, (
            f"Enhanced chunking throughput {throughput:.0f} docs/sec is too slow"
        )

    def test_ast_parsing_performance(self):
        """Test AST parsing performance (target: <100ms per source file)."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.AST,
            enable_ast_chunking=True,
            chunk_size=1600,
            max_function_chunk_size=3200,
        )
        chunker = EnhancedChunker(config)

        # Create realistic source files of varying sizes
        source_files = []

        # Small file (~1KB)
        small_file = """
def hello_world():
    '''A simple hello world function.'''
    print("Hello, World!")
    return "Hello"

class Greeter:
    '''A simple greeter class.'''

    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"Hello, {self.name}!"
"""
        source_files.append(("small.py", small_file))

        # Medium file (~5KB)
        medium_file = (
            small_file * 10
            + """
def complex_function(data):
    '''A more complex function with loops and conditions.'''
    result = []
    for item in data:
        if isinstance(item, str):
            result.append(item.upper())
        elif isinstance(item, int):
            if item > 0:
                result.append(item * 2)
            else:
                result.append(0)
        else:
            result.append(str(item))
    return result

class DataProcessor:
    '''A data processing class.'''

    def __init__(self):
        self.processors = {}

    def register_processor(self, name, func):
        self.processors[name] = func

    def process(self, name, data):
        if name in self.processors:
            return self.processors[name](data)
        raise ValueError(f"Unknown processor: {name}")
"""
        )
        source_files.append(("medium.py", medium_file))

        # Large file (~20KB)
        large_file = (
            medium_file * 4
            + """
class AdvancedProcessor(DataProcessor):
    '''An advanced data processor with many methods.'''

    def __init__(self):
        super().__init__()
        self.cache = {}

    def cached_process(self, name, data):
        key = (name, str(data))
        if key in self.cache:
            return self.cache[key]

        result = self.process(name, data)
        self.cache[key] = result
        return result

    def batch_process(self, operations):
        results = []
        for name, data in operations:
            try:
                result = self.cached_process(name, data)
                results.append(result)
            except Exception as e:
                results.append(f"Error: {e}")
        return results

    def clear_cache(self):
        self.cache.clear()
"""
            + ("\n    # More code...\n" * 100)
        )
        source_files.append(("large.py", large_file))

        # Measure AST parsing performance
        parsing_times = []

        for filename, content in source_files:
            # Multiple runs for more accurate measurement
            for _ in range(3):
                start_time = time.time()

                chunks = chunker.chunk_content(
                    content,
                    title=f"Source: {filename}",
                    url=f"file://{filename}",
                    language="python",
                )

                end_time = time.time()
                elapsed_ms = (end_time - start_time) * 1000
                parsing_times.append(elapsed_ms)

                # Ensure parsing actually worked
                assert len(chunks) > 0
                # Should have some code chunks for Python files
                has_code = any(
                    chunk.get("chunk_type") == "code"
                    for chunk in chunks
                    if isinstance(chunk, dict)
                )
                if chunker.parsers:  # Only check if AST parsing is available
                    assert has_code, f"No code chunks found in {filename}"

        avg_parsing_time = sum(parsing_times) / len(parsing_times)
        max_parsing_time = max(parsing_times)

        print("\nAST parsing performance:")
        print(f"Average parsing time: {avg_parsing_time:.1f}ms")
        print(f"Maximum parsing time: {max_parsing_time:.1f}ms")
        print(f"Parsing times: {[f'{t:.1f}ms' for t in parsing_times]}")

        # Performance target: <100ms per source file
        assert avg_parsing_time < 100, (
            f"Average AST parsing time {avg_parsing_time:.1f}ms exceeds target of 100ms"
        )
        assert max_parsing_time < 200, (
            f"Maximum AST parsing time {max_parsing_time:.1f}ms is too slow"
        )

    def test_chunking_memory_efficiency(self, sample_documents):
        """Test that chunking doesn't consume excessive memory."""
        import os

        import psutil

        config = ChunkingConfig(strategy=ChunkingStrategy.ENHANCED)
        chunker = EnhancedChunker(config)

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process all documents
        all_chunks = []
        for doc in sample_documents:
            chunks = chunker.chunk_content(
                doc["content"], title=doc["title"], url=doc["url"]
            )
            all_chunks.extend(chunks)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        print("\nMemory usage:")
        print(f"Initial memory: {initial_memory:.1f}MB")
        print(f"Final memory: {final_memory:.1f}MB")
        print(f"Memory increase: {memory_increase:.1f}MB")
        print(
            f"Processed {len(all_chunks)} chunks from {len(sample_documents)} documents"
        )

        # Memory increase should be reasonable (less than 50MB for this test)
        assert memory_increase < 50, (
            f"Memory increase of {memory_increase:.1f}MB is too high"
        )

    def test_concurrent_chunking_performance(self, sample_documents):
        """Test chunking performance with concurrent operations."""
        import concurrent.futures

        config = ChunkingConfig(strategy=ChunkingStrategy.ENHANCED)

        def chunk_document(doc):
            chunker = EnhancedChunker(config)  # Each thread gets its own chunker
            return chunker.chunk_content(
                doc["content"], title=doc["title"], url=doc["url"]
            )

        start_time = time.time()

        # Use ThreadPoolExecutor for concurrent chunking
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(chunk_document, doc) for doc in sample_documents[:50]
            ]
            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        end_time = time.time()
        elapsed = end_time - start_time
        throughput = len(sample_documents[:50]) / elapsed

        print("\nConcurrent chunking performance:")
        print(f"Processed {len(sample_documents[:50])} documents in {elapsed:.2f}s")
        print(f"Throughput: {throughput:.0f} documents/second")

        # Should handle concurrency well
        assert len(results) == 50
        assert all(len(chunks) > 0 for chunks in results)


class TestChunkingStressTests:
    """Stress tests for chunking system."""

    def test_very_large_document_chunking(self):
        """Test chunking of very large documents."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.ENHANCED, chunk_size=2000, chunk_overlap=200
        )
        chunker = EnhancedChunker(config)

        # Create a very large document (1MB)
        large_content = "# Very Large Document\n\n" + ("This is line content. " * 50000)

        start_time = time.time()
        chunks = chunker.chunk_content(large_content, title="Large Doc", url="test.md")
        end_time = time.time()

        elapsed = end_time - start_time
        content_size_mb = len(large_content) / 1024 / 1024

        print("\nLarge document chunking:")
        print(f"Document size: {content_size_mb:.1f}MB")
        print(f"Chunking time: {elapsed:.2f}s")
        print(f"Generated {len(chunks)} chunks")
        print(f"Processing speed: {content_size_mb / elapsed:.1f}MB/s")

        # Should complete within reasonable time
        assert elapsed < 10, (
            f"Large document chunking took {elapsed:.2f}s, which is too slow"
        )
        assert len(chunks) > 100, "Should generate many chunks for large document"

        # Check chunk integrity
        total_content_length = sum(len(chunk["content"]) for chunk in chunks)
        assert total_content_length > len(large_content) * 0.8, (
            "Significant content loss in chunking"
        )

    def test_many_small_documents_chunking(self):
        """Test chunking many small documents quickly."""
        config = ChunkingConfig(strategy=ChunkingStrategy.BASIC)
        chunker = EnhancedChunker(config)

        # Generate many small documents
        small_docs = []
        for i in range(2000):
            small_docs.append(
                f"Document {i}: This is a short document with minimal content."
            )

        start_time = time.time()
        all_chunks = []

        for i, content in enumerate(small_docs):
            chunks = chunker.chunk_content(content, title=f"Doc {i}", url=f"doc{i}.txt")
            all_chunks.extend(chunks)

        end_time = time.time()
        elapsed = end_time - start_time
        throughput = len(small_docs) / elapsed

        print("\nMany small documents:")
        print(f"Processed {len(small_docs)} documents in {elapsed:.2f}s")
        print(f"Throughput: {throughput:.0f} documents/second")
        print(f"Generated {len(all_chunks)} total chunks")

        # Should meet high-throughput target
        assert throughput > 2000, (
            f"Small document throughput {throughput:.0f} docs/sec is below target"
        )

    def test_complex_code_chunking_performance(self):
        """Test performance on complex code with many functions and classes."""
        config = ChunkingConfig(strategy=ChunkingStrategy.AST, enable_ast_chunking=True)
        chunker = EnhancedChunker(config)

        # Generate complex Python code
        complex_code = """
# Complex Python module with many functions and classes

import os
import sys
from typing import List, Dict, Any, Optional
"""

        # Add many functions
        for i in range(50):
            complex_code += f"""
def function_{i}(param1: str, param2: int = {i}) -> Optional[str]:
    '''Function {i} with type hints and complex logic.'''
    if param2 > {i // 2}:
        result = param1.upper() * param2
        for j in range(param2):
            if j % 2 == 0:
                result += str(j)
        return result
    else:
        return None
"""

        # Add many classes
        for i in range(20):
            complex_code += f"""
class Class_{i}:
    '''Class {i} with multiple methods.'''

    def __init__(self, value: int = {i}):
        self.value = value
        self.data: Dict[str, Any] = {{}}

    def method_{i}_a(self) -> int:
        return self.value * 2

    def method_{i}_b(self, factor: float) -> float:
        return self.value * factor

    def method_{i}_c(self, items: List[str]) -> List[str]:
        return [item + str(self.value) for item in items]
"""

        start_time = time.time()
        chunks = chunker.chunk_content(
            complex_code, title="Complex Code", url="complex.py", language="python"
        )
        end_time = time.time()

        elapsed_ms = (end_time - start_time) * 1000
        code_size_kb = len(complex_code) / 1024

        print("\nComplex code chunking:")
        print(f"Code size: {code_size_kb:.1f}KB")
        print(f"Chunking time: {elapsed_ms:.1f}ms")
        print(f"Generated {len(chunks)} chunks")
        print(f"Processing speed: {code_size_kb / elapsed_ms * 1000:.1f}KB/s")

        # Should meet AST parsing target
        assert elapsed_ms < 500, (
            f"Complex code chunking took {elapsed_ms:.1f}ms, which is too slow"
        )
        assert len(chunks) > 20, "Should generate many chunks for complex code"

        # Verify code chunks were created (if AST parsing is available)
        if chunker.parsers:
            code_chunks = [
                c
                for c in chunks
                if isinstance(c, dict) and c.get("chunk_type") == "code"
            ]
            assert len(code_chunks) > 10, (
                "Should create many code chunks for complex code"
            )
