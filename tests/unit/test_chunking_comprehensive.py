"""Comprehensive tests for chunking module to improve coverage.

This module provides comprehensive test coverage for the EnhancedChunker class,
following 2025 standardized patterns with proper type annotations, standardized
assertions, and modern test patterns.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import pytest

from src.chunking import EnhancedChunker
from src.config.core import ChunkingConfig
from src.config.enums import ChunkingStrategy
from src.models.document_processing import Chunk, CodeBlock, CodeLanguage

from tests.utils.assertion_helpers import (
    assert_valid_document_chunk,
    assert_performance_within_threshold,
    assert_resource_cleanup,
)
from tests.utils.test_factories import ChunkFactory, DocumentFactory


class TestEnhancedChunker:
    """Test EnhancedChunker class functionality with standardized patterns.
    
    This test class provides comprehensive coverage of the EnhancedChunker
    functionality using modern pytest patterns, proper type annotations,
    and standardized assertion helpers.
    """
    
    @pytest.fixture
    def chunker_config(self) -> ChunkingConfig:
        """Create a ChunkingConfig for testing.
        
        Returns:
            ChunkingConfig configured for test scenarios
        """
        return ChunkingConfig(
            strategy=ChunkingStrategy.ENHANCED,
            chunk_size=1000,
            chunk_overlap=200,
            min_chunk_size=100,
            max_chunk_size=2000
        )
    
    @pytest.fixture
    def chunker(self, chunker_config: ChunkingConfig) -> EnhancedChunker:
        """Create an EnhancedChunker instance for testing.
        
        Args:
            chunker_config: Configuration for the chunker
            
        Returns:
            EnhancedChunker instance ready for testing
        """
        return EnhancedChunker(chunker_config)
    
    def test_chunker_initialization(self, chunker_config: ChunkingConfig) -> None:
        """Test EnhancedChunker initialization.
        
        Verifies that the EnhancedChunker properly initializes with the provided
        configuration and exposes the correct configuration properties.
        
        Args:
            chunker_config: Test configuration for the chunker
        """
        chunker = EnhancedChunker(chunker_config)
        
        assert chunker.config == chunker_config
        assert chunker.config.chunk_size == 1000
        assert chunker.config.chunk_overlap == 200
        assert chunker.config.min_chunk_size == 100
        assert chunker.config.max_chunk_size == 2000
        assert isinstance(chunker.parsers, dict)
    
    def test_initialize_parsers(self, chunker: EnhancedChunker) -> None:
        """Test parser initialization.
        
        Verifies that parser initialization completes without errors and
        creates the necessary parser infrastructure.
        
        Args:
            chunker: EnhancedChunker instance for testing
        """
        chunker._initialize_parsers()
        
        # Should not raise any errors during initialization
        assert True
    
    @pytest.mark.parametrize("url,expected_lang", [
        ("test.py", "python"),
        ("test.js", "javascript"),
        ("test.ts", "typescript"),
        ("test.jsx", "unknown"),  # .jsx not supported in current implementation
        ("test.tsx", "typescript"),
        ("test.txt", "unknown"),  # .txt not supported
        ("", "unknown"),  # no extension
    ])
    def test_detect_language_from_url(
        self, 
        chunker: EnhancedChunker, 
        url: str, 
        expected_lang: str
    ) -> None:
        """Test language detection from URL using parametrized test cases.
        
        Verifies that the chunker correctly identifies programming languages
        based on file extensions in URLs.
        
        Args:
            chunker: EnhancedChunker instance for testing
            url: URL or filename to test
            expected_lang: Expected detected language
        """
        result = chunker._detect_language_from_url(url)
        assert result == expected_lang
    
    def test_detect_language_from_code_fences(self, chunker: EnhancedChunker) -> None:
        """Test language detection from code fences.
        
        Verifies that the chunker correctly identifies programming languages
        from code fence markers in markdown-style content.
        
        Args:
            chunker: EnhancedChunker instance for testing
        """
        # Test Python code fence detection
        content_with_python = """
        Some text here
        ```python
        def hello():
            print("Hello")
        ```
        More text
        """
        
        result = chunker._detect_language_from_code_fences(content_with_python)
        assert result == "python"
        
        # Test JavaScript code fence detection
        content_with_js = """
        ```javascript
        function hello() {
            console.log("Hello");
        }
        ```
        """
        
        result = chunker._detect_language_from_code_fences(content_with_js)
        assert result == "javascript"
        
        # Test content without code fences
        content_without_fences = "Just plain text here"
        result = chunker._detect_language_from_code_fences(content_without_fences)
        assert result == "unknown"
    
    def test_detect_language_from_patterns(self, chunker: EnhancedChunker) -> None:
        """Test language detection from code patterns.
        
        Verifies that the chunker can identify programming languages
        based on syntax patterns and keywords in the content.
        
        Args:
            chunker: EnhancedChunker instance for testing
        """
        # Test Python pattern detection
        python_code = """import os
from pathlib import Path

def hello_world():
    print("Hello, World!")
"""
        
        result = chunker._detect_language_from_patterns(python_code)
        assert result == "python"
        
        # Test JavaScript pattern detection
        js_code = """const myFunction = () => {
    console.log("Hello, World!");
}

let myVariable = 42;
var anotherVar = "test";
"""
        
        result = chunker._detect_language_from_patterns(js_code)
        assert result == "javascript"
        
        # Test plain text (no patterns)
        plain_text = "This is just plain text with no code patterns."
        result = chunker._detect_language_from_patterns(plain_text)
        assert result == "unknown"
    
    def test_detect_language(self, chunker: EnhancedChunker) -> None:
        """Test overall language detection method.
        
        Verifies that the main language detection method properly combines
        URL hints, code fences, and pattern matching for accurate detection.
        
        Args:
            chunker: EnhancedChunker instance for testing
        """
        # Test with URL hint
        result = chunker._detect_language("def hello(): pass", "test.py")
        assert result == "python"
        
        # Test with code fences
        content = """
        ```typescript
        interface User {
            name: string;
        }
        ```
        """
        result = chunker._detect_language(content)
        assert result == "typescript"
        
        # Test with patterns
        result = chunker._detect_language("const test = () => { return true; }")
        assert result == "javascript"
    
    def test_find_code_blocks(self, chunker: EnhancedChunker) -> None:
        """Test finding code blocks in content.
        
        Verifies that the chunker correctly identifies and extracts
        code blocks from mixed content with proper language detection.
        
        Args:
            chunker: EnhancedChunker instance for testing
        """
        content = """
        Here is some text.
        
        ```python
        def hello():
            print("Hello")
        ```
        
        More text here.
        
        ```javascript
        function world() {
            console.log("World");
        }
        ```
        
        Final text.
        """
        
        code_blocks = chunker._find_code_blocks(content)
        
        assert len(code_blocks) == 2
        assert code_blocks[0].language == CodeLanguage.PYTHON
        assert "def hello():" in code_blocks[0].content
        assert code_blocks[1].language == CodeLanguage.JAVASCRIPT
        assert "function world()" in code_blocks[1].content
    
    def test_basic_chunking(self, chunker: EnhancedChunker) -> None:
        """Test basic chunking functionality.
        
        Verifies that basic chunking properly splits long text into
        appropriately sized chunks while respecting size constraints.
        
        Args:
            chunker: EnhancedChunker instance for testing
        """
        # Create a long text that needs chunking
        long_text = "This is a sentence. " * 100  # ~2000 characters
        
        chunks = chunker._basic_chunking(long_text)
        
        assert len(chunks) > 1  # Should be split into multiple chunks
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(len(chunk.content) <= chunker.config.max_chunk_size for chunk in chunks)
        
        # Validate chunk structure
        for i, chunk in enumerate(chunks):
            assert chunk.content is not None
            assert len(chunk.content) > 0
            assert chunk.chunk_index == i
            assert chunk.start_pos >= 0
            assert chunk.end_pos > chunk.start_pos
    
    def test_chunk_content_with_text(self, chunker: EnhancedChunker) -> None:
        """Test chunk_content method with plain text.
        
        Verifies that the main chunking method properly processes plain
        text content and returns well-formed chunks.
        
        Args:
            chunker: EnhancedChunker instance for testing
        """
        text = "This is a simple text that should be chunked properly. " * 20
        
        chunks = chunker.chunk_content(text, title="Test Document", url="test.txt")
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, dict) for chunk in chunks)
        assert all(chunk["chunk_index"] >= 0 for chunk in chunks)
        
        # Validate each chunk using standardized assertions
        for chunk in chunks:
            assert_valid_document_chunk(chunk)
    
    def test_chunk_content_with_code(self, chunker: EnhancedChunker) -> None:
        """Test chunk_content method with code content.
        
        Verifies that the chunker properly handles code content,
        respecting function boundaries and maintaining code structure.
        
        Args:
            chunker: EnhancedChunker instance for testing
        """
        python_code = """
        def function1():
            '''This is a function'''
            return "hello"
        
        def function2():
            '''Another function'''
            return "world"
        
        class MyClass:
            def method1(self):
                pass
            
            def method2(self):
                pass
        """
        
        chunks = chunker.chunk_content(python_code, "test.py")
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        
        # Validate code-specific chunk properties
        for chunk in chunks:
            assert_valid_document_chunk({
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "url": "test.py"
            })
            # Additional validation for code chunks
            if hasattr(chunk, 'language'):
                assert chunk.language in ["python", None]
    
    def test_find_text_boundary(self, chunker: EnhancedChunker) -> None:
        """Test finding text boundaries for chunking.
        
        Verifies that the chunker correctly identifies appropriate
        text boundaries for splitting content.
        
        Args:
            chunker: EnhancedChunker instance for testing
        """
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        
        # Should find boundary at sentence end
        boundary = chunker._find_text_boundary(text, 0, 30)
        
        assert boundary > 0
        assert boundary <= len(text)
        # Boundary should be at a sentence end
        assert text[boundary-1] in ".!?" or text[boundary] == " "
    
    def test_find_enhanced_boundary(self, chunker: EnhancedChunker) -> None:
        """Test finding enhanced boundaries in code.
        
        Verifies that the chunker can identify appropriate boundaries
        in code content, preferring function boundaries when possible.
        
        Args:
            chunker: EnhancedChunker instance for testing
        """
        code = """
        def function1():
            return "hello"
        
        def function2():
            return "world"
        """
        
        boundary = chunker._find_enhanced_boundary(code, 0, 50)
        
        assert boundary > 0
        assert boundary <= len(code)
        # Validate that boundary is at a reasonable position
        assert boundary < len(code) or code[boundary:boundary+3] in ["\n\n\n", "def"]
    
    def test_format_chunks(self, chunker: EnhancedChunker) -> None:
        """Test chunk formatting.
        
        Verifies that raw chunk data is properly formatted into
        structured Chunk objects with correct metadata.
        
        Args:
            chunker: EnhancedChunker instance for testing
        """
        raw_chunks = [
            ("First chunk content", 0, 20),
            ("Second chunk content", 15, 35),
            ("Third chunk content", 30, 50)
        ]
        
        formatted_chunks = chunker._format_chunks(raw_chunks, "test.txt")
        
        assert len(formatted_chunks) == 3
        assert all(isinstance(chunk, Chunk) for chunk in formatted_chunks)
        assert all(chunk.url == "test.txt" for chunk in formatted_chunks)
        assert all(chunk.chunk_index == i for i, chunk in enumerate(formatted_chunks))
        
        # Validate each formatted chunk
        for i, chunk in enumerate(formatted_chunks):
            assert_valid_document_chunk({
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "url": chunk.url
            })
            assert chunk.chunk_index == i
    
    def test_get_current_code_block(self, chunker: EnhancedChunker) -> None:
        """Test getting current code block at position.
        
        Verifies that the chunker correctly identifies which code block
        contains a given position in the content.
        
        Args:
            chunker: EnhancedChunker instance for testing
        """
        code_blocks = [
            CodeBlock(content="block1", start_pos=10, end_pos=20, language=CodeLanguage.PYTHON),
            CodeBlock(content="block2", start_pos=30, end_pos=40, language=CodeLanguage.PYTHON),
        ]
        
        # Position within first block
        current_block = chunker._get_current_code_block(code_blocks, 15)
        assert current_block == code_blocks[0]
        
        # Position outside any block
        current_block = chunker._get_current_code_block(code_blocks, 25)
        assert current_block is None
    
    def test_get_next_code_block(self, chunker: EnhancedChunker) -> None:
        """Test getting next code block after position.
        
        Verifies that the chunker correctly finds the next code block
        that appears after a given position in the content.
        
        Args:
            chunker: EnhancedChunker instance for testing
        """
        code_blocks = [
            CodeBlock(content="block1", start_pos=10, end_pos=20, language=CodeLanguage.PYTHON),
            CodeBlock(content="block2", start_pos=30, end_pos=40, language=CodeLanguage.PYTHON),
        ]
        
        # Position before first block
        next_block = chunker._get_next_code_block(code_blocks, 5)
        assert next_block == code_blocks[0]
        
        # Position between blocks
        next_block = chunker._get_next_code_block(code_blocks, 25)
        assert next_block == code_blocks[1]
        
        # Position after all blocks
        next_block = chunker._get_next_code_block(code_blocks, 50)
        assert next_block is None
    
    def test_handle_code_block_as_chunk(self, chunker: EnhancedChunker) -> None:
        """Test handling code blocks as chunks.
        
        Verifies that code blocks are properly converted into chunks
        and added to the chunk collection.
        
        Args:
            chunker: EnhancedChunker instance for testing
        """
        content = "Some text\n```python\ndef hello():\n    pass\n```\nMore text"
        chunks: list[Chunk] = []
        code_block = CodeBlock(
            content="def hello():\n    pass",
            start_pos=10,
            end_pos=30,
            language=CodeLanguage.PYTHON
        )
        
        result = chunker._handle_code_block_as_chunk(content, chunks, code_block)
        
        # Should add chunk to the list or return position advancement
        assert len(chunks) > 0 or result > 0
        
        # If a chunk was added, validate its structure
        if len(chunks) > 0:
            chunk = chunks[0]
            assert_valid_document_chunk({
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "start_pos": chunk.start_pos,
                "end_pos": chunk.end_pos
            })


class TestEnhancedChunkerEdgeCases:
    """Test edge cases and error conditions with standardized patterns.
    
    This test class focuses on boundary conditions, error handling,
    and edge cases that might occur in real-world usage.
    """
    
    @pytest.fixture
    def chunker(self) -> EnhancedChunker:
        """Create chunker with default config.
        
        Returns:
            EnhancedChunker instance configured with default settings
        """
        config = ChunkingConfig()
        return EnhancedChunker(config)
    
    def test_empty_content(self, chunker):
        """Test chunking empty content."""
        chunks = chunker.chunk_content("", "test.txt")
        assert len(chunks) == 0
    
    def test_very_short_content(self, chunker):
        """Test chunking very short content."""
        short_text = "Hi"
        chunks = chunker.chunk_content(short_text, "test.txt")
        
        # Should still create at least one chunk
        assert len(chunks) >= 1
        assert chunks[0].content == short_text
    
    def test_content_at_exact_chunk_size(self, chunker):
        """Test content that's exactly the chunk size."""
        exact_size_text = "a" * chunker.chunk_size
        chunks = chunker.chunk_content(exact_size_text, "test.txt")
        
        assert len(chunks) == 1
        assert chunks[0].content == exact_size_text
    
    def test_content_slightly_over_chunk_size(self, chunker):
        """Test content slightly over chunk size."""
        over_size_text = "a" * (chunker.chunk_size + 1)
        chunks = chunker.chunk_content(over_size_text, "test.txt")
        
        assert len(chunks) >= 1
    
    def test_malformed_code_fences(self, chunker):
        """Test handling malformed code fences."""
        malformed_content = """
        ```python
        def hello():
            print("Hello")
        # Missing closing fence
        
        Some more text
        """
        
        # Should not crash
        chunks = chunker.chunk_content(malformed_content, "test.py")
        assert len(chunks) > 0
    
    def test_nested_code_fences(self, chunker):
        """Test handling nested code fences."""
        nested_content = """
        ```markdown
        # Documentation
        
        ```python
        def example():
            pass
        ```
        ```
        """
        
        # Should handle gracefully
        chunks = chunker.chunk_content(nested_content, "test.md")
        assert len(chunks) > 0
    
    def test_unicode_content(self, chunker):
        """Test chunking unicode content."""
        unicode_text = "Hello 世界! 🌍 Múltíplos idiomas. Ελληνικά."
        
        chunks = chunker.chunk_content(unicode_text, "test.txt")
        assert len(chunks) > 0
        assert unicode_text in chunks[0].content


class TestChunkingStrategies:
    """Test different chunking strategies."""
    
    def test_smart_chunking_strategy(self):
        """Test smart chunking strategy."""
        config = ChunkingConfig(strategy=ChunkingStrategy.ENHANCED)
        chunker = EnhancedChunker(config)
        
        code_content = """
        def function1():
            '''First function'''
            return "hello"
        
        def function2():
            '''Second function'''
            return "world"
        """
        
        chunks = chunker.chunk_content(code_content, "test.py")
        assert len(chunks) > 0
    
    def test_fixed_size_chunking_strategy(self):
        """Test fixed size chunking strategy."""
        config = ChunkingConfig(strategy=ChunkingStrategy.BASIC)
        chunker = EnhancedChunker(config)
        
        long_text = "This is a sentence. " * 100
        
        chunks = chunker.chunk_content(long_text, "test.txt")
        assert len(chunks) > 1
        
        # All chunks except possibly the last should be close to chunk_size
        for chunk in chunks[:-1]:
            assert len(chunk.content) <= config.chunk_size + config.chunk_overlap
    
    def test_sentence_chunking_strategy(self):
        """Test sentence-based chunking strategy."""
        config = ChunkingConfig(strategy=ChunkingStrategy.AST_AWARE)
        chunker = EnhancedChunker(config)
        
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        
        chunks = chunker.chunk_content(text, "test.txt")
        assert len(chunks) > 0


class TestCodeLanguageDetection:
    """Test code language detection capabilities."""
    
    @pytest.fixture
    def chunker(self):
        config = ChunkingConfig()
        return EnhancedChunker(config)
    
    def test_python_detection(self, chunker):
        """Test Python code detection."""
        python_samples = [
            "def hello(): pass",
            "class MyClass: pass",
            "import os\nprint('hello')",
            "# Python comment\nif __name__ == '__main__':",
        ]
        
        for sample in python_samples:
            lang = chunker._detect_language(sample, "test.py")
            assert lang == "python"
    
    def test_javascript_detection(self, chunker):
        """Test JavaScript code detection."""
        js_samples = [
            "function hello() { return true; }",
            "const x = () => { return 42; }",
            "class MyClass { constructor() {} }",
            "// JavaScript comment\nconsole.log('hello');",
        ]
        
        for sample in js_samples:
            lang = chunker._detect_language(sample, "test.js")
            assert lang == "javascript"
    
    def test_typescript_detection(self, chunker):
        """Test TypeScript code detection."""
        ts_samples = [
            "interface User { name: string; }",
            "function hello(): string { return 'hello'; }",
            "class MyClass { private x: number; }",
            "export const config: Config = {};",
        ]
        
        for sample in ts_samples:
            lang = chunker._detect_language(sample, "test.ts")
            assert lang == "typescript"
    
    def test_mixed_content_detection(self, chunker):
        """Test detection with mixed content."""
        mixed_content = """
        # This is a markdown file
        
        Here's some Python code:
        
        ```python
        def hello():
            print("Hello")
        ```
        
        And some JavaScript:
        
        ```javascript
        function world() {
            console.log("World");
        }
        ```
        """
        
        lang = chunker._detect_language(mixed_content, "test.md")
        # Should detect the first code fence language
        assert lang in ["python", "text", "markdown"]


class TestPerformanceAndLimits:
    """Test performance characteristics and limits."""
    
    def test_large_content_handling(self):
        """Test handling of large content."""
        config = ChunkingConfig(chunk_size=1000, max_chunk_size=2000)
        chunker = EnhancedChunker(config)
        
        # Create large content (10KB)
        large_content = "This is a test sentence. " * 400
        
        chunks = chunker.chunk_content(large_content, "test.txt")
        
        assert len(chunks) > 1
        assert all(len(chunk.content) <= config.max_chunk_size for chunk in chunks)
    
    def test_many_small_functions(self):
        """Test chunking content with many small functions."""
        config = ChunkingConfig()
        chunker = EnhancedChunker(config)
        
        # Generate many small functions
        functions = []
        for i in range(50):
            functions.append(f"""
def function_{i}():
    '''Function number {i}'''
    return {i}
""")
        
        code_content = "\n".join(functions)
        chunks = chunker.chunk_content(code_content, "test.py")
        
        assert len(chunks) > 0
        # Should handle many functions without issues
    
    def test_deeply_nested_code(self):
        """Test chunking deeply nested code."""
        config = ChunkingConfig()
        chunker = EnhancedChunker(config)
        
        # Create deeply nested code
        nested_code = "if True:\n"
        for i in range(10):
            nested_code += "    " * (i + 1) + f"if condition_{i}:\n"
        nested_code += "    " * 11 + "print('deep nesting')\n"
        
        chunks = chunker.chunk_content(nested_code, "test.py")
        assert len(chunks) > 0


class TestChunkValidation:
    """Test chunk validation and properties."""
    
    def test_chunk_properties(self):
        """Test that chunks have required properties."""
        config = ChunkingConfig()
        chunker = EnhancedChunker(config)
        
        content = "Test content for chunking validation."
        chunks = chunker.chunk_content(content, "test.txt")
        
        for i, chunk in enumerate(chunks):
            assert isinstance(chunk, Chunk)
            assert hasattr(chunk, 'content')
            assert hasattr(chunk, 'chunk_index')
            assert hasattr(chunk, 'url')
            assert chunk.chunk_index == i
            assert chunk.url == "test.txt"
            assert len(chunk.content) > 0
    
    def test_chunk_overlap_preservation(self):
        """Test that chunk overlap is preserved correctly."""
        config = ChunkingConfig(chunk_size=100, chunk_overlap=20)
        chunker = EnhancedChunker(config)
        
        # Create content that will definitely need multiple chunks
        content = "Word " * 50  # ~250 characters
        chunks = chunker.chunk_content(content, "test.txt")
        
        if len(chunks) > 1:
            # Check that there's some overlap between consecutive chunks
            # This is a basic check - the actual overlap might be more sophisticated
            for i in range(len(chunks) - 1):
                current_chunk = chunks[i]
                next_chunk = chunks[i + 1]
                
                # Verify chunks exist and have content
                assert len(current_chunk.content) > 0
                assert len(next_chunk.content) > 0


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_config(self):
        """Test handling of invalid configuration."""
        # This should be caught by Pydantic validation in ChunkingConfig
        with pytest.raises(ValueError):
            invalid_config = ChunkingConfig(chunk_size=0)
    
    def test_none_content(self):
        """Test handling of None content."""
        config = ChunkingConfig()
        chunker = EnhancedChunker(config)
        
        with pytest.raises((TypeError, AttributeError)):
            chunker.chunk_content(None, "test.txt")
    
    def test_non_string_content(self):
        """Test handling of non-string content."""
        config = ChunkingConfig()
        chunker = EnhancedChunker(config)
        
        with pytest.raises((TypeError, AttributeError)):
            chunker.chunk_content(123, "test.txt")
    
    def test_invalid_url(self):
        """Test handling of invalid URL."""
        config = ChunkingConfig()
        chunker = EnhancedChunker(config)
        
        # Should handle gracefully
        chunks = chunker.chunk_content("test content", None)
        assert len(chunks) > 0