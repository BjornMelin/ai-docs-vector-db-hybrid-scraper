"""Unit tests for chunking module."""

from src.chunking import EnhancedChunker
from src.config.enums import ChunkingStrategy
from src.config.models import ChunkingConfig
from src.models.document_processing import Chunk
from src.models.document_processing import CodeBlock


class TestChunkingConfig:
    """Test ChunkingConfig model."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ChunkingConfig()
        assert config.chunk_size == 1600
        assert config.chunk_overlap == 320
        assert config.strategy == ChunkingStrategy.ENHANCED
        assert config.enable_ast_chunking is True
        assert config.preserve_function_boundaries is True
        assert config.preserve_code_blocks is True
        assert config.max_function_chunk_size == 3200
        assert config.detect_language is True
        assert config.fallback_to_text_chunking is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = ChunkingConfig(
            chunk_size=2000,
            chunk_overlap=400,
            strategy=ChunkingStrategy.BASIC,
            enable_ast_chunking=False,
            preserve_function_boundaries=False,
            supported_languages=["python"],
        )
        assert config.chunk_size == 2000
        assert config.chunk_overlap == 400
        assert config.strategy == ChunkingStrategy.BASIC
        assert config.enable_ast_chunking is False
        assert config.supported_languages == ["python"]

    def test_supported_languages_default(self):
        """Test default supported languages."""
        config = ChunkingConfig()
        assert "python" in config.supported_languages
        assert "javascript" in config.supported_languages
        assert "typescript" in config.supported_languages
        assert "markdown" in config.supported_languages


class TestCodeBlock:
    """Test CodeBlock dataclass."""

    def test_code_block_creation(self):
        """Test creating a code block."""
        block = CodeBlock(
            language="python",
            content="def hello(): pass",
            start_pos=0,
            end_pos=17,
            fence_type="```",
        )
        assert block.language == "python"
        assert block.content == "def hello(): pass"
        assert block.start_pos == 0
        assert block.end_pos == 17
        assert block.fence_type == "```"

    def test_code_block_with_tilde_fence(self):
        """Test code block with tilde fence."""
        block = CodeBlock(
            language="javascript",
            content="console.log('test');",
            start_pos=10,
            end_pos=30,
            fence_type="~~~",
        )
        assert block.fence_type == "~~~"


class TestChunk:
    """Test Chunk dataclass."""

    def test_minimal_chunk(self):
        """Test creating a minimal chunk."""
        chunk = Chunk(
            content="This is a test chunk",
            start_pos=0,
            end_pos=20,
            chunk_index=0,
        )
        assert chunk.content == "This is a test chunk"
        assert chunk.start_pos == 0
        assert chunk.end_pos == 20
        assert chunk.chunk_index == 0
        assert chunk.total_chunks == 0
        assert chunk.char_count == 0
        assert chunk.token_estimate == 0
        assert chunk.chunk_type == "text"
        assert chunk.language is None
        assert chunk.has_code is False
        assert chunk.metadata is None

    def test_full_chunk(self):
        """Test creating a chunk with all fields."""
        metadata = {"is_function": True, "function_name": "test_func"}
        chunk = Chunk(
            content="def test_func(): pass",
            start_pos=100,
            end_pos=121,
            chunk_index=2,
            total_chunks=5,
            char_count=21,
            token_estimate=5,
            chunk_type="code",
            language="python",
            has_code=True,
            metadata=metadata,
        )
        assert chunk.total_chunks == 5
        assert chunk.char_count == 21
        assert chunk.token_estimate == 5
        assert chunk.chunk_type == "code"
        assert chunk.language == "python"
        assert chunk.has_code is True
        assert chunk.metadata == metadata


class TestEnhancedChunker:
    """Test EnhancedChunker class."""

    def test_initialization(self):
        """Test chunker initialization."""
        config = ChunkingConfig()
        chunker = EnhancedChunker(config)
        assert chunker.config == config
        assert isinstance(chunker.parsers, dict)

    def test_initialization_with_multiple_languages(self):
        """Test chunker initialization with multiple language support."""
        config = ChunkingConfig(
            enable_ast_chunking=True,
            supported_languages=["python", "javascript", "typescript", "markdown"],
        )
        chunker = EnhancedChunker(config)
        assert chunker.config == config
        assert isinstance(chunker.parsers, dict)
        # Parsers may or may not be loaded depending on availability

    def test_detect_language_from_url(self):
        """Test language detection from URL."""
        config = ChunkingConfig()
        chunker = EnhancedChunker(config)

        assert chunker._detect_language_from_url("test.py") == "python"
        assert chunker._detect_language_from_url("script.js") == "javascript"
        assert chunker._detect_language_from_url("app.ts") == "typescript"
        assert chunker._detect_language_from_url("README.md") == "markdown"
        assert chunker._detect_language_from_url("test.txt") == "unknown"

    def test_detect_language_from_code_fences(self):
        """Test language detection from code fences."""
        config = ChunkingConfig()
        chunker = EnhancedChunker(config)

        # Single Python code block
        content = """
```python
def hello():
    print("Hello")
```
"""
        assert chunker._detect_language_from_code_fences(content) == "python"

        # Multiple JavaScript blocks
        content = """
```javascript
console.log("test");
```
More text
```javascript
function test() {}
```
"""
        assert chunker._detect_language_from_code_fences(content) == "javascript"

        # Mixed languages (most common wins)
        content = """
```python
x = 1
```
```javascript
var x = 1;
```
```javascript
let y = 2;
```
"""
        assert chunker._detect_language_from_code_fences(content) == "javascript"

    def test_detect_language_from_patterns(self):
        """Test language detection from code patterns."""
        config = ChunkingConfig()
        chunker = EnhancedChunker(config)

        # Python patterns
        python_content = """
import os
from pathlib import Path

def main():
    pass
"""
        assert chunker._detect_language_from_patterns(python_content) == "python"

        # JavaScript patterns
        js_content = """
const express = require('express');
let app = express();
var port = 3000;
"""
        assert chunker._detect_language_from_patterns(js_content) == "javascript"

    def test_find_code_blocks(self):
        """Test finding code blocks in content."""
        config = ChunkingConfig()
        chunker = EnhancedChunker(config)

        content = """
Some text before.

```python
def hello():
    print("Hello")
```

Middle text.

~~~javascript
console.log("test");
~~~

End text.
"""
        blocks = chunker._find_code_blocks(content)
        assert len(blocks) == 2

        # First block
        assert blocks[0].language == "python"
        assert "def hello():" in blocks[0].content
        assert blocks[0].fence_type == "```"

        # Second block
        assert blocks[1].language == "javascript"
        assert "console.log" in blocks[1].content
        assert blocks[1].fence_type == "~~~"

    def test_basic_chunking(self):
        """Test basic chunking without code awareness."""
        config = ChunkingConfig(
            chunk_size=50,
            chunk_overlap=10,
            strategy=ChunkingStrategy.BASIC,
        )
        chunker = EnhancedChunker(config)

        content = "This is a test content that should be chunked into smaller pieces based on the chunk size configuration."

        chunks = chunker._basic_chunking(content)
        assert len(chunks) > 1

        # Check first chunk
        assert chunks[0].chunk_type == "text"
        assert len(chunks[0].content) <= config.chunk_size

    def test_enhanced_chunking_preserves_code_blocks(self):
        """Test that enhanced chunking preserves code blocks."""
        config = ChunkingConfig(
            chunk_size=100,
            chunk_overlap=20,
            preserve_code_blocks=True,
        )
        chunker = EnhancedChunker(config)

        content = """
This is some text before the code.

```python
def important_function():
    # This function should not be split
    result = complex_calculation()
    return result
```

This is text after the code.
"""
        chunks = chunker._enhanced_chunking(content)

        # Find the code chunk
        code_chunks = [c for c in chunks if c.chunk_type == "code"]
        assert len(code_chunks) >= 1

        # Verify the code block is intact
        code_chunk = code_chunks[0]
        assert "def important_function():" in code_chunk.content
        assert "return result" in code_chunk.content
        assert code_chunk.language == "python"

    def test_chunk_content_main_entry(self):
        """Test the main chunk_content method."""
        config = ChunkingConfig(chunk_size=100, chunk_overlap=20)
        chunker = EnhancedChunker(config)

        content = """
# Python Tutorial

This is a Python tutorial about functions.

```python
def greet(name):
    return f"Hello, {name}!"
```

Functions are reusable blocks of code.
"""
        chunks = chunker.chunk_content(
            content,
            title="Python Tutorial",
            url="https://example.com/tutorial.py",
        )

        assert len(chunks) > 0
        assert all(isinstance(chunk, dict) for chunk in chunks)

        # Check that chunks have required fields
        for chunk in chunks:
            assert "content" in chunk
            assert "title" in chunk
            assert "url" in chunk
            assert chunk["title"] == "Python Tutorial" or "Part" in chunk["title"]
            assert chunk["url"] == "https://example.com/tutorial.py"

    def test_find_enhanced_boundary(self):
        """Test finding enhanced boundaries for chunking."""
        config = ChunkingConfig()
        chunker = EnhancedChunker(config)

        # Test with various boundary types
        content = """First sentence. Second sentence! Third question?

New paragraph starts here.

def function():
    pass

Another paragraph."""

        # Should prefer paragraph boundary
        boundary = chunker._find_enhanced_boundary(content, 0, 30)
        assert (
            content[boundary - 2 : boundary] == ". "
            or content[boundary - 2 : boundary] == "! "
        )

    def test_chunk_large_code_block_by_functions(self):
        """Test chunking large code blocks by function boundaries."""
        config = ChunkingConfig(chunk_size=400, chunk_overlap=50)
        chunker = EnhancedChunker(config)

        large_code = """
def function_one():
    # First function
    x = 1
    y = 2
    return x + y

def function_two():
    # Second function
    a = 10
    b = 20
    return a * b

def function_three():
    # Third function
    result = []
    for i in range(10):
        result.append(i)
    return result
"""
        chunks = chunker._chunk_large_code_block(large_code, 0, "python")

        # Should split by functions
        assert len(chunks) >= 3

        # Each chunk should contain one function
        assert "function_one" in chunks[0].content
        assert "function_two" in chunks[1].content
        assert "function_three" in chunks[2].content

    def test_format_chunks(self):
        """Test formatting chunks for output."""
        config = ChunkingConfig()
        chunker = EnhancedChunker(config)

        chunks = [
            Chunk(
                content="Test content",
                start_pos=0,
                end_pos=12,
                chunk_index=0,
                total_chunks=1,
                char_count=12,
                token_estimate=3,
                chunk_type="text",
            )
        ]

        formatted = chunker._format_chunks(chunks, "Test Title", "http://test.com")

        assert len(formatted) == 1
        assert formatted[0]["content"] == "Test content"
        assert formatted[0]["title"] == "Test Title"
        assert formatted[0]["url"] == "http://test.com"
        assert formatted[0]["chunk_index"] == 0
        assert formatted[0]["total_chunks"] == 1
