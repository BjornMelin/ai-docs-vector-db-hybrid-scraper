"""Comprehensive tests for chunking edge cases and coverage improvement."""

from unittest.mock import MagicMock
from unittest.mock import patch

from src.chunking import EnhancedChunker
from src.config.enums import ChunkingStrategy
from src.config.models import ChunkingConfig


class TestChunkingEdgeCases:
    """Test edge cases and error paths for improved coverage."""

    def test_empty_content(self):
        """Test chunking empty content."""
        config = ChunkingConfig()
        chunker = EnhancedChunker(config)

        chunks = chunker.chunk_content("")
        assert len(chunks) == 0

    def test_whitespace_only_content(self):
        """Test chunking whitespace-only content."""
        config = ChunkingConfig()
        chunker = EnhancedChunker(config)

        chunks = chunker.chunk_content("   \n\n   \t   ")
        # Should produce empty chunks or filter out whitespace
        assert len(chunks) == 0

    def test_very_long_content_without_boundaries(self):
        """Test chunking very long content without good boundaries."""
        config = ChunkingConfig(chunk_size=50, chunk_overlap=10)
        chunker = EnhancedChunker(config)

        # Create content with no good boundaries
        content = "a" * 200
        chunks = chunker.chunk_content(content)

        assert len(chunks) >= 3  # Should be split into multiple chunks
        assert all(
            len(chunk["content"]) <= 60 for chunk in chunks
        )  # Respect size + overlap

    def test_ast_strategy_without_tree_sitter(self):
        """Test AST strategy when tree-sitter is not available."""
        config = ChunkingConfig(strategy=ChunkingStrategy.AST)

        with patch("src.chunking.TREE_SITTER_AVAILABLE", False):
            chunker = EnhancedChunker(config)

            content = """
def test_function():
    return "test"
"""
            chunks = chunker.chunk_content(content, language="python")

            # Should fall back to enhanced chunking
            assert len(chunks) >= 1
            assert all(isinstance(chunk, dict) for chunk in chunks)

    def test_detect_language_unknown_extension(self):
        """Test language detection with unknown file extension."""
        config = ChunkingConfig()
        chunker = EnhancedChunker(config)

        lang = chunker._detect_language_from_url("test.unknown")
        assert lang == "unknown"

    def test_detect_language_no_extension(self):
        """Test language detection without file extension."""
        config = ChunkingConfig()
        chunker = EnhancedChunker(config)

        lang = chunker._detect_language_from_url("test")
        assert lang == "unknown"

    def test_detect_language_from_code_fences_no_language(self):
        """Test language detection from code fences without language specifier."""
        config = ChunkingConfig()
        chunker = EnhancedChunker(config)

        content = """
```
some code without language
```
"""
        lang = chunker._detect_language_from_code_fences(content)
        assert lang == "unknown"

    def test_detect_language_from_patterns_no_match(self):
        """Test language detection from patterns with no recognizable patterns."""
        config = ChunkingConfig()
        chunker = EnhancedChunker(config)

        content = "This is just plain text with no code patterns."
        lang = chunker._detect_language_from_patterns(content)
        assert lang == "unknown"

    def test_chunk_large_code_block_no_functions(self):
        """Test chunking large code block without function patterns."""
        config = ChunkingConfig(chunk_size=200, chunk_overlap=40)
        chunker = EnhancedChunker(config)

        # Large code without function patterns
        large_code = "x = 1\n" * 50  # 150 chars
        chunks = chunker._chunk_large_code_block(large_code, 0, "unknown")

        assert len(chunks) >= 1
        assert all(chunk.chunk_type == "code" for chunk in chunks)

    def test_basic_chunking_no_good_boundaries(self):
        """Test basic chunking when no good boundaries are found."""
        config = ChunkingConfig(
            chunk_size=100, chunk_overlap=20, strategy=ChunkingStrategy.BASIC
        )
        chunker = EnhancedChunker(config)

        # Content without good sentence boundaries
        content = "word " * 30  # 150 chars, all short words
        chunks = chunker._basic_chunking(content)

        assert len(chunks) >= 2
        assert all(chunk.chunk_type == "text" for chunk in chunks)

    def test_enhanced_chunking_no_code_blocks(self):
        """Test enhanced chunking with no code blocks."""
        config = ChunkingConfig(preserve_code_blocks=False)
        chunker = EnhancedChunker(config)

        content = """
Some text content.

```python
def test():
    pass
```

More text.
"""
        chunks = chunker._enhanced_chunking(content)

        # Should treat everything as regular text
        assert all(chunk.chunk_type == "text" for chunk in chunks)

    def test_ast_chunking_with_unsupported_language(self):
        """Test AST chunking with a language we don't support."""
        config = ChunkingConfig(strategy=ChunkingStrategy.AST)
        chunker = EnhancedChunker(config)

        content = """
module Main where
main = putStrLn "Hello, Haskell!"
"""
        chunks = chunker._ast_based_chunking(content, "haskell")

        # Should fall back to enhanced chunking
        assert len(chunks) >= 1

    def test_get_next_code_block_empty_list(self):
        """Test getting next code block from empty list."""
        config = ChunkingConfig()
        chunker = EnhancedChunker(config)

        result = chunker._get_next_code_block([], 0)
        assert result is None

    def test_get_next_code_block_no_match(self):
        """Test getting next code block when current position is past all blocks."""
        config = ChunkingConfig()
        chunker = EnhancedChunker(config)

        from src.models.document_processing import CodeBlock

        code_blocks = [
            CodeBlock("python", "test", 10, 20, "```"),
            CodeBlock("python", "test2", 30, 40, "```"),
        ]

        # Position past all blocks
        result = chunker._get_next_code_block(code_blocks, 50)
        assert result is None

    def test_handle_code_block_as_chunk_regular_size(self):
        """Test handling regular-sized code block."""
        config = ChunkingConfig()
        chunker = EnhancedChunker(config)

        from src.models.document_processing import CodeBlock

        # Create a normal-sized code block
        content = "def test():\n    return 'hello'"
        chunks = []

        code_block = CodeBlock("python", content, 0, len(content), "```")

        chunker._handle_code_block_as_chunk(content, chunks, code_block)

        # Should create one chunk for normal-sized code
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "code"
        assert chunks[0].language == "python"

    def test_format_chunks_with_metadata(self):
        """Test chunk formatting with metadata."""
        config = ChunkingConfig()
        chunker = EnhancedChunker(config)

        from src.models.document_processing import Chunk

        chunk = Chunk(
            content="test content",
            start_pos=0,
            end_pos=12,
            chunk_index=0,
            total_chunks=1,
            char_count=12,
            token_estimate=3,
            chunk_type="code",
            language="python",
            has_code=True,
            metadata={"function_name": "test"},
        )

        formatted = chunker._format_chunks([chunk], "Test Title", "http://test.com")

        assert len(formatted) == 1
        assert formatted[0]["language"] == "python"
        assert formatted[0]["metadata"]["function_name"] == "test"

    def test_chunk_content_with_detection_disabled(self):
        """Test chunking with language detection disabled."""
        config = ChunkingConfig(detect_language=False)
        chunker = EnhancedChunker(config)

        content = """
def test_function():
    return "test"
"""
        chunks = chunker.chunk_content(content, url="test.py")

        # Should chunk without language detection
        assert len(chunks) >= 1

    def test_chunk_text_content_edge_cases(self):
        """Test _chunk_text_content with edge cases."""
        config = ChunkingConfig(chunk_size=20, chunk_overlap=5)
        chunker = EnhancedChunker(config)

        # Short content that doesn't need chunking
        chunks = chunker._chunk_text_content("short", 0, 5)
        assert len(chunks) == 1
        assert chunks[0].content == "short"

        # Content exactly at chunk size
        content = "a" * 20
        chunks = chunker._chunk_text_content(content, 0, 20)
        assert len(chunks) == 1
        assert len(chunks[0].content) == 20

    def test_find_text_boundary_edge_cases(self):
        """Test text boundary finding with edge cases."""
        config = ChunkingConfig()
        chunker = EnhancedChunker(config)

        # Very short content
        boundary = chunker._find_text_boundary("ab", 0, 2)
        assert boundary == 2

        # Content with only commas (lowest priority boundary)
        content = "a, b, c, d, e, f, g, h, i, j"
        boundary = chunker._find_text_boundary(content, 0, 15)
        assert boundary <= 15

    def test_enhanced_boundary_no_patterns(self):
        """Test enhanced boundary finding with no pattern matches."""
        config = ChunkingConfig()
        chunker = EnhancedChunker(config)

        # Content with no boundary patterns
        content = "abcdefghijklmnopqrstuvwxyz"
        boundary = chunker._find_enhanced_boundary(content, 0, 15)
        assert boundary == 15  # Should return target_end if no patterns found


class TestLanguageSpecificParsing:
    """Test language-specific parsing features."""

    def test_javascript_patterns(self):
        """Test JavaScript language detection patterns."""
        config = ChunkingConfig()
        chunker = EnhancedChunker(config)

        js_content = """
const myFunc = async () => {
    return await fetch('/api/data');
};

var oldStyle = function() {
    console.log('old style');
};
"""
        lang = chunker._detect_language_from_patterns(js_content)
        assert lang == "javascript"

    def test_typescript_extension_detection(self):
        """Test TypeScript file extension detection."""
        config = ChunkingConfig()
        chunker = EnhancedChunker(config)

        assert chunker._detect_language_from_url("component.tsx") == "typescript"
        assert chunker._detect_language_from_url("module.ts") == "typescript"

    def test_markdown_extension_detection(self):
        """Test Markdown file extension detection."""
        config = ChunkingConfig()
        chunker = EnhancedChunker(config)

        assert chunker._detect_language_from_url("README.md") == "markdown"

    def test_multiple_code_fence_languages(self):
        """Test detection with multiple different code fence languages."""
        config = ChunkingConfig()
        chunker = EnhancedChunker(config)

        content = """
```python
def python_func():
    pass
```

```bash
echo "shell command"
```

```python
def another_python_func():
    pass
```
"""
        # Python appears twice, should be detected
        lang = chunker._detect_language_from_code_fences(content)
        assert lang == "python"


class TestErrorHandling:
    """Test error handling and fallback scenarios."""

    def test_ast_parsing_exception(self):
        """Test handling of AST parsing exceptions."""
        config = ChunkingConfig(strategy=ChunkingStrategy.AST)
        chunker = EnhancedChunker(config)

        # Mock parser to raise exception
        mock_parser = MagicMock()
        mock_parser.parse.side_effect = Exception("Parse error")
        chunker.parsers = {"python": mock_parser}

        content = """
def test_function():
    return "test"
"""
        # Should fallback to enhanced chunking without error
        chunks = chunker._ast_based_chunking(content, "python")
        assert len(chunks) >= 1

    def test_parser_initialization_with_missing_module(self):
        """Test parser initialization when language module is missing."""
        config = ChunkingConfig(
            enable_ast_chunking=True, supported_languages=["python", "nonexistent"]
        )

        # Should handle missing language gracefully
        chunker = EnhancedChunker(config)
        # Only available languages should be loaded
        assert "nonexistent" not in chunker.parsers

    def test_extract_code_units_with_exception(self):
        """Test code unit extraction with parsing exception."""
        config = ChunkingConfig()
        chunker = EnhancedChunker(config)

        # Mock node that raises exception during traversal
        mock_node = MagicMock()
        mock_node.children = [MagicMock()]
        mock_node.children[0].type = "function_definition"
        mock_node.children[0].children = []
        mock_node.children[0].start_byte = 0
        mock_node.children[0].end_byte = 10

        # Should handle gracefully
        units = chunker._extract_code_units(mock_node, "test content", "python")
        assert isinstance(units, list)
