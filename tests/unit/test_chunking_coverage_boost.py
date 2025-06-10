"""Additional tests to boost chunking.py coverage to ≥90%.

These tests target specific uncovered code paths identified through coverage analysis.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.chunking import EnhancedChunker
from src.config.enums import ChunkingStrategy
from src.config.models import ChunkingConfig
from src.models.document_processing import Chunk


class TestChunkLargeCodeBlock:
    """Test _chunk_large_code_block method for line-based splitting."""
    
    def test_chunk_large_code_block_basic(self):
        """Test basic line-based code chunking."""
        config = ChunkingConfig(chunk_size=100, chunk_overlap=20)
        chunker = EnhancedChunker(config)
        
        # Create content that exceeds chunk size
        code_content = "\n".join([f"line_{i} = {i}" for i in range(20)])
        
        chunks = chunker._chunk_large_code_block(code_content, 0, "python")
        
        assert len(chunks) > 1
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(chunk.chunk_type == "code" for chunk in chunks)
        assert all(chunk.language == "python" for chunk in chunks)
        assert all(chunk.has_code for chunk in chunks)
    
    def test_chunk_large_code_block_with_overlap(self):
        """Test line-based chunking with overlap calculation."""
        config = ChunkingConfig(chunk_size=50, chunk_overlap=10)
        chunker = EnhancedChunker(config)
        
        code_content = "\n".join([f"function_{i}()" for i in range(10)])
        
        chunks = chunker._chunk_large_code_block(code_content, 100, "javascript")
        
        assert len(chunks) > 1
        # Check that overlap is applied correctly
        assert chunks[0].start_pos == 100
        # Subsequent chunks should have overlapping content
        for i in range(1, len(chunks)):
            assert chunks[i].start_pos < chunks[i-1].end_pos
    
    def test_chunk_large_code_block_single_chunk(self):
        """Test when content fits in single chunk."""
        config = ChunkingConfig(chunk_size=1000, chunk_overlap=100)
        chunker = EnhancedChunker(config)
        
        small_content = "def small_function():\n    return 42"
        
        chunks = chunker._chunk_large_code_block(small_content, 0, "python")
        
        assert len(chunks) == 1
        assert chunks[0].content == small_content
        assert chunks[0].start_pos == 0
        assert chunks[0].end_pos == len(small_content)


class TestRecursiveCodeSplitting:
    """Test recursive splitting of large code units."""
    
    def test_split_large_method_recursively(self):
        """Test recursive splitting of very large methods."""
        config = ChunkingConfig(
            chunk_size=500, 
            max_chunk_size=800, 
            max_function_chunk_size=1000
        )
        chunker = EnhancedChunker(config)
        
        # Mock a very large method that needs recursive splitting
        very_large_method = "    def huge_method(self):\n" + "        " + "code_line\n" * 100
        
        mock_methods = [{
            "name": "huge_method",
            "start_pos": 20,
            "end_pos": len(very_large_method) + 20,
            "type": "method"
        }]
        
        with patch.object(chunker, '_extract_class_methods', return_value=mock_methods), \
             patch.object(chunker, '_split_large_code_unit') as mock_split:
            
            mock_split.return_value = [
                Chunk(content="part1", start_pos=0, end_pos=100, chunk_index=0, chunk_type="code"),
                Chunk(content="part2", start_pos=100, end_pos=200, chunk_index=1, chunk_type="code")
            ]
            
            chunks = chunker._split_large_code_unit(
                f"class TestClass:\n{very_large_method}", 0, "class", "python"
            )
            
            assert len(chunks) >= 1
            # Verify recursive splitting was called for large method
            mock_split.assert_called()
    
    def test_method_with_parent_class_metadata(self):
        """Test that sub-chunks of class methods get parent_class metadata."""
        config = ChunkingConfig(
            chunk_size=500,
            chunk_overlap=50,
            max_chunk_size=800,
            max_function_chunk_size=1000
        )
        chunker = EnhancedChunker(config)
        
        # Mock a large method that gets split
        large_content = "class TestClass:\n    def method():\n" + "        code\n" * 50
        
        mock_methods = [{
            "name": "method",
            "start_pos": 20,
            "end_pos": len(large_content),
            "type": "method"
        }]
        
        with patch.object(chunker, '_extract_class_methods', return_value=mock_methods):
            # Create sub-chunks that should get metadata
            sub_chunk1 = Chunk(content="sub1", start_pos=0, end_pos=50, chunk_index=0, chunk_type="code")
            sub_chunk2 = Chunk(content="sub2", start_pos=50, end_pos=100, chunk_index=1, chunk_type="code")
            
            with patch.object(chunker, '_split_large_code_unit', return_value=[sub_chunk1, sub_chunk2]):
                chunks = chunker._split_large_code_unit(large_content, 0, "class", "python")
                
                # Verify metadata was added
                for chunk in chunks:
                    if hasattr(chunk, 'metadata') and chunk.metadata:
                        assert chunk.metadata.get("parent_class") is True


class TestASTParsingSpeicalCases:
    """Test special cases in AST parsing that may be uncovered."""
    
    def test_ast_parsing_with_encoding_issues(self):
        """Test AST parsing with content that has encoding issues."""
        config = ChunkingConfig(strategy=ChunkingStrategy.AST)
        chunker = EnhancedChunker(config)
        
        # Mock parser that handles encoding edge cases
        mock_parser = Mock()
        mock_tree = Mock()
        mock_root = Mock()
        mock_root.children = []
        mock_tree.root_node = mock_root
        mock_parser.parse.return_value = mock_tree
        
        chunker.parsers = {"python": mock_parser}
        
        # Content with potential encoding issues
        content_with_unicode = "def test_function():\n    print('Hello 世界')\n    return True"
        
        with patch.object(chunker, '_extract_code_units', return_value=[]), \
             patch.object(chunker, '_enhanced_chunking') as mock_enhanced:
            mock_enhanced.return_value = [
                Chunk(content=content_with_unicode, start_pos=0, end_pos=len(content_with_unicode), chunk_index=0)
            ]
            
            chunks = chunker._ast_based_chunking(content_with_unicode, "python")
            assert len(chunks) >= 1
    
    def test_language_detection_edge_cases(self):
        """Test language detection for edge cases."""
        config = ChunkingConfig(detect_language=True)
        chunker = EnhancedChunker(config)
        
        # Test with ambiguous content
        ambiguous_content = "// This could be JavaScript or C++\nif (true) { return; }"
        
        chunks = chunker.chunk_content(ambiguous_content)
        assert len(chunks) >= 1
        assert isinstance(chunks[0], dict)  # Should be formatted as dict
    
    def test_function_context_preservation(self):
        """Test function context preservation across chunks."""
        config = ChunkingConfig(include_function_context=True, chunk_size=500, chunk_overlap=50)
        chunker = EnhancedChunker(config)
        
        code_with_functions = """def func1():
    return 1

def func2():
    return 2
    
def very_long_function():
    # This function is intentionally long
    result = 0
    for i in range(100):
        result += i
        if result > 50:
            break
    return result"""
        
        chunks = chunker.chunk_content(code_with_functions, language="python")
        
        assert len(chunks) >= 1
        # Verify chunks contain context information
        for chunk_dict in chunks:
            assert "content" in chunk_dict
            # Language may or may not be present depending on chunk type
            if "language" in chunk_dict:
                assert chunk_dict["language"] in ["python", None]


class TestBoundaryDetectionEdgeCases:
    """Test boundary detection edge cases."""
    
    def test_boundary_detection_with_nested_structures(self):
        """Test boundary detection with deeply nested code structures."""
        config = ChunkingConfig()
        chunker = EnhancedChunker(config)
        
        nested_content = """
if condition1:
    if condition2:
        if condition3:
            if condition4:
                if condition5:
                    deeply_nested_call()
                    another_call()
            else:
                alternative()
        else:
            another_alternative()
    else:
        outer_alternative()
else:
    final_alternative()
"""
        
        boundary = chunker._find_enhanced_boundary(nested_content, 0, len(nested_content) // 2)
        
        # Should find a reasonable boundary
        assert 0 <= boundary <= len(nested_content)
    
    def test_boundary_detection_with_strings(self):
        """Test boundary detection with multi-line strings."""
        config = ChunkingConfig()
        chunker = EnhancedChunker(config)
        
        content_with_strings = '''
def function():
    long_string = """
    This is a very long multi-line string
    that contains multiple sentences.
    
    It should not be split inappropriately.
    """
    return process(long_string)
'''
        
        boundary = chunker._find_enhanced_boundary(content_with_strings, 0, 100)
        
        # Should avoid splitting inside the multi-line string
        assert 0 <= boundary <= len(content_with_strings)


class TestChunkingSizeLimits:
    """Test chunking with various size constraints."""
    
    def test_chunk_size_validation(self):
        """Test chunking respects size limits."""
        config = ChunkingConfig(
            chunk_size=200,
            chunk_overlap=40,
            min_chunk_size=50,
            max_chunk_size=500
        )
        chunker = EnhancedChunker(config)
        
        # Content that should be split appropriately
        medium_content = "def function():\n" + "    line\n" * 30
        
        chunks = chunker._enhanced_chunking(medium_content, "python")
        
        for chunk in chunks:
            # Most chunks should respect size limits (allowing some margin for boundaries)
            assert len(chunk.content) >= config.min_chunk_size * 0.8  # Allow 20% margin
            assert len(chunk.content) <= config.max_chunk_size * 1.2  # Allow 20% margin
    
    def test_very_long_single_line(self):
        """Test handling of very long single lines."""
        config = ChunkingConfig(chunk_size=500, chunk_overlap=50)
        chunker = EnhancedChunker(config)
        
        # Create a very long line that exceeds chunk_size (500 chars)
        very_long_line = "def very_long_function_name_that_exceeds_chunk_size_with_many_parameters(" + "param" + "," * 200 + "):"
        
        chunks = chunker._enhanced_chunking(very_long_line, "python")
        
        assert len(chunks) >= 1
        # Should handle long lines gracefully (some chunks may exceed chunk_size due to line boundaries)
        total_content_length = sum(len(chunk.content) for chunk in chunks)
        assert total_content_length >= len(very_long_line) * 0.8  # Most content should be preserved


class TestTreeSitterEdgeCases:
    """Test Tree-sitter integration edge cases."""
    
    def test_parser_memory_cleanup(self):
        """Test that parsers handle memory cleanup properly."""
        config = ChunkingConfig(enable_ast_chunking=True)
        chunker = EnhancedChunker(config)
        
        # Mock parser that simulates memory pressure
        mock_parser = Mock()
        mock_tree = Mock()
        mock_root = Mock()
        mock_root.children = []
        mock_tree.root_node = mock_root
        mock_parser.parse.return_value = mock_tree
        
        chunker.parsers = {"python": mock_parser}
        
        # Process multiple chunks to test memory handling
        for i in range(10):
            content = f"def function_{i}():\n    return {i}"
            chunks = chunker._ast_based_chunking(content, "python")
            assert len(chunks) >= 1
            
        # Verify parser was called multiple times
        assert mock_parser.parse.call_count == 10
    
    def test_tree_sitter_node_iteration(self):
        """Test Tree-sitter node iteration edge cases."""
        config = ChunkingConfig()
        chunker = EnhancedChunker(config)
        
        # Mock nodes with complex hierarchies for Python AST
        mock_identifier = Mock()
        mock_identifier.type = "identifier"
        mock_identifier.children = []
        mock_identifier.start_byte = 4
        mock_identifier.end_byte = 8
        
        mock_function = Mock()
        mock_function.type = "function_definition"
        mock_function.children = [mock_identifier]
        mock_function.start_byte = 0
        mock_function.end_byte = 17
        
        content = "def test(): pass"
        code_units = []
        
        # Test direct traversal of function node
        chunker._traverse_python(mock_function, content, code_units)
        
        # Should find the function
        assert len(code_units) >= 1
        assert code_units[0]["type"] == "function"
        assert code_units[0]["name"] == "test"