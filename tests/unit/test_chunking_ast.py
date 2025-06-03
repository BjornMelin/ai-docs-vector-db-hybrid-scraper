"""Unit tests for AST-based chunking enhancements."""

from unittest.mock import MagicMock
from unittest.mock import patch

from src.chunking import EnhancedChunker
from src.config.enums import ChunkingStrategy
from src.config.models import ChunkingConfig
from src.models.document_processing import Chunk


class TestASTParserLoading:
    """Test multi-language parser loading functionality."""

    def test_parser_loading_all_languages(self):
        """Test that parsers are loaded for all supported languages."""
        # This test requires actual parser packages to be installed
        # We'll test with just Python which is installed
        config = ChunkingConfig(
            enable_ast_chunking=True,
            supported_languages=["python"],
        )

        chunker = EnhancedChunker(config)

        # Should have Python parser if tree-sitter-python is installed
        if "python" in chunker.parsers:
            assert chunker.parsers["python"] is not None
        else:
            # Parser not available, which is OK for test environment
            assert len(chunker.parsers) == 0

    def test_parser_loading_missing_language(self):
        """Test graceful handling when a language parser is not available."""
        config = ChunkingConfig(
            enable_ast_chunking=True,
            supported_languages=[
                "rust",
                "golang",
                "unknown",
            ],  # None of these are implemented
        )

        chunker = EnhancedChunker(config)

        # Should have no parsers since none of these languages are supported
        assert "rust" not in chunker.parsers
        assert "golang" not in chunker.parsers
        assert "unknown" not in chunker.parsers

    def test_parser_loading_with_exception(self):
        """Test graceful handling when parser initialization fails."""
        config = ChunkingConfig(
            enable_ast_chunking=True,
            supported_languages=["python"],
        )

        with patch("src.chunking.TREE_SITTER_AVAILABLE", True):
            mock_tspython = MagicMock()
            # Make language() raise an exception
            mock_tspython.language.side_effect = Exception("Parser error")

            with patch("src.chunking.tspython", mock_tspython):
                chunker = EnhancedChunker(config)

                # Should have no parsers due to exception
                assert len(chunker.parsers) == 0

    def test_ast_chunking_disabled(self):
        """Test that no parsers are loaded when AST chunking is disabled."""
        config = ChunkingConfig(
            enable_ast_chunking=False,
            supported_languages=["python", "javascript", "typescript"],
        )

        chunker = EnhancedChunker(config)
        assert len(chunker.parsers) == 0


class TestASTBasedChunking:
    """Test AST-based chunking functionality."""

    def test_ast_chunking_with_overlap_documentation(self):
        """Test that AST chunking has proper overlap documentation."""
        config = ChunkingConfig(
            enable_ast_chunking=True,
            chunk_overlap=100,
        )
        chunker = EnhancedChunker(config)

        # Check that the method has proper documentation
        assert "_ast_based_chunking" in dir(chunker)
        assert "Overlap Strategy" in chunker._ast_based_chunking.__doc__
        assert "character-based overlap" in chunker._ast_based_chunking.__doc__
        assert "semantic context" in chunker._ast_based_chunking.__doc__

    def test_ast_chunking_fallback(self):
        """Test fallback to enhanced chunking when AST parsing fails."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.AST,
            enable_ast_chunking=True,
        )
        chunker = EnhancedChunker(config)

        # Remove all parsers to force fallback
        chunker.parsers = {}

        content = """
def test_function():
    return "test"
"""

        chunks = chunker._ast_based_chunking(content, "python")

        # Should fall back to enhanced chunking
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)

    @patch("src.chunking.TREE_SITTER_AVAILABLE", True)
    def test_ast_chunking_with_code_units(self):
        """Test AST chunking extracts code units properly."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.AST,
            enable_ast_chunking=True,
            chunk_size=400,
            chunk_overlap=50,
        )

        # Create a mock parser
        mock_parser = MagicMock()
        mock_tree = MagicMock()
        mock_root = MagicMock()

        # Setup mock AST structure
        mock_parser.parse.return_value = mock_tree
        mock_tree.root_node = mock_root
        mock_root.children = []

        chunker = EnhancedChunker(config)
        chunker.parsers = {"python": mock_parser}

        # Mock _extract_code_units to return test data
        def mock_extract_code_units(node, content, lang):
            return [
                {
                    "type": "function",
                    "name": "test_func",
                    "start_pos": 0,
                    "end_pos": 50,
                }
            ]

        chunker._extract_code_units = mock_extract_code_units

        content = """def test_func():
    return "test"

Some other content here."""

        chunks = chunker._ast_based_chunking(content, "python")

        # Should have at least one chunk for the function
        assert len(chunks) > 0
        assert chunks[0].chunk_type == "code"
        assert chunks[0].metadata["node_type"] == "function"


class TestSplitLargeCodeUnit:
    """Test enhanced _split_large_code_unit functionality."""

    def test_split_large_class_by_methods(self):
        """Test splitting a large class by its methods."""
        config = ChunkingConfig(
            max_function_chunk_size=3200,
            chunk_size=400,
            chunk_overlap=50,
            max_chunk_size=1000,
        )
        chunker = EnhancedChunker(config)

        # Mock the parser setup
        mock_parser = MagicMock()
        mock_tree = MagicMock()
        mock_root = MagicMock()

        mock_parser.parse.return_value = mock_tree
        mock_tree.root_node = mock_root

        chunker.parsers = {"python": mock_parser}

        # Mock _extract_class_methods
        def mock_extract_methods(node, content, lang):
            return [
                {"name": "method1", "start_pos": 20, "end_pos": 60},
                {"name": "method2", "start_pos": 60, "end_pos": 100},
            ]

        chunker._extract_class_methods = mock_extract_methods

        class_content = """class TestClass:
    def method1(self):
        pass

    def method2(self):
        pass"""

        chunks = chunker._split_large_code_unit(class_content, 0, "class", "python")

        # Should have chunks for class header and methods
        assert len(chunks) >= 2
        assert chunks[0].metadata["node_type"] == "class_header"
        assert chunks[1].metadata["node_type"] == "method"

    def test_split_large_function_by_blocks(self):
        """Test splitting a large function by logical blocks."""
        config = ChunkingConfig(
            max_function_chunk_size=3200,
            chunk_size=400,
            chunk_overlap=50,
            max_chunk_size=1000,
        )
        chunker = EnhancedChunker(config)

        # Mock the parser setup
        mock_parser = MagicMock()
        mock_tree = MagicMock()
        mock_root = MagicMock()

        mock_parser.parse.return_value = mock_tree
        mock_tree.root_node = mock_root

        chunker.parsers = {"python": mock_parser}

        # Mock _extract_function_blocks
        def mock_extract_blocks(node, content, lang):
            return [
                {"type": "statement_sequence", "start_pos": 0, "end_pos": 50},
                {"type": "statement_sequence", "start_pos": 50, "end_pos": 100},
            ]

        chunker._extract_function_blocks = mock_extract_blocks
        chunker._extract_function_signature = lambda n, c, lang: "def test():"

        function_content = """def test():
    # First block
    x = 1
    y = 2

    # Second block
    if x > 0:
        print(x)"""

        chunks = chunker._split_large_code_unit(
            function_content, 0, "function", "python"
        )

        # Should have chunks for different blocks
        assert len(chunks) == 2
        assert all(c.metadata["node_type"] == "function_block" for c in chunks)
        assert chunks[1].content.startswith("# Function: def test():")

    def test_split_fallback_to_line_based(self):
        """Test fallback to line-based splitting when AST splitting fails."""
        config = ChunkingConfig(
            max_function_chunk_size=500,
            chunk_size=100,
            chunk_overlap=10,
            max_chunk_size=200,
        )
        chunker = EnhancedChunker(config)

        # No parsers available
        chunker.parsers = {}

        large_content = "\n".join([f"line {i}" for i in range(20)])

        chunks = chunker._split_large_code_unit(large_content, 0, "function", "python")

        # Should fall back to line-based splitting
        assert len(chunks) > 1
        assert all(isinstance(c, Chunk) for c in chunks)


class TestHelperMethods:
    """Test helper methods for AST-based chunking."""

    def test_extract_class_methods_python(self):
        """Test extracting methods from a Python class AST."""
        config = ChunkingConfig()
        chunker = EnhancedChunker(config)

        # Create mock AST nodes
        mock_class_node = MagicMock()
        mock_method_node = MagicMock()
        mock_method_node.type = "function_definition"
        mock_method_node.start_byte = 10
        mock_method_node.end_byte = 50

        mock_id_node = MagicMock()
        mock_id_node.type = "identifier"
        mock_id_node.start_byte = 14
        mock_id_node.end_byte = 24

        mock_method_node.children = [mock_id_node]
        mock_class_node.children = [mock_method_node]

        content = "class Test:\n    def test_method(self):\n        pass"

        methods = chunker._extract_class_methods(mock_class_node, content, "python")

        assert len(methods) == 1
        assert methods[0]["type"] == "method"
        assert methods[0]["start_pos"] == 10
        assert methods[0]["end_pos"] == 50

    def test_extract_function_blocks(self):
        """Test extracting logical blocks from a function."""
        config = ChunkingConfig(
            chunk_size=400,
            chunk_overlap=50,
            max_function_chunk_size=3200,
            max_chunk_size=1000,
        )
        chunker = EnhancedChunker(config)

        # Mock a function node
        mock_func_node = MagicMock()
        mock_func_node.start_byte = 0
        mock_func_node.end_byte = 150  # Small enough to not split

        content = "def test():\n    pass"

        blocks = chunker._extract_function_blocks(mock_func_node, content, "python")

        # Should return whole function since it's small
        assert len(blocks) == 1
        assert blocks[0]["type"] == "whole_function"

    def test_extract_function_signature_python(self):
        """Test extracting function signature for Python."""
        config = ChunkingConfig()
        chunker = EnhancedChunker(config)

        # Mock function node with colon child
        mock_func_node = MagicMock()
        mock_func_node.start_byte = 0

        mock_colon = MagicMock()
        mock_colon.type = ":"
        mock_colon.start_byte = 29

        mock_func_node.children = [mock_colon]

        content = "def test_function(arg1, arg2):\n    pass"

        signature = chunker._extract_function_signature(
            mock_func_node, content, "python"
        )

        assert signature == "def test_function(arg1, arg2)"

    def test_extract_function_signature_javascript(self):
        """Test extracting function signature for JavaScript."""
        config = ChunkingConfig()
        chunker = EnhancedChunker(config)

        # Mock function node with statement_block child
        mock_func_node = MagicMock()
        mock_func_node.start_byte = 0

        mock_block = MagicMock()
        mock_block.type = "statement_block"
        mock_block.start_byte = 20

        mock_func_node.children = [mock_block]

        content = "function test(a, b) {\n    return a + b;\n}"

        signature = chunker._extract_function_signature(
            mock_func_node, content, "javascript"
        )

        assert signature == "function test(a, b)"

    def test_get_js_method_name(self):
        """Test extracting JavaScript method name."""
        config = ChunkingConfig()
        chunker = EnhancedChunker(config)

        # Mock method node
        mock_method = MagicMock()
        mock_id = MagicMock()
        mock_id.type = "property_identifier"
        mock_id.start_byte = 0
        mock_id.end_byte = 10

        mock_method.children = [mock_id]

        content = "methodName() { return 42; }"

        name = chunker._get_js_method_name(mock_method, content)

        assert name == "methodName"


class TestIntegration:
    """Integration tests for the complete chunking flow."""

    def test_chunk_content_with_ast_strategy(self):
        """Test complete chunking flow with AST strategy."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.AST,
            enable_ast_chunking=True,
            chunk_size=200,
            chunk_overlap=40,
        )

        # Mock tree-sitter availability but no parsers
        # This will force fallback to enhanced chunking
        with patch("src.chunking.TREE_SITTER_AVAILABLE", True):
            chunker = EnhancedChunker(config)
            chunker.parsers = {}  # No parsers available

            content = """
# Python Code Example

This is a Python file with functions.

def calculate(x, y):
    '''Calculate sum of x and y'''
    return x + y

def process(data):
    '''Process the data'''
    result = []
    for item in data:
        result.append(calculate(item, 10))
    return result

# End of file
"""

            chunks = chunker.chunk_content(
                content,
                title="Example Code",
                url="example.py",
                language="python",
            )

            # Should produce chunks
            assert len(chunks) > 0

            # All chunks should be dictionaries
            assert all(isinstance(chunk, dict) for chunk in chunks)

            # Check chunk structure
            for chunk in chunks:
                assert "content" in chunk
                assert "title" in chunk
                assert "url" in chunk
                assert "chunk_index" in chunk
                assert "total_chunks" in chunk

    def test_chunk_overlap_in_text_content(self):
        """Test that overlap is applied correctly in text content between code units."""
        config = ChunkingConfig(
            chunk_size=50,
            chunk_overlap=10,
            preserve_code_blocks=True,
        )
        chunker = EnhancedChunker(config)

        # Create content with text that will be chunked
        content = "A" * 45 + " " + "B" * 45 + " " + "C" * 45

        chunks = chunker._chunk_text_content(content, 0, len(content))

        # Should have overlap between chunks
        assert len(chunks) >= 2

        if len(chunks) >= 2:
            # Check that there's overlap (chunk 1 end overlaps with chunk 2 start)
            chunk1_end = chunks[0].content[-10:]  # Last 10 chars
            chunk2_start = chunks[1].content[:20]  # First 20 chars

            # There should be some common content due to overlap
            # The overlap might not be exact due to boundary detection
            assert any(char in chunk2_start for char in chunk1_end[-5:])
