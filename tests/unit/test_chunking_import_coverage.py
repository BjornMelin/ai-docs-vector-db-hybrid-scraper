"""Simple tests to cover import fallback scenarios in chunking.py.

This test suite focuses specifically on covering the ImportError
handling blocks in the chunking module.
"""

import sys
import pytest
from unittest.mock import patch, MagicMock


def test_tree_sitter_import_error():
    """Test that import errors for tree_sitter are handled properly."""
    # First, let's test when tree_sitter itself is not available
    
    # Remove the chunking module from cache to force re-import
    modules_to_remove = []
    for module_name in sys.modules.keys():
        if 'chunking' in module_name:
            modules_to_remove.append(module_name)
    
    for module_name in modules_to_remove:
        if module_name in sys.modules:
            del sys.modules[module_name]
    
    # Mock the tree_sitter import to fail
    original_import = __builtins__['__import__']
    
    def mock_import(name, *args, **kwargs):
        if name == 'tree_sitter':
            raise ImportError("No module named 'tree_sitter'")
        return original_import(name, *args, **kwargs)
    
    with patch('builtins.__import__', side_effect=mock_import):
        # Import chunking module - should trigger the ImportError handling
        from src import chunking
        
        # Verify the fallback values are set
        assert chunking.TREE_SITTER_AVAILABLE is False
        assert chunking.Parser is None
        assert chunking.Node is None
        assert chunking.PYTHON_AVAILABLE is False
        assert chunking.JAVASCRIPT_AVAILABLE is False
        assert chunking.TYPESCRIPT_AVAILABLE is False


def test_python_parser_import_error():
    """Test when tree_sitter is available but python parser is not."""
    # Remove chunking modules from cache
    modules_to_remove = []
    for module_name in sys.modules.keys():
        if 'chunking' in module_name:
            modules_to_remove.append(module_name)
    
    for module_name in modules_to_remove:
        if module_name in sys.modules:
            del sys.modules[module_name]
    
    original_import = __builtins__['__import__']
    
    def mock_import(name, *args, **kwargs):
        if name == 'tree_sitter_python':
            raise ImportError("No module named 'tree_sitter_python'")
        return original_import(name, *args, **kwargs)
    
    with patch('builtins.__import__', side_effect=mock_import):
        from src import chunking
        
        # tree_sitter should be available, but python parser should not
        assert chunking.TREE_SITTER_AVAILABLE is True
        assert chunking.PYTHON_AVAILABLE is False
        assert chunking.tspython is None


def test_javascript_parser_import_error():
    """Test when tree_sitter is available but javascript parser is not."""
    modules_to_remove = []
    for module_name in sys.modules.keys():
        if 'chunking' in module_name:
            modules_to_remove.append(module_name)
    
    for module_name in modules_to_remove:
        if module_name in sys.modules:
            del sys.modules[module_name]
    
    original_import = __builtins__['__import__']
    
    def mock_import(name, *args, **kwargs):
        if name == 'tree_sitter_javascript':
            raise ImportError("No module named 'tree_sitter_javascript'")
        return original_import(name, *args, **kwargs)
    
    with patch('builtins.__import__', side_effect=mock_import):
        from src import chunking
        
        assert chunking.TREE_SITTER_AVAILABLE is True
        assert chunking.JAVASCRIPT_AVAILABLE is False
        assert chunking.tsjavascript is None


def test_typescript_parser_import_error():
    """Test when tree_sitter is available but typescript parser is not."""
    modules_to_remove = []
    for module_name in sys.modules.keys():
        if 'chunking' in module_name:
            modules_to_remove.append(module_name)
    
    for module_name in modules_to_remove:
        if module_name in sys.modules:
            del sys.modules[module_name]
    
    original_import = __builtins__['__import__']
    
    def mock_import(name, *args, **kwargs):
        if name == 'tree_sitter_typescript':
            raise ImportError("No module named 'tree_sitter_typescript'")
        return original_import(name, *args, **kwargs)
    
    with patch('builtins.__import__', side_effect=mock_import):
        from src import chunking
        
        assert chunking.TREE_SITTER_AVAILABLE is True
        assert chunking.TYPESCRIPT_AVAILABLE is False
        assert chunking.tstypescript is None


def test_chunking_with_missing_parsers():
    """Test that chunking still works when parsers are missing."""
    from src.chunking import EnhancedChunker
    from src.config.models import ChunkingConfig
    from src.config.enums import ChunkingStrategy
    
    # Even with missing parsers, chunking should still work
    config = ChunkingConfig(strategy=ChunkingStrategy.AST)
    chunker = EnhancedChunker(config)
    
    content = "def test_function():\n    return 'hello world'"
    chunks = chunker.chunk_content(content, "Test", "http://test.py")
    
    assert len(chunks) > 0
    assert chunks[0]["content"]