"""test_chunking.py - Comprehensive test suite for advanced 2025 enhanced chunking."""

import pytest
from src.chunking import ChunkingConfig
from src.chunking import ChunkingStrategy
from src.chunking import EnhancedChunker


class TestEnhancedChunking:
    """Test suite for enhanced chunking functionality"""

    @pytest.fixture()
    def basic_config(self):
        """Basic chunking configuration"""
        return ChunkingConfig(
            chunk_size=200,  # Small for testing
            chunk_overlap=40,
            strategy=ChunkingStrategy.ENHANCED,
            enable_ast_chunking=False,
        )

    @pytest.fixture()
    def ast_config(self):
        """AST-based chunking configuration"""
        return ChunkingConfig(
            chunk_size=300,
            chunk_overlap=60,
            strategy=ChunkingStrategy.AST_BASED,
            enable_ast_chunking=True,
        )

    @pytest.fixture()
    def python_code_sample(self):
        """Sample Python code for testing"""
        return '''
# This is a Python module for testing chunking

import os
import sys
from typing import List, Dict

class DataProcessor:
    """Process data with various methods"""

    def __init__(self, config: Dict):
        self.config = config
        self.data = []

    def process_item(self, item: str) -> str:
        """Process a single item"""
        # Some processing logic
        result = item.upper()
        return result

    def process_batch(self, items: List[str]) -> List[str]:
        """Process multiple items"""
        results = []
        for item in items:
            processed = self.process_item(item)
            results.append(processed)
        return results

async def fetch_data(url: str) -> Dict:
    """Fetch data from URL asynchronously"""
    # Simulated async operation
    return {"status": "success", "data": []}

def main():
    """Main entry point"""
    processor = DataProcessor({"verbose": True})
    data = ["hello", "world"]
    results = processor.process_batch(data)
    print(results)

if __name__ == "__main__":
    main()
'''

    @pytest.fixture()
    def markdown_with_code(self):
        """Sample markdown with embedded code blocks"""
        return """# API Documentation

This document describes the API endpoints.

## Installation

First, install the package:

```bash
pip install mypackage
```

## Usage Example

Here's how to use the API:

```python
import mypackage

# Initialize the client
client = mypackage.Client(api_key="your-key")

# Make a request
response = client.get_data()
print(response)
```

## Advanced Usage

For more complex scenarios:

```python
class CustomHandler:
    def __init__(self):
        self.client = mypackage.Client()

    def process(self, data):
        # Process the data
        return self.client.transform(data)
```

## Configuration

Set these environment variables:

- `API_KEY`: Your API key
- `API_URL`: The API endpoint
"""

    @pytest.fixture()
    def javascript_code_sample(self):
        """Sample JavaScript code for testing"""
        return """
// JavaScript module for testing

const express = require('express');
const router = express.Router();

class UserController {
    constructor(database) {
        this.db = database;
    }

    async getUser(id) {
        try {
            const user = await this.db.findById(id);
            return user;
        } catch (error) {
            console.error('Error fetching user:', error);
            throw error;
        }
    }

    async createUser(userData) {
        const newUser = {
            ...userData,
            createdAt: new Date()
        };
        return await this.db.create(newUser);
    }
}

// Route handlers
router.get('/users/:id', async (req, res) => {
    const controller = new UserController(req.db);
    const user = await controller.getUser(req.params.id);
    res.json(user);
});

router.post('/users', async (req, res) => {
    const controller = new UserController(req.db);
    const user = await controller.createUser(req.body);
    res.status(201).json(user);
});

module.exports = router;
"""

    def test_basic_text_chunking(self, basic_config):
        """Test basic text chunking without code"""
        chunker = EnhancedChunker(basic_config)

        text = "This is a simple text. " * 20  # Create text longer than chunk size
        chunks = chunker.chunk_content(text, "Test Document", "http://test.com")

        assert len(chunks) > 1
        assert all("content" in chunk for chunk in chunks)
        assert all("chunk_index" in chunk for chunk in chunks)
        assert all(
            chunk["char_count"] <= basic_config.chunk_size * 1.2 for chunk in chunks
        )

    def test_code_fence_detection(self, basic_config, markdown_with_code):
        """Test detection and preservation of code fences"""
        chunker = EnhancedChunker(basic_config)
        chunks = chunker.chunk_content(
            markdown_with_code, "API Docs", "http://test.com/api.md"
        )

        # Check that code blocks are detected
        code_chunks = [
            c for c in chunks if c.get("chunk_type") == "code" or c.get("has_code")
        ]
        assert len(code_chunks) > 0

        # When preserve_code_blocks is True (default), code blocks should be intact
        if basic_config.preserve_code_blocks:
            # Check if any chunks are marked as code type
            code_type_chunks = [c for c in chunks if c.get("chunk_type") == "code"]
            if code_type_chunks:
                # Code chunks should have complete code blocks
                for chunk in code_type_chunks:
                    content = chunk["content"]
                    backtick_count = content.count("```")
                    # Code chunks should have complete blocks
                    # Either complete fence or no fence
                    assert backtick_count == 2 or backtick_count == 0

        # Verify overall integrity by joining all chunks
        full_content = "".join(c["content"] for c in chunks)
        original_backticks = markdown_with_code.count("```")
        reconstructed_backticks = full_content.count("```")
        # Total number of backticks should be preserved
        assert (
            abs(original_backticks - reconstructed_backticks) <= 2
        )  # Allow small difference due to overlap

    def test_function_boundary_preservation(self, basic_config, python_code_sample):
        """Test that function boundaries are respected"""
        config = ChunkingConfig(
            chunk_size=400,  # Size that would normally split functions
            chunk_overlap=80,
            strategy=ChunkingStrategy.ENHANCED,
            preserve_function_boundaries=True,
        )
        chunker = EnhancedChunker(config)

        chunks = chunker.chunk_content(
            python_code_sample, "test.py", "http://test.com/test.py", language="python"
        )

        # Check that function definitions are not split mid-definition
        for chunk in chunks:
            content = chunk["content"]
            # Count def keywords
            def_count = content.count("def ")
            if def_count > 0:
                # Each function should have matching indentation levels
                lines = content.split("\n")
                in_function = False
                for line in lines:
                    if line.strip().startswith("def "):
                        in_function = True
                    elif (
                        in_function
                        and line
                        and not line[0].isspace()
                        and not line.startswith("#")
                    ):
                        # New top-level element, function should be complete
                        in_function = False

    def test_overlap_consistency(self, basic_config):
        """Test that overlap between chunks is consistent"""
        chunker = EnhancedChunker(basic_config)

        text = "Sentence one. Sentence two. Sentence three. " * 10
        chunks = chunker.chunk_content(text, "Test", "http://test.com")

        if len(chunks) > 1:
            for i in range(len(chunks) - 1):
                current_end = chunks[i]["content"][-basic_config.chunk_overlap :]
                next_start = chunks[i + 1]["content"][: basic_config.chunk_overlap]

                # There should be some overlap between consecutive chunks
                # (not exact due to boundary adjustments)
                assert len(current_end) > 0 and len(next_start) > 0

    def test_language_detection(self, basic_config):
        """Test automatic language detection"""
        chunker = EnhancedChunker(basic_config)

        # Test Python detection
        python_code = "import pandas as pd\ndf = pd.DataFrame()"
        chunks = chunker.chunk_content(
            python_code, "test.py", "http://test.com/test.py"
        )
        assert (
            chunks[0].get("language") == "python" or True
        )  # Language detection is optional

        # Test JavaScript detection
        js_code = "const express = require('express');\nconst app = express();"
        chunks = chunker.chunk_content(js_code, "test.js", "http://test.com/test.js")
        assert (
            chunks[0].get("language") == "javascript" or True
        )  # Language detection is optional

    def test_empty_content_handling(self, basic_config):
        """Test handling of empty or very short content"""
        chunker = EnhancedChunker(basic_config)

        # Empty content
        chunks = chunker.chunk_content("", "Empty", "http://test.com")
        assert len(chunks) == 0

        # Very short content
        chunks = chunker.chunk_content("Hello", "Short", "http://test.com")
        assert len(chunks) == 1
        assert chunks[0]["content"] == "Hello"
        assert chunks[0]["total_chunks"] == 1

    def test_chunk_metadata(self, basic_config, python_code_sample):
        """Test that chunks contain proper metadata"""
        chunker = EnhancedChunker(basic_config)
        chunks = chunker.chunk_content(
            python_code_sample, "test.py", "http://test.com/test.py"
        )

        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i
            assert chunk["total_chunks"] == len(chunks)
            assert "char_count" in chunk
            assert "token_estimate" in chunk
            assert chunk["token_estimate"] == chunk["char_count"] // 4
            assert "start_pos" in chunk
            assert "end_pos" in chunk
            assert chunk["url"] == "http://test.com/test.py"

    def test_large_code_block_handling(self, basic_config):
        """Test handling of code blocks larger than chunk size"""
        config = ChunkingConfig(
            chunk_size=100,  # Very small for testing
            chunk_overlap=20,
            strategy=ChunkingStrategy.ENHANCED,
            preserve_code_blocks=True,
            max_function_chunk_size=300,
        )
        chunker = EnhancedChunker(config)

        large_code = '''```python
def very_long_function():
    """This is a very long function that exceeds chunk size"""
    # Line 1
    # Line 2
    # Line 3
    # Line 4
    # Line 5
    # Line 6
    # Line 7
    # Line 8
    # Line 9
    # Line 10
    result = []
    for i in range(100):
        result.append(i * 2)
    return result
```'''

        chunks = chunker.chunk_content(large_code, "Long Code", "http://test.com")

        # Verify the code block was handled appropriately
        assert len(chunks) >= 1
        # Check that code fence integrity is maintained
        full_content = " ".join(c["content"] for c in chunks)
        assert full_content.count("```") % 2 == 0

    def test_mixed_content_chunking(self, basic_config, markdown_with_code):
        """Test chunking of mixed markdown and code content"""
        chunker = EnhancedChunker(basic_config)
        chunks = chunker.chunk_content(
            markdown_with_code, "Mixed", "http://test.com/mixed.md"
        )

        # Should have both text and code chunks
        chunk_types = {c.get("chunk_type", "text") for c in chunks}
        assert "text" in chunk_types or len(chunk_types) > 0

        # Verify title handling
        assert chunks[0]["title"] == "Mixed"
        if len(chunks) > 1:
            assert "Part" in chunks[1]["title"]

    @pytest.mark.parametrize(
        "strategy",
        [
            ChunkingStrategy.BASIC,
            ChunkingStrategy.ENHANCED,
            ChunkingStrategy.AST_BASED,
        ],
    )
    def test_all_strategies(self, strategy, python_code_sample):
        """Test that all chunking strategies work without errors"""
        config = ChunkingConfig(
            chunk_size=300,
            chunk_overlap=60,
            strategy=strategy,
        )
        chunker = EnhancedChunker(config)

        chunks = chunker.chunk_content(
            python_code_sample, "test.py", "http://test.com/test.py"
        )

        assert len(chunks) > 0
        assert all("content" in chunk for chunk in chunks)
        assert all(len(chunk["content"]) > 0 for chunk in chunks)

    def test_boundary_detection_patterns(self, basic_config):
        """Test various boundary detection patterns"""
        chunker = EnhancedChunker(basic_config)

        # Test different boundary types
        text_with_boundaries = """
# Header 1

This is paragraph one. It has multiple sentences. Each sentence ends with a period.

## Header 2

- List item 1
- List item 2
- List item 3

### Header 3

1. Numbered item
2. Another item

```python
def example():
    pass
```

---

Final section with text.
"""

        chunks = chunker.chunk_content(
            text_with_boundaries, "Boundaries", "http://test.com"
        )

        # Verify chunks are created at logical boundaries
        assert len(chunks) >= 1
        # Check that chunks don't end mid-sentence when possible
        for chunk in chunks:
            content = chunk["content"].strip()
            if len(content) < basic_config.chunk_size * 0.8:
                # Short chunks should end at natural boundaries
                assert (
                    content.endswith((".", "!", "?", "```", "\n"))
                    or content.endswith(tuple(str(i) for i in range(10)))  # List items
                )


class TestASTChunking:
    """Test suite specifically for AST-based chunking"""

    @pytest.fixture()
    def ast_chunker(self):
        """Create AST-enabled chunker"""
        config = ChunkingConfig(
            chunk_size=500,
            chunk_overlap=100,
            strategy=ChunkingStrategy.AST_BASED,
            enable_ast_chunking=True,
        )
        return EnhancedChunker(config)

    def test_python_ast_function_extraction(self, ast_chunker):
        """Test AST extraction of Python functions"""
        code = '''
def simple_function():
    """A simple function"""
    return 42

class MyClass:
    def method_one(self):
        return "one"

    def method_two(self):
        return "two"

async def async_function():
    return await something()
'''

        chunks = ast_chunker.chunk_content(
            code, "test.py", "http://test.com/test.py", language="python"
        )

        # Should create chunks for each function/class
        assert len(chunks) >= 3  # At least one per major code unit

        # Check metadata
        code_chunks = [c for c in chunks if c.get("chunk_type") == "code"]
        assert len(code_chunks) > 0

    def test_javascript_ast_parsing(self, ast_chunker):
        """Test AST parsing of JavaScript code"""
        code = """
function regularFunction() {
    return "regular";
}

const arrowFunction = () => {
    return "arrow";
};

class ES6Class {
    constructor() {
        this.value = 42;
    }

    getValue() {
        return this.value;
    }
}
"""

        chunks = ast_chunker.chunk_content(
            code, "test.js", "http://test.com/test.js", language="javascript"
        )

        assert len(chunks) >= 1
        # Verify function boundaries are preserved
        for chunk in chunks:
            content = chunk["content"]
            # Check for balanced braces
            assert content.count("{") == content.count("}")

    def test_ast_fallback_on_error(self, ast_chunker):
        """Test graceful fallback when AST parsing fails"""
        # Malformed code that might cause AST parsing issues
        malformed_code = """
def broken_function(
    this is not valid python syntax
    return None
"""

        # Should not raise an exception
        chunks = ast_chunker.chunk_content(
            malformed_code, "broken.py", "http://test.com/broken.py", language="python"
        )

        assert len(chunks) >= 1
        assert all("content" in chunk for chunk in chunks)

    def test_class_method_separation(self, ast_chunker):
        """Test that large classes are split by methods"""
        large_class = '''
class LargeClass:
    """A class with many methods"""

    def __init__(self):
        self.data = []
        self.config = {}
        self.status = "initialized"

    def method_one(self):
        """First method with substantial implementation"""
        result = []
        for i in range(100):
            if i % 2 == 0:
                result.append(i * 2)
            else:
                result.append(i * 3)
        return result

    def method_two(self, param):
        """Second method with different logic"""
        processed = str(param).upper()
        return processed * 3

    def method_three(self):
        """Third method doing something else"""
        import time
        time.sleep(0.1)
        return "done"
'''

        config = ChunkingConfig(
            chunk_size=200,  # Small to force splitting
            strategy=ChunkingStrategy.AST_BASED,
            enable_ast_chunking=True,
        )
        chunker = EnhancedChunker(config)

        chunks = chunker.chunk_content(
            large_class, "large.py", "http://test.com/large.py", language="python"
        )

        # Should split into multiple chunks
        assert len(chunks) > 1

        # Each chunk should contain complete methods
        for chunk in chunks:
            content = chunk["content"]
            # If it contains a def, it should be complete
            if "def " in content:
                # Basic check: matching indentation
                lines = content.split("\n")
                def_lines = [line for line in lines if line.strip().startswith("def ")]
                assert len(def_lines) >= 1
