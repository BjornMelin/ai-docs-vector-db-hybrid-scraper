"""Tests for content intelligence classifiers."""

from unittest.mock import AsyncMock

import pytest

from src.services.content_intelligence.classifiers import ContentClassifier
from src.services.content_intelligence.models import ContentClassification
from src.services.content_intelligence.models import ContentType


@pytest.fixture
def mock_embedding_manager():
    """Create mock embedding manager."""
    manager = AsyncMock()

    # Create a mock embeddings result with proper structure for classifier
    def mock_generate_embeddings(*args, **kwargs):
        # Return different embeddings for different text types
        texts = kwargs.get("texts", args[0] if args else [])
        len(texts)

        # Generate mock embeddings based on content
        embeddings = []
        for i, text in enumerate(texts):
            if i == 0:  # Content embedding - analyze content to determine type
                content = text.lower()
                if "def " in content and "return" in content:
                    # Code content
                    embeddings.append([0.9, 0.1, 0.1])
                elif (
                    "documentation" in content
                    or "api" in content
                    or "getting started" in content
                ):
                    # Documentation content
                    embeddings.append([0.1, 0.9, 0.1])
                elif "tutorial" in content or "step" in content or "guide" in content:
                    # Tutorial content
                    embeddings.append([0.1, 0.1, 0.9])
                else:
                    # Default embedding
                    embeddings.append([0.5, 0.5, 0.5])
            else:  # Reference embeddings
                ref_text = text.lower()
                if "code" in ref_text:
                    embeddings.append([0.95, 0.05, 0.05])  # Code reference
                elif "documentation" in ref_text:
                    embeddings.append([0.05, 0.95, 0.05])  # Documentation reference
                elif "tutorial" in ref_text:
                    embeddings.append([0.05, 0.05, 0.95])  # Tutorial reference
                else:
                    embeddings.append([0.1, 0.1, 0.1])  # Other reference

        return {"success": True, "embeddings": embeddings}

    manager.generate_embeddings = AsyncMock(side_effect=mock_generate_embeddings)
    return manager


@pytest.fixture
def classifier(mock_embedding_manager):
    """Create ContentClassifier instance with mocked dependencies."""
    return ContentClassifier(embedding_manager=mock_embedding_manager)


class TestContentClassifier:
    """Test ContentClassifier functionality."""

    async def test_initialize(self, classifier):
        """Test classifier initialization."""
        await classifier.initialize()
        assert classifier._initialized is True

    async def test_classify_documentation_content(self, classifier):
        """Test classification of documentation content."""
        content = """
        # API Documentation

        ## Getting Started

        This guide will help you get started with our API.

        ### Authentication

        To authenticate, include your API key in the header:

        ```
        Authorization: Bearer your-api-key
        ```

        ### Making Requests

        All requests should be made to the base URL: https://api.example.com
        """
        url = "https://docs.example.com/api"

        await classifier.initialize()
        result = await classifier.classify_content(content, url)

        assert isinstance(result, ContentClassification)
        assert result.primary_type == ContentType.DOCUMENTATION
        assert ContentType.DOCUMENTATION in result.confidence_scores
        assert result.confidence_scores[ContentType.DOCUMENTATION] > 0.5

    async def test_classify_code_content(self, classifier):
        """Test classification of code content."""
        content = """
        def calculate_fibonacci(n):
            '''Calculate the nth Fibonacci number.'''
            if n <= 1:
                return n
            return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

        class DataProcessor:
            def __init__(self, config):
                self.config = config
                self.data = []

            def process(self, input_data):
                # Process the input data
                for item in input_data:
                    self.data.append(self.transform(item))
                return self.data
        """
        url = "https://github.com/user/repo/blob/main/utils.py"

        await classifier.initialize()
        result = await classifier.classify_content(content, url)

        assert result.primary_type == ContentType.CODE
        assert ContentType.CODE in result.confidence_scores
        assert result.confidence_scores[ContentType.CODE] > 0.5

    async def test_classify_faq_content(self, classifier):
        """Test classification of FAQ content."""
        content = """
        # Frequently Asked Questions

        ## Q: How do I reset my password?
        A: Click on "Forgot Password" on the login page and follow the instructions.

        ## Q: What payment methods do you accept?
        A: We accept credit cards, PayPal, and bank transfers.

        ## Q: How can I contact support?
        A: You can reach our support team at support@example.com or through the chat widget.

        ## Q: Is there a mobile app available?
        A: Yes, our mobile app is available on both iOS and Android.
        """
        url = "https://example.com/faq"

        await classifier.initialize()
        result = await classifier.classify_content(content, url)

        assert result.primary_type == ContentType.FAQ
        assert ContentType.FAQ in result.confidence_scores

    async def test_classify_tutorial_content(self, classifier):
        """Test classification of tutorial content."""
        content = """
        # Step-by-Step Guide: Building Your First React App

        In this tutorial, we'll walk through creating a simple React application.

        ## Step 1: Set up your development environment

        First, make sure you have Node.js installed on your computer.

        ## Step 2: Create a new React project

        Run the following command in your terminal:
        ```
        npx create-react-app my-first-app
        ```

        ## Step 3: Navigate to your project directory

        Change into the project directory:
        ```
        cd my-first-app
        ```

        ## Step 4: Start the development server

        Run the development server:
        ```
        npm start
        ```

        Congratulations! You've created your first React app.
        """
        url = "https://blog.example.com/react-tutorial"

        await classifier.initialize()
        result = await classifier.classify_content(content, url)

        assert result.primary_type == ContentType.TUTORIAL
        assert ContentType.TUTORIAL in result.confidence_scores

    async def test_classify_blog_content(self, classifier):
        """Test classification of blog content."""
        content = """
        # The Future of Artificial Intelligence in Healthcare

        Published on March 15, 2024 by Dr. Jane Smith

        Artificial Intelligence is revolutionizing the healthcare industry in ways we never imagined.
        From diagnostic imaging to drug discovery, AI is making healthcare more precise, efficient, and accessible.

        In this post, I'll explore the latest developments and share my thoughts on what lies ahead.

        ## Current Applications

        AI is already being used in several areas of healthcare:
        - Medical imaging analysis
        - Drug discovery and development
        - Personalized treatment plans
        - Predictive analytics for patient outcomes

        What do you think about these developments? Share your thoughts in the comments below.
        """
        url = "https://healthblog.example.com/ai-in-healthcare"

        await classifier.initialize()
        result = await classifier.classify_content(content, url)

        assert result.primary_type == ContentType.BLOG
        assert ContentType.BLOG in result.confidence_scores

    async def test_classify_forum_content(self, classifier):
        """Test classification of forum content."""
        content = """
        **Topic: Need help with Python error - TypeError: 'NoneType' object is not subscriptable**

        Posted by user123 - 2 hours ago

        Hi everyone, I'm getting this error in my Python code and can't figure out what's wrong:

        ```
        TypeError: 'NoneType' object is not subscriptable
        ```

        Here's my code:
        ```python
        def get_data():
            # some code here
            return result[0]
        ```

        ---

        **Reply by expert_dev - 1 hour ago**

        This error occurs when you're trying to access an element of a variable that is None.
        Check if your `result` variable is actually returning data.

        ---

        **Reply by user123 - 30 minutes ago**

        @expert_dev Thanks! That was exactly the issue. I was not handling the case where the function returns None.
        """
        url = "https://forum.example.com/thread/12345"

        await classifier.initialize()
        result = await classifier.classify_content(content, url)

        assert result.primary_type == ContentType.FORUM
        assert ContentType.FORUM in result.confidence_scores

    async def test_classify_news_content(self, classifier):
        """Test classification of news content."""
        content = """
        # Tech Giant Announces Breakthrough in Quantum Computing

        NEW YORK, March 15, 2024 - A major technology company announced today that it has achieved
        a significant breakthrough in quantum computing, potentially bringing quantum computers
        closer to mainstream adoption.

        The announcement was made during the company's annual developer conference, where CEO John Doe
        revealed that their new quantum processor can perform certain calculations 1000 times faster
        than traditional computers.

        "This represents a major milestone in our quantum computing research," said Doe during his keynote speech.

        Industry experts are calling this development a game-changer for fields including cryptography,
        drug discovery, and financial modeling.

        The company's stock price rose 5% following the announcement.
        """
        url = "https://news.example.com/tech-breakthrough"

        await classifier.initialize()
        result = await classifier.classify_content(content, url)

        assert result.primary_type == ContentType.NEWS
        assert ContentType.NEWS in result.confidence_scores

    async def test_classify_with_url_hints(self, classifier):
        """Test that URL patterns influence classification."""
        content = "Some generic content that could be anything."

        await classifier.initialize()

        # Test documentation URL
        doc_result = await classifier.classify_content(
            content, "https://docs.example.com/guide"
        )
        # Should lean towards documentation due to URL

        # Test GitHub URL
        code_result = await classifier.classify_content(
            content, "https://github.com/user/repo/blob/main/file.py"
        )
        # Should lean towards code due to URL

        # Test blog URL
        blog_result = await classifier.classify_content(
            content, "https://blog.example.com/post"
        )
        # Should lean towards blog due to URL

        assert isinstance(doc_result, ContentClassification)
        assert isinstance(code_result, ContentClassification)
        assert isinstance(blog_result, ContentClassification)

    async def test_classify_mixed_content(self, classifier):
        """Test classification of content with mixed characteristics."""
        content = """
        # How to Implement Authentication in Your API

        This tutorial will show you how to add authentication to your REST API.

        ## Step 1: Install the required packages

        ```bash
        npm install jsonwebtoken bcryptjs
        ```

        ## Step 2: Create the authentication middleware

        ```javascript
        const jwt = require('jsonwebtoken');

        function authenticateToken(req, res, next) {
            const authHeader = req.headers['authorization'];
            const token = authHeader && authHeader.split(' ')[1];

            if (token == null) return res.sendStatus(401);

            jwt.verify(token, process.env.ACCESS_TOKEN_SECRET, (err, user) => {
                if (err) return res.sendStatus(403);
                req.user = user;
                next();
            });
        }
        ```

        This middleware checks for a valid JWT token in the Authorization header.
        """
        url = "https://tutorial.example.com/api-auth"

        await classifier.initialize()
        result = await classifier.classify_content(content, url)

        # Should classify as tutorial (primary) with code as secondary
        assert result.primary_type == ContentType.TUTORIAL
        assert (
            ContentType.CODE in result.secondary_types
            or ContentType.CODE in result.confidence_scores
        )

    async def test_classify_programming_language_detection(self, classifier):
        """Test programming language detection in code content."""
        python_content = """
        class Calculator:
            def add(self, a, b):
                return a + b

            def multiply(self, a, b):
                return a * b

        if __name__ == "__main__":
            calc = Calculator()
            print(calc.add(2, 3))
        """

        javascript_content = """
        class Calculator {
            add(a, b) {
                return a + b;
            }

            multiply(a, b) {
                return a * b;
            }
        }

        const calc = new Calculator();
        console.log(calc.add(2, 3));
        """

        await classifier.initialize()

        python_result = await classifier.classify_content(
            python_content, "https://example.com/code.py"
        )
        javascript_result = await classifier.classify_content(
            javascript_content, "https://example.com/code.js"
        )

        assert python_result.primary_type == ContentType.CODE
        assert javascript_result.primary_type == ContentType.CODE

        # Both should be classified as code but may have different language-specific metadata

    async def test_classify_unknown_content(self, classifier):
        """Test classification of content that doesn't fit clear categories."""
        content = """
        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor
        incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud
        exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
        """
        url = "https://example.com/random"

        await classifier.initialize()
        result = await classifier.classify_content(content, url)

        # Should either classify as UNKNOWN or have low confidence scores
        assert (
            result.primary_type == ContentType.UNKNOWN
            or max(result.confidence_scores.values()) < 0.6
        )

    async def test_classify_empty_content(self, classifier):
        """Test classification with empty or minimal content."""
        await classifier.initialize()

        result = await classifier.classify_content("", "https://example.com")
        assert result.primary_type == ContentType.UNKNOWN

        result = await classifier.classify_content("   ", "https://example.com")
        assert result.primary_type == ContentType.UNKNOWN

    async def test_classify_with_title(self, classifier):
        """Test classification when title is provided."""
        content = "Some content here."
        url = "https://example.com/page"
        title = "API Reference Documentation"

        await classifier.initialize()
        result = await classifier.classify_content(content, url, title)

        # Title should influence classification towards REFERENCE/DOCUMENTATION
        assert result.primary_type in [ContentType.REFERENCE, ContentType.DOCUMENTATION]
        assert result.classification_reasoning is not None

    async def test_classifier_not_initialized(self, classifier):
        """Test that classifier raises error when not initialized."""
        with pytest.raises(RuntimeError, match="ContentClassifier not initialized"):
            await classifier.classify_content("test content", "https://example.com")

    async def test_embedding_generation_error(self, classifier, mock_embedding_manager):
        """Test handling of embedding generation errors."""
        mock_embedding_manager.generate_embeddings.side_effect = Exception(
            "Embedding error"
        )

        await classifier.initialize()
        result = await classifier.classify_content(
            "test content", "https://example.com"
        )

        # Should fall back to heuristic classification
        assert isinstance(result, ContentClassification)
        assert result.primary_type is not None

    async def test_cleanup(self, classifier):
        """Test classifier cleanup."""
        await classifier.initialize()
        await classifier.cleanup()
        assert classifier._initialized is False
