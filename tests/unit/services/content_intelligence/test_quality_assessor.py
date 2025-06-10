"""Tests for content intelligence quality assessor."""

from unittest.mock import AsyncMock, MagicMock
import pytest
from datetime import datetime, timezone

from src.services.content_intelligence.quality_assessor import QualityAssessor
from src.services.content_intelligence.models import QualityScore


@pytest.fixture
def mock_embedding_manager():
    """Create mock embedding manager."""
    manager = AsyncMock()
    manager.generate_embeddings = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    return manager


@pytest.fixture
def quality_assessor(mock_embedding_manager):
    """Create QualityAssessor instance with mocked dependencies."""
    return QualityAssessor(embedding_manager=mock_embedding_manager)


class TestQualityAssessor:
    """Test QualityAssessor functionality."""

    async def test_initialize(self, quality_assessor):
        """Test quality assessor initialization."""
        await quality_assessor.initialize()
        assert quality_assessor._initialized is True

    async def test_assess_high_quality_content(self, quality_assessor):
        """Test assessment of high-quality content."""
        content = """
        # Complete Guide to Machine Learning
        
        Machine learning is a subset of artificial intelligence that enables computers to learn and make 
        decisions without being explicitly programmed. This comprehensive guide covers all the essential 
        concepts you need to understand.
        
        ## Table of Contents
        1. Introduction to Machine Learning
        2. Types of Machine Learning
        3. Popular Algorithms
        4. Implementation Examples
        5. Best Practices
        
        ## Introduction to Machine Learning
        
        Machine learning algorithms build mathematical models based on training data to make predictions 
        or decisions without being explicitly programmed to do so. The field has applications in computer 
        vision, natural language processing, email filtering, and many other areas.
        
        ### Supervised Learning
        Supervised learning uses labeled training data to learn a mapping from inputs to outputs.
        
        ### Unsupervised Learning
        Unsupervised learning finds hidden patterns in data without labeled examples.
        
        ### Reinforcement Learning
        Reinforcement learning learns optimal actions through trial and error interactions with an environment.
        
        ## Popular Algorithms
        
        ### Linear Regression
        Linear regression models the relationship between variables using linear equations.
        
        ### Decision Trees
        Decision trees use a tree-like model to make decisions based on feature values.
        
        ### Neural Networks
        Neural networks are inspired by biological neural networks and consist of interconnected nodes.
        
        This content was last updated on March 15, 2024.
        """
        
        await quality_assessor.initialize()
        result = await quality_assessor.assess_quality(content)
        
        assert isinstance(result, QualityScore)
        assert result.overall_score > 0.7  # High quality
        assert result.completeness > 0.8  # Well-structured and complete
        assert result.structure_quality > 0.8  # Good organization with headings
        assert result.readability > 0.7  # Clear and readable
        assert result.meets_threshold is True

    async def test_assess_low_quality_content(self, quality_assessor):
        """Test assessment of low-quality content."""
        content = """
        this is some text without much structure or useful information it doesnt have proper 
        capitalization or punctuation and its very hard to read because its just one long 
        sentence that goes on and on without any breaks or formatting
        """
        
        await quality_assessor.initialize()
        result = await quality_assessor.assess_quality(content)
        
        assert isinstance(result, QualityScore)
        assert result.overall_score < 0.6  # Low quality
        assert result.structure_quality < 0.5  # Poor structure
        assert result.readability < 0.6  # Poor readability
        assert len(result.quality_issues) > 0  # Should identify issues
        assert len(result.improvement_suggestions) > 0  # Should provide suggestions

    async def test_assess_incomplete_content(self, quality_assessor):
        """Test assessment of incomplete content."""
        content = "This is a very short"  # Incomplete sentence
        
        await quality_assessor.initialize()
        result = await quality_assessor.assess_quality(content)
        
        assert result.completeness < 0.5  # Low completeness
        assert "incomplete" in str(result.quality_issues).lower() or "short" in str(result.quality_issues).lower()

    async def test_assess_with_confidence_threshold(self, quality_assessor):
        """Test assessment with different confidence thresholds."""
        content = """
        # Medium Quality Content
        
        This content has decent structure and length, but could be improved.
        It has some good points but lacks depth in certain areas.
        
        The formatting is okay and it's readable, but not exceptional.
        """
        
        await quality_assessor.initialize()
        
        # Test with low threshold
        result_low = await quality_assessor.assess_quality(content, confidence_threshold=0.5)
        
        # Test with high threshold
        result_high = await quality_assessor.assess_quality(content, confidence_threshold=0.9)
        
        assert result_low.threshold == 0.5
        assert result_high.threshold == 0.9
        assert result_low.meets_threshold != result_high.meets_threshold  # Different thresholds should give different results

    async def test_assess_with_query_context(self, quality_assessor):
        """Test assessment with query context for relevance scoring."""
        content = """
        # Python Programming Guide
        
        Python is a high-level programming language known for its simplicity and readability.
        It's widely used in web development, data science, and automation.
        
        ## Basic Syntax
        
        Variables in Python are created by assignment:
        ```python
        x = 5
        name = "Alice"
        ```
        
        ## Data Types
        
        Python has several built-in data types:
        - Integers (int)
        - Floating-point numbers (float)
        - Strings (str)
        - Lists (list)
        - Dictionaries (dict)
        """
        
        await quality_assessor.initialize()
        
        # Test with relevant query
        result_relevant = await quality_assessor.assess_quality(
            content, 
            query_context="Python programming basics"
        )
        
        # Test with irrelevant query
        result_irrelevant = await quality_assessor.assess_quality(
            content, 
            query_context="JavaScript web development"
        )
        
        assert result_relevant.relevance > result_irrelevant.relevance

    async def test_assess_with_extraction_metadata(self, quality_assessor):
        """Test assessment with extraction metadata."""
        content = "Content that was extracted successfully."
        
        extraction_metadata = {
            "extraction_method": "crawl4ai",
            "success": True,
            "load_time_ms": 150,
            "status_code": 200,
            "content_length": len(content),
        }
        
        await quality_assessor.initialize()
        result = await quality_assessor.assess_quality(
            content, 
            extraction_metadata=extraction_metadata
        )
        
        assert result.confidence > 0.5  # Good extraction metadata should boost confidence

    async def test_assess_freshness_scoring(self, quality_assessor):
        """Test freshness scoring based on content timestamps."""
        # Recent content
        recent_content = """
        # Latest AI Developments
        
        Published on March 15, 2024
        
        The latest developments in AI are happening rapidly...
        Last updated: March 15, 2024
        """
        
        # Old content
        old_content = """
        # Web Development Guide
        
        Published on January 1, 2010
        
        This guide covers web development as of 2010...
        Last updated: January 1, 2010
        """
        
        await quality_assessor.initialize()
        
        recent_result = await quality_assessor.assess_quality(recent_content)
        old_result = await quality_assessor.assess_quality(old_content)
        
        # Recent content should have higher freshness score
        assert recent_result.freshness > old_result.freshness

    async def test_assess_duplicate_detection(self, quality_assessor):
        """Test duplicate content detection."""
        content = """
        This is some unique content that should be assessed for quality.
        It contains information that is not duplicated elsewhere.
        """
        
        existing_content = [
            "This is some unique content that should be assessed for quality.",
            "Completely different content about something else.",
            "Another piece of unrelated content.",
        ]
        
        await quality_assessor.initialize()
        result = await quality_assessor.assess_quality(
            content, 
            existing_content=existing_content
        )
        
        # Should detect high similarity with first existing content
        assert result.duplicate_similarity > 0.7

    async def test_assess_code_content_quality(self, quality_assessor):
        """Test quality assessment of code content."""
        good_code = """
        def calculate_fibonacci(n: int) -> int:
            '''
            Calculate the nth Fibonacci number using dynamic programming.
            
            Args:
                n: The position in the Fibonacci sequence
                
            Returns:
                The nth Fibonacci number
                
            Raises:
                ValueError: If n is negative
            '''
            if n < 0:
                raise ValueError("n must be non-negative")
            
            if n <= 1:
                return n
            
            a, b = 0, 1
            for _ in range(2, n + 1):
                a, b = b, a + b
            
            return b
        
        # Example usage
        result = calculate_fibonacci(10)
        print(f"The 10th Fibonacci number is: {result}")
        """
        
        bad_code = """
        def fib(n):
            if n<=1:return n
            return fib(n-1)+fib(n-2)
        """
        
        await quality_assessor.initialize()
        
        good_result = await quality_assessor.assess_quality(good_code)
        bad_result = await quality_assessor.assess_quality(bad_code)
        
        # Good code should score higher on structure and readability
        assert good_result.structure_quality > bad_result.structure_quality
        assert good_result.readability > bad_result.readability

    async def test_assess_with_extraction_errors(self, quality_assessor):
        """Test assessment when extraction had errors."""
        content = "Partial content due to extraction issues..."
        
        extraction_metadata = {
            "extraction_method": "crawl4ai",
            "success": False,
            "error": "Timeout during extraction",
            "status_code": 408,
            "partial_content": True,
        }
        
        await quality_assessor.initialize()
        result = await quality_assessor.assess_quality(
            content, 
            extraction_metadata=extraction_metadata
        )
        
        # Poor extraction should reduce confidence
        assert result.confidence < 0.6
        assert "extraction" in str(result.quality_issues).lower()

    async def test_quality_issues_identification(self, quality_assessor):
        """Test identification of specific quality issues."""
        problematic_content = """
        this text has many issues it has no capitalization no punctuation and no structure whatsoever
        also it repeats the same words the same words over and over again
        it also contains some suspicious content like click here for amazing deals!!!
        """
        
        await quality_assessor.initialize()
        result = await quality_assessor.assess_quality(problematic_content)
        
        issues = [issue.lower() for issue in result.quality_issues]
        
        # Should identify various issues
        assert any("structure" in issue or "formatting" in issue for issue in issues)
        assert any("readability" in issue or "capitalization" in issue for issue in issues)
        assert len(result.improvement_suggestions) > 0

    async def test_improvement_suggestions(self, quality_assessor):
        """Test generation of improvement suggestions."""
        improvable_content = """
        some content that could be better
        
        it has minimal structure and could use more detail
        """
        
        await quality_assessor.initialize()
        result = await quality_assessor.assess_quality(improvable_content)
        
        suggestions = [suggestion.lower() for suggestion in result.improvement_suggestions]
        
        # Should provide actionable suggestions
        assert len(suggestions) > 0
        assert any("structure" in suggestion or "detail" in suggestion or "formatting" in suggestion for suggestion in suggestions)

    async def test_assess_not_initialized_error(self, quality_assessor):
        """Test that assessor raises error when not initialized."""
        with pytest.raises(RuntimeError, match="QualityAssessor not initialized"):
            await quality_assessor.assess_quality("test content")

    async def test_embedding_error_handling(self, quality_assessor, mock_embedding_manager):
        """Test handling of embedding generation errors."""
        mock_embedding_manager.generate_embeddings.side_effect = Exception("Embedding error")
        
        await quality_assessor.initialize()
        result = await quality_assessor.assess_quality("test content")
        
        # Should still return a valid QualityScore even with embedding errors
        assert isinstance(result, QualityScore)
        assert result.confidence < 0.8  # Confidence should be reduced due to error

    async def test_cleanup(self, quality_assessor):
        """Test quality assessor cleanup."""
        await quality_assessor.initialize()
        await quality_assessor.cleanup()
        assert quality_assessor._initialized is False

    async def test_quality_metrics_calculation(self, quality_assessor):
        """Test individual quality metrics calculation."""
        content = """
        # Well-Structured Article
        
        This article demonstrates good structure with proper headings, clear paragraphs,
        and comprehensive content coverage.
        
        ## Introduction
        
        The introduction provides context and sets expectations for the reader.
        
        ## Main Content
        
        The main content is well-organized with:
        1. Clear structure
        2. Proper formatting
        3. Comprehensive coverage
        4. Good readability
        
        ## Conclusion
        
        The conclusion summarizes key points effectively.
        """
        
        await quality_assessor.initialize()
        result = await quality_assessor.assess_quality(content)
        
        # Verify all metrics are calculated
        assert 0 <= result.completeness <= 1
        assert 0 <= result.relevance <= 1
        assert 0 <= result.confidence <= 1
        assert 0 <= result.structure_quality <= 1
        assert 0 <= result.readability <= 1
        assert 0 <= result.duplicate_similarity <= 1
        
        # Overall score should be reasonable average
        calculated_average = (
            result.completeness + result.relevance + result.confidence + 
            result.structure_quality + result.readability
        ) / 5
        
        # Overall score should be close to the average of individual metrics
        assert abs(result.overall_score - calculated_average) < 0.2