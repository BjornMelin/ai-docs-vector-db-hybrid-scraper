"""Portfolio-worthy examples showcasing the progressive API design.

This module demonstrates the sophisticated API design through practical
examples that show both simplicity and advanced capabilities.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime, timedelta

from .simple import AIDocSystem, SimpleSearchResult
from .builders import (
    AIDocSystemBuilder,
    AdvancedConfigBuilder,
    SearchConfigBuilder,
    EmbeddingConfigBuilder,
)
from .types import SearchOptions, SearchStrategy, QualityTier, ContentType
from .factory import discover_features, create_system, get_quick_examples
from .protocols import SearchProtocol, EmbeddingProtocol


logger = logging.getLogger(__name__)


class ProgressiveAPIShowcase:
    """Portfolio showcase of progressive API design patterns.
    
    This class demonstrates the sophistication of the API through
    practical examples that progress from simple to advanced usage.
    """
    
    @staticmethod
    async def example_30_second_success():
        """Demonstrate 30-second success pattern.
        
        This example shows how users can get immediate value
        with minimal setup and no configuration.
        """
        print("=== 30-Second Success Pattern ===")
        
        # One line setup - everything works immediately
        system = AIDocSystem.quick_start()
        
        async with system:
            # Add some sample content
            await system.add_document(
                content="Machine learning is a subset of artificial intelligence that enables computers to learn without explicit programming.",
                title="What is Machine Learning?",
                metadata={"category": "AI", "difficulty": "beginner"}
            )
            
            # Immediate search results
            results = await system.search("artificial intelligence")
            
            print(f"Found {len(results)} results:")
            for result in results:
                print(f"  - {result.title}: {result.content[:50]}...")
                
            # Progressive feature discovery
            features = system.discover_features()
            print(f"\nAvailable next steps: {features['next_steps']}")
    
    @staticmethod
    async def example_progressive_builders():
        """Demonstrate progressive builder patterns.
        
        This example shows how the builder pattern reveals
        sophisticated features through natural progression.
        """
        print("\n=== Progressive Builder Patterns ===")
        
        # Start simple, add features progressively
        system = (AIDocSystemBuilder()
            .with_embedding_provider("fastembed")  # Choose provider
            .with_quality_tier("balanced")         # Set quality
            .with_cache(                           # Add caching
                enabled=True,
                ttl=3600,
                max_size=1000
            )
            .with_monitoring(                      # Add monitoring
                enabled=True,
                track_costs=False,
                budget_limit=None
            )
            .build())
        
        async with system:
            # Use advanced search options
            search_options = (SearchConfigBuilder()
                .with_strategy("hybrid")
                .with_reranking(enabled=True)
                .with_similarity_threshold(0.7)
                .with_embeddings(include=True)
                .with_analysis(include=True)
                .build())
            
            # Sophisticated search
            results = await system.search(
                "machine learning algorithms",
                limit=5,
                options=search_options
            )
            
            # Access progressive features
            for result in results:
                analysis = result.get_analysis()
                suggestions = result.get_suggestions()
                print(f"Result: {result.title}")
                print(f"  Analysis: {analysis}")
                print(f"  Suggestions: {suggestions}")
            
            # System statistics
            stats = await system.get_stats()
            print(f"\nSystem Stats: {stats}")
    
    @staticmethod
    async def example_advanced_configuration():
        """Demonstrate advanced configuration patterns.
        
        This example shows the full power of the configuration
        system for expert users.
        """
        print("\n=== Advanced Configuration Patterns ===")
        
        # Expert-level configuration
        system = (AIDocSystemBuilder()
            .with_advanced_config(
                lambda config: config
                    .with_embedding_config(
                        lambda emb: emb
                            .with_provider("openai")
                            .with_model("text-embedding-3-large")
                            .with_batch_size(64)
                            .with_custom_preprocessing({
                                "clean_html": True,
                                "extract_code": True,
                                "normalize_whitespace": True
                            })
                    )
                    .with_search_config(
                        lambda search: search
                            .with_strategy("semantic")
                            .with_reranking(enabled=True)
                            .with_custom_weights({
                                "semantic": 0.7,
                                "keyword": 0.2,
                                "freshness": 0.1
                            })
                            .with_fusion_algorithm("reciprocal_rank")
                    )
                    .with_cache_config(
                        lambda cache: cache
                            .with_caching(enabled=True)
                            .with_ttl(7200)
                            .with_embedding_cache(enabled=True)
                            .with_compression(enabled=True)
                    )
                    .with_monitoring_config(
                        lambda mon: mon
                            .with_monitoring(enabled=True)
                            .with_performance_tracking(
                                latency=True,
                                throughput=True,
                                errors=True
                            )
                            .with_cost_tracking(
                                enabled=True,
                                alerts=True,
                                budget_limit=100.0
                            )
                            .with_detailed_tracing(enabled=True)
                    )
                    .with_experimental_features({
                        "adaptive_chunking": True,
                        "neural_reranking": True,
                        "query_expansion": True
                    })
                    .with_debug_mode(enabled=True)
            )
            .build())
        
        print("Advanced system configured with:")
        print("  - OpenAI embeddings with custom preprocessing")
        print("  - Semantic search with custom weights")
        print("  - Advanced caching with compression")
        print("  - Full monitoring and cost tracking")
        print("  - Experimental features enabled")
    
    @staticmethod
    async def example_batch_processing():
        """Demonstrate sophisticated batch processing.
        
        This example shows how to efficiently process
        large amounts of content with progress tracking.
        """
        print("\n=== Batch Processing Example ===")
        
        system = (AIDocSystemBuilder()
            .with_embedding_provider("fastembed")
            .with_cache(enabled=True, ttl=3600)
            .build())
        
        async with system:
            # Sample URLs for demonstration
            urls = [
                "https://docs.python.org/3/tutorial/",
                "https://fastapi.tiangolo.com/tutorial/",
                "https://docs.pydantic.dev/latest/",
            ]
            
            # Progress tracking callback
            def progress_callback(progress: float, message: str):
                print(f"Progress: {progress:.1%} - {message}")
            
            # Batch processing with progress
            document_ids = await system.add_documents_from_urls(
                urls,
                batch_size=2,
                progress_callback=progress_callback
            )
            
            print(f"Processed {len(document_ids)} documents")
            
            # Batch search across all documents
            search_queries = [
                "python functions",
                "async programming", 
                "data validation"
            ]
            
            print("\nBatch search results:")
            for query in search_queries:
                results = await system.search(query, limit=3)
                print(f"\nQuery: '{query}'")
                for i, result in enumerate(results, 1):
                    print(f"  {i}. {result.title} (score: {result.score:.3f})")
    
    @staticmethod
    async def example_feature_discovery():
        """Demonstrate feature discovery capabilities.
        
        This example shows how users can discover and learn
        about available features programmatically.
        """
        print("\n=== Feature Discovery Example ===")
        
        # Discover system capabilities
        discovery = discover_features()
        
        # Basic features for new users
        basic_features = discovery.get_basic_features()
        print("Basic Features:")
        for feature in basic_features:
            print(f"  - {feature.name}: {feature.description}")
            print(f"    Example: {feature.example}")
        
        # Progressive features for growing users
        progressive_features = discovery.get_progressive_features()
        print(f"\nProgressive Features ({len(progressive_features)} available):")
        for feature in progressive_features[:3]:  # Show first 3
            print(f"  - {feature.name}: {feature.description}")
        
        # Provider discovery
        providers = discovery.discover_embedding_providers()
        print(f"\nAvailable Providers:")
        for provider in providers:
            print(f"  - {provider.name}: {provider.description}")
            print(f"    Cost: {provider.cost_model}, Performance: {provider.performance_tier.value}")
        
        # Learning path
        learning_path = discovery.get_learning_path("intermediate")
        print(f"\nRecommended learning path:")
        for i, step in enumerate(learning_path, 1):
            print(f"  {i}. {step}")
    
    @staticmethod
    async def example_error_handling():
        """Demonstrate sophisticated error handling patterns.
        
        This example shows how the API provides helpful
        error messages and recovery suggestions.
        """
        print("\n=== Error Handling Example ===")
        
        try:
            # This will fail - system not initialized
            system = AIDocSystem()
            await system.search("test query")
        except RuntimeError as e:
            print(f"Expected error: {e}")
            print("  → Solution: Use 'async with system:' or call 'await system.initialize()'")
        
        try:
            # This will fail - invalid provider
            system = create_system(provider="invalid_provider")
        except Exception as e:
            print(f"Configuration error: {e}")
            print("  → Solution: Use discovery.discover_embedding_providers() to see options")
        
        # Graceful degradation example
        system = AIDocSystem.quick_start()
        async with system:
            # Search with fallback behavior
            try:
                # Try advanced search first
                options = SearchOptions(
                    strategy=SearchStrategy.SEMANTIC,
                    rerank=True,
                    similarity_threshold=0.9
                )
                results = await system.search("test", options=options)
                if not results:
                    print("No results with advanced options, trying basic search...")
                    results = await system.search("test")
                    
            except Exception as e:
                print(f"Advanced search failed: {e}")
                print("Falling back to basic search...")
                results = await system.search("test")
            
            print(f"Final results: {len(results)} found")
    
    @staticmethod
    async def example_custom_protocols():
        """Demonstrate custom protocol implementation.
        
        This example shows how expert users can implement
        custom search and embedding providers.
        """
        print("\n=== Custom Protocol Example ===")
        
        class CustomSearchProvider:
            """Example custom search provider implementation."""
            
            def __init__(self):
                self._documents = {}
                self._next_id = 1
            
            async def search(
                self,
                query: str,
                *,
                limit: int = 10,
                options = None,
            ) -> List[Dict[str, Any]]:
                """Custom search implementation."""
                # Simple keyword matching for demo
                results = []
                for doc_id, doc in self._documents.items():
                    if query.lower() in doc["content"].lower():
                        results.append({
                            "content": doc["content"],
                            "title": doc.get("title", ""),
                            "score": 0.5,  # Simple scoring
                            "metadata": doc.get("metadata", {})
                        })
                
                return results[:limit]
            
            async def add_document(
                self,
                content: str,
                *,
                metadata = None,
            ) -> str:
                """Add document to custom provider."""
                doc_id = str(self._next_id)
                self._next_id += 1
                
                self._documents[doc_id] = {
                    "content": content,
                    "metadata": metadata or {}
                }
                
                return doc_id
            
            async def get_stats(self) -> Dict[str, Any]:
                """Get provider statistics."""
                return {
                    "total_documents": len(self._documents),
                    "provider_type": "custom",
                }
        
        # Create and use custom provider
        custom_provider = CustomSearchProvider()
        
        # Add documents
        await custom_provider.add_document(
            content="This is a custom document about machine learning",
            metadata={"source": "custom"}
        )
        
        # Search
        results = await custom_provider.search("machine learning")
        print(f"Custom provider found {len(results)} results")
        
        stats = await custom_provider.get_stats()
        print(f"Custom provider stats: {stats}")
    
    @staticmethod
    async def run_all_examples():
        """Run all examples to demonstrate the full API capabilities."""
        print("🚀 Progressive API Design Showcase")
        print("=" * 50)
        
        examples = [
            ProgressiveAPIShowcase.example_30_second_success,
            ProgressiveAPIShowcase.example_progressive_builders,
            ProgressiveAPIShowcase.example_advanced_configuration,
            ProgressiveAPIShowcase.example_feature_discovery,
            ProgressiveAPIShowcase.example_error_handling,
            ProgressiveAPIShowcase.example_custom_protocols,
        ]
        
        for example in examples:
            try:
                await example()
                await asyncio.sleep(0.1)  # Brief pause between examples
            except Exception as e:
                logger.error(f"Example {example.__name__} failed: {e}")
                print(f"⚠️  Example failed: {e}")
        
        print("\n✅ Progressive API Showcase Complete!")
        print("\nKey Design Principles Demonstrated:")
        print("  ✓ 30-second success with one-line setup")
        print("  ✓ Progressive disclosure through builders")
        print("  ✓ Advanced configuration for experts")
        print("  ✓ Feature discovery and learning paths")
        print("  ✓ Sophisticated error handling")
        print("  ✓ Custom protocol implementation")
        print("  ✓ Enterprise-level type safety")
        print("  ✓ Clean dependency injection patterns")


# Quick examples for documentation
QUICK_EXAMPLES = {
    "basic_usage": '''
# 30-second setup and search
system = AIDocSystem.quick_start()
async with system:
    results = await system.search("machine learning")
    print(f"Found {len(results)} results")
    ''',
    
    "progressive_builder": '''
# Progressive configuration
system = (AIDocSystem.builder()
    .with_embedding_provider("openai")
    .with_cache(enabled=True, ttl=3600)
    .with_monitoring(enabled=True)
    .build())
    ''',
    
    "advanced_search": '''
# Sophisticated search options
options = SearchOptions(
    strategy=SearchStrategy.SEMANTIC,
    rerank=True,
    include_embeddings=True,
    similarity_threshold=0.7
)
results = await system.search("query", options=options)
    ''',
    
    "feature_discovery": '''
# Discover available features
discovery = discover_features()
providers = discovery.discover_embedding_providers()
path = discovery.get_learning_path("intermediate")
    ''',
}


if __name__ == "__main__":
    # Run the showcase if this file is executed directly
    asyncio.run(ProgressiveAPIShowcase.run_all_examples())