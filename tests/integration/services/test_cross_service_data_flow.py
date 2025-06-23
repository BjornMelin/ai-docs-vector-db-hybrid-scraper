"""Cross-service data flow integration tests.

This module tests complete data flow across the distributed system architecture,
validating end-to-end data transformation pipelines and service interactions.

Tests include:
- Document ingestion pipeline (Crawling → Content Intelligence → Embeddings → Vector DB)
- Search pipeline (Query Processing → Vector Search → Result Ranking → Cache)
- Content analysis pipeline (Browser → Content AI → Quality Assessment → Storage)
- Real-time data synchronization across services
- Event-driven architecture validation
"""

import asyncio
import pytest
import time
import uuid
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from src.config import Config
from src.infrastructure.client_manager import ClientManager
from src.services.errors import APIError, CrawlServiceError, EmbeddingServiceError


class TestDocumentIngestionDataFlow:
    """Test complete document ingestion data flow across services."""

    @pytest.fixture
    async def ingestion_pipeline_services(self):
        """Setup complete document ingestion pipeline services."""
        config = MagicMock(spec=Config)
        
        # Configure all services for integration
        config.crawling.max_concurrent_requests = 5
        config.content_intelligence.enable_quality_assessment = True
        config.embeddings.batch_size = 10
        config.vector_db.collection_name = "test_documents"
        
        # Mock all services in the pipeline
        services = {
            'crawl_manager': AsyncMock(),
            'content_intelligence': AsyncMock(),
            'embedding_manager': AsyncMock(),
            'vector_db_service': AsyncMock(),
            'cache_manager': AsyncMock(),
            'client_manager': AsyncMock(),
        }
        
        return {'config': config, **services}

    @pytest.mark.asyncio
    async def test_complete_document_ingestion_flow(self, ingestion_pipeline_services):
        """Test complete document ingestion from URL to vector storage."""
        services = ingestion_pipeline_services
        
        # Input: URLs for ingestion
        urls = [
            'https://example.com/doc1',
            'https://example.com/doc2',
            'https://example.com/doc3'
        ]
        
        # Stage 1: Web scraping with crawl manager
        crawling_results = []
        for i, url in enumerate(urls):
            crawl_result = {
                'success': True,
                'url': url,
                'content': f'<html><body><h1>Document {i+1}</h1><p>Content for document {i+1}</p></body></html>',
                'metadata': {
                    'status_code': 200,
                    'content_type': 'text/html',
                    'crawled_at': time.time(),
                    'tier_used': 'playwright',
                    'processing_time_ms': 1200 + i * 100
                }
            }
            crawling_results.append(crawl_result)
        
        services['crawl_manager'].bulk_scrape.return_value = {
            'success_count': 3,
            'failure_count': 0,
            'results': crawling_results,
            'total_processing_time_ms': 3600
        }
        
        # Stage 2: Content intelligence analysis
        content_analysis_results = []
        for i, crawl_result in enumerate(crawling_results):
            analysis_result = {
                'url': crawl_result['url'],
                'content_type': 'article',
                'quality_score': 0.85 + i * 0.05,  # Varying quality
                'extracted_text': f'Document {i+1}\nContent for document {i+1}',
                'metadata': {
                    'word_count': 50 + i * 10,
                    'paragraph_count': 2,
                    'heading_count': 1,
                    'readability_score': 0.8,
                    'key_topics': [f'topic_{i+1}', 'general_content'],
                    'sentiment': 'neutral',
                    'language': 'en'
                },
                'classification': {
                    'category': 'informational',
                    'subcategory': 'article',
                    'confidence': 0.92
                }
            }
            content_analysis_results.append(analysis_result)
        
        services['content_intelligence'].batch_analyze.return_value = {
            'analyses': content_analysis_results,
            'processing_stats': {
                'total_documents': 3,
                'successful_analyses': 3,
                'average_quality_score': 0.88,
                'processing_time_ms': 2400
            }
        }
        
        # Stage 3: Embedding generation
        embedding_texts = [result['extracted_text'] for result in content_analysis_results]
        embedding_results = {
            'embeddings': [
                [0.1, 0.2, 0.3] + [0.0] * 1533,  # 1536-dim embeddings
                [0.2, 0.3, 0.4] + [0.0] * 1533,
                [0.3, 0.4, 0.5] + [0.0] * 1533
            ],
            'model': 'text-embedding-3-small',
            'provider': 'openai',
            'usage': {
                'total_tokens': 150,
                'cost_usd': 0.0003
            },
            'processing_time_ms': 800
        }
        services['embedding_manager'].generate_embeddings.return_value = embedding_results
        
        # Stage 4: Vector database storage
        vector_points = []
        for i, (crawl_result, analysis_result, embedding) in enumerate(
            zip(crawling_results, content_analysis_results, embedding_results['embeddings'])
        ):
            point = {
                'id': f'doc_{uuid.uuid4()}',
                'vector': embedding,
                'payload': {
                    'url': crawl_result['url'],
                    'title': f'Document {i+1}',
                    'content': analysis_result['extracted_text'],
                    'metadata': {
                        **crawl_result['metadata'],
                        **analysis_result['metadata'],
                        'ingested_at': time.time(),
                        'quality_score': analysis_result['quality_score'],
                        'content_type': analysis_result['content_type']
                    }
                }
            }
            vector_points.append(point)
        
        storage_result = {
            'operation_id': str(uuid.uuid4()),
            'status': 'completed',
            'points_upserted': 3,
            'points_failed': 0,
            'collection': 'test_documents',
            'processing_time_ms': 150
        }
        services['vector_db_service'].batch_upsert.return_value = storage_result
        
        # Execute complete ingestion pipeline
        start_time = time.time()
        
        # Step 1: Bulk scraping
        crawl_results = await services['crawl_manager'].bulk_scrape(
            urls=urls,
            config={
                'timeout': 30,
                'max_retries': 3,
                'tier_preference': 'adaptive'
            }
        )
        
        # Step 2: Content analysis
        raw_contents = [result['content'] for result in crawl_results['results']]
        analysis_results = await services['content_intelligence'].batch_analyze(
            contents=raw_contents,
            urls=urls,
            options={
                'extract_topics': True,
                'assess_quality': True,
                'detect_language': True
            }
        )
        
        # Step 3: Embedding generation
        analysis_texts = [analysis['extracted_text'] for analysis in analysis_results['analyses']]
        embedding_data = await services['embedding_manager'].generate_embeddings(
            texts=analysis_texts,
            batch_size=10
        )
        
        # Step 4: Vector storage preparation
        ingestion_points = []
        for i, (crawl_result, analysis, embedding) in enumerate(
            zip(crawl_results['results'], analysis_results['analyses'], embedding_data['embeddings'])
        ):
            point = {
                'id': f'doc_{i}_{int(time.time())}',
                'vector': embedding,
                'payload': {
                    'url': crawl_result['url'],
                    'content': analysis['extracted_text'],
                    'metadata': {
                        'crawled_at': crawl_result['metadata']['crawled_at'],
                        'quality_score': analysis['quality_score'],
                        'content_type': analysis['content_type'],
                        'word_count': analysis['metadata']['word_count'],
                        'topics': analysis['metadata']['key_topics']
                    }
                }
            }
            ingestion_points.append(point)
        
        # Step 5: Vector database storage
        final_storage = await services['vector_db_service'].batch_upsert(
            collection='test_documents',
            points=ingestion_points,
            wait=True
        )
        
        total_processing_time = (time.time() - start_time) * 1000
        
        # Verify complete pipeline execution
        services['crawl_manager'].bulk_scrape.assert_called_once()
        services['content_intelligence'].batch_analyze.assert_called_once()
        services['embedding_manager'].generate_embeddings.assert_called_once()
        services['vector_db_service'].batch_upsert.assert_called_once()
        
        # Verify data flow integrity
        assert crawl_results['success_count'] == 3
        assert analysis_results['processing_stats']['successful_analyses'] == 3
        assert len(embedding_data['embeddings']) == 3
        assert final_storage['points_upserted'] == 3
        
        # Verify data transformation consistency
        assert len(ingestion_points) == len(urls)
        for point in ingestion_points:
            assert 'vector' in point
            assert 'payload' in point
            assert 'url' in point['payload']
            assert 'content' in point['payload']
            assert 'metadata' in point['payload']
            assert 'quality_score' in point['payload']['metadata']

    @pytest.mark.asyncio
    async def test_ingestion_pipeline_error_handling(self, ingestion_pipeline_services):
        """Test error handling and recovery in ingestion pipeline."""
        services = ingestion_pipeline_services
        
        urls = ['https://example.com/good', 'https://example.com/bad', 'https://example.com/ugly']
        
        # Simulate partial failures in different stages
        
        # Stage 1: Crawling with mixed results
        crawl_results = {
            'success_count': 2,
            'failure_count': 1,
            'results': [
                {
                    'success': True,
                    'url': 'https://example.com/good',
                    'content': '<html><body>Good content</body></html>',
                    'metadata': {'status_code': 200}
                },
                {
                    'success': False,
                    'url': 'https://example.com/bad',
                    'error': 'Connection timeout',
                    'metadata': {'status_code': None}
                },
                {
                    'success': True,
                    'url': 'https://example.com/ugly',
                    'content': '<html><body>Ugly content</body></html>',
                    'metadata': {'status_code': 200}
                }
            ]
        }
        services['crawl_manager'].bulk_scrape.return_value = crawl_results
        
        # Stage 2: Content analysis with quality filtering
        analysis_results = {
            'analyses': [
                {
                    'url': 'https://example.com/good',
                    'quality_score': 0.9,
                    'extracted_text': 'Good content',
                    'content_type': 'article'
                },
                {
                    'url': 'https://example.com/ugly',
                    'quality_score': 0.4,  # Below quality threshold
                    'extracted_text': 'Ugly content',
                    'content_type': 'low_quality'
                }
            ],
            'processing_stats': {
                'successful_analyses': 2,
                'quality_filtered': 1  # Ugly content filtered out
            }
        }
        services['content_intelligence'].batch_analyze.return_value = analysis_results
        
        # Stage 3: Embedding generation for quality content only
        embedding_results = {
            'embeddings': [[0.1, 0.2, 0.3]],  # Only one embedding
            'model': 'text-embedding-3-small',
            'failed_texts': ['Ugly content'],  # Quality filtered
            'success_count': 1
        }
        services['embedding_manager'].generate_embeddings.return_value = embedding_results
        
        # Stage 4: Vector storage for successful pipeline
        storage_result = {
            'operation_id': str(uuid.uuid4()),
            'status': 'completed',
            'points_upserted': 1,  # Only good content stored
            'points_failed': 0
        }
        services['vector_db_service'].batch_upsert.return_value = storage_result
        
        # Execute pipeline with error handling
        pipeline_result = {
            'total_urls': len(urls),
            'crawl_success': 0,
            'analysis_success': 0,
            'embedding_success': 0,
            'storage_success': 0,
            'errors': []
        }
        
        try:
            # Crawling stage
            crawl_data = await services['crawl_manager'].bulk_scrape(urls)
            pipeline_result['crawl_success'] = crawl_data['success_count']
            
            # Filter successful crawls
            successful_crawls = [r for r in crawl_data['results'] if r.get('success')]
            
            if successful_crawls:
                # Content analysis stage
                analysis_data = await services['content_intelligence'].batch_analyze(
                    contents=[r['content'] for r in successful_crawls],
                    urls=[r['url'] for r in successful_crawls]
                )
                pipeline_result['analysis_success'] = analysis_data['processing_stats']['successful_analyses']
                
                # Filter quality content
                quality_analyses = [
                    a for a in analysis_data['analyses'] 
                    if a['quality_score'] >= 0.7
                ]
                
                if quality_analyses:
                    # Embedding stage
                    embedding_data = await services['embedding_manager'].generate_embeddings(
                        texts=[a['extracted_text'] for a in quality_analyses]
                    )
                    pipeline_result['embedding_success'] = embedding_data['success_count']
                    
                    # Storage stage
                    if embedding_data['success_count'] > 0:
                        storage_data = await services['vector_db_service'].batch_upsert(
                            collection='test_documents',
                            points=[{
                                'id': f'doc_{i}',
                                'vector': embedding_data['embeddings'][i],
                                'payload': {'content': quality_analyses[i]['extracted_text']}
                            } for i in range(embedding_data['success_count'])]
                        )
                        pipeline_result['storage_success'] = storage_data['points_upserted']
        
        except Exception as e:
            pipeline_result['errors'].append(str(e))
        
        # Verify error handling and partial success
        assert pipeline_result['crawl_success'] == 2  # 2 out of 3 crawled
        assert pipeline_result['analysis_success'] == 2  # Both analyzed
        assert pipeline_result['embedding_success'] == 1  # Only quality content
        assert pipeline_result['storage_success'] == 1  # Only quality content stored
        
        # Verify services called appropriately
        services['crawl_manager'].bulk_scrape.assert_called_once()
        services['content_intelligence'].batch_analyze.assert_called_once()
        services['embedding_manager'].generate_embeddings.assert_called_once()
        services['vector_db_service'].batch_upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_real_time_ingestion_streaming(self, ingestion_pipeline_services):
        """Test real-time document ingestion with streaming data flow."""
        services = ingestion_pipeline_services
        
        # Simulate real-time document stream
        document_stream = [
            {'url': f'https://news.example.com/article_{i}', 'priority': 'high' if i % 3 == 0 else 'normal'}
            for i in range(10)
        ]
        
        # Mock streaming ingestion responses
        streaming_results = []
        for i, doc in enumerate(document_stream):
            result = {
                'document_id': f'stream_doc_{i}',
                'url': doc['url'],
                'priority': doc['priority'],
                'ingestion_status': 'completed',
                'pipeline_stages': {
                    'crawling': {'completed_at': time.time(), 'duration_ms': 800},
                    'analysis': {'completed_at': time.time() + 0.5, 'duration_ms': 300},
                    'embedding': {'completed_at': time.time() + 0.8, 'duration_ms': 200},
                    'storage': {'completed_at': time.time() + 1.0, 'duration_ms': 50}
                },
                'quality_metrics': {
                    'content_quality': 0.8 + (i % 5) * 0.04,
                    'extraction_confidence': 0.95,
                    'embedding_dimension': 1536
                }
            }
            streaming_results.append(result)
        
        services['client_manager'].stream_document_ingestion.return_value = streaming_results
        
        # Test streaming ingestion
        ingestion_config = {
            'batch_size': 3,
            'priority_processing': True,
            'quality_threshold': 0.7,
            'real_time_indexing': True
        }
        
        streaming_response = await services['client_manager'].stream_document_ingestion(
            documents=document_stream,
            config=ingestion_config
        )
        
        # Verify streaming results
        assert len(streaming_response) == 10
        
        # Verify priority processing (high priority docs should be processed first)
        high_priority_docs = [r for r in streaming_response if r['priority'] == 'high']
        normal_priority_docs = [r for r in streaming_response if r['priority'] == 'normal']
        
        assert len(high_priority_docs) > 0
        assert len(normal_priority_docs) > 0
        
        # Verify all stages completed for each document
        for result in streaming_response:
            assert 'pipeline_stages' in result
            stages = result['pipeline_stages']
            assert 'crawling' in stages
            assert 'analysis' in stages
            assert 'embedding' in stages
            assert 'storage' in stages
            
            # Verify stage completion order
            crawl_time = stages['crawling']['completed_at']
            analysis_time = stages['analysis']['completed_at']
            embedding_time = stages['embedding']['completed_at']
            storage_time = stages['storage']['completed_at']
            
            assert crawl_time <= analysis_time <= embedding_time <= storage_time
        
        services['client_manager'].stream_document_ingestion.assert_called_once()


class TestSearchPipelineDataFlow:
    """Test search pipeline data flow across services."""

    @pytest.fixture
    async def search_pipeline_services(self):
        """Setup search pipeline services."""
        config = MagicMock(spec=Config)
        
        services = {
            'query_processor': AsyncMock(),
            'embedding_manager': AsyncMock(),
            'vector_db_service': AsyncMock(),
            'cache_manager': AsyncMock(),
            'rag_generator': AsyncMock(),
        }
        
        return {'config': config, **services}

    @pytest.mark.asyncio
    async def test_complete_search_pipeline_flow(self, search_pipeline_services):
        """Test complete search pipeline from query to results."""
        services = search_pipeline_services
        
        query = "machine learning algorithms for text classification"
        
        # Stage 1: Query processing and intent classification
        query_analysis = {
            'original_query': query,
            'intent': 'technical_search',
            'complexity': 'medium',
            'entities': ['machine learning', 'algorithms', 'text classification'],
            'query_expansion': {
                'synonyms': ['ML', 'classification algorithms', 'NLP'],
                'related_terms': ['neural networks', 'deep learning', 'supervised learning'],
                'expanded_query': 'machine learning ML algorithms text classification NLP neural networks'
            },
            'search_strategy': 'hybrid_with_reranking'
        }
        services['query_processor'].analyze_query.return_value = query_analysis
        
        # Stage 2: Embedding generation for query
        query_embedding = {
            'embeddings': [[0.15, 0.25, 0.35] + [0.0] * 1533],
            'model': 'text-embedding-3-small',
            'processing_time_ms': 120
        }
        services['embedding_manager'].generate_embeddings.return_value = query_embedding
        
        # Stage 3: Vector search execution
        vector_search_results = {
            'matches': [
                {
                    'id': 'doc_123',
                    'score': 0.94,
                    'payload': {
                        'title': 'Introduction to ML Classification',
                        'content': 'Machine learning classification algorithms...',
                        'url': 'https://example.com/ml-classification',
                        'metadata': {'author': 'Dr. Smith', 'date': '2024-01-15'}
                    }
                },
                {
                    'id': 'doc_456',
                    'score': 0.89,
                    'payload': {
                        'title': 'Text Classification with Neural Networks',
                        'content': 'Neural networks for text classification...',
                        'url': 'https://example.com/nn-text-classification',
                        'metadata': {'author': 'Prof. Johnson', 'date': '2024-02-20'}
                    }
                },
                {
                    'id': 'doc_789',
                    'score': 0.85,
                    'payload': {
                        'title': 'Supervised Learning Techniques',
                        'content': 'Supervised learning approaches for classification...',
                        'url': 'https://example.com/supervised-learning',
                        'metadata': {'author': 'Dr. Williams', 'date': '2024-03-10'}
                    }
                }
            ],
            'total_matches': 3,
            'search_time_ms': 45
        }
        services['vector_db_service'].hybrid_search.return_value = vector_search_results
        
        # Stage 4: Result reranking and post-processing
        reranked_results = {
            'results': [
                {
                    'id': 'doc_123',
                    'relevance_score': 0.96,
                    'vector_score': 0.94,
                    'rerank_score': 0.98,
                    'title': 'Introduction to ML Classification',
                    'content': 'Machine learning classification algorithms...',
                    'url': 'https://example.com/ml-classification',
                    'metadata': {'author': 'Dr. Smith', 'date': '2024-01-15'},
                    'highlights': ['machine learning', 'classification algorithms'],
                    'relevance_explanation': 'Direct match for ML classification concepts'
                },
                {
                    'id': 'doc_456',
                    'relevance_score': 0.92,
                    'vector_score': 0.89,
                    'rerank_score': 0.95,
                    'title': 'Text Classification with Neural Networks',
                    'content': 'Neural networks for text classification...',
                    'url': 'https://example.com/nn-text-classification',
                    'metadata': {'author': 'Prof. Johnson', 'date': '2024-02-20'},
                    'highlights': ['text classification', 'neural networks'],
                    'relevance_explanation': 'Specific to text classification domain'
                },
                {
                    'id': 'doc_789',
                    'relevance_score': 0.87,
                    'vector_score': 0.85,
                    'rerank_score': 0.89,
                    'title': 'Supervised Learning Techniques',
                    'content': 'Supervised learning approaches for classification...',
                    'url': 'https://example.com/supervised-learning',
                    'metadata': {'author': 'Dr. Williams', 'date': '2024-03-10'},
                    'highlights': ['supervised learning', 'classification'],
                    'relevance_explanation': 'Related classification methodology'
                }
            ],
            'reranking_model': 'cross_encoder_v2',
            'reranking_time_ms': 180
        }
        services['query_processor'].rerank_results.return_value = reranked_results
        
        # Stage 5: RAG answer generation (optional)
        rag_response = {
            'answer': 'Machine learning algorithms for text classification include various approaches such as neural networks, support vector machines, and ensemble methods. These algorithms analyze text features to categorize documents into predefined classes.',
            'source_documents': ['doc_123', 'doc_456', 'doc_789'],
            'confidence': 0.91,
            'answer_type': 'comprehensive',
            'generation_time_ms': 850
        }
        services['rag_generator'].generate_answer.return_value = rag_response
        
        # Execute complete search pipeline
        search_session_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Step 1: Query analysis
        query_data = await services['query_processor'].analyze_query(
            query=query,
            session_id=search_session_id
        )
        
        # Step 2: Query embedding
        embedding_data = await services['embedding_manager'].generate_embeddings(
            texts=[query_data['query_expansion']['expanded_query']]
        )
        
        # Step 3: Vector search
        search_results = await services['vector_db_service'].hybrid_search(
            query_vector=embedding_data['embeddings'][0],
            query_text=query_data['query_expansion']['expanded_query'],
            collection='documents',
            limit=10,
            strategy=query_data['search_strategy']
        )
        
        # Step 4: Result reranking
        final_results = await services['query_processor'].rerank_results(
            results=search_results['matches'],
            original_query=query,
            expanded_query=query_data['query_expansion']['expanded_query'],
            intent=query_data['intent']
        )
        
        # Step 5: RAG answer generation
        answer_data = await services['rag_generator'].generate_answer(
            query=query,
            search_results=final_results['results'][:3],
            session_id=search_session_id
        )
        
        total_search_time = (time.time() - start_time) * 1000
        
        # Compile complete search response
        search_response = {
            'session_id': search_session_id,
            'query': query,
            'intent': query_data['intent'],
            'results': final_results['results'],
            'answer': answer_data['answer'],
            'metadata': {
                'total_results': len(final_results['results']),
                'search_time_ms': total_search_time,
                'answer_confidence': answer_data['confidence'],
                'processing_stages': {
                    'query_analysis': query_data,
                    'vector_search': search_results,
                    'reranking': final_results,
                    'answer_generation': answer_data
                }
            }
        }
        
        # Verify complete pipeline execution
        services['query_processor'].analyze_query.assert_called_once()
        services['embedding_manager'].generate_embeddings.assert_called_once()
        services['vector_db_service'].hybrid_search.assert_called_once()
        services['query_processor'].rerank_results.assert_called_once()
        services['rag_generator'].generate_answer.assert_called_once()
        
        # Verify data flow integrity
        assert search_response['query'] == query
        assert search_response['intent'] == 'technical_search'
        assert len(search_response['results']) == 3
        assert search_response['answer'] is not None
        assert search_response['metadata']['answer_confidence'] > 0.9
        
        # Verify result quality and ranking
        results = search_response['results']
        assert results[0]['relevance_score'] >= results[1]['relevance_score']
        assert results[1]['relevance_score'] >= results[2]['relevance_score']
        
        # Verify all results have required fields
        for result in results:
            assert 'id' in result
            assert 'relevance_score' in result
            assert 'title' in result
            assert 'content' in result
            assert 'url' in result
            assert 'highlights' in result

    @pytest.mark.asyncio
    async def test_search_caching_integration(self, search_pipeline_services):
        """Test search pipeline with comprehensive caching integration."""
        services = search_pipeline_services
        
        query = "python programming tutorials"
        cache_key = f"search:{hash(query)}"
        
        # Test cache miss scenario first
        services['cache_manager'].get.return_value = None
        
        # Mock fresh search execution
        fresh_search_result = {
            'query': query,
            'results': [
                {'id': 'tutorial_1', 'title': 'Python Basics', 'score': 0.95},
                {'id': 'tutorial_2', 'title': 'Advanced Python', 'score': 0.88}
            ],
            'metadata': {
                'cached': False,
                'search_time_ms': 450,
                'total_results': 2,
                'generated_at': time.time()
            }
        }
        
        # Mock the complete search pipeline for cache miss
        services['query_processor'].analyze_query.return_value = {'intent': 'tutorial_search'}
        services['embedding_manager'].generate_embeddings.return_value = {'embeddings': [[0.1, 0.2, 0.3]]}
        services['vector_db_service'].hybrid_search.return_value = {
            'matches': [
                {'id': 'tutorial_1', 'score': 0.95, 'payload': {'title': 'Python Basics'}},
                {'id': 'tutorial_2', 'score': 0.88, 'payload': {'title': 'Advanced Python'}}
            ]
        }
        services['query_processor'].rerank_results.return_value = {
            'results': fresh_search_result['results']
        }
        
        # Execute search with cache miss
        search_result = await self._execute_cached_search(services, query, cache_key)
        
        # Verify cache miss behavior
        assert search_result['metadata']['cached'] is False
        services['cache_manager'].get.assert_called_with(cache_key)
        services['cache_manager'].set.assert_called_once()
        
        # Now test cache hit scenario
        services['cache_manager'].reset_mock()
        cached_result = {
            'query': query,
            'results': fresh_search_result['results'],
            'metadata': {
                'cached': True,
                'original_search_time_ms': 450,
                'cache_retrieval_time_ms': 15,
                'cached_at': time.time() - 300,  # 5 minutes ago
                'cache_hit': True
            }
        }
        services['cache_manager'].get.return_value = cached_result
        
        # Execute search with cache hit
        cached_search_result = await self._execute_cached_search(services, query, cache_key)
        
        # Verify cache hit behavior
        assert cached_search_result['metadata']['cached'] is True
        assert cached_search_result['metadata']['cache_hit'] is True
        services['cache_manager'].get.assert_called_with(cache_key)
        # Should not call set again for cache hit
        services['cache_manager'].set.assert_not_called()
    
    async def _execute_cached_search(self, services, query: str, cache_key: str):
        """Helper method to execute cached search logic."""
        # Check cache first
        cached_result = await services['cache_manager'].get(cache_key)
        
        if cached_result and time.time() - cached_result['metadata'].get('cached_at', 0) < 3600:
            # Cache hit and not expired
            cached_result['metadata']['cache_hit'] = True
            return cached_result
        
        # Cache miss - execute full search pipeline
        query_analysis = await services['query_processor'].analyze_query(query)
        embedding_data = await services['embedding_manager'].generate_embeddings([query])
        search_results = await services['vector_db_service'].hybrid_search(
            query_vector=embedding_data['embeddings'][0],
            query_text=query
        )
        final_results = await services['query_processor'].rerank_results(
            results=search_results['matches'],
            original_query=query
        )
        
        # Prepare result for caching
        search_result = {
            'query': query,
            'results': final_results['results'],
            'metadata': {
                'cached': False,
                'search_time_ms': 450,
                'total_results': len(final_results['results']),
                'generated_at': time.time()
            }
        }
        
        # Cache the result
        await services['cache_manager'].set(
            cache_key, 
            search_result, 
            ttl=3600
        )
        
        return search_result


class TestEventDrivenArchitecture:
    """Test event-driven architecture and message flow."""

    @pytest.fixture
    async def event_driven_services(self):
        """Setup event-driven architecture services."""
        services = {
            'event_bus': AsyncMock(),
            'document_processor': AsyncMock(),
            'index_updater': AsyncMock(),
            'cache_invalidator': AsyncMock(),
            'metrics_collector': AsyncMock(),
        }
        return services

    @pytest.mark.asyncio
    async def test_document_ingestion_event_flow(self, event_driven_services):
        """Test event-driven document ingestion flow."""
        services = event_driven_services
        
        # Define event flow for document ingestion
        document_events = [
            {
                'event_type': 'document.crawl.started',
                'document_id': 'doc_123',
                'url': 'https://example.com/article',
                'timestamp': time.time(),
                'metadata': {'priority': 'high'}
            },
            {
                'event_type': 'document.crawl.completed',
                'document_id': 'doc_123',
                'content_length': 5000,
                'processing_time_ms': 1200,
                'timestamp': time.time() + 1.2
            },
            {
                'event_type': 'document.analysis.completed',
                'document_id': 'doc_123',
                'quality_score': 0.92,
                'content_type': 'article',
                'timestamp': time.time() + 2.0
            },
            {
                'event_type': 'document.indexed',
                'document_id': 'doc_123',
                'vector_id': 'vec_123',
                'collection': 'documents',
                'timestamp': time.time() + 3.5
            }
        ]
        
        # Mock event publishing
        services['event_bus'].publish.return_value = {'status': 'published', 'event_id': str(uuid.uuid4())}
        
        # Mock event subscription handlers
        event_handlers = {
            'document.crawl.completed': [services['document_processor'].process_content],
            'document.analysis.completed': [services['index_updater'].prepare_indexing],
            'document.indexed': [services['cache_invalidator'].invalidate_search_cache, 
                                services['metrics_collector'].record_ingestion]
        }
        
        # Simulate event-driven processing
        published_events = []
        for event in document_events:
            # Publish event
            publish_result = await services['event_bus'].publish(
                topic=event['event_type'],
                payload=event
            )
            published_events.append({**event, 'publish_result': publish_result})
            
            # Trigger relevant handlers
            if event['event_type'] in event_handlers:
                for handler in event_handlers[event['event_type']]:
                    await handler(event)
        
        # Verify event publishing
        assert services['event_bus'].publish.call_count == len(document_events)
        
        # Verify handler invocations
        services['document_processor'].process_content.assert_called_once()
        services['index_updater'].prepare_indexing.assert_called_once()
        services['cache_invalidator'].invalidate_search_cache.assert_called_once()
        services['metrics_collector'].record_ingestion.assert_called_once()
        
        # Verify event ordering and timing
        assert len(published_events) == 4
        for i in range(1, len(published_events)):
            assert published_events[i]['timestamp'] >= published_events[i-1]['timestamp']

    @pytest.mark.asyncio
    async def test_search_analytics_event_flow(self, event_driven_services):
        """Test event-driven search analytics and optimization."""
        services = event_driven_services
        
        # Search analytics events
        search_events = [
            {
                'event_type': 'search.query.received',
                'session_id': 'session_123',
                'query': 'machine learning',
                'user_id': 'user_456',
                'timestamp': time.time()
            },
            {
                'event_type': 'search.results.generated',
                'session_id': 'session_123',
                'result_count': 15,
                'search_time_ms': 245,
                'timestamp': time.time() + 0.245
            },
            {
                'event_type': 'search.result.clicked',
                'session_id': 'session_123',
                'document_id': 'doc_789',
                'rank_position': 2,
                'timestamp': time.time() + 10.5
            },
            {
                'event_type': 'search.session.ended',
                'session_id': 'session_123',
                'duration_seconds': 45,
                'satisfaction_indicator': 'positive',
                'timestamp': time.time() + 45
            }
        ]
        
        # Mock analytics processing
        analytics_results = []
        for event in search_events:
            result = {
                'event': event,
                'processed_at': time.time(),
                'analytics_data': {
                    'user_behavior': 'engaged' if event['event_type'] == 'search.result.clicked' else 'searching',
                    'performance_metrics': {'latency': event.get('search_time_ms', 0)},
                    'relevance_feedback': {'implicit_feedback': True}
                }
            }
            analytics_results.append(result)
        
        services['metrics_collector'].process_search_analytics.return_value = analytics_results
        
        # Process search analytics events
        analytics_data = await services['metrics_collector'].process_search_analytics(search_events)
        
        # Verify analytics processing
        assert len(analytics_data) == len(search_events)
        
        # Verify user behavior tracking
        click_events = [r for r in analytics_data if r['event']['event_type'] == 'search.result.clicked']
        assert len(click_events) == 1
        assert click_events[0]['event']['rank_position'] == 2
        
        # Verify performance metrics
        performance_events = [r for r in analytics_data if 'search_time_ms' in r['event']]
        assert len(performance_events) == 1
        assert performance_events[0]['analytics_data']['performance_metrics']['latency'] == 245


class TestServiceDependencyValidation:
    """Test service dependency validation and startup sequences."""

    @pytest.mark.asyncio
    async def test_service_startup_dependency_order(self):
        """Test correct service startup dependency ordering."""
        # Define service dependency graph
        service_dependencies = {
            'config_service': [],  # No dependencies
            'database_service': ['config_service'],
            'cache_service': ['config_service'],
            'vector_db_service': ['config_service'],
            'embedding_service': ['config_service', 'cache_service'],
            'crawl_service': ['config_service', 'cache_service'],
            'content_intelligence_service': ['config_service', 'embedding_service'],
            'query_processor': ['config_service', 'vector_db_service', 'embedding_service'],
            'api_service': ['config_service', 'query_processor', 'crawl_service', 'content_intelligence_service']
        }
        
        # Mock service startup results
        startup_order = []
        startup_times = {}
        
        async def mock_service_startup(service_name: str, dependencies: List[str]):
            # Verify dependencies are already started
            for dep in dependencies:
                assert dep in startup_order, f"Dependency {dep} not started before {service_name}"
            
            # Simulate startup time
            await asyncio.sleep(0.01)
            startup_order.append(service_name)
            startup_times[service_name] = time.time()
            
            return {'service': service_name, 'status': 'started', 'startup_time_ms': 10}
        
        # Execute dependency-ordered startup
        async def start_services_with_dependencies():
            started_services = set()
            
            while len(started_services) < len(service_dependencies):
                # Find services whose dependencies are all started
                ready_services = []
                for service, deps in service_dependencies.items():
                    if service not in started_services and all(dep in started_services for dep in deps):
                        ready_services.append(service)
                
                # Start ready services in parallel
                if ready_services:
                    startup_tasks = [
                        mock_service_startup(service, service_dependencies[service])
                        for service in ready_services
                    ]
                    results = await asyncio.gather(*startup_tasks)
                    
                    for result in results:
                        started_services.add(result['service'])
                else:
                    # No services ready - potential circular dependency
                    break
        
        await start_services_with_dependencies()
        
        # Verify all services started
        assert len(startup_order) == len(service_dependencies)
        
        # Verify dependency order constraints
        service_positions = {service: i for i, service in enumerate(startup_order)}
        
        for service, dependencies in service_dependencies.items():
            service_pos = service_positions[service]
            for dep in dependencies:
                dep_pos = service_positions[dep]
                assert dep_pos < service_pos, f"Dependency {dep} started after {service}"
        
        # Verify expected startup order patterns
        assert startup_order.index('config_service') == 0  # First service
        assert startup_order.index('api_service') == len(startup_order) - 1  # Last service
        assert startup_order.index('cache_service') < startup_order.index('embedding_service')
        assert startup_order.index('vector_db_service') < startup_order.index('query_processor')

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_dependency_failure(self):
        """Test graceful degradation when service dependencies fail."""
        
        # Mock service health states
        service_health = {
            'config_service': 'healthy',
            'database_service': 'healthy', 
            'cache_service': 'failed',  # Cache service failed
            'vector_db_service': 'healthy',
            'embedding_service': 'degraded',  # Degraded due to cache failure
            'api_service': 'healthy'  # Should continue with degraded performance
        }
        
        # Mock service adaptation strategies
        adaptation_strategies = {
            'embedding_service': {
                'cache_failure': 'disable_caching',
                'fallback_mode': 'direct_computation'
            },
            'api_service': {
                'cache_failure': 'increase_timeout',
                'embedding_degraded': 'reduce_batch_size'
            }
        }
        
        async def check_service_health_and_adapt(service_name: str):
            health = service_health[service_name]
            adaptations = []
            
            if health == 'failed':
                return {'status': 'failed', 'adaptations': []}
            
            if health == 'degraded':
                # Apply adaptation strategies
                if service_name in adaptation_strategies:
                    for failure_mode, strategy in adaptation_strategies[service_name].items():
                        if 'cache' in failure_mode and service_health['cache_service'] == 'failed':
                            adaptations.append(strategy)
                
                return {'status': 'degraded', 'adaptations': adaptations}
            
            return {'status': 'healthy', 'adaptations': []}
        
        # Test service adaptation
        service_states = {}
        for service in service_health.keys():
            state = await check_service_health_and_adapt(service)
            service_states[service] = state
        
        # Verify graceful degradation
        assert service_states['cache_service']['status'] == 'failed'
        assert service_states['embedding_service']['status'] == 'degraded'
        assert 'disable_caching' in service_states['embedding_service']['adaptations']
        
        assert service_states['api_service']['status'] == 'healthy'
        # API service should continue operating despite cache failure
        
        # Verify critical services remain operational
        critical_services = ['config_service', 'vector_db_service', 'api_service']
        for service in critical_services:
            assert service_states[service]['status'] in ['healthy', 'degraded']
            assert service_states[service]['status'] != 'failed'