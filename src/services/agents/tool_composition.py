"""Tool Composition Engine for dynamic tool orchestration.

This module provides intelligent tool selection, composition, and execution
for agentic RAG workflows with performance optimization and error handling.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field

from src.infrastructure.client_manager import ClientManager

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Categories for tool classification."""
    SEARCH = "search"
    EMBEDDING = "embedding"
    FILTERING = "filtering"
    ANALYTICS = "analytics"
    CONTENT_INTELLIGENCE = "content_intelligence"
    RAG = "rag"
    QUERY_PROCESSING = "query_processing"


class ToolPriority(Enum):
    """Priority levels for tool execution."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ToolMetadata:
    """Metadata for tool registration and management."""
    name: str
    category: ToolCategory
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    performance_metrics: Dict[str, float]
    dependencies: List[str]
    priority: ToolPriority = ToolPriority.NORMAL
    estimated_latency_ms: float = 100.0
    cost_estimate: float = 0.01
    reliability_score: float = 0.95


class ToolExecutionResult(BaseModel):
    """Result from tool execution."""
    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ToolChainStep(BaseModel):
    """Single step in a tool execution chain."""
    tool_name: str
    input_mapping: Dict[str, str]  # Map output keys to input keys
    output_key: str  # Key to store result under
    parallel: bool = False  # Whether this step can run in parallel
    optional: bool = False  # Whether failure should stop the chain


class ToolCompositionEngine:
    """Engine for dynamic tool composition and orchestration."""
    
    def __init__(self, client_manager: ClientManager):
        """Initialize the tool composition engine.
        
        Args:
            client_manager: Client manager for accessing services
        """
        self.client_manager = client_manager
        self.tool_registry: Dict[str, ToolMetadata] = {}
        self.execution_graph: Dict[str, List[str]] = {}
        self.performance_history: List[Dict[str, Any]] = []
        
        # Tool execution functions
        self.tool_executors: Dict[str, Callable] = {}
        
        # Performance tracking
        self.execution_stats: Dict[str, Dict[str, float]] = {}
        
    async def initialize(self) -> None:
        """Initialize the composition engine with existing tools."""
        await self.register_existing_tools()
        logger.info(f"Tool composition engine initialized with {len(self.tool_registry)} tools")
        
    async def register_existing_tools(self) -> None:
        """Register all existing MCP tools with metadata."""
        # Register search tools
        await self._register_search_tools()
        
        # Register embedding tools
        await self._register_embedding_tools()
        
        # Register content intelligence tools
        await self._register_content_intelligence_tools()
        
        # Register RAG tools
        await self._register_rag_tools()
        
        # Register analytics tools
        await self._register_analytics_tools()
        
    async def _register_search_tools(self) -> None:
        """Register search-related tools."""
        search_tools = {
            "hybrid_search": ToolMetadata(
                name="hybrid_search",
                category=ToolCategory.SEARCH,
                description="Hybrid vector and text search with fusion",
                input_schema={
                    "query": "str",
                    "collection": "str", 
                    "limit": "int",
                    "score_threshold": "Optional[float]"
                },
                output_schema={"results": "List[SearchResult]"},
                performance_metrics={"avg_latency_ms": 150.0, "accuracy_score": 0.85},
                dependencies=["embedding_manager", "qdrant_service"],
                estimated_latency_ms=150.0,
                cost_estimate=0.02
            ),
            "hyde_search": ToolMetadata(
                name="hyde_search",
                category=ToolCategory.SEARCH,
                description="Hypothetical Document Embeddings search",
                input_schema={
                    "query": "str",
                    "collection": "str",
                    "domain": "Optional[str]",
                    "num_generations": "Optional[int]"
                },
                output_schema={"results": "List[SearchResult]"},
                performance_metrics={"avg_latency_ms": 300.0, "accuracy_score": 0.92},
                dependencies=["hyde_engine", "embedding_manager"],
                estimated_latency_ms=300.0,
                cost_estimate=0.05
            ),
            "multi_stage_search": ToolMetadata(
                name="multi_stage_search",
                category=ToolCategory.SEARCH,
                description="Multi-stage retrieval with different strategies",
                input_schema={
                    "query": "str",
                    "collection": "str",
                    "stages": "List[Dict[str, Any]]",
                    "fusion_algorithm": "Optional[str]"
                },
                output_schema={"results": "List[SearchResult]"},
                performance_metrics={"avg_latency_ms": 400.0, "accuracy_score": 0.90},
                dependencies=["qdrant_service", "embedding_manager"],
                estimated_latency_ms=400.0,
                cost_estimate=0.08
            ),
            "filtered_search": ToolMetadata(
                name="filtered_search",
                category=ToolCategory.FILTERING,
                description="Search with advanced filtering capabilities",
                input_schema={
                    "query": "str",
                    "collection": "str",
                    "filters": "Dict[str, Any]",
                    "limit": "int"
                },
                output_schema={"results": "List[SearchResult]"},
                performance_metrics={"avg_latency_ms": 120.0, "accuracy_score": 0.82},
                dependencies=["qdrant_service"],
                estimated_latency_ms=120.0,
                cost_estimate=0.015
            )
        }
        
        self.tool_registry.update(search_tools)
        
        # Register tool executors
        self.tool_executors.update({
            "hybrid_search": self._execute_hybrid_search,
            "hyde_search": self._execute_hyde_search,
            "multi_stage_search": self._execute_multi_stage_search,
            "filtered_search": self._execute_filtered_search
        })
        
    async def _register_embedding_tools(self) -> None:
        """Register embedding-related tools."""
        embedding_tools = {
            "generate_embeddings": ToolMetadata(
                name="generate_embeddings",
                category=ToolCategory.EMBEDDING,
                description="Generate dense and sparse embeddings",
                input_schema={
                    "texts": "List[str]",
                    "generate_sparse": "Optional[bool]"
                },
                output_schema={"embeddings": "List[List[float]]"},
                performance_metrics={"avg_latency_ms": 200.0, "accuracy_score": 0.95},
                dependencies=["embedding_manager"],
                estimated_latency_ms=200.0,
                cost_estimate=0.01
            ),
            "rerank_results": ToolMetadata(
                name="rerank_results",
                category=ToolCategory.EMBEDDING,
                description="Rerank search results for better relevance",
                input_schema={
                    "query": "str",
                    "results": "List[Dict[str, Any]]"
                },
                output_schema={"reranked_results": "List[Dict[str, Any]]"},
                performance_metrics={"avg_latency_ms": 100.0, "accuracy_score": 0.88},
                dependencies=["embedding_manager"],
                estimated_latency_ms=100.0,
                cost_estimate=0.005
            )
        }
        
        self.tool_registry.update(embedding_tools)
        self.tool_executors.update({
            "generate_embeddings": self._execute_generate_embeddings,
            "rerank_results": self._execute_rerank_results
        })
        
    async def _register_content_intelligence_tools(self) -> None:
        """Register content intelligence tools."""
        content_tools = {
            "classify_content": ToolMetadata(
                name="classify_content",
                category=ToolCategory.CONTENT_INTELLIGENCE,
                description="Classify content type and quality",
                input_schema={
                    "content": "str",
                    "metadata": "Optional[Dict[str, Any]]"
                },
                output_schema={"classification": "Dict[str, float]"},
                performance_metrics={"avg_latency_ms": 50.0, "accuracy_score": 0.88},
                dependencies=["content_intelligence_service"],
                estimated_latency_ms=50.0,
                cost_estimate=0.002
            ),
            "assess_content_quality": ToolMetadata(
                name="assess_content_quality",
                category=ToolCategory.CONTENT_INTELLIGENCE,
                description="Assess content quality and reliability",
                input_schema={
                    "content": "str",
                    "source_metadata": "Optional[Dict[str, Any]]"
                },
                output_schema={"quality_assessment": "Dict[str, float]"},
                performance_metrics={"avg_latency_ms": 75.0, "accuracy_score": 0.85},
                dependencies=["content_intelligence_service"],
                estimated_latency_ms=75.0,
                cost_estimate=0.003
            )
        }
        
        self.tool_registry.update(content_tools)
        self.tool_executors.update({
            "classify_content": self._execute_classify_content,
            "assess_content_quality": self._execute_assess_content_quality
        })
        
    async def _register_rag_tools(self) -> None:
        """Register RAG-related tools."""
        rag_tools = {
            "generate_rag_answer": ToolMetadata(
                name="generate_rag_answer",
                category=ToolCategory.RAG,
                description="Generate contextual answers from search results",
                input_schema={
                    "query": "str",
                    "search_results": "List[Dict[str, Any]]",
                    "max_tokens": "Optional[int]",
                    "temperature": "Optional[float]"
                },
                output_schema={"answer": "str", "sources": "List[Dict[str, Any]]"},
                performance_metrics={"avg_latency_ms": 800.0, "accuracy_score": 0.90},
                dependencies=["rag_generator"],
                estimated_latency_ms=800.0,
                cost_estimate=0.10
            )
        }
        
        self.tool_registry.update(rag_tools)
        self.tool_executors["generate_rag_answer"] = self._execute_generate_rag_answer
        
    async def _register_analytics_tools(self) -> None:
        """Register analytics and monitoring tools."""
        analytics_tools = {
            "analyze_query_performance": ToolMetadata(
                name="analyze_query_performance",
                category=ToolCategory.ANALYTICS,
                description="Analyze query performance metrics",
                input_schema={
                    "query": "str",
                    "results": "List[Dict[str, Any]]",
                    "execution_time": "float"
                },
                output_schema={"analytics": "Dict[str, Any]"},
                performance_metrics={"avg_latency_ms": 25.0, "accuracy_score": 0.95},
                dependencies=["analytics_service"],
                estimated_latency_ms=25.0,
                cost_estimate=0.001
            )
        }
        
        self.tool_registry.update(analytics_tools)
        self.tool_executors["analyze_query_performance"] = self._execute_analyze_query_performance
        
    async def compose_tool_chain(
        self,
        goal: str,
        constraints: Dict[str, Any],
        available_tools: Optional[List[str]] = None
    ) -> List[ToolChainStep]:
        """Intelligently compose tool chains to achieve goals.
        
        Args:
            goal: High-level goal description
            constraints: Performance and quality constraints
            available_tools: Optional list of available tools
            
        Returns:
            Optimized tool chain
        """
        # Analyze goal to determine required capabilities
        goal_analysis = await self._analyze_goal(goal, constraints)
        
        # Select optimal tools based on analysis
        selected_tools = await self._select_optimal_tools(
            goal_analysis, constraints, available_tools
        )
        
        # Create execution chain with dependencies
        tool_chain = await self._create_execution_chain(selected_tools, goal_analysis)
        
        logger.info(f"Composed tool chain with {len(tool_chain)} steps for goal: {goal}")
        
        return tool_chain
        
    async def execute_tool_chain(
        self,
        chain: List[ToolChainStep],
        input_data: Dict[str, Any],
        timeout_seconds: float = 30.0
    ) -> Dict[str, Any]:
        """Execute a composed tool chain with error handling and optimization.
        
        Args:
            chain: Tool chain to execute
            input_data: Initial input data
            timeout_seconds: Execution timeout
            
        Returns:
            Chain execution results
        """
        execution_id = str(uuid4())
        start_time = time.time()
        
        logger.info(f"Executing tool chain {execution_id} with {len(chain)} steps")
        
        # Initialize execution context
        context = {
            "execution_id": execution_id,
            "input_data": input_data.copy(),
            "results": {},
            "errors": [],
            "execution_times": {},
            "step_count": 0
        }
        
        try:
            # Execute chain steps
            for step_idx, step in enumerate(chain):
                step_start = time.time()
                
                try:
                    # Check timeout
                    if time.time() - start_time > timeout_seconds:
                        raise TimeoutError(f"Tool chain execution exceeded {timeout_seconds}s timeout")
                        
                    # Execute step
                    step_result = await self._execute_chain_step(step, context)
                    
                    # Store result
                    context["results"][step.output_key] = step_result.result
                    context["execution_times"][step.tool_name] = step_result.execution_time_ms
                    context["step_count"] += 1
                    
                    # Handle step failure
                    if not step_result.success:
                        if step.optional:
                            logger.warning(f"Optional step {step.tool_name} failed: {step_result.error}")
                            context["errors"].append({
                                "step": step_idx,
                                "tool": step.tool_name,
                                "error": step_result.error,
                                "optional": True
                            })
                        else:
                            raise RuntimeError(f"Required step {step.tool_name} failed: {step_result.error}")
                            
                except Exception as e:
                    if step.optional:
                        logger.warning(f"Optional step {step.tool_name} failed: {e}")
                        context["errors"].append({
                            "step": step_idx,
                            "tool": step.tool_name,
                            "error": str(e),
                            "optional": True
                        })
                    else:
                        raise
                        
                step_time = (time.time() - step_start) * 1000
                logger.debug(f"Step {step_idx} ({step.tool_name}) completed in {step_time:.1f}ms")
                
            # Calculate final metrics
            total_time = (time.time() - start_time) * 1000
            
            execution_result = {
                "success": True,
                "execution_id": execution_id,
                "results": context["results"],
                "metadata": {
                    "total_execution_time_ms": total_time,
                    "steps_executed": context["step_count"],
                    "step_execution_times": context["execution_times"],
                    "errors": context["errors"],
                    "chain_length": len(chain)
                }
            }
            
            # Record performance for learning
            await self._record_chain_performance(chain, execution_result)
            
            logger.info(f"Tool chain {execution_id} completed successfully in {total_time:.1f}ms")
            
            return execution_result
            
        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            
            logger.error(f"Tool chain {execution_id} failed after {total_time:.1f}ms: {e}")
            
            return {
                "success": False,
                "execution_id": execution_id,
                "error": str(e),
                "partial_results": context["results"],
                "metadata": {
                    "total_execution_time_ms": total_time,
                    "steps_executed": context["step_count"],
                    "errors": context["errors"],
                    "failed_at_step": context["step_count"]
                }
            }
            
    async def _analyze_goal(self, goal: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze goal to determine required capabilities."""
        goal_lower = goal.lower()
        
        # Basic goal analysis
        analysis = {
            "primary_action": "search",  # Default
            "complexity": "moderate",
            "required_capabilities": [],
            "performance_priority": "balanced"
        }
        
        # Determine primary action
        if "search" in goal_lower or "find" in goal_lower:
            analysis["primary_action"] = "search"
            analysis["required_capabilities"].append("search")
        elif "generate" in goal_lower or "create" in goal_lower:
            analysis["primary_action"] = "generate"
            analysis["required_capabilities"].extend(["search", "rag"])
        elif "analyze" in goal_lower or "evaluate" in goal_lower:
            analysis["primary_action"] = "analyze"
            analysis["required_capabilities"].extend(["search", "analytics"])
            
        # Determine complexity
        complexity_indicators = {
            "simple": ["basic", "simple", "quick"],
            "complex": ["comprehensive", "detailed", "thorough", "analyze"]
        }
        
        for level, indicators in complexity_indicators.items():
            if any(indicator in goal_lower for indicator in indicators):
                analysis["complexity"] = level
                break
                
        # Check performance constraints
        if constraints.get("max_latency_ms", 1000) < 500:
            analysis["performance_priority"] = "speed"
        elif constraints.get("min_quality_score", 0.7) > 0.9:
            analysis["performance_priority"] = "quality"
            
        return analysis
        
    async def _select_optimal_tools(
        self,
        goal_analysis: Dict[str, Any],
        constraints: Dict[str, Any],
        available_tools: Optional[List[str]] = None
    ) -> List[str]:
        """Select optimal tools based on goal analysis and constraints."""
        required_capabilities = goal_analysis["required_capabilities"]
        performance_priority = goal_analysis["performance_priority"]
        complexity = goal_analysis["complexity"]
        
        # Filter tools by availability
        candidate_tools = []
        for tool_name, metadata in self.tool_registry.items():
            if available_tools is None or tool_name in available_tools:
                candidate_tools.append((tool_name, metadata))
                
        # Score tools based on requirements
        tool_scores = {}
        for tool_name, metadata in candidate_tools:
            score = 0.0
            
            # Capability match score
            if metadata.category.value in required_capabilities:
                score += 10.0
                
            # Performance priority score
            if performance_priority == "speed":
                score += 5.0 / (metadata.estimated_latency_ms / 100.0)
            elif performance_priority == "quality":
                score += metadata.performance_metrics.get("accuracy_score", 0.5) * 5.0
            else:  # balanced
                score += (
                    metadata.performance_metrics.get("accuracy_score", 0.5) * 2.5 +
                    2.5 / (metadata.estimated_latency_ms / 100.0)
                )
                
            # Complexity match score
            if complexity == "simple" and metadata.estimated_latency_ms < 200:
                score += 2.0
            elif complexity == "complex" and metadata.category in [ToolCategory.RAG, ToolCategory.ANALYTICS]:
                score += 2.0
                
            # Reliability score
            score += metadata.reliability_score
            
            tool_scores[tool_name] = score
            
        # Select top tools based on scores
        sorted_tools = sorted(tool_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build tool selection based on primary action
        selected_tools = []
        primary_action = goal_analysis["primary_action"]
        
        if primary_action == "search":
            # Select best search tool
            search_tools = [t for t, _ in sorted_tools if self.tool_registry[t].category == ToolCategory.SEARCH]
            if search_tools:
                selected_tools.append(search_tools[0])
        elif primary_action == "generate":
            # Select search + RAG tools
            search_tools = [t for t, _ in sorted_tools if self.tool_registry[t].category == ToolCategory.SEARCH]
            rag_tools = [t for t, _ in sorted_tools if self.tool_registry[t].category == ToolCategory.RAG]
            if search_tools:
                selected_tools.append(search_tools[0])
            if rag_tools:
                selected_tools.append(rag_tools[0])
        elif primary_action == "analyze":
            # Select search + analytics tools
            search_tools = [t for t, _ in sorted_tools if self.tool_registry[t].category == ToolCategory.SEARCH]
            analytics_tools = [t for t, _ in sorted_tools if self.tool_registry[t].category == ToolCategory.ANALYTICS]
            if search_tools:
                selected_tools.append(search_tools[0])
            if analytics_tools:
                selected_tools.append(analytics_tools[0])
                
        # Add content intelligence if beneficial
        if complexity == "complex":
            content_tools = [t for t, _ in sorted_tools if self.tool_registry[t].category == ToolCategory.CONTENT_INTELLIGENCE]
            if content_tools:
                selected_tools.append(content_tools[0])
                
        return selected_tools
        
    async def _create_execution_chain(
        self,
        selected_tools: List[str],
        goal_analysis: Dict[str, Any]
    ) -> List[ToolChainStep]:
        """Create execution chain with proper dependencies."""
        chain = []
        
        # Create steps based on selected tools and dependencies
        for idx, tool_name in enumerate(selected_tools):
            metadata = self.tool_registry[tool_name]
            
            # Determine input mapping based on tool type and position
            input_mapping = {}
            if idx == 0:
                # First tool uses original input
                input_mapping = {"query": "input_data.query", "collection": "input_data.collection"}
            else:
                # Subsequent tools may use previous results
                if metadata.category == ToolCategory.RAG:
                    input_mapping = {
                        "query": "input_data.query",
                        "search_results": f"results.{selected_tools[idx-1]}_results"
                    }
                elif metadata.category == ToolCategory.ANALYTICS:
                    input_mapping = {
                        "query": "input_data.query",
                        "results": f"results.{selected_tools[idx-1]}_results"
                    }
                    
            step = ToolChainStep(
                tool_name=tool_name,
                input_mapping=input_mapping,
                output_key=f"{tool_name}_results",
                parallel=metadata.category in [ToolCategory.CONTENT_INTELLIGENCE, ToolCategory.ANALYTICS],
                optional=metadata.category == ToolCategory.CONTENT_INTELLIGENCE
            )
            
            chain.append(step)
            
        return chain
        
    async def _execute_chain_step(
        self,
        step: ToolChainStep,
        context: Dict[str, Any]
    ) -> ToolExecutionResult:
        """Execute a single step in the tool chain."""
        tool_name = step.tool_name
        
        # Get tool executor
        executor = self.tool_executors.get(tool_name)
        if not executor:
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                error=f"No executor found for tool {tool_name}",
                execution_time_ms=0.0
            )
            
        # Build input parameters from mapping
        input_params = {}
        for param_name, source_path in step.input_mapping.items():
            try:
                # Simple path resolution (can be enhanced)
                if source_path.startswith("input_data."):
                    key = source_path.replace("input_data.", "")
                    input_params[param_name] = context["input_data"].get(key)
                elif source_path.startswith("results."):
                    key = source_path.replace("results.", "")
                    input_params[param_name] = context["results"].get(key)
                else:
                    input_params[param_name] = source_path
            except Exception as e:
                return ToolExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    error=f"Input mapping failed: {e}",
                    execution_time_ms=0.0
                )
                
        # Execute tool
        start_time = time.time()
        try:
            result = await executor(**input_params)
            execution_time = (time.time() - start_time) * 1000
            
            return ToolExecutionResult(
                tool_name=tool_name,
                success=True,
                result=result,
                execution_time_ms=execution_time
            )
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                error=str(e),
                execution_time_ms=execution_time
            )
            
    async def _record_chain_performance(
        self,
        chain: List[ToolChainStep],
        result: Dict[str, Any]
    ) -> None:
        """Record chain performance for learning."""
        performance_record = {
            "chain_signature": "_".join([step.tool_name for step in chain]),
            "success": result["success"],
            "execution_time_ms": result["metadata"]["total_execution_time_ms"],
            "steps_count": len(chain),
            "timestamp": time.time()
        }
        
        self.performance_history.append(performance_record)
        
        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
            
    # Tool executor implementations
    async def _execute_hybrid_search(self, query: str, collection: str, **kwargs) -> List[Dict[str, Any]]:
        """Execute hybrid search tool."""
        # This would integrate with the actual hybrid search implementation
        # For now, return a mock result
        return [
            {
                "id": "doc_1",
                "content": f"Mock result for query: {query}",
                "score": 0.85,
                "metadata": {"collection": collection}
            }
        ]
        
    async def _execute_hyde_search(self, query: str, collection: str, **kwargs) -> List[Dict[str, Any]]:
        """Execute HyDE search tool."""
        # This would integrate with the actual HyDE search implementation
        return [
            {
                "id": "hyde_1",
                "content": f"HyDE result for query: {query}",
                "score": 0.92,
                "metadata": {"collection": collection, "method": "hyde"}
            }
        ]
        
    async def _execute_multi_stage_search(self, query: str, collection: str, **kwargs) -> List[Dict[str, Any]]:
        """Execute multi-stage search tool."""
        return [
            {
                "id": "multi_1",
                "content": f"Multi-stage result for query: {query}",
                "score": 0.90,
                "metadata": {"collection": collection, "method": "multi_stage"}
            }
        ]
        
    async def _execute_filtered_search(self, query: str, collection: str, **kwargs) -> List[Dict[str, Any]]:
        """Execute filtered search tool."""
        return [
            {
                "id": "filtered_1",
                "content": f"Filtered result for query: {query}",
                "score": 0.82,
                "metadata": {"collection": collection, "method": "filtered"}
            }
        ]
        
    async def _execute_generate_embeddings(self, texts: List[str], **kwargs) -> Dict[str, Any]:
        """Execute embedding generation tool."""
        return {
            "embeddings": [[0.1] * 384 for _ in texts],  # Mock embeddings
            "count": len(texts)
        }
        
    async def _execute_rerank_results(self, query: str, results: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """Execute result reranking tool."""
        # Simple mock reranking
        return sorted(results, key=lambda x: x.get("score", 0), reverse=True)
        
    async def _execute_classify_content(self, content: str, **kwargs) -> Dict[str, float]:
        """Execute content classification tool."""
        return {
            "technical": 0.7,
            "informational": 0.8,
            "quality_score": 0.85
        }
        
    async def _execute_assess_content_quality(self, content: str, **kwargs) -> Dict[str, float]:
        """Execute content quality assessment tool."""
        return {
            "readability": 0.8,
            "completeness": 0.9,
            "accuracy": 0.85,
            "overall_quality": 0.85
        }
        
    async def _execute_generate_rag_answer(self, query: str, search_results: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Execute RAG answer generation tool."""
        return {
            "answer": f"Generated answer for: {query}",
            "confidence": 0.9,
            "sources": search_results[:3],  # Top 3 sources
            "generation_time_ms": 800.0
        }
        
    async def _execute_analyze_query_performance(self, query: str, results: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Execute query performance analysis tool."""
        return {
            "result_count": len(results),
            "avg_score": sum(r.get("score", 0) for r in results) / len(results) if results else 0,
            "query_complexity": "moderate",
            "performance_rating": 0.85
        }
        
    def get_tool_metadata(self, tool_name: str) -> Optional[ToolMetadata]:
        """Get metadata for a specific tool."""
        return self.tool_registry.get(tool_name)
        
    def list_tools_by_category(self, category: ToolCategory) -> List[str]:
        """List all tools in a specific category."""
        return [
            name for name, metadata in self.tool_registry.items()
            if metadata.category == category
        ]
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the composition engine."""
        if not self.performance_history:
            return {"message": "No performance data available"}
            
        recent_history = self.performance_history[-100:]  # Last 100 executions
        
        return {
            "total_executions": len(self.performance_history),
            "recent_success_rate": sum(1 for r in recent_history if r["success"]) / len(recent_history),
            "avg_execution_time_ms": sum(r["execution_time_ms"] for r in recent_history) / len(recent_history),
            "most_used_tools": self._get_most_used_tools(),
            "performance_trends": self._analyze_performance_trends()
        }
        
    def _get_most_used_tools(self) -> List[Tuple[str, int]]:
        """Get most frequently used tools."""
        tool_usage = {}
        for record in self.performance_history:
            chain_signature = record["chain_signature"]
            tools = chain_signature.split("_")
            for tool in tools:
                tool_usage[tool] = tool_usage.get(tool, 0) + 1
                
        return sorted(tool_usage.items(), key=lambda x: x[1], reverse=True)[:10]
        
    def _analyze_performance_trends(self) -> Dict[str, float]:
        """Analyze performance trends over time."""
        if len(self.performance_history) < 10:
            return {"trend": "insufficient_data"}
            
        recent_avg = sum(
            r["execution_time_ms"] for r in self.performance_history[-50:]
        ) / min(50, len(self.performance_history))
        
        older_avg = sum(
            r["execution_time_ms"] for r in self.performance_history[-100:-50]
        ) / min(50, len(self.performance_history) - 50) if len(self.performance_history) > 50 else recent_avg
        
        return {
            "recent_avg_time_ms": recent_avg,
            "older_avg_time_ms": older_avg,
            "performance_change_pct": ((older_avg - recent_avg) / older_avg * 100) if older_avg > 0 else 0
        }