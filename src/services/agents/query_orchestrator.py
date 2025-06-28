"""Query Orchestrator Agent for intelligent query processing coordination.

This agent serves as the master coordinator for query processing workflows,
analyzing queries, delegating to specialists, and ensuring optimal results.
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .core import BaseAgent, BaseAgentDependencies

if BaseAgent.__module__ != 'src.services.agents.core':
    # Handle import path for development
    try:
        from pydantic_ai import RunContext
        PYDANTIC_AI_AVAILABLE = True
    except ImportError:
        PYDANTIC_AI_AVAILABLE = False
        RunContext = None

logger = logging.getLogger(__name__)


class QueryOrchestrator(BaseAgent):
    """Master agent that coordinates query processing workflow.
    
    This agent analyzes incoming queries, determines optimal processing strategies,
    delegates tasks to specialized agents, and coordinates multi-stage retrieval
    when needed.
    """
    
    def __init__(self, model: str = "gpt-4"):
        """Initialize the Query Orchestrator.
        
        Args:
            model: LLM model to use for orchestration decisions
        """
        super().__init__(
            name="query_orchestrator",
            model=model,
            temperature=0.1,  # Low temperature for consistent decisions
            max_tokens=1500
        )
        
        # Strategy effectiveness tracking
        self.strategy_performance: Dict[str, Dict[str, float]] = {}
        
    def get_system_prompt(self) -> str:
        """Define system prompt for the Query Orchestrator."""
        return """You are a Query Orchestrator responsible for coordinating intelligent query processing.

Your core responsibilities:
1. Analyze incoming queries to determine complexity, intent, and optimal processing strategy
2. Delegate to specialized agents based on query characteristics and performance requirements
3. Coordinate multi-stage retrieval when single-stage search is insufficient
4. Ensure response quality while optimizing for latency and cost
5. Learn from past performance to improve future orchestration decisions

Query Analysis Framework:
- SIMPLE: Direct factual queries that can be answered with basic search
- MODERATE: Queries requiring some analysis or multi-source information
- COMPLEX: Queries needing deep analysis, reasoning, or multi-step processing

Processing Strategies:
- FAST: Prioritize speed over depth, use cached results when possible
- BALANCED: Balance quality and performance for most queries
- COMPREHENSIVE: Maximize quality for complex or critical queries

Available Specialists:
- retrieval_specialist: Optimize search strategies and parameters
- answer_generator: Generate high-quality contextual answers
- tool_selector: Choose optimal tools for specific tasks

Always provide structured decisions with confidence scores and reasoning."""
        
    async def initialize_tools(self, deps: BaseAgentDependencies) -> None:
        """Initialize Query Orchestrator tools.
        
        Args:
            deps: Agent dependencies containing client manager and config
        """
        if not PYDANTIC_AI_AVAILABLE or self.agent is None:
            logger.warning("Pydantic-AI not available, tools will use fallback mode")
            return
            
        @self.agent.tool
        async def analyze_query_intent(
            ctx: RunContext[BaseAgentDependencies], 
            query: str,
            user_context: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            """Analyze query to determine intent and optimal processing strategy.
            
            Args:
                ctx: Runtime context with dependencies
                query: User query to analyze
                user_context: Optional user context for personalization
                
            Returns:
                Analysis results with strategy recommendations
            """
            # Update tool usage stats
            ctx.deps.session_state.increment_tool_usage("analyze_query_intent")
            
            # Basic intent classification logic
            query_lower = query.lower()
            
            # Determine complexity
            complexity_indicators = {
                "simple": ["what is", "who is", "when did", "where is", "define"],
                "moderate": ["how to", "why does", "compare", "difference between"],
                "complex": ["analyze", "evaluate", "recommend", "strategy", "multiple"]
            }
            
            complexity = "moderate"  # Default
            for level, indicators in complexity_indicators.items():
                if any(indicator in query_lower for indicator in indicators):
                    complexity = level
                    break
                    
            # Determine domain if possible
            domains = {
                "technical": ["code", "programming", "api", "database", "algorithm"],
                "business": ["market", "revenue", "strategy", "customer", "sales"],
                "academic": ["research", "study", "theory", "academic", "paper"]
            }
            
            domain = "general"
            for domain_name, keywords in domains.items():
                if any(keyword in query_lower for keyword in keywords):
                    domain = domain_name
                    break
                    
            # Recommend processing strategy
            if complexity == "simple":
                strategy = "fast"
            elif complexity == "complex":
                strategy = "comprehensive"
            else:
                strategy = "balanced"
                
            # Check for multi-step requirements
            multi_step_indicators = [
                "and then", "after that", "step by step", "process",
                "first", "second", "finally"
            ]
            requires_multi_step = any(
                indicator in query_lower for indicator in multi_step_indicators
            )
            
            analysis_result = {
                "query": query,
                "complexity": complexity,
                "domain": domain,
                "strategy": strategy,
                "requires_multi_step": requires_multi_step,
                "confidence": 0.8,  # Basic confidence scoring
                "reasoning": f"Classified as {complexity} based on query patterns",
                "estimated_tokens": len(query.split()) * 4,  # Rough estimate
                "recommended_tools": self._recommend_tools(complexity, domain)
            }
            
            return analysis_result
            
        @self.agent.tool
        async def delegate_to_specialist(
            ctx: RunContext[BaseAgentDependencies],
            agent_type: str,
            task_data: Dict[str, Any],
            priority: str = "normal"
        ) -> Dict[str, Any]:
            """Delegate specific tasks to specialized agents.
            
            Args:
                ctx: Runtime context
                agent_type: Type of specialist agent to delegate to
                task_data: Data for the task
                priority: Task priority level
                
            Returns:
                Delegation result
            """
            ctx.deps.session_state.increment_tool_usage("delegate_to_specialist")
            
            # For now, simulate delegation (will be enhanced with actual agents)
            delegation_result = {
                "agent_type": agent_type,
                "task_id": str(uuid4()),
                "status": "delegated",
                "priority": priority,
                "task_data": task_data,
                "estimated_completion_time": self._estimate_completion_time(
                    agent_type, task_data
                )
            }
            
            logger.info(
                f"Delegated task to {agent_type} with priority {priority}"
            )
            
            return delegation_result
            
        @self.agent.tool
        async def coordinate_multi_stage_search(
            ctx: RunContext[BaseAgentDependencies],
            query: str,
            collection: str,
            stages: List[Dict[str, Any]]
        ) -> Dict[str, Any]:
            """Coordinate multi-stage search operations.
            
            Args:
                ctx: Runtime context
                query: Original query
                collection: Target collection
                stages: List of search stages to coordinate
                
            Returns:
                Coordinated search results
            """
            ctx.deps.session_state.increment_tool_usage("coordinate_multi_stage_search")
            
            # Get search service from client manager
            try:
                search_orchestrator = await ctx.deps.client_manager.get_search_orchestrator()
                
                # Execute coordinated search
                search_request = {
                    "query": query,
                    "collection_name": collection,
                    "enable_federation": len(stages) > 1,
                    "enable_clustering": True,
                    "mode": "enhanced"
                }
                
                # This would use the existing SearchOrchestrator
                # For now, return a structured response
                coordination_result = {
                    "query": query,
                    "collection": collection,
                    "stages_executed": len(stages),
                    "status": "completed",
                    "results_count": 0,  # Would be populated by actual search
                    "processing_time_ms": 0.0
                }
                
                return coordination_result
                
            except Exception as e:
                logger.error(f"Multi-stage search coordination failed: {e}")
                return {
                    "status": "failed",
                    "error": str(e),
                    "query": query,
                    "collection": collection
                }
                
        @self.agent.tool
        async def evaluate_strategy_performance(
            ctx: RunContext[BaseAgentDependencies],
            strategy: str,
            results: Dict[str, Any],
            user_feedback: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            """Evaluate the performance of a chosen strategy.
            
            Args:
                ctx: Runtime context
                strategy: Strategy that was used
                results: Results from strategy execution
                user_feedback: Optional user feedback
                
            Returns:
                Performance evaluation
            """
            ctx.deps.session_state.increment_tool_usage("evaluate_strategy_performance")
            
            # Calculate performance metrics
            latency = results.get("processing_time_ms", 0.0)
            quality_score = results.get("quality_score", 0.5)
            cost = results.get("cost_estimate", 0.0)
            
            # Composite performance score
            performance_score = (
                (1.0 - min(latency / 1000.0, 1.0)) * 0.3 +  # Latency weight
                quality_score * 0.6 +  # Quality weight  
                (1.0 - min(cost / 0.1, 1.0)) * 0.1  # Cost weight
            )
            
            # Update strategy performance tracking
            if strategy not in self.strategy_performance:
                self.strategy_performance[strategy] = {
                    "total_uses": 0,
                    "avg_performance": 0.0,
                    "avg_latency": 0.0,
                    "avg_quality": 0.0
                }
                
            stats = self.strategy_performance[strategy]
            stats["total_uses"] += 1
            
            # Update running averages
            alpha = 0.1  # Learning rate
            stats["avg_performance"] = (
                (1 - alpha) * stats["avg_performance"] + 
                alpha * performance_score
            )
            stats["avg_latency"] = (
                (1 - alpha) * stats["avg_latency"] + 
                alpha * latency
            )
            stats["avg_quality"] = (
                (1 - alpha) * stats["avg_quality"] + 
                alpha * quality_score
            )
            
            evaluation = {
                "strategy": strategy,
                "performance_score": performance_score,
                "metrics": {
                    "latency_ms": latency,
                    "quality_score": quality_score,
                    "cost_estimate": cost
                },
                "strategy_stats": stats.copy(),
                "recommendation": self._get_strategy_recommendation(stats)
            }
            
            return evaluation
            
    def _recommend_tools(self, complexity: str, domain: str) -> List[str]:
        """Recommend tools based on query characteristics.
        
        Args:
            complexity: Query complexity level
            domain: Query domain
            
        Returns:
            List of recommended tool names
        """
        base_tools = ["hybrid_search"]
        
        if complexity == "complex":
            base_tools.extend(["hyde_search", "multi_stage_search"])
            
        if domain == "technical":
            base_tools.append("content_classification")
            
        return base_tools
        
    def _estimate_completion_time(
        self, 
        agent_type: str, 
        task_data: Dict[str, Any]
    ) -> float:
        """Estimate task completion time for an agent type.
        
        Args:
            agent_type: Type of agent
            task_data: Task data
            
        Returns:
            Estimated completion time in seconds
        """
        base_times = {
            "retrieval_specialist": 2.0,
            "answer_generator": 3.0,
            "tool_selector": 1.0
        }
        
        base_time = base_times.get(agent_type, 2.0)
        
        # Adjust based on task complexity
        complexity_multipliers = {
            "simple": 0.5,
            "moderate": 1.0,
            "complex": 2.0
        }
        
        complexity = task_data.get("complexity", "moderate")
        multiplier = complexity_multipliers.get(complexity, 1.0)
        
        return base_time * multiplier
        
    def _get_strategy_recommendation(self, stats: Dict[str, float]) -> str:
        """Get recommendation based on strategy performance.
        
        Args:
            stats: Strategy performance statistics
            
        Returns:
            Recommendation string
        """
        if stats["avg_performance"] > 0.8:
            return "continue_using"
        elif stats["avg_performance"] > 0.6:
            return "monitor_performance"
        else:
            return "consider_alternative"
            
    async def orchestrate_query(
        self, 
        query: str,
        collection: str = "documentation",
        user_context: Optional[Dict[str, Any]] = None,
        performance_requirements: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """High-level method to orchestrate a complete query processing workflow.
        
        Args:
            query: User query to process
            collection: Target collection
            user_context: Optional user context
            performance_requirements: Optional performance constraints
            
        Returns:
            Complete orchestration result
        """
        if not self._initialized:
            raise RuntimeError("Agent not initialized")
            
        # Create execution context
        execution_context = {
            "query": query,
            "collection": collection,
            "user_context": user_context or {},
            "performance_requirements": performance_requirements or {},
            "orchestration_id": str(uuid4())
        }
        
        logger.info(
            f"Starting query orchestration for: {query[:50]}..."
        )
        
        try:
            if PYDANTIC_AI_AVAILABLE and self.agent is not None:
                # Use Pydantic-AI for orchestration
                result = await self.agent.run(
                    f"""Orchestrate processing for this query: {query}
                    
                    Collection: {collection}
                    User context: {user_context}
                    Performance requirements: {performance_requirements}
                    
                    Provide a complete orchestration plan and execute it."""
                )
                
                return {
                    "success": True,
                    "result": result.data,
                    "orchestration_id": execution_context["orchestration_id"]
                }
            else:
                # Fallback orchestration logic
                return await self._fallback_orchestration(execution_context)
                
        except Exception as e:
            logger.error(f"Query orchestration failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "orchestration_id": execution_context["orchestration_id"]
            }
            
    async def _fallback_orchestration(
        self, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fallback orchestration when Pydantic-AI is not available.
        
        Args:
            context: Execution context
            
        Returns:
            Fallback orchestration result
        """
        query = context["query"]
        
        # Simple fallback logic
        analysis = {
            "complexity": "moderate",
            "strategy": "balanced",
            "recommended_tools": ["hybrid_search"]
        }
        
        return {
            "success": True,
            "result": {
                "analysis": analysis,
                "orchestration_plan": f"Process query '{query}' using balanced strategy",
                "fallback_used": True
            },
            "orchestration_id": context["orchestration_id"]
        }