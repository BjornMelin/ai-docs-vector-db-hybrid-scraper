# G1 Research Report: Pydantic-AI Native Tool Composition Analysis

## Executive Summary

**Can we use Pydantic-AI Agents framework for Tool Composition instead of CrewAI/LangChain?**

**Answer: YES - Pydantic-AI provides comprehensive native capabilities for tool composition and multi-agent orchestration without requiring additional frameworks.**

Pydantic-AI offers four levels of multi-agent application complexity, native workflow patterns, and robust state management that can fully replace external orchestration frameworks for our use case.

## Key Findings

### 1. Native Agent Orchestration Capabilities

Pydantic-AI provides **four levels of multi-agent complexity**:

1. **Single Agent Workflows** - Basic agent operations
2. **Agent Delegation** - Agents delegate work to other agents and regain control
3. **Programmatic Agent Hand-off** - Sequential agent execution with application control
4. **Graph-based Control Flow** - Complex state machine orchestration

#### Agent Delegation Example
```python
joke_selection_agent = Agent(
    'openai:gpt-4o',
    system_prompt='Use the `joke_factory` to generate jokes, then choose the best.'
)

joke_generation_agent = Agent(
    'google-gla:gemini-1.5-flash', 
    output_type=list[str]
)

@joke_selection_agent.tool
async def joke_factory(ctx: RunContext[None], count: int) -> list[str]:
    # Agent delegates work to another agent
    r = await joke_generation_agent.run(
        f'Please generate {count} jokes.',
        usage=ctx.usage  # Pass usage context between agents
    )
    return r.output
```

#### Programmatic Hand-off Example
```python
flight_search_agent = Agent[None, Union[FlightDetails, Failed]](
    'openai:gpt-4o',
    system_prompt='Use the "flight_search" tool to find a flight'
)

seat_preference_agent = Agent[None, Union[SeatPreference, Failed]](
    'openai:gpt-4o',
    system_prompt='Extract the user\'s seat preference'
)

async def main():
    usage: Usage = Usage()
    
    # Sequential agent execution
    flight_details = await find_flight(usage)
    seat_preference = await find_seat(usage)
```

### 2. Native Tool Composition Patterns

Pydantic-AI supports **five comprehensive workflow patterns**:

#### Pattern 1: Prompt Chaining
```python
# Decompose tasks into smaller parts
outline_agent = Agent(
    'openai:gpt-4o',
    system_prompt='Create email outlines'
)

email_agent = Agent(
    'openai:gpt-4o', 
    system_prompt='Write emails from outlines'
)

# Chain execution
outline = await outline_agent.run("Create sales email outline")
email = await email_agent.run(f"Write email from: {outline.output}")
```

#### Pattern 2: Routing Workflow
```python
class TaskType(Enum):
    SUPPORT = "support"
    SALES = "sales"
    TECHNICAL = "technical"

router_agent = Agent(
    'openai:gpt-4o',
    output_type=TaskType,
    system_prompt='Classify incoming requests'
)

# Route to specialized agents based on classification
task_type = await router_agent.run(user_query)
if task_type == TaskType.SUPPORT:
    result = await support_agent.run(user_query)
elif task_type == TaskType.SALES:
    result = await sales_agent.run(user_query)
```

#### Pattern 3: Parallelization Workflow
```python
async def parallel_analysis(document: str):
    # Run multiple agents simultaneously
    tasks = [
        sentiment_agent.run(document),
        topic_agent.run(document),
        summary_agent.run(document)
    ]
    
    # Aggregate results
    sentiment, topics, summary = await asyncio.gather(*tasks)
    return combine_results(sentiment, topics, summary)
```

#### Pattern 4: Orchestrator-Workers Pattern
```python
orchestrator_agent = Agent(
    'openai:gpt-4o',
    system_prompt='Assign tasks to worker agents'
)

@orchestrator_agent.tool
async def delegate_research(ctx: RunContext, topic: str) -> dict:
    # Assign work to specialized workers
    tasks = [
        technical_worker.run(f"Technical analysis of {topic}"),
        market_worker.run(f"Market analysis of {topic}"),
        competitive_worker.run(f"Competitive analysis of {topic}")
    ]
    
    results = await asyncio.gather(*tasks)
    return synthesizer_agent.run(combine_worker_outputs(results))
```

#### Pattern 5: Evaluator-Optimizer Pattern
```python
async def iterative_improvement(initial_content: str, max_iterations: int = 3):
    current_content = initial_content
    
    for i in range(max_iterations):
        # Generate improved version
        improved = await generator_agent.run(current_content)
        
        # Evaluate quality
        evaluation = await evaluator_agent.run(improved.output)
        
        if evaluation.score > threshold:
            break
            
        current_content = improved.output
    
    return current_content
```

### 3. State Management & Coordination

#### Native State Management Features:
- **Stateless Agent Design**: Agents are global and stateless by design
- **RunContext**: Provides request-scoped state and dependency injection
- **Usage Tracking**: Comprehensive usage tracking across agent interactions
- **Dependency Injection**: Optional system for providing data/services to agents

```python
class DatabaseDependency:
    def __init__(self, connection_string: str):
        self.db = Database(connection_string)

@agent.system_prompt
async def system_prompt(ctx: RunContext[DatabaseDependency]) -> str:
    # Access injected dependencies
    user_count = await ctx.deps.db.count_users()
    return f"You are a support agent. We have {user_count} users."

@agent.tool
async def lookup_user(ctx: RunContext[DatabaseDependency], user_id: int) -> User:
    # Tools can access the same dependencies
    return await ctx.deps.db.get_user(user_id)
```

### 4. Error Handling & Resilience

#### Native Error Handling:
```python
from pydantic_ai import Agent
from pydantic import BaseModel

class Failed(BaseModel):
    error_message: str
    retry_suggested: bool

agent = Agent[None, Union[SuccessResult, Failed]](
    'openai:gpt-4o',
    system_prompt='Process requests with error handling'
)

result = await agent.run(user_input)
if isinstance(result.output, Failed):
    if result.output.retry_suggested:
        # Implement retry logic
        result = await agent.run(user_input)
```

### 5. Integration with FastMCP

Pydantic-AI integrates seamlessly with our existing FastMCP stack:

```python
from fastmcp import FastMCP
from pydantic_ai import Agent

# Create MCP server
mcp = FastMCP("AI-Tool-Composition-Server")

# Create Pydantic-AI agents
coordinator_agent = Agent('openai:gpt-4o')
specialist_agent = Agent('anthropic:claude-3.5-sonnet')

@mcp.tool()
async def orchestrate_workflow(task_description: str) -> dict:
    """Orchestrate multi-agent workflow using Pydantic-AI."""
    
    # Use coordinator agent to plan
    plan = await coordinator_agent.run(
        f"Create execution plan for: {task_description}"
    )
    
    # Execute plan with specialist agents
    results = []
    for step in plan.output.steps:
        result = await specialist_agent.run(step)
        results.append(result.output)
    
    return {"plan": plan.output, "results": results}
```

## Architecture Comparison

### Current Approach (CrewAI/LangChain)
```
FastMCP Server → CrewAI Framework → LangChain Tools → LLM Providers
```

### Proposed Pydantic-AI Approach
```
FastMCP Server → Pydantic-AI Agents → Direct LLM Providers
```

## Advantages of Native Pydantic-AI Approach

### 1. **Simplified Architecture**
- Eliminates external framework dependencies
- Reduces complexity and potential failure points
- Maintains type safety throughout the stack

### 2. **Better Integration**
- Seamless integration with existing Pydantic v2 models
- Natural fit with FastMCP server architecture
- Consistent error handling and validation patterns

### 3. **Performance Benefits**
- Fewer abstraction layers
- Direct model provider access
- Reduced overhead from framework translation

### 4. **Maintainability**
- Single framework to learn and maintain
- Consistent patterns across the application
- Python-native control flow

### 5. **Enterprise Features**
- Built-in usage tracking and monitoring
- Structured output validation
- Dependency injection for testing and configuration

## Limitations & Considerations

### 1. **Framework Maturity**
- Pydantic-AI is newer than CrewAI/LangChain
- Smaller ecosystem of pre-built integrations
- Less community documentation and examples

### 2. **Advanced Features**
- May lack some specialized features from mature frameworks
- Need to implement some patterns manually
- Limited pre-built agent templates

### 3. **Learning Curve**
- Team needs to learn Pydantic-AI patterns
- Different mental model from traditional frameworks
- Requires understanding of pydantic-graph for complex workflows

## Recommendation

**PROCEED with Pydantic-AI native implementation** for the following reasons:

1. **Perfect Alignment**: Matches our existing Pydantic v2 + FastMCP architecture
2. **Simplified Stack**: Eliminates framework complexity while maintaining functionality
3. **Type Safety**: Maintains end-to-end type safety and validation
4. **Performance**: Reduces overhead and improves response times
5. **Future-Proof**: Aligns with modern Python AI development patterns

## Implementation Strategy

### Phase 1: Core Agent Framework
```python
# Create base agent classes
from pydantic_ai import Agent
from typing import Union, Any
from pydantic import BaseModel

class TaskResult(BaseModel):
    success: bool
    data: Any
    metadata: dict

class ToolCompositionAgent:
    def __init__(self, model: str, system_prompt: str):
        self.agent = Agent[None, Union[TaskResult, Failed]](
            model=model,
            system_prompt=system_prompt
        )
    
    async def execute(self, task: str, context: dict = None) -> TaskResult:
        result = await self.agent.run(task, deps=context)
        return result.output
```

### Phase 2: Workflow Orchestration
```python
class WorkflowOrchestrator:
    def __init__(self):
        self.agents = {}
        self.workflows = {}
    
    def register_agent(self, name: str, agent: ToolCompositionAgent):
        self.agents[name] = agent
    
    async def execute_workflow(self, workflow_name: str, input_data: dict):
        workflow = self.workflows[workflow_name]
        return await workflow.execute(input_data, self.agents)
```

### Phase 3: FastMCP Integration
```python
@mcp.tool()
async def compose_tools(
    workflow_name: str,
    input_data: dict,
    orchestrator: WorkflowOrchestrator = Depends(get_orchestrator)
) -> dict:
    """Execute composed tool workflow using Pydantic-AI agents."""
    result = await orchestrator.execute_workflow(workflow_name, input_data)
    return result.model_dump()
```

## Conclusion

Pydantic-AI provides comprehensive native capabilities for tool composition and multi-agent orchestration that fully meet our requirements. The framework's alignment with our existing architecture, combined with its powerful workflow patterns and state management, makes it the optimal choice for implementing tool composition without external dependencies.

The native approach will result in a simpler, more maintainable, and better-performing system while maintaining all the functionality we need for sophisticated AI agent workflows.