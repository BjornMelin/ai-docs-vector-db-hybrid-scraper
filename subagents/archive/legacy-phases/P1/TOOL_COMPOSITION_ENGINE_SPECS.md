# Tool Composition Engine - Technical Specifications

## Overview

The Tool Composition Engine enables sophisticated orchestration of multiple MCP tools to create complex workflows. It provides parallel execution, dependency management, conditional logic, and intelligent optimization for enterprise-grade tool composition.

## Architecture Components

### 1. Workflow Orchestrator Core

```python
# src/mcp_tools/composition/orchestrator.py

import asyncio
import logging
import time
from typing import Dict, List, Optional, Set, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

logger = logging.getLogger(__name__)

class ExecutionMode(str, Enum):
    """Workflow execution modes."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    PIPELINE = "pipeline"
    SCATTER_GATHER = "scatter_gather"

class ExecutionState(str, Enum):
    """Task execution states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

@dataclass
class ExecutionContext:
    """Context for workflow execution."""
    workflow_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: Optional[float] = None
    max_retries: int = 3
    retry_backoff: float = 1.0

@dataclass
class TaskResult:
    """Result of a single task execution."""
    task_id: str
    tool_id: str
    state: ExecutionState
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    memory_used: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
class WorkflowOrchestrator:
    """Advanced tool composition and workflow execution engine."""
    
    def __init__(self, 
                 discovery_engine: 'ToolDiscoveryEngine',
                 config: 'CompositionConfig'):
        self.discovery_engine = discovery_engine
        self.config = config
        self.execution_engine = ExecutionEngine(config)
        self.dependency_resolver = DependencyResolver()
        self.workflow_optimizer = WorkflowOptimizer()
        self.active_workflows: Dict[str, 'WorkflowExecution'] = {}
        self.workflow_templates: Dict[str, 'WorkflowTemplate'] = {}
        self.performance_monitor = WorkflowPerformanceMonitor()
        
    async def create_workflow(self, 
                            workflow_spec: 'WorkflowSpecification',
                            context: ExecutionContext) -> 'Workflow':
        """Create optimized workflow from specification."""
        
        # Validate workflow specification
        validation_result = await self._validate_workflow_spec(workflow_spec)
        if not validation_result.is_valid:
            raise ValueError(f"Invalid workflow: {validation_result.errors}")
        
        # Resolve tool dependencies
        dependency_graph = await self.dependency_resolver.resolve_dependencies(
            workflow_spec.tasks
        )
        
        # Optimize workflow structure
        optimized_spec = await self.workflow_optimizer.optimize(
            workflow_spec, dependency_graph
        )
        
        # Create workflow instance
        workflow = Workflow(
            id=str(uuid.uuid4()),
            specification=optimized_spec,
            dependency_graph=dependency_graph,
            context=context,
            created_at=time.time()
        )
        
        logger.info(f"Created workflow {workflow.id} with {len(optimized_spec.tasks)} tasks")
        return workflow
    
    async def execute_workflow(self, 
                             workflow: 'Workflow') -> 'WorkflowResult':
        """Execute workflow with advanced orchestration."""
        
        workflow_execution = WorkflowExecution(
            workflow=workflow,
            orchestrator=self
        )
        
        self.active_workflows[workflow.id] = workflow_execution
        
        try:
            # Start performance monitoring
            self.performance_monitor.start_monitoring(workflow.id)
            
            # Execute workflow
            result = await workflow_execution.execute()
            
            # Complete performance monitoring
            performance_metrics = self.performance_monitor.complete_monitoring(workflow.id)
            result.performance_metrics = performance_metrics
            
            logger.info(f"Workflow {workflow.id} completed in {result.total_execution_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Workflow {workflow.id} failed: {e}")
            raise
        finally:
            # Cleanup
            if workflow.id in self.active_workflows:
                del self.active_workflows[workflow.id]
    
    async def pause_workflow(self, workflow_id: str) -> bool:
        """Pause a running workflow."""
        if workflow_id not in self.active_workflows:
            return False
        
        workflow_execution = self.active_workflows[workflow_id]
        return await workflow_execution.pause()
    
    async def resume_workflow(self, workflow_id: str) -> bool:
        """Resume a paused workflow."""
        if workflow_id not in self.active_workflows:
            return False
        
        workflow_execution = self.active_workflows[workflow_id]
        return await workflow_execution.resume()
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow."""
        if workflow_id not in self.active_workflows:
            return False
        
        workflow_execution = self.active_workflows[workflow_id]
        return await workflow_execution.cancel()
    
    async def get_workflow_status(self, workflow_id: str) -> Optional['WorkflowStatus']:
        """Get current status of a workflow."""
        if workflow_id not in self.active_workflows:
            return None
        
        workflow_execution = self.active_workflows[workflow_id]
        return await workflow_execution.get_status()
    
    async def create_workflow_template(self, 
                                     name: str,
                                     workflow_spec: 'WorkflowSpecification') -> str:
        """Create reusable workflow template."""
        template_id = str(uuid.uuid4())
        template = WorkflowTemplate(
            id=template_id,
            name=name,
            specification=workflow_spec,
            created_at=time.time()
        )
        
        self.workflow_templates[template_id] = template
        logger.info(f"Created workflow template: {name}")
        return template_id
    
    async def instantiate_template(self, 
                                 template_id: str,
                                 context: ExecutionContext,
                                 parameters: Dict[str, Any] = None) -> 'Workflow':
        """Create workflow instance from template."""
        if template_id not in self.workflow_templates:
            raise ValueError(f"Template not found: {template_id}")
        
        template = self.workflow_templates[template_id]
        
        # Apply parameters to template
        workflow_spec = await self._apply_template_parameters(
            template.specification, parameters or {}
        )
        
        return await self.create_workflow(workflow_spec, context)


class WorkflowExecution:
    """Manages execution of a single workflow."""
    
    def __init__(self, workflow: 'Workflow', orchestrator: WorkflowOrchestrator):
        self.workflow = workflow
        self.orchestrator = orchestrator
        self.task_results: Dict[str, TaskResult] = {}
        self.execution_state = ExecutionState.PENDING
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.pause_event = asyncio.Event()
        self.cancel_event = asyncio.Event()
        self.pause_event.set()  # Start unpaused
        
    async def execute(self) -> 'WorkflowResult':
        """Execute the workflow."""
        self.execution_state = ExecutionState.RUNNING
        self.start_time = time.time()
        
        try:
            # Get execution plan
            execution_plan = await self._create_execution_plan()
            
            # Execute based on workflow mode
            if self.workflow.specification.execution_mode == ExecutionMode.SEQUENTIAL:
                await self._execute_sequential(execution_plan)
            elif self.workflow.specification.execution_mode == ExecutionMode.PARALLEL:
                await self._execute_parallel(execution_plan)
            elif self.workflow.specification.execution_mode == ExecutionMode.PIPELINE:
                await self._execute_pipeline(execution_plan)
            elif self.workflow.specification.execution_mode == ExecutionMode.SCATTER_GATHER:
                await self._execute_scatter_gather(execution_plan)
            else:
                raise ValueError(f"Unsupported execution mode: {self.workflow.specification.execution_mode}")
            
            self.execution_state = ExecutionState.COMPLETED
            
        except asyncio.CancelledError:
            self.execution_state = ExecutionState.CANCELLED
            raise
        except Exception as e:
            self.execution_state = ExecutionState.FAILED
            logger.error(f"Workflow execution failed: {e}")
            raise
        finally:
            self.end_time = time.time()
        
        # Create result
        return WorkflowResult(
            workflow_id=self.workflow.id,
            state=self.execution_state,
            task_results=self.task_results,
            start_time=self.start_time,
            end_time=self.end_time,
            total_execution_time=self.end_time - self.start_time,
            context=self.workflow.context
        )
    
    async def _execute_sequential(self, execution_plan: 'ExecutionPlan') -> None:
        """Execute tasks sequentially."""
        for task_group in execution_plan.execution_groups:
            for task in task_group.tasks:
                await self._check_pause_cancel()
                
                result = await self._execute_single_task(task)
                self.task_results[task.id] = result
                
                if result.state == ExecutionState.FAILED and task.fail_fast:
                    raise RuntimeError(f"Task {task.id} failed: {result.error}")
    
    async def _execute_parallel(self, execution_plan: 'ExecutionPlan') -> None:
        """Execute tasks in parallel where possible."""
        for task_group in execution_plan.execution_groups:
            await self._check_pause_cancel()
            
            # Execute all tasks in group concurrently
            tasks = [self._execute_single_task(task) for task in task_group.tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for task, result in zip(task_group.tasks, results):
                if isinstance(result, Exception):
                    self.task_results[task.id] = TaskResult(
                        task_id=task.id,
                        tool_id=task.tool_id,
                        state=ExecutionState.FAILED,
                        error=str(result)
                    )
                    if task.fail_fast:
                        raise result
                else:
                    self.task_results[task.id] = result
    
    async def _execute_pipeline(self, execution_plan: 'ExecutionPlan') -> None:
        """Execute tasks as a pipeline with data flow."""
        pipeline_context = {}
        
        for task_group in execution_plan.execution_groups:
            await self._check_pause_cancel()
            
            # Execute tasks with pipeline context
            for task in task_group.tasks:
                # Add pipeline context to task input
                task_input = {**task.input_data, **pipeline_context}
                
                result = await self._execute_single_task(task, task_input)
                self.task_results[task.id] = result
                
                # Update pipeline context with result
                if result.state == ExecutionState.COMPLETED and result.result:
                    pipeline_context[f"{task.id}_result"] = result.result
                
                if result.state == ExecutionState.FAILED and task.fail_fast:
                    raise RuntimeError(f"Pipeline task {task.id} failed: {result.error}")
    
    async def _execute_scatter_gather(self, execution_plan: 'ExecutionPlan') -> None:
        """Execute scatter-gather pattern."""
        # Scatter phase - distribute work
        scatter_tasks = []
        for task_group in execution_plan.execution_groups[:-1]:  # All except gather
            await self._check_pause_cancel()
            
            for task in task_group.tasks:
                scatter_tasks.append(self._execute_single_task(task))
        
        # Execute scatter tasks
        scatter_results = await asyncio.gather(*scatter_tasks, return_exceptions=True)
        
        # Process scatter results
        gather_input = {}
        for i, (task_group, results_slice) in enumerate(zip(execution_plan.execution_groups[:-1], 
                                                           self._chunk_list(scatter_results, 
                                                                          len(execution_plan.execution_groups[:-1])))):
            for task, result in zip(task_group.tasks, results_slice):
                if isinstance(result, Exception):
                    self.task_results[task.id] = TaskResult(
                        task_id=task.id,
                        tool_id=task.tool_id,
                        state=ExecutionState.FAILED,
                        error=str(result)
                    )
                else:
                    self.task_results[task.id] = result
                    if result.state == ExecutionState.COMPLETED:
                        gather_input[task.id] = result.result
        
        # Gather phase - consolidate results
        gather_group = execution_plan.execution_groups[-1]
        for task in gather_group.tasks:
            await self._check_pause_cancel()
            
            # Pass all scatter results to gather task
            gather_task_input = {**task.input_data, "scatter_results": gather_input}
            result = await self._execute_single_task(task, gather_task_input)
            self.task_results[task.id] = result
    
    async def _execute_single_task(self, 
                                 task: 'WorkflowTask',
                                 custom_input: Dict[str, Any] = None) -> TaskResult:
        """Execute a single workflow task."""
        start_time = time.time()
        
        try:
            # Get tool instance
            tool_instance = self.orchestrator.discovery_engine.active_tools.get(task.tool_id)
            if not tool_instance:
                raise ValueError(f"Tool not found: {task.tool_id}")
            
            # Prepare input
            task_input = custom_input or task.input_data
            
            # Apply input transformations
            if task.input_transformations:
                task_input = await self._apply_transformations(task_input, task.input_transformations)
            
            # Execute tool with retry logic
            result = await self._execute_with_retry(
                tool_instance.callable,
                task_input,
                max_retries=task.max_retries,
                retry_backoff=task.retry_backoff
            )
            
            # Apply output transformations
            if task.output_transformations:
                result = await self._apply_transformations(result, task.output_transformations)
            
            execution_time = time.time() - start_time
            
            return TaskResult(
                task_id=task.id,
                tool_id=task.tool_id,
                state=ExecutionState.COMPLETED,
                result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Task {task.id} failed: {e}")
            
            return TaskResult(
                task_id=task.id,
                tool_id=task.tool_id,
                state=ExecutionState.FAILED,
                error=str(e),
                execution_time=execution_time
            )
    
    async def _check_pause_cancel(self) -> None:
        """Check for pause/cancel signals."""
        # Check for cancellation
        if self.cancel_event.is_set():
            raise asyncio.CancelledError("Workflow cancelled")
        
        # Wait if paused
        await self.pause_event.wait()
    
    async def pause(self) -> bool:
        """Pause workflow execution."""
        self.pause_event.clear()
        return True
    
    async def resume(self) -> bool:
        """Resume workflow execution."""
        self.pause_event.set()
        return True
    
    async def cancel(self) -> bool:
        """Cancel workflow execution."""
        self.cancel_event.set()
        return True
```

### 2. Workflow Specification DSL

```python
# src/mcp_tools/composition/dsl.py

class WorkflowBuilder:
    """Fluent API for building complex workflows."""
    
    def __init__(self):
        self.tasks: List['WorkflowTask'] = []
        self.current_group = 0
        self.execution_mode = ExecutionMode.SEQUENTIAL
        self.global_config: Dict[str, Any] = {}
        
    def parallel(self, *tool_calls: Union[str, 'ToolCall']) -> 'WorkflowBuilder':
        """Execute tools in parallel."""
        self.execution_mode = ExecutionMode.PARALLEL
        
        for tool_call in tool_calls:
            task = self._create_task(tool_call)
            task.execution_group = self.current_group
            self.tasks.append(task)
        
        self.current_group += 1
        return self
    
    def sequence(self, *tool_calls: Union[str, 'ToolCall']) -> 'WorkflowBuilder':
        """Execute tools sequentially."""
        for tool_call in tool_calls:
            task = self._create_task(tool_call)
            task.execution_group = self.current_group
            self.tasks.append(task)
            self.current_group += 1
        
        return self
    
    def pipeline(self, *tool_calls: Union[str, 'ToolCall']) -> 'WorkflowBuilder':
        """Execute tools as pipeline with data flow."""
        self.execution_mode = ExecutionMode.PIPELINE
        
        for i, tool_call in enumerate(tool_calls):
            task = self._create_task(tool_call)
            task.execution_group = i
            
            # Set up pipeline data flow
            if i > 0:
                previous_task_id = self.tasks[-1].id
                task.input_transformations.append(
                    InputTransformation(
                        source=f"{previous_task_id}_result",
                        target="input",
                        transformation="passthrough"
                    )
                )
            
            self.tasks.append(task)
        
        self.current_group = len(tool_calls)
        return self
    
    def conditional(self, 
                   condition: Union[str, Callable],
                   true_branch: 'WorkflowBuilder',
                   false_branch: Optional['WorkflowBuilder'] = None) -> 'WorkflowBuilder':
        """Execute conditional branching."""
        
        # Create condition evaluation task
        condition_task = WorkflowTask(
            id=f"condition_{self.current_group}",
            tool_id="evaluate_condition",
            input_data={"condition": condition},
            execution_group=self.current_group
        )
        self.tasks.append(condition_task)
        self.current_group += 1
        
        # Add true branch tasks
        true_tasks = true_branch.build().tasks
        for task in true_tasks:
            task.execution_group += self.current_group
            task.conditions = [ConditionalExecution(
                condition_task_id=condition_task.id,
                condition_value=True
            )]
            self.tasks.append(task)
        
        # Add false branch tasks if provided
        if false_branch:
            false_tasks = false_branch.build().tasks
            for task in false_tasks:
                task.execution_group += self.current_group
                task.conditions = [ConditionalExecution(
                    condition_task_id=condition_task.id,
                    condition_value=False
                )]
                self.tasks.append(task)
        
        self.current_group += max(len(true_tasks), len(false_tasks) if false_branch else 0)
        return self
    
    def scatter_gather(self, 
                      scatter_tasks: List[Union[str, 'ToolCall']],
                      gather_task: Union[str, 'ToolCall']) -> 'WorkflowBuilder':
        """Execute scatter-gather pattern."""
        self.execution_mode = ExecutionMode.SCATTER_GATHER
        
        # Add scatter tasks
        for tool_call in scatter_tasks:
            task = self._create_task(tool_call)
            task.execution_group = self.current_group
            self.tasks.append(task)
        
        self.current_group += 1
        
        # Add gather task
        gather_task_obj = self._create_task(gather_task)
        gather_task_obj.execution_group = self.current_group
        self.tasks.append(gather_task_obj)
        
        self.current_group += 1
        return self
    
    def retry(self, attempts: int, backoff: float = 1.0) -> 'WorkflowBuilder':
        """Add retry logic to last added tasks."""
        for task in self.tasks:
            if task.execution_group == self.current_group - 1:
                task.max_retries = attempts
                task.retry_backoff = backoff
        
        return self
    
    def timeout(self, seconds: float) -> 'WorkflowBuilder':
        """Add timeout to last added tasks."""
        for task in self.tasks:
            if task.execution_group == self.current_group - 1:
                task.timeout_seconds = seconds
        
        return self
    
    def fail_fast(self, enabled: bool = True) -> 'WorkflowBuilder':
        """Configure fail-fast behavior."""
        for task in self.tasks:
            if task.execution_group == self.current_group - 1:
                task.fail_fast = enabled
        
        return self
    
    def transform_input(self, transformation: 'InputTransformation') -> 'WorkflowBuilder':
        """Add input transformation to last added tasks."""
        for task in self.tasks:
            if task.execution_group == self.current_group - 1:
                task.input_transformations.append(transformation)
        
        return self
    
    def transform_output(self, transformation: 'OutputTransformation') -> 'WorkflowBuilder':
        """Add output transformation to last added tasks."""
        for task in self.tasks:
            if task.execution_group == self.current_group - 1:
                task.output_transformations.append(transformation)
        
        return self
    
    def set_priority(self, priority: int) -> 'WorkflowBuilder':
        """Set execution priority for last added tasks."""
        for task in self.tasks:
            if task.execution_group == self.current_group - 1:
                task.priority = priority
        
        return self
    
    def with_config(self, **kwargs) -> 'WorkflowBuilder':
        """Set global workflow configuration."""
        self.global_config.update(kwargs)
        return self
    
    def build(self) -> 'WorkflowSpecification':
        """Build the workflow specification."""
        return WorkflowSpecification(
            name=self.global_config.get('name', 'untitled_workflow'),
            description=self.global_config.get('description', ''),
            execution_mode=self.execution_mode,
            tasks=self.tasks,
            global_timeout=self.global_config.get('global_timeout'),
            max_concurrency=self.global_config.get('max_concurrency', 10),
            resource_limits=self.global_config.get('resource_limits', {}),
            metadata=self.global_config
        )
    
    def _create_task(self, tool_call: Union[str, 'ToolCall']) -> 'WorkflowTask':
        """Create workflow task from tool call."""
        if isinstance(tool_call, str):
            # Simple tool name
            return WorkflowTask(
                id=f"task_{len(self.tasks)}",
                tool_id=tool_call,
                input_data={},
                execution_group=self.current_group
            )
        else:
            # Structured tool call
            return WorkflowTask(
                id=tool_call.id or f"task_{len(self.tasks)}",
                tool_id=tool_call.tool_id,
                input_data=tool_call.input_data,
                execution_group=self.current_group
            )


# Usage Examples:

# Simple parallel execution
workflow = (WorkflowBuilder()
    .parallel("search_documents", "get_embeddings", "analyze_content")
    .retry(3, backoff=2.0)
    .timeout(30.0)
    .build())

# Complex pipeline with transformations
workflow = (WorkflowBuilder()
    .sequence("extract_urls")
    .transform_output(OutputTransformation(
        source="result.urls",
        target="url_list",
        transformation="extract_field"
    ))
    .parallel("crawl_url_1", "crawl_url_2", "crawl_url_3")
    .pipeline("aggregate_content", "generate_summary", "create_embeddings")
    .with_config(
        name="content_processing_pipeline",
        description="Extract, crawl, and process content from URLs",
        max_concurrency=5
    )
    .build())

# Conditional workflow
search_workflow = WorkflowBuilder().sequence("search_documents").retry(2)
fallback_workflow = WorkflowBuilder().sequence("fallback_search", "generate_suggestions")

workflow = (WorkflowBuilder()
    .conditional(
        condition="lambda ctx: len(ctx.get('search_results', [])) > 0",
        true_branch=search_workflow,
        false_branch=fallback_workflow
    )
    .sequence("format_response")
    .build())

# Scatter-gather pattern
workflow = (WorkflowBuilder()
    .scatter_gather(
        scatter_tasks=["search_news", "search_docs", "search_web"],
        gather_task="aggregate_results"
    )
    .sequence("rank_results", "format_response")
    .build())
```

### 3. Workflow Models

```python
# src/mcp_tools/composition/models.py

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, Callable
from enum import Enum
import uuid

class ToolCall(BaseModel):
    """Structured tool call specification."""
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_id: str = Field(description="ID of the tool to call")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data for the tool")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class InputTransformation(BaseModel):
    """Input data transformation specification."""
    source: str = Field(description="Source field path")
    target: str = Field(description="Target field path")
    transformation: str = Field(description="Transformation type")
    parameters: Dict[str, Any] = Field(default_factory=dict)

class OutputTransformation(BaseModel):
    """Output data transformation specification."""
    source: str = Field(description="Source field path")
    target: str = Field(description="Target field path")
    transformation: str = Field(description="Transformation type")
    parameters: Dict[str, Any] = Field(default_factory=dict)

class ConditionalExecution(BaseModel):
    """Conditional execution specification."""
    condition_task_id: str = Field(description="Task ID that provides condition result")
    condition_value: Any = Field(description="Value to match for execution")
    operator: str = Field(default="equals", description="Comparison operator")

class WorkflowTask(BaseModel):
    """Individual task within a workflow."""
    id: str = Field(description="Unique task identifier")
    tool_id: str = Field(description="Tool to execute")
    input_data: Dict[str, Any] = Field(default_factory=dict)
    execution_group: int = Field(default=0, description="Execution group for parallelization")
    dependencies: List[str] = Field(default_factory=list, description="Task dependencies")
    conditions: List[ConditionalExecution] = Field(default_factory=list)
    
    # Execution configuration
    max_retries: int = Field(default=0, description="Maximum retry attempts")
    retry_backoff: float = Field(default=1.0, description="Retry backoff factor")
    timeout_seconds: Optional[float] = Field(default=None)
    fail_fast: bool = Field(default=False, description="Fail entire workflow on task failure")
    priority: int = Field(default=0, description="Task execution priority")
    
    # Transformations
    input_transformations: List[InputTransformation] = Field(default_factory=list)
    output_transformations: List[OutputTransformation] = Field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

class WorkflowSpecification(BaseModel):
    """Complete workflow specification."""
    name: str = Field(description="Workflow name")
    description: str = Field(default="", description="Workflow description")
    version: str = Field(default="1.0.0", description="Workflow version")
    
    # Execution configuration
    execution_mode: ExecutionMode = Field(default=ExecutionMode.SEQUENTIAL)
    tasks: List[WorkflowTask] = Field(description="Workflow tasks")
    
    # Resource limits
    global_timeout: Optional[float] = Field(default=None, description="Global workflow timeout")
    max_concurrency: int = Field(default=10, description="Maximum concurrent tasks")
    resource_limits: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_by: Optional[str] = Field(default=None)
    created_at: Optional[float] = Field(default=None)

class Workflow(BaseModel):
    """Workflow instance."""
    id: str = Field(description="Unique workflow instance ID")
    specification: WorkflowSpecification = Field(description="Workflow specification")
    dependency_graph: 'DependencyGraph' = Field(description="Resolved dependency graph")
    context: ExecutionContext = Field(description="Execution context")
    created_at: float = Field(description="Creation timestamp")
    
    class Config:
        arbitrary_types_allowed = True

class WorkflowResult(BaseModel):
    """Result of workflow execution."""
    workflow_id: str
    state: ExecutionState
    task_results: Dict[str, TaskResult]
    start_time: float
    end_time: float
    total_execution_time: float
    context: ExecutionContext
    performance_metrics: Optional[Dict[str, Any]] = None
    error_summary: Optional[str] = None
    
    def get_successful_results(self) -> Dict[str, Any]:
        """Get results from successful tasks."""
        return {
            task_id: result.result 
            for task_id, result in self.task_results.items()
            if result.state == ExecutionState.COMPLETED and result.result is not None
        }
    
    def get_failed_tasks(self) -> List[str]:
        """Get list of failed task IDs."""
        return [
            task_id for task_id, result in self.task_results.items()
            if result.state == ExecutionState.FAILED
        ]
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary statistics."""
        total_tasks = len(self.task_results)
        successful_tasks = len([r for r in self.task_results.values() if r.state == ExecutionState.COMPLETED])
        failed_tasks = len([r for r in self.task_results.values() if r.state == ExecutionState.FAILED])
        
        return {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0,
            "total_execution_time": self.total_execution_time,
            "avg_task_time": sum(r.execution_time for r in self.task_results.values()) / total_tasks if total_tasks > 0 else 0
        }
```

### 4. Performance Optimization

```python
# src/mcp_tools/composition/optimization.py

class WorkflowOptimizer:
    """Intelligent workflow optimization engine."""
    
    async def optimize(self, 
                      workflow_spec: WorkflowSpecification,
                      dependency_graph: 'DependencyGraph') -> WorkflowSpecification:
        """Optimize workflow for performance."""
        
        optimized_spec = workflow_spec.copy()
        
        # Apply optimization strategies
        optimized_spec = await self._optimize_parallelization(optimized_spec, dependency_graph)
        optimized_spec = await self._optimize_resource_allocation(optimized_spec)
        optimized_spec = await self._optimize_data_flow(optimized_spec)
        optimized_spec = await self._optimize_caching(optimized_spec)
        
        return optimized_spec
    
    async def _optimize_parallelization(self, 
                                      spec: WorkflowSpecification,
                                      dependency_graph: 'DependencyGraph') -> WorkflowSpecification:
        """Optimize task parallelization based on dependencies."""
        
        # Analyze dependency graph to find parallelization opportunities
        parallel_groups = dependency_graph.find_parallel_groups()
        
        # Update execution groups for optimal parallelization
        for group_id, tasks in enumerate(parallel_groups):
            for task_id in tasks:
                task = next(t for t in spec.tasks if t.id == task_id)
                task.execution_group = group_id
        
        return spec
    
    async def _optimize_resource_allocation(self, 
                                          spec: WorkflowSpecification) -> WorkflowSpecification:
        """Optimize resource allocation based on task characteristics."""
        
        # Analyze task resource requirements
        for task in spec.tasks:
            tool_metadata = await self._get_tool_metadata(task.tool_id)
            if tool_metadata:
                performance_profile = tool_metadata.performance_profile
                
                # Adjust priority based on resource intensity
                if performance_profile.cpu_intensive:
                    task.priority = max(task.priority, 5)
                if performance_profile.io_intensive:
                    task.priority = min(task.priority, 2)
        
        return spec
    
    async def _optimize_data_flow(self, 
                                spec: WorkflowSpecification) -> WorkflowSpecification:
        """Optimize data flow between tasks."""
        
        # Analyze data dependencies and optimize transformations
        data_flow_graph = await self._build_data_flow_graph(spec)
        
        # Optimize transformations
        for task in spec.tasks:
            if task.input_transformations:
                task.input_transformations = await self._optimize_transformations(
                    task.input_transformations, data_flow_graph
                )
        
        return spec
    
    async def _optimize_caching(self, 
                              spec: WorkflowSpecification) -> WorkflowSpecification:
        """Add intelligent caching to workflow tasks."""
        
        for task in spec.tasks:
            tool_metadata = await self._get_tool_metadata(task.tool_id)
            if tool_metadata and self._is_cacheable(tool_metadata):
                # Add caching metadata
                task.metadata["cache_enabled"] = True
                task.metadata["cache_ttl"] = self._calculate_cache_ttl(tool_metadata)
        
        return spec
```

## Integration Points

### 1. MCP Server Integration

The composition engine integrates seamlessly with the MCP server:
- Workflows are exposed as high-level MCP tools
- Individual workflow steps use existing MCP tools
- Results are streamed back through MCP protocol

### 2. Performance Monitoring

All workflow executions are monitored for:
- Task-level performance metrics
- Resource utilization tracking
- Bottleneck identification
- Optimization recommendations

### 3. Error Handling and Recovery

Advanced error handling features:
- Circuit breaker pattern for failing tools
- Automatic retry with exponential backoff
- Graceful degradation strategies
- Workflow rollback capabilities

## Benefits

1. **Complex Task Orchestration**: Handle sophisticated multi-tool workflows
2. **Performance Optimization**: Automatic parallelization and resource optimization
3. **Flexible Execution**: Support multiple execution patterns (sequential, parallel, pipeline, etc.)
4. **Error Resilience**: Robust error handling and recovery mechanisms
5. **Developer Experience**: Intuitive DSL for workflow construction
6. **Enterprise Ready**: Comprehensive monitoring and management capabilities

This tool composition engine enables the MCP server to handle enterprise-grade workflows with sophisticated orchestration, optimization, and monitoring capabilities.