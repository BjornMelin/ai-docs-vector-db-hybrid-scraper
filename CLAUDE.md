# CLAUDE.md

> **Inheritance**
> This repo inherits the global Simplicity Charter (`~/.claude/CLAUDE.md`),  
> which defines all coding, linting, testing, security and library choices  
> (uv, ruff, Pydantic v2, ≥ 80% pytest-cov, secrets in .env, etc.).
> Google format docstrings for Python code.
> Project-specific implementation context is maintained in:  
> • `TODO.md` - comprehensive task list with V1/V2 implementation roadmap  
> • `TODO-V2.md` - future feature roadmap post-V1 release  
> Always read these before coding.

## Core Project Commands

**Note:** Prefix all Python and test commands with `uv` since we're in a uv-managed environment.

- Test: `uv run pytest --cov=src`
- Benchmark: `uv run pytest tests/benchmarks/ --benchmark-only`
- Lint: `ruff check . --fix && ruff format .`
- Services: `./scripts/start-services.sh`
- Health: `curl localhost:6333/health`

## Development Workflow

### TDD Process

1. **Write tests** based on expected input/output pairs, commit
2. **Run tests** to confirm they fail (no implementation yet)
3. **Write code** to pass tests without modifying tests
4. **Iterate** until all tests pass, then commit
5. **Typecheck** when done with code changes

### Testing Quality Standards

**Testing Anti-Patterns to AVOID:**

❌ **Coverage-Driven Testing**: Never create tests solely to hit coverage metrics or line numbers  
❌ **Implementation Detail Testing**: Don't test private methods or internal implementation details  
❌ **Heavy Internal Mocking**: Avoid over-mocking internal components; mock at boundaries  
❌ **Shared Mutable State**: Tests that depend on execution order or share state between functions  
❌ **Magic Values**: Hardcoded test data without clear business meaning  
❌ **Timing-Dependent Tests**: Don't rely on real timers or timing for async/cache tests  
❌ **Giant Test Functions**: Tests that verify multiple behaviors in a single function

**Testing Best Practices to FOLLOW:**

✅ **Functional Organization**: Structure tests by business functionality (Unit, Integration, E2E)  
✅ **Behavior-Driven Testing**: Test observable behavior, not implementation details  
✅ **Boundary Mocking**: Mock external services (APIs, databases) not internal logic  
✅ **AAA Pattern**: Arrange, Act, Assert structure for clear test flow  
✅ **Property-Based Testing**: Use `hypothesis` for data generation and edge case discovery  
✅ **Async Test Patterns**: Proper `pytest-asyncio` usage with `respx` for HTTP mocking  
✅ **Descriptive Test Names**: Names that explain business value and expected behavior

**Test Organization Standards:**

```text
tests/
├── conftest.py              # Shared fixtures and configuration
├── unit/                    # Fast, isolated tests (<100ms each)
│   ├── test_embeddings.py   # Core business logic
│   ├── test_search.py       # Search algorithms
│   └── test_chunking.py     # Text processing
├── integration/             # Component integration (<5s each)
│   ├── test_api_endpoints.py
│   ├── test_vector_db.py
│   └── test_crawling.py
├── e2e/                     # Full workflow tests
│   └── test_complete_pipeline.py
└── fixtures/                # Test data and samples
    ├── sample_documents.json
    └── test_embeddings.pkl
```

**AI/ML Testing Patterns:**

```python
# Mock expensive operations at boundaries
@pytest.fixture
def mock_openai_embeddings(respx_mock):
    respx_mock.post("https://api.openai.com/v1/embeddings").mock(
        return_value=httpx.Response(200, json={"data": [{"embedding": [0.1] * 1536}]})
    )

# Test properties, not exact values
@pytest.mark.parametrize("text", ["short", "a much longer text with more content"])
def test_embedding_dimensions(text, embedding_service):
    """Embeddings should have consistent dimensions regardless of input length"""
    embedding = embedding_service.generate_embedding(text)
    assert len(embedding) == 1536
    assert all(isinstance(x, float) for x in embedding)

# Use hypothesis for property-based testing
from hypothesis import given, strategies as st

@given(st.lists(st.text(min_size=1), min_size=1, max_size=10))
def test_batch_processing_preserves_count(documents, processor):
    """Processing N documents should return N results"""
    results = processor.process_batch(documents)
    assert len(results) == len(documents)
```

**Test Quality Checklist:**

- [ ] Tests organized by functionality, not coverage metrics
- [ ] All scenarios represent realistic usage patterns
- [ ] `respx` used for HTTP mocking with proper setup/teardown
- [ ] Proper async/await patterns with `pytest-asyncio`
- [ ] No artificial timing dependencies or private method testing
- [ ] Test names describe business value and expected behavior
- [ ] 80%+ coverage achieved through meaningful scenarios, not line-targeting
- [ ] AI/ML operations tested for properties (dimensions, types) not exact values

### Tool Usage

- **Parallel execution:** Invoke multiple independent tools simultaneously, not sequentially
- **Thinking:** Reflect on tool results quality before proceeding; use thinking to plan next steps
- **Testing:** Prefer running single tests over full test suite for performance
- **Use checklists** for complex workflows - create Markdown files to track progress on multi-step tasks

### Frontend Development Instructions

When implementing frontend/visual components:

- Don't hold back. Give it your all.
- Include all relevant user interactions and interface elements
- Add hover states, smooth transitions, and thoughtful micro-interactions
- Apply visual design principles: establish clear hierarchy, use effective contrast, maintain visual balance, and create purposeful movement
- Prioritize user experience through polished details and responsive behavior

### Extended Thinking

Use these commands to trigger progressively deeper analysis:

- `think` - basic extended thinking for evaluating alternatives
- `think hard` - increased thinking budget for complex problems
- `think harder` - higher computation time for thorough evaluation
- `ultrathink` - maximum thinking budget for the most complex scenarios

Each level allocates progressively more thinking budget for Claude to use.

### File Management

- **Avoid temporary files:** Don't create temporary scripts or helper files for iteration
- **Clean up:** If temporary files are created, remove them at task completion
- **Use exact paths:** Reference files like `src/services/embeddings/manager.py`

### Guidelines

- **Conventional commits:** Use conventional-commits format for branch names, commits, pull requests, and GitHub issues
- **Targets:** Set measurable goals like "reduce latency to <100ms for 95th percentile"

---

## Task Master AI - Claude Code Integration Guide

### Key Files & Project Structure

#### Core Files

- `.taskmaster/tasks/tasks.json` - Main task data file (auto-managed)
- `.taskmaster/config.json` - AI model configuration (use `task-master models` to modify)
- `.taskmaster/docs/prd.txt` - Product Requirements Document for parsing
- `.taskmaster/tasks/*.txt` - Individual task files (auto-generated from tasks.json)

## Task Master AI - MCP Integration

**Essential MCP Tools & CLI Commands:**

```javascript
// === PROJECT SETUP ===
initialize_project; // task-master init
parse_prd; // task-master parse-prd .taskmaster/docs/prd.txt
parse_prd({ append: true }); // task-master parse-prd --append (for additional PRDs)

// === DAILY WORKFLOW ===
next_task; // task-master next (find next available task)
get_task({ id: "1.2" }); // task-master show 1.2 (view task details)
set_task_status({ id: "1.2", status: "in-progress" }); // task-master set-status --id=1.2 --status=in-progress
set_task_status({ id: "1.2", status: "done" }); // task-master set-status --id=1.2 --status=done

// === TASK MANAGEMENT ===
get_tasks; // task-master list (show all tasks)
add_task({ prompt: "description", research: true }); // task-master add-task --prompt="..." --research
expand_task({ id: "1", research: true }); // task-master expand --id=1 --research
expand_all({ research: true }); // task-master expand --all --research
update_task({ id: "1", prompt: "changes" }); // task-master update-task --id=1 --prompt="..."
update_subtask({ id: "1.2", prompt: "notes" }); // task-master update-subtask --id=1.2 --prompt="..."
update({ from: "3", prompt: "changes" }); // task-master update --from=3 --prompt="..."

// === ANALYSIS & ORGANIZATION ===
analyze_project_complexity({ research: true }); // task-master analyze-complexity --research
complexity_report; // task-master complexity-report
add_dependency({ id: "2", dependsOn: "1" }); // task-master add-dependency --id=2 --depends-on=1
move_task({ from: "2", to: "3" }); // task-master move --from=2 --to=3

// === MAINTENANCE ===
generate; // task-master generate (regenerate task files)
validate_dependencies; // task-master validate-dependencies
fix_dependencies; // task-master fix-dependencies
help; // shows available commands
```

**Quick Start Workflow:**

1. `initialize_project` → `parse_prd` → `analyze_project_complexity` → `expand_all`
2. Daily: `next_task` → `get_task` → work → `update_subtask` → `set_task_status`

## Task Structure & IDs

### Task ID Format

- Main tasks: `1`, `2`, `3`, etc.
- Subtasks: `1.1`, `1.2`, `2.1`, etc.
- Sub-subtasks: `1.1.1`, `1.1.2`, etc.

### Task Status Values

- `pending` - Ready to work on
- `in-progress` - Currently being worked on
- `done` - Completed and verified
- `deferred` - Postponed
- `cancelled` - No longer needed
- `blocked` - Waiting on external factors

### Task Fields

```json
{
  "id": "1.2",
  "title": "Implement user authentication",
  "description": "Set up JWT-based auth system",
  "status": "pending",
  "priority": "high",
  "dependencies": ["1.1"],
  "details": "Use bcrypt for hashing, JWT for tokens...",
  "testStrategy": "Unit tests for auth functions, integration tests for login flow",
  "subtasks": []
}
```

## Implementation Workflow

**Iterative Development Process:**

1. `get_task({id: "subtask-id"})` - Understand requirements
2. Explore codebase and plan implementation
3. `update_subtask({id: "1.2", prompt: "detailed plan"})` - Log plan
4. `set_task_status({id: "1.2", status: "in-progress"})` - Start work
5. Implement code following logged plan
6. `update_subtask({id: "1.2", prompt: "what worked/didn't work"})` - Log progress
7. `set_task_status({id: "1.2", status: "done"})` - Complete task

## Important Notes

**AI-Powered Operations** (may take up to a minute): `parse_prd`, `analyze_project_complexity`, `expand_task`, `expand_all`, `add_task`, `update`, `update_task`, `update_subtask`

**File Management:**

- Never manually edit `tasks.json` or `.taskmaster/config.json`
- Task markdown files are auto-generated
- Use `generate` to regenerate task files after manual changes

**Research Mode:** Add `{research: true}` to any MCP call for enhanced AI analysis
