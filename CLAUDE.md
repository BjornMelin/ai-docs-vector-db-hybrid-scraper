# CLAUDE.md

## Commands
- Test: `uv run pytest --cov=src`
- Lint: `ruff check . --fix && ruff format .`  
- Services: `./scripts/start-services.sh`
- Health: `curl localhost:6333/health`

## Workflow
- Use parallel tool calling for file reads and quality checks
- Say "Create plan but DO NOT code until approved" for complex tasks
- Write tests first, run quality checks before commits
- Use `/clear` between major task switches

## Guidelines  
- Use exact file paths: `src/services/embeddings/manager.py`
- Be specific: "test Qdrant hybrid search on localhost:6333"
- Set targets: "reduce latency to <100ms for 95th percentile"