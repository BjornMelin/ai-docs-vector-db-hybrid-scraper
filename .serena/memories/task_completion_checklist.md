# Task Completion Checklist

## Before Completion
1. **Format Code**: `ruff format .`
2. **Fix Linting**: `ruff check . --fix`
3. **Type Check**: `mypy src/ --config-file pyproject.toml`
4. **Run Tests**: `uv run pytest --cov=src`
5. **Validate Coverage**: Ensure ≥ 80% coverage
6. **Check Services**: `./scripts/start-services.sh` and `curl localhost:6333/health`

## Quality Gates
- [ ] No ruff linting errors
- [ ] No mypy type errors  
- [ ] All tests passing
- [ ] Coverage ≥ 80%
- [ ] No security issues (secrets in code)
- [ ] Docstrings for new functions/classes
- [ ] Type hints for all parameters/returns

## Security Checklist
- [ ] No hardcoded credentials
- [ ] Environment variables for secrets
- [ ] Input validation with Pydantic
- [ ] No SQL injection risks
- [ ] HTTPS for external requests

## Documentation
- [ ] Update CLAUDE.md if workflow changes
- [ ] Update TODO.md for completed items
- [ ] Add/update docstrings for new code
- [ ] Update type hints as needed

## Final Steps
- [ ] Commit changes with conventional commit format
- [ ] Run final test suite
- [ ] Verify no temporary files created