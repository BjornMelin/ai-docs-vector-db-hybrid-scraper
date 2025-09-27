.PHONY: quality-unit
quality-unit:
	uv run ruff format src tests/unit --check
	uv run ruff check src tests/unit
	uv run pylint src tests/unit
	uv run pyright src tests/unit
	uv run pytest tests/unit -q
