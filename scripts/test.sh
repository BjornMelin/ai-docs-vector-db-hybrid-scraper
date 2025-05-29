#!/bin/bash
# Test runner script with clean output options

case "$1" in
"quick" | "unit")
    echo "🧪 Running unit tests..."
    uv run pytest tests/unit/ --tb=line --no-header -v
    ;;
"integration")
    echo "🔗 Running integration tests..."
    uv run pytest tests/integration/ --tb=line --no-header -v
    ;;
"performance")
    echo "⚡ Running performance tests..."
    uv run pytest tests/performance/ --tb=line --no-header -v
    ;;
"coverage" | "cov")
    echo "📊 Running tests with coverage..."
    uv run pytest --cov=src --cov-report=term-missing --tb=short
    ;;
"failed" | "lf")
    echo "❌ Running only failed tests..."
    uv run pytest --lf --tb=short -v
    ;;
"clean" | "summary")
    echo "📋 Running tests with clean summary..."
    uv run pytest tests/unit/ --tb=line -q
    ;;
"durations" | "slow")
    echo "⏱️  Finding slow tests..."
    uv run pytest --durations=10 --tb=no
    ;;
*)
    echo "🚀 Test Runner Options:"
    echo "  ./scripts/test.sh quick       - Fast unit tests only"
    echo "  ./scripts/test.sh integration - Integration tests"
    echo "  ./scripts/test.sh performance - Performance tests"
    echo "  ./scripts/test.sh coverage    - Tests with coverage report"
    echo "  ./scripts/test.sh failed      - Only previously failed tests"
    echo "  ./scripts/test.sh clean       - Clean summary (no random chars)"
    echo "  ./scripts/test.sh slow        - Show slowest tests"
    echo ""
    echo "Examples:"
    echo "  ./scripts/test.sh quick        # Clean, fast unit tests"
    echo "  ./scripts/test.sh clean        # Summary with pass/fail counts only"
    echo "  ./scripts/test.sh coverage     # Full coverage report"
    ;;
esac
