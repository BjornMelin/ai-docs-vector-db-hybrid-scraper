#!/bin/bash
# Start application in simple mode

set -e

echo "🚀 Starting AI Docs Vector DB in Simple Mode"
echo "📊 Mode: Solo Developer Optimized (25K lines)"
echo "🎯 Target: Minimal complexity, fast startup"
echo ""

# Set simple mode environment
export AI_DOCS_MODE=simple

# Load simple mode configuration
if [ -f .env.simple ]; then
    echo "📄 Loading simple mode configuration from .env.simple"
    export $(cat .env.simple | grep -v '^#' | xargs)
else
    echo "⚠️  .env.simple not found, using environment defaults"
fi

# Display mode information
echo "Mode Configuration:"
echo "  - Max Concurrent Crawls: ${AI_DOCS_CRAWL4AI__MAX_CONCURRENT_CRAWLS:-5}"
echo "  - Max Memory Usage: ${AI_DOCS_PERFORMANCE__MAX_MEMORY_USAGE_MB:-500}MB"
echo "  - Cache Size: ${AI_DOCS_CACHE__LOCAL_MAX_MEMORY_MB:-50}MB"
echo "  - Advanced Monitoring: ${AI_DOCS_MONITORING__ENABLE_METRICS:-false}"
echo "  - Distributed Cache: ${AI_DOCS_CACHE__ENABLE_DRAGONFLY_CACHE:-false}"
echo ""

# Check if running with Docker Compose
if [ "$1" = "--docker" ]; then
    echo "🐳 Starting with Docker Compose (simple mode)"
    docker-compose -f docker-compose.simple.yml up --build
elif [ "$1" = "--docker-bg" ]; then
    echo "🐳 Starting with Docker Compose in background (simple mode)"
    docker-compose -f docker-compose.simple.yml up --build -d
else
    # Check if virtual environment is activated
    if [ -z "$VIRTUAL_ENV" ] && [ ! -f "pyproject.toml" ]; then
        echo "⚠️  No virtual environment detected. Consider using uv or activating a venv."
    fi
    
    # Start services if needed
    echo "🔧 Starting required services..."
    if ! curl -s http://localhost:6333/health > /dev/null 2>&1; then
        echo "📦 Starting Qdrant vector database..."
        docker run -d --name qdrant-simple -p 6333:6333 qdrant/qdrant:latest || echo "Qdrant may already be running"
    else
        echo "✅ Qdrant already running"
    fi
    
    # Wait for services
    echo "⏳ Waiting for services to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:6333/health > /dev/null 2>&1; then
            echo "✅ Services ready!"
            break
        fi
        if [ $i -eq 30 ]; then
            echo "❌ Services failed to start within 30 seconds"
            exit 1
        fi
        sleep 1
    done
    
    echo ""
    echo "🎉 Starting application in simple mode..."
    
    # Start with uv if available, otherwise fallback to python
    if command -v uv &> /dev/null; then
        echo "📦 Using uv to run application..."
        uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
    else
        echo "🐍 Using python to run application..."
        python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
    fi
fi

echo ""
echo "🌐 Application running at: http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo "❤️  Health Check: http://localhost:8000/health"
echo "ℹ️  Mode Info: http://localhost:8000/mode"