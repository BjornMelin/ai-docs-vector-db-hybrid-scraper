#!/bin/bash
# Start application in enterprise mode

set -e

echo "🚀 Starting AI Docs Vector DB in Enterprise Mode"
echo "📊 Mode: Full Feature Portfolio (70K lines)"
echo "🎯 Target: Complete enterprise capabilities"
echo ""

# Set enterprise mode environment
export AI_DOCS_MODE=enterprise

# Load enterprise mode configuration
if [ -f .env.enterprise ]; then
    echo "📄 Loading enterprise mode configuration from .env.enterprise"
    export $(cat .env.enterprise | grep -v '^#' | xargs)
else
    echo "⚠️  .env.enterprise not found, using environment defaults"
fi

# Display mode information
echo "Mode Configuration:"
echo "  - Max Concurrent Crawls: ${AI_DOCS_CRAWL4AI__MAX_CONCURRENT_CRAWLS:-50}"
echo "  - Max Memory Usage: ${AI_DOCS_PERFORMANCE__MAX_MEMORY_USAGE_MB:-4000}MB"
echo "  - Cache Size: ${AI_DOCS_CACHE__LOCAL_MAX_MEMORY_MB:-1000}MB"
echo "  - Advanced Monitoring: ${AI_DOCS_MONITORING__ENABLE_METRICS:-true}"
echo "  - Distributed Cache: ${AI_DOCS_CACHE__ENABLE_DRAGONFLY_CACHE:-true}"
echo "  - Observability: ${AI_DOCS_OBSERVABILITY__ENABLED:-true}"
echo "  - A/B Testing: ${AI_DOCS_DEPLOYMENT__ENABLE_AB_TESTING:-true}"
echo "  - Deployment Features: ${AI_DOCS_DEPLOYMENT__ENABLE_DEPLOYMENT_SERVICES:-true}"
echo ""

# Check if running with Docker Compose
if [ "$1" = "--docker" ]; then
    echo "🐳 Starting with Docker Compose (enterprise mode)"
    docker-compose -f docker-compose.enterprise.yml up --build
elif [ "$1" = "--docker-bg" ]; then
    echo "🐳 Starting with Docker Compose in background (enterprise mode)"
    docker-compose -f docker-compose.enterprise.yml up --build -d
else
    # Check if virtual environment is activated
    if [ -z "$VIRTUAL_ENV" ] && [ ! -f "pyproject.toml" ]; then
        echo "⚠️  No virtual environment detected. Consider using uv or activating a venv."
    fi
    
    # Start services if needed
    echo "🔧 Starting required services..."
    
    # Start Qdrant
    if ! curl -s http://localhost:6333/health > /dev/null 2>&1; then
        echo "📦 Starting Qdrant vector database..."
        docker run -d --name qdrant-enterprise -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest || echo "Qdrant may already be running"
    else
        echo "✅ Qdrant already running"
    fi
    
    # Start Redis/Dragonfly
    if ! redis-cli ping > /dev/null 2>&1; then
        echo "📦 Starting Redis/Dragonfly cache..."
        docker run -d --name redis-enterprise -p 6379:6379 docker.dragonflydb.io/dragonflydb/dragonfly dragonfly --logtostderr || echo "Redis may already be running"
    else
        echo "✅ Redis already running"
    fi
    
    # Wait for services
    echo "⏳ Waiting for services to be ready..."
    for i in {1..30}; do
        qdrant_ready=false
        redis_ready=false
        
        if curl -s http://localhost:6333/health > /dev/null 2>&1; then
            qdrant_ready=true
        fi
        
        if redis-cli ping > /dev/null 2>&1; then
            redis_ready=true
        fi
        
        if [ "$qdrant_ready" = true ] && [ "$redis_ready" = true ]; then
            echo "✅ All services ready!"
            break
        fi
        
        if [ $i -eq 30 ]; then
            echo "❌ Services failed to start within 30 seconds"
            echo "   Qdrant ready: $qdrant_ready"
            echo "   Redis ready: $redis_ready"
            exit 1
        fi
        sleep 1
    done
    
    echo ""
    echo "🎉 Starting application in enterprise mode..."
    
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
echo "📊 RedDoc Documentation: http://localhost:8000/redoc"
echo "❤️  Health Check: http://localhost:8000/health"
echo "ℹ️  Mode Info: http://localhost:8000/mode"
echo "📈 Metrics (if enabled): http://localhost:9090"
echo "📊 Grafana (if enabled): http://localhost:3000"
echo "🔍 Jaeger Tracing (if enabled): http://localhost:16686"