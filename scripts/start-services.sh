#!/bin/bash
# Service startup script for AI Documentation Vector Database

echo "🚀 Starting AI Documentation Vector Database Services..."

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Docker is not running. Please start Docker Desktop."
    exit 1
fi

# Start Qdrant service
echo "🐳 Starting Qdrant vector database..."
docker-compose up -d qdrant-vector-db

# Wait for Qdrant to be ready
echo "⏳ Waiting for Qdrant to be ready..."
until curl -s http://localhost:6333/health > /dev/null; do
    sleep 2
    echo "   Checking Qdrant health..."
done

echo "✅ Services Status:"
echo "   Qdrant: $(curl -s http://localhost:6333/health | jq -r .status 2>/dev/null || echo "Running")"
echo "   Data Directory: $(realpath ~/.qdrant_data)"

# Show collection stats if available
if curl -s http://localhost:6333/collections/documents > /dev/null 2>&1; then
    echo "📊 Vector Database Stats:"
    COLLECTION_INFO=$(curl -s http://localhost:6333/collections/documents)
    echo "   Collection: documents"
    echo "   Total Vectors: $(echo $COLLECTION_INFO | jq -r .result.points_count 2>/dev/null || echo "N/A")"
    echo "   Status: $(echo $COLLECTION_INFO | jq -r .result.status 2>/dev/null || echo "Ready")"
else
    echo "📊 Vector Database: Ready for first use"
fi

echo ""
echo "🎯 Next Steps:"
echo "   1. Set environment variables: export OPENAI_API_KEY='your_key'"
echo "   2. Run bulk scraper: python src/crawl4ai_bulk_embedder.py"
echo "   3. Configure Claude Desktop with provided MCP config"
echo ""
echo "🔗 Access Points:"
echo "   Qdrant Dashboard: http://localhost:6333/dashboard"
echo "   Health Check: http://localhost:6333/health"
