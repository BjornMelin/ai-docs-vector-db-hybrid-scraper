#!/bin/bash
# AI Documentation Vector Database Hybrid Scraper Setup Script
# Optimized for Python 3.13 and uv package manager

set -e

echo "🚀 Setting up AI Documentation Vector Database Hybrid Scraper..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if running in WSL
if ! grep -q microsoft /proc/version 2>/dev/null; then
    echo -e "${RED}❌ This setup is optimized for WSL2. Please run in WSL environment.${NC}"
    exit 1
fi

echo -e "${BLUE}📋 Checking system requirements...${NC}"

# Check Node.js
if ! command -v node &>/dev/null; then
    echo -e "${RED}❌ Node.js is required but not installed.${NC}"
    exit 1
fi

# Check Docker
if ! command -v docker &>/dev/null; then
    echo -e "${RED}❌ Docker is required but not installed.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ System requirements met${NC}"

# Install UV package manager (the fastest Python package manager)
echo -e "${BLUE}🦀 Installing UV package manager...${NC}"
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
    echo -e "${GREEN}✅ UV installed successfully${NC}"
else
    echo -e "${GREEN}✅ UV already installed${NC}"
fi

# Install Python 3.13 with UV
echo -e "${BLUE}🐍 Installing Python 3.13 with UV...${NC}"
uv python install 3.13

# Pin Python 3.13 for this project
echo -e "${BLUE}📌 Pinning Python 3.13 for project...${NC}"
uv python pin 3.13

# Create data directory
echo -e "${BLUE}📁 Creating data directory...${NC}"
mkdir -p ~/.qdrant_data
echo -e "${GREEN}✅ Data directory created at $(realpath ~/.qdrant_data)${NC}"

# Initialize UV project
echo -e "${BLUE}🔧 Initializing UV project...${NC}"
if [ ! -f "pyproject.toml" ]; then
    uv init --no-readme --python 3.13
fi

# Install project dependencies with UV
echo -e "${BLUE}📦 Installing project dependencies with UV...${NC}"
uv add crawl4ai[all] qdrant-client openai pandas numpy python-dotenv colorlog tqdm playwright httpx pydantic click rich
uv add --dev pytest pytest-asyncio black ruff

# Setup Crawl4AI
echo -e "${BLUE}🕷️ Setting up Crawl4AI...${NC}"
uv run crawl4ai-setup

# Install Firecrawl MCP server
echo -e "${BLUE}🔥 Installing Firecrawl MCP server...${NC}"
npm install -g firecrawl-mcp

# Make scripts executable
echo -e "${BLUE}🔧 Making scripts executable...${NC}"
chmod +x scripts/*.sh

# Start Docker services
echo -e "${BLUE}🐳 Starting Qdrant service...${NC}"
docker-compose up -d

# Wait for Qdrant to be ready
echo -e "${BLUE}⏳ Waiting for Qdrant to be ready...${NC}"
until curl -s http://localhost:6333/health >/dev/null; do
    sleep 2
done

echo -e "${GREEN}✅ Setup complete!${NC}"
echo -e "${YELLOW}🔑 Don't forget to set your API keys:${NC}"
echo -e "   export OPENAI_API_KEY='your_openai_api_key'"
echo -e "   export FIRECRAWL_API_KEY='your_firecrawl_api_key'"
echo ""
echo -e "${BLUE}🚀 Quick start with UV:${NC}"
echo -e "   1. Set your API keys (above)"
echo -e "   2. Run: uv run python src/crawl4ai_bulk_embedder.py"
echo -e "   3. Configure Claude Desktop with config/claude-mcp-config.json"
echo ""
echo -e "${GREEN}✨ Using Python $(uv python find --quiet) with UV package manager${NC}"
