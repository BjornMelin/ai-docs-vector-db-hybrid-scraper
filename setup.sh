#!/bin/bash
# AI Documentation Vector Database Hybrid Scraper Setup Script

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

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is required but not installed.${NC}"
    exit 1
fi

# Check Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js is required but not installed.${NC}"
    exit 1
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is required but not installed.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ System requirements met${NC}"

# Create data directory
echo -e "${BLUE}📁 Creating data directory...${NC}"
mkdir -p ~/.qdrant_data
echo -e "${GREEN}✅ Data directory created at $(realpath ~/.qdrant_data)${NC}"

# Install Python dependencies
echo -e "${BLUE}🐍 Installing Python dependencies...${NC}"
pip install -r requirements.txt

# Install Crawl4AI setup
echo -e "${BLUE}🕷️ Setting up Crawl4AI...${NC}"
crawl4ai-setup

# Install UV for MCP servers
echo -e "${BLUE}📦 Installing UV package manager...${NC}"
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

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
until curl -s http://localhost:6333/health > /dev/null; do
    sleep 2
done

echo -e "${GREEN}✅ Setup complete!${NC}"
echo -e "${YELLOW}🔑 Don't forget to set your API keys:${NC}"
echo -e "   export OPENAI_API_KEY='your_openai_api_key'"
echo -e "   export FIRECRAWL_API_KEY='your_firecrawl_api_key'"
echo ""
echo -e "${BLUE}🚀 Quick start:${NC}"
echo -e "   1. Set your API keys (above)"
echo -e "   2. Run: python src/crawl4ai_bulk_embedder.py"
echo -e "   3. Configure Claude Desktop with config/claude-mcp-config.json"
