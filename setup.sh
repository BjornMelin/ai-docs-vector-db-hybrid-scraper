#!/bin/bash
# AI Documentation Vector Database Hybrid Scraper Setup Script
# Optimized for Python 3.11 and uv package manager with profile support

set -e

# Profile-aware setup with modern wizard integration
PROFILE=""
SKIP_WIZARD=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --profile|-p)
            PROFILE="$2"
            shift 2
            ;;
        --skip-wizard)
            SKIP_WIZARD=true
            shift
            ;;
        --help|-h)
            echo "AI Documentation Vector Database Hybrid Scraper Setup Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --profile, -p PROFILE    Use a specific configuration profile"
            echo "                          (personal, development, production, testing, local-only, minimal)"
            echo "  --skip-wizard           Skip the configuration wizard"
            echo "  --help, -h              Show this help message"
            echo ""
            echo "Available profiles:"
            echo "  personal     - Recommended for individual developers (default)"
            echo "  development  - Local development with debugging enabled"
            echo "  production   - High-performance production deployment"
            echo "  testing      - CI/CD pipeline and automated testing"
            echo "  local-only   - Privacy-focused, no cloud services"
            echo "  minimal      - Quick start with essential settings only"
            echo ""
            echo "Examples:"
            echo "  $0                        # Interactive wizard (recommended)"
            echo "  $0 --profile personal     # Quick setup with personal profile"
            echo "  $0 --profile production   # Production deployment setup"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

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

# Install Python 3.11 with UV
echo -e "${BLUE}🐍 Installing Python 3.11 with UV...${NC}"
uv python install 3.11

# Pin Python 3.11 for this project
echo -e "${BLUE}📌 Pinning Python 3.11 for project...${NC}"
uv python pin 3.11

# Create data directory
echo -e "${BLUE}📁 Creating data directory...${NC}"
mkdir -p ~/.qdrant_data
echo -e "${GREEN}✅ Data directory created at $(realpath ~/.qdrant_data)${NC}"

# Initialize UV project
echo -e "${BLUE}🔧 Initializing UV project...${NC}"
if [ ! -f "pyproject.toml" ]; then
    uv init --no-readme --python 3.11
fi

# Install project dependencies with UV
echo -e "${BLUE}📦 Installing project dependencies with UV...${NC}"
uv sync

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
docker compose --profile simple up -d qdrant

# Wait up to one minute for Qdrant to be ready
echo -e "${BLUE}⏳ Waiting for Qdrant to be ready...${NC}"
for attempt in {1..30}; do
    if curl --fail --silent http://localhost:6333/readyz >/dev/null; then
        break
    fi
    if [ "$attempt" -eq 30 ]; then
        echo -e "${RED}Qdrant did not become ready within 60 seconds.${NC}"
        exit 1
    fi
    sleep 2
done

# Run configuration wizard
if [ "$SKIP_WIZARD" = false ]; then
    echo -e "${BLUE}🧙 Running configuration wizard...${NC}"
    if [ -n "$PROFILE" ]; then
        echo -e "${YELLOW}📋 Using profile: $PROFILE${NC}"
        uv run python -m src.cli.main setup --profile "$PROFILE"
    else
        echo -e "${YELLOW}🎯 Interactive wizard (recommended for first-time setup)${NC}"
        uv run python -m src.cli.main setup
    fi
    echo -e "${GREEN}✅ Configuration complete!${NC}"
else
    echo -e "${YELLOW}⚠️ Wizard skipped - you'll need to configure manually${NC}"
fi

echo ""
echo -e "${GREEN}✅ Setup complete!${NC}"

# Show profile-specific next steps
if [ -n "$PROFILE" ]; then
    echo -e "${BLUE}🎯 Profile '$PROFILE' configured successfully${NC}"
fi

EMBEDDING_PROVIDER="${AI_DOCS_EMBEDDING_PROVIDER:-fastembed}"
CRAWL_PROVIDER="${AI_DOCS_CRAWL_PROVIDER:-crawl4ai}"

echo -e "${YELLOW}🔑 Provider credentials:${NC}"
if [ -f ".env.$PROFILE" ]; then
    echo -e "   📄 Environment variables template created: .env.$PROFILE"
    echo -e "   💡 Edit .env.$PROFILE if you select a hosted provider, then:"
    echo -e "      source .env.$PROFILE"
fi
if [ "$EMBEDDING_PROVIDER" = "openai" ]; then
    echo -e "   export AI_DOCS_OPENAI__API_KEY='your_openai_api_key'"
else
    echo -e "   ✅ $EMBEDDING_PROVIDER embeddings don't require provider credentials."
fi
if [ "$CRAWL_PROVIDER" = "firecrawl" ]; then
    echo -e "   export AI_DOCS_BROWSER__FIRECRAWL__API_KEY='your_firecrawl_api_key'"
else
    echo -e "   ✅ $CRAWL_PROVIDER crawling doesn't require provider credentials."
fi

echo ""
echo -e "${BLUE}🚀 Next Steps:${NC}"
echo -e "   1. Add provider credentials only if required above"
echo -e "   2. Test configuration: uv run python -m src.cli.main config validate"
echo -e "   3. Check system status: uv run python -m src.cli.main status"
echo -e "   4. Create your first collection: uv run python -m src.cli.main database create my-docs"
echo ""
echo -e "${GREEN}✨ Using Python $(uv python find --quiet) with UV package manager${NC}"
echo -e "${BLUE}💡 For help: uv run python -m src.cli.main --help${NC}"
