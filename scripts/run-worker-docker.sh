#!/bin/bash

# Run the ARQ worker using Docker Compose

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting ARQ task queue worker with Docker...${NC}"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running${NC}"
    exit 1
fi

# Check if services are running
if ! docker-compose ps | grep -q "dragonfly.*Up"; then
    echo -e "${YELLOW}Starting required services...${NC}"
    docker-compose up -d qdrant dragonfly
    sleep 5
fi

# Build and start the worker
echo -e "${GREEN}Building and starting worker container...${NC}"
docker-compose --profile worker up --build task-worker

# Note: The worker will run until stopped with Ctrl+C