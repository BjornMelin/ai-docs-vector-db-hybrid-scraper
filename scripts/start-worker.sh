#!/bin/bash

# Start the ARQ worker for background task processing

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting ARQ task queue worker...${NC}"

# Check if Redis/DragonflyDB is running
if ! nc -z localhost 6379 2>/dev/null; then
    echo -e "${RED}Error: Redis/DragonflyDB is not running on port 6379${NC}"
    echo -e "${YELLOW}Please start services first with: ./scripts/start-services.sh${NC}"
    exit 1
fi

# Set Python path to include src directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Start the worker
echo -e "${GREEN}Starting worker with settings from src.services.task_queue.worker${NC}"
uv run arq src.services.task_queue.worker.WorkerSettings

# Note: The worker will run continuously until stopped with Ctrl+C