#!/bin/bash
# Docker Build Test Script for AI Docs Vector DB Hybrid Scraper
# Tests multiple Dockerfile variants to identify and resolve build issues

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

print_status $BLUE "=== Docker Build Test Suite ==="
print_status $BLUE "Testing multiple Dockerfile variants to resolve UV build issues"
echo

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    print_status $RED "❌ Docker is not installed or not available in PATH"
    exit 1
fi

print_status $GREEN "✅ Docker is available"

# Array of Dockerfiles to test
declare -a dockerfiles=(
    "Dockerfile:ai-docs-original"
    "Dockerfile.fixed:ai-docs-fixed" 
    "Dockerfile.simple:ai-docs-simple"
)

# Test each Dockerfile
for dockerfile_pair in "${dockerfiles[@]}"; do
    IFS=':' read -r dockerfile tag <<< "$dockerfile_pair"
    
    print_status $YELLOW "🔨 Testing $dockerfile -> $tag"
    
    # Check if Dockerfile exists
    if [[ ! -f "$dockerfile" ]]; then
        print_status $RED "❌ $dockerfile not found, skipping..."
        continue
    fi
    
    # Attempt to build the Docker image
    if docker build -f "$dockerfile" -t "$tag" .; then
        print_status $GREEN "✅ $dockerfile built successfully"
        
        # Test if the image can start and respond
        print_status $YELLOW "🧪 Testing container startup..."
        
        # Start container in background
        if container_id=$(docker run -d -p 8000:8000 "$tag"); then
            sleep 10  # Give it time to start
            
            # Test if the application responds
            if curl -f http://localhost:8000/api/v1/config/status >/dev/null 2>&1; then
                print_status $GREEN "✅ Container started and responds to health check"
            else
                print_status $YELLOW "⚠️  Container started but health check failed"
                docker logs "$container_id" | tail -20
            fi
            
            # Clean up
            docker stop "$container_id" >/dev/null
            docker rm "$container_id" >/dev/null
        else
            print_status $RED "❌ Container failed to start"
        fi
        
    else
        print_status $RED "❌ $dockerfile build failed"
        
        # Show last few lines of build output for debugging
        print_status $YELLOW "Last build output:"
        docker build -f "$dockerfile" -t "$tag" . 2>&1 | tail -20 || true
    fi
    
    echo "----------------------------------------"
done

print_status $BLUE "=== Docker Build Test Summary ==="

# List successful builds
print_status $YELLOW "Built images:"
docker images | grep ai-docs || print_status $RED "No ai-docs images found"

echo
print_status $BLUE "=== Cleanup Instructions ==="
print_status $YELLOW "To clean up test images, run:"
print_status $YELLOW "  docker rmi ai-docs-original ai-docs-fixed ai-docs-simple"
print_status $YELLOW "To clean up all unused images:"
print_status $YELLOW "  docker image prune -f"

echo
print_status $GREEN "Docker build testing complete!"