#!/bin/bash

# ML Application Monitoring Stack Shutdown Script
# Stops Prometheus, Grafana, and Alertmanager services

set -e

echo "🛑 Stopping ML Application Monitoring Stack..."

# Stop monitoring services
echo "📊 Stopping Prometheus, Grafana, and Alertmanager..."
docker-compose -f docker-compose.monitoring.yml down

echo "🧹 Cleaning up (optional - keeps data volumes)..."
echo "To remove monitoring data volumes, run:"
echo "docker-compose -f docker-compose.monitoring.yml down -v"

echo ""
echo "✅ Monitoring stack stopped successfully!"