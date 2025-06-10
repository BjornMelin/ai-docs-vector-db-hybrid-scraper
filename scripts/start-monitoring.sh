#!/bin/bash

# ML Application Monitoring Stack Startup Script
# Starts Prometheus, Grafana, and Alertmanager for comprehensive observability

set -e

echo "🚀 Starting ML Application Monitoring Stack..."

# Create monitoring network if it doesn't exist
echo "📡 Creating monitoring network..."
docker network create monitoring-observability-vect_18478f295ebe501a_monitoring 2>/dev/null || echo "Network already exists"

# Start monitoring stack
echo "📊 Starting Prometheus, Grafana, and Alertmanager..."
docker-compose -f docker-compose.monitoring.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check service health
echo "🩺 Checking service health..."

# Check Prometheus
if curl -s http://localhost:9090/-/healthy > /dev/null; then
    echo "✅ Prometheus is healthy"
else
    echo "❌ Prometheus is not responding"
fi

# Check Grafana
if curl -s http://localhost:3000/api/health > /dev/null; then
    echo "✅ Grafana is healthy"
else
    echo "❌ Grafana is not responding"
fi

# Check Alertmanager
if curl -s http://localhost:9093/-/healthy > /dev/null; then
    echo "✅ Alertmanager is healthy"
else
    echo "❌ Alertmanager is not responding"
fi

echo ""
echo "🎯 Monitoring Stack URLs:"
echo "📊 Prometheus: http://localhost:9090"
echo "📈 Grafana: http://localhost:3000 (admin/admin123)"
echo "🚨 Alertmanager: http://localhost:9093"
echo ""
echo "📋 Next steps:"
echo "1. Start main application: docker-compose up -d"
echo "2. Access Grafana dashboards to view metrics"
echo "3. Configure alert notifications in Alertmanager"
echo ""
echo "✨ Monitoring stack is ready!"