#!/bin/bash
echo "=== Docker Container Status ==="
docker ps | grep trading-system

echo ""
echo "=== Chart API Logs (last 20 lines) ==="
docker logs trading-system-chart-api --tail 20

echo ""
echo "=== Frontend Logs (last 20 lines) ==="
docker logs trading-system-frontend --tail 20

echo ""
echo "=== Testing API Health ==="
curl -s http://localhost:5000/api/health || echo "API not responding on port 5000"
curl -s http://localhost:3000/api/health || echo "API not responding through nginx"

