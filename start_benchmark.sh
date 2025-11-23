#!/bin/bash

echo "======================================"
echo "SEMANTIX BENCHMARK SYSTEM"
echo "======================================"
echo ""
echo "Starting components..."
echo ""

# Start web dashboard in background
echo "[1/2] Starting web dashboard server..."
python3 src/web_dashboard.py &
DASHBOARD_PID=$!

# Wait for server to start
sleep 3

# Open browser
echo "[2/2] Opening browser..."
open http://localhost:8080 2>/dev/null || xdg-open http://localhost:8080 2>/dev/null || echo "Please open: http://localhost:8080"

# Run benchmark
echo ""
echo "Starting benchmark..."
echo "======================================"
python3 run_benchmark.py

# Cleanup
kill $DASHBOARD_PID 2>/dev/null
