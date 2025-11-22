#!/bin/bash
# Semantix Demo Runner
# Runs different ablation modes for comparison

echo "========================================"
echo "Semantix Demo Script"
echo "========================================"
echo ""
echo "This script will run Semantix in different modes."
echo "Close the matplotlib window after each run to proceed."
echo ""

# Check if python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3.8+."
    exit 1
fi

# Check dependencies
echo "Checking dependencies..."
python3 -c "import pybullet, numpy, matplotlib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Missing dependencies. Installing..."
    pip3 install -r requirements.txt
fi

echo ""
echo "========================================"
echo "Run 1: FULL MODE (default)"
echo "All utility terms: hazard + entropy + glow"
echo "========================================"
read -p "Press Enter to start..."
python3 scout_semantix.py

echo ""
echo "========================================"
echo "Run 2: HAZARD ONLY"
echo "Greedy hazard-seeking (no exploration)"
echo "========================================"
read -p "Press Enter to start..."
# Note: Toggle ablation mode by editing ABLATION_MODE in scout_semantix.py
# Or implement command-line args
python3 -c "
import scout_semantix
scout_semantix.ABLATION_MODE = 'hazard_only'
scout_semantix.main()
"

echo ""
echo "========================================"
echo "Run 3: ENTROPY ONLY"
echo "Pure frontier exploration"
echo "========================================"
read -p "Press Enter to start..."
python3 -c "
import scout_semantix
scout_semantix.ABLATION_MODE = 'entropy_only'
scout_semantix.main()
"

echo ""
echo "========================================"
echo "Demo complete!"
echo "========================================"
echo "Compare the three runs:"
echo "  - FULL: balanced exploration + exploitation"
echo "  - HAZARD_ONLY: fastest hazard discovery"
echo "  - ENTROPY_ONLY: best coverage, may miss hazards"
