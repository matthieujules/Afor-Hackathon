#!/usr/bin/env python3
"""
Launcher for Semantix Panopticon simulation.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run
from scout_semantix import main

if __name__ == "__main__":
    main()
