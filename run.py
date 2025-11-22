#!/usr/bin/env python3
"""
Launcher script for Semantix with Web Dashboard

Starts both the FastAPI web server and PyBullet simulation.
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def main():
    # Get project root
    project_root = Path(__file__).parent

    print("="*60)
    print("Semantix Web Dashboard Launcher")
    print("="*60)

    # Step 1: Start web dashboard server in background
    print("\n[1/2] Starting web dashboard server...")
    web_server = subprocess.Popen(
        [sys.executable, str(project_root / "src" / "web_dashboard.py")],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # Wait for server to start
    print("    Waiting for server to start (2 seconds)...")
    time.sleep(2)

    # Check if server started successfully
    if web_server.poll() is not None:
        print("    ERROR: Web server failed to start!")
        print("    Check that FastAPI and uvicorn are installed:")
        print("    pip install -r requirements.txt")
        return 1

    print("    Web dashboard server started successfully")
    print("    Dashboard URL: http://localhost:8080")

    # Step 2: Start PyBullet simulation
    print("\n[2/2] Starting PyBullet simulation...")
    print("    Open http://localhost:8080 in your browser to view dashboard")
    print("="*60 + "\n")

    try:
        # Run simulation in foreground (so we can see output and Ctrl+C works)
        sim_process = subprocess.run(
            [sys.executable, str(project_root / "src" / "scout_semantix.py")],
            cwd=str(project_root)
        )
        return_code = sim_process.returncode

    except KeyboardInterrupt:
        print("\n[EXIT] Keyboard interrupt - shutting down...")
        return_code = 0

    finally:
        # Cleanup: terminate web server
        print("[EXIT] Stopping web server...")
        web_server.terminate()
        try:
            web_server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("[EXIT] Force killing web server...")
            web_server.kill()

    print("[EXIT] Shutdown complete")
    return return_code

if __name__ == "__main__":
    sys.exit(main())
