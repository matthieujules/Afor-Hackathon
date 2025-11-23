#!/usr/bin/env python3
"""
Combined Web Dashboard + Benchmark Runner
Starts the web server and runs the benchmark, sending live updates to browser clients.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import multiprocessing as mp
import time
import subprocess
import webbrowser

def start_web_server():
    """Start the FastAPI web dashboard server"""
    import uvicorn
    from src.web_dashboard import app

    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")

def start_benchmark():
    """Start the benchmark runner"""
    # Give server time to start
    time.sleep(2)

    # Open browser
    webbrowser.open('http://localhost:8080')

    # Run benchmark
    import run_benchmark
    run_benchmark.run_benchmark(max_saccades=50)

if __name__ == "__main__":
    print("="*60)
    print("SEMANTIX WEB BENCHMARK")
    print("="*60)
    print("Starting web server on http://localhost:8080")
    print("Benchmark will start automatically...")
    print("="*60 + "\n")

    # Start web server in separate process
    server_process = mp.Process(target=start_web_server, daemon=True)
    server_process.start()

    try:
        # Run benchmark in main process
        start_benchmark()
    except KeyboardInterrupt:
        print("\n[EXIT] Shutting down...")
    finally:
        server_process.terminate()
        server_process.join()
