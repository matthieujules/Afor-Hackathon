#!/usr/bin/env python3
"""
Web Dashboard Server for Semantix
FastAPI + WebSocket server that receives visualization data and serves it to browsers.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import asyncio
import json
import base64
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Semantix Dashboard")

# Active WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.latest_data = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")

        # Send latest data if available
        if self.latest_data:
            try:
                await websocket.send_json(self.latest_data)
            except Exception as e:
                logger.error(f"Error sending initial data: {e}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, data: dict):
        """Broadcast data to all connected clients"""
        self.latest_data = data
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            if conn in self.active_connections:
                self.active_connections.remove(conn)

manager = ConnectionManager()

# Serve static files (HTML, CSS, JS)
static_path = Path(__file__).parent.parent / "static"
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

@app.get("/")
async def get_dashboard():
    """Serve the main dashboard HTML"""
    html_path = static_path / "dashboard.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    else:
        return HTMLResponse(
            content="""
            <html>
                <head><title>Semantix Dashboard</title></head>
                <body>
                    <h1>Dashboard Not Found</h1>
                    <p>Please ensure static/dashboard.html exists.</p>
                </body>
            </html>
            """,
            status_code=404
        )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for browser clients"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle pings
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.websocket("/ws/data")
async def data_websocket(websocket: WebSocket):
    """WebSocket endpoint for simulation to push data"""
    await websocket.accept()
    logger.info("Simulation connected to data channel")

    try:
        while True:
            # Receive data from simulation
            data = await websocket.receive_json()

            # Broadcast to all browser clients
            await manager.broadcast(data)

    except WebSocketDisconnect:
        logger.info("Simulation disconnected from data channel")
    except Exception as e:
        logger.error(f"Data channel error: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "active_connections": len(manager.active_connections),
        "has_data": manager.latest_data is not None
    }

def start_server(host="127.0.0.1", port=8080):
    """Start the FastAPI server"""
    import uvicorn
    logger.info(f"Starting Semantix Dashboard on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    start_server()
