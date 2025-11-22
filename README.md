# Semantix: "The Panopticon"

**Stationary Active Vision Scout with Semantic Curiosity**

A fixed-base robot that uses gaze control (360° rotation) to build semantic understanding through **Visual Continuity** and **DINOv3 semantic analysis**.

## Quick Start

```bash
# Install dependencies
pip3 install -r requirements.txt

# Start web dashboard server
python3 src/web_dashboard.py &

# Start simulation (uses DINOv2 by default)
python3 src/scout_semantix.py

# Open browser to: http://localhost:8080

# To use DINOv3 instead
USE_DINOV3=1 python3 src/scout_semantix.py
```

## Project Structure

```
Vision_Test/
├── src/                    # Source code
│   ├── scout_semantix.py       # Main simulation (586 lines)
│   ├── web_dashboard.py        # FastAPI web dashboard server
│   ├── vision_alternatives.py  # DINOv2/v3 clients (320 lines)
│   └── vlm_client.py           # VLM interface (optional)
├── static/                 # Web dashboard assets
│   ├── dashboard.html
│   └── dashboard.css
├── tests/                  # Test suite
│   ├── test_components.py
│   ├── test_dinov3.py
│   └── test_dashboard.py
├── docs/                   # Documentation
├── archive/                # Archived files
├── assets/                 # 3D models
├── logs/                   # Simulation logs
├── requirements.txt        # Python dependencies
├── CLAUDE.md              # Developer guide
└── README.md              # This file
```

## Visualization Dashboard

**Web-based dashboard (http://localhost:8080)** eliminates macOS GUI threading conflicts.

**Architecture:**
- **FastAPI + WebSocket:** Real-time data streaming from simulation
- **Browser-based UI:** 6-panel layout (2×3 grid)
- **Headless matplotlib:** Agg backend for server-side rendering

**Panels:**
- **Live Feed:** ~10fps real-time camera view
- **Analysis Snapshot:** Frame during deep analysis (every 5s)
- **PCA Rainbow:** DINOv2/v3 semantic regions (Meta's signature viz)
- **Attention Map:** Where DINOv2/v3 focuses attention
- **Patch Similarity:** Semantic coherence heatmap
- **Metrics Panel:** Real-time stats (interest score, heading, decision)

## Key Features

- **Fixed-Base Panopticon:** Robot anchored at origin, rotates 360°
- **DINOv2/v3 Semantic Analysis:** PCA embeddings, attention maps, patch similarity
- **Visual Continuity:** Predicts where interest continues off-screen ("lead direction")
- **Glow Projection:** Paints curiosity into unseen areas based on predictions
- **Saccadic Planning:** Maximizes information gain (entropy + glow)
- **Web Dashboard:** Browser-based real-time visualization via WebSocket
- **macOS Compatible:** Solves PyBullet + matplotlib GUI threading conflicts

## Technical Details

See [CLAUDE.md](CLAUDE.md) for complete architecture documentation.
