# Semantix: "The Panopticon"

**Stationary Active Vision Scout with Semantic Curiosity**

A fixed-base robot that uses gaze control (360° rotation) to build semantic understanding through **Visual Continuity** and **DINOv3 semantic analysis**.

## Quick Start

```bash
# Install dependencies
pip3 install -r requirements.txt

# Run simulation (DINOv3 requires Hugging Face auth)
USE_DINOV3=1 python3 run.py

# Or use DINOv2 (no auth required)
python3 run.py
```

## Project Structure

```
Vision_Test/
├── run.py                  # Main launcher
├── src/                    # Source code
│   ├── scout_semantix.py   # Main simulation (450 lines)
│   ├── vision_alternatives.py  # DINOv2/v3 clients (360 lines)
│   └── vlm_client.py       # VLM interface (optional)
├── tests/                  # Test suite
│   ├── test_components.py
│   └── test_dinov3.py
├── docs/                   # Documentation
├── assets/                 # 3D models
├── logs/                   # Simulation logs
├── requirements.txt
├── CLAUDE.md              # Developer guide
└── README.md              # This file
```

## Visualization Dashboard

**5-Panel Layout (2×3 grid):**
- **Live Feed:** ~10fps real-time camera view
- **Analysis Snapshot:** Frame during deep analysis
- **PCA Rainbow:** DINOv3 semantic regions (Meta's signature viz)
- **Attention Map:** Where DINOv3 focuses
- **Patch Similarity:** Semantic coherence

## Key Features

- **Fixed-Base Panopticon:** Robot anchored at origin, rotates 360°
- **DINOv3 Semantic Analysis:** PCA embeddings, attention maps, similarity
- **Visual Continuity:** Predicts where interest continues off-screen
- **Glow Projection:** Paints curiosity into unseen areas
- **Saccadic Planning:** Maximizes information gain

## Technical Details

See [CLAUDE.md](CLAUDE.md) for complete architecture documentation.
