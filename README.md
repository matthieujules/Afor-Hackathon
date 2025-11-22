# Semantix: "The Panopticon"

**Stationary Active Vision Scout with Semantic Curiosity**

Instead of moving physically, the robot rotates its view 360° to explore. It uses vision models to predict where interesting content continues off-screen ("White Glow"), then follows those trails rather than exhaustively scanning.

## Quick Start

```bash
# Install dependencies
pip3 install -r requirements.txt

# Run the simulation
python3 scout_semantix.py
```

**First run:** DINOv2 model downloads automatically (~84MB). If you see SSL errors on macOS, run:
```bash
/Applications/Python\ 3.11/Install\ Certificates.command
```

## What You'll See

**Two windows will open:**

1. **PyBullet 3D Simulation**: R2D2 robot in a warehouse environment with cluttered and empty zones
2. **Matplotlib Dashboard**: Three real-time panels showing:
   - Semantic interest map (what the robot has learned)
   - Predictive glow map (where it expects to find interest)
   - Live camera feed (what the robot currently sees)

**Console output every ~5 seconds:**
```
Analyzing view at 45.0°...
  Interest: 0.73 | Lead: right
  Decision: Curiosity (Following Glow)
  Saccading to 90.0° (Ent: 12.3, Glow: 18.7)
```

## How It Works

### The Loop (0.2 Hz)

```
Capture Frame → Vision Analysis (DINOv2)
                    ↓
            Update Semantic Maps
                    ↓
            Project "Glow" (Visual Continuity)
                    ↓
            Plan Next Best View
                    ↓
            Saccade (Rotate to New Angle)
```

### Key Innovation: Visual Continuity

When the vision model detects that interesting content continues off-screen (e.g., "cable runs left"), the system:
1. Projects a cone of predicted utility ("White Glow") into unseen areas in that direction
2. Weights that direction higher when planning the next view
3. Follows features to their conclusion rather than systematically scanning

This mimics human visual search behavior.

## Vision Modes

**DINOv2 (default)** - Local model, no auth
```bash
python3 scout_semantix.py
```

**DINOv3** - Better model, requires [Hugging Face](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) auth (see CLAUDE.md)
```bash
USE_DINOV3=1 python3 scout_semantix.py
```

**VLM** - Gemini/OpenAI (requires API key)
```bash
export GEMINI_API_KEY="your_key"
USE_VLM=1 python3 scout_semantix.py
```

## Key Parameters

`scout_semantix.py:35-50`:

```python
FRAME_PERIOD = 5.0       # Seconds between analyses
FOV_DEGREES = 60         # Camera field of view
MAX_RANGE = 8.0          # Vision range (meters)

LAMBDA_INTEREST = 1.0    # Weight on interesting content
LAMBDA_ENTROPY = 0.8     # Weight on exploring unknown
LAMBDA_GLOW = 1.5        # Weight on following predictions (HIGHEST)
```

**The glow weight is highest** - robot prefers following visual trails over random exploration.

## Files

```
scout_semantix.py          # Main simulation
vision_alternatives.py     # DINOv2/v3 clients
vlm_client.py              # VLM interface
tests/                     # Test suite
CLAUDE.md                  # Technical docs
```

See **CLAUDE.md** for architecture, troubleshooting, and development details.
