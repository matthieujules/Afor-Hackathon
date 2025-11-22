# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Semantix: "The Panopticon"** is a stationary active vision research prototype. The robot is fixed in place and uses gaze control (360° rotation) to build a semantic understanding of its environment using **Semantic Curiosity** and **Visual Continuity**.

Core innovation: Instead of exhaustively scanning, the agent follows visual trails by predicting where interesting content continues off-screen ("White Glow"), mimicking human visual search patterns.

## Running the Project

### Quick Start

```bash
# Install dependencies
pip3 install -r requirements.txt

# Run with DINOv2 (default - local model, no auth)
python3 scout_semantix.py

# Run tests
python3 tests/test_components.py
python3 tests/test_dinov3.py
```

### Vision Model Options

**DINOv2 (Default)** - Local vision transformer, no authentication
```bash
python3 scout_semantix.py
```

**DINOv3** - Requires Hugging Face authentication (see `docs/DINOV3_SETUP.md`)
```bash
USE_DINOV3=1 python3 scout_semantix.py
```

**VLM Mode** - Uses Gemini/OpenAI APIs
```bash
export GEMINI_API_KEY="your_key"
USE_VLM=1 python3 scout_semantix.py
```

## Architecture

### Core Loop (0.2 Hz - Every 5 Seconds)

```
Frame Capture (320x320 RGB from PyBullet)
            ↓
Vision Analysis (DINOv2: interest_score + lead_direction)
            ↓
Update Semantic Maps (interest_map, seen_map)
            ↓
Project "White Glow" (continuity prediction into unseen areas)
            ↓
Calculate Next Best View (entropy + glow utility)
            ↓
Saccade (rotate robot to winning angle)
```

### Key Components

**`scout_semantix.py`** - Main simulation (373 lines)
- PyBullet environment setup (warehouse with cluttered/empty zones)
- Semantic mapping: 64×64 grid representing 20m × 20m world
- Saccadic planning: evaluates 16 candidate angles (22.5° apart)
- Real-time visualization: 3-panel matplotlib dashboard
- Robot: R2D2 URDF, fixed at origin, rotates head only

**`vision_alternatives.py`** - Local vision models (320 lines)
- `DinoV2Client`: Main vision client supporting both DINOv2 and DINOv3
  - DINOv2: torch.hub, ViT-S/14, 16×16 patch grid (256 patches)
  - DINOv3: Hugging Face Transformers, ViT-S/16, 14×14 patch grid (196 patches)
  - Includes robust fallback to numpy edge detection
  - SSL certificate workaround for macOS torch.hub issues
- All clients implement `analyze_scene(image_rgb)` → `{interest_score, lead_direction, hazard_score}`

**`vlm_client.py`** - VLM interface (161 lines)
- Unified interface for Gemini, OpenAI, or mock analysis
- Returns structured JSON: interest_score (0-1), hazard_score (0-1), lead_direction
- Mock mode uses heuristics: red detection for hazards, variance for interest

**`tests/test_components.py`** - Component validation
- Bayesian mapping (Beta posteriors, entropy calculations)
- Convolution (Gaussian kernel for glow projection)
- FOV projection (raycasting to grid)
- Utility function (hazard + entropy + glow weighting)

**`tests/test_dinov3.py`** - Vision client integration tests
- Tests DINOv2/v3 model loading and fallback behavior
- Validates interface contract and score calibration

### Critical Parameters

Location: `scout_semantix.py:35-50`

```python
# World Configuration
GRID_SIZE = 64           # 64×64 grid
WORLD_SIZE = 20.0        # ±10m square world
CELL_SIZE = 0.3125       # WORLD_SIZE / GRID_SIZE

# Vision System
FRAME_PERIOD = 5.0       # Seconds between frames (0.2 Hz)
FOV_DEGREES = 60         # Camera field of view
MAX_RANGE = 8.0          # Vision range in meters
FOV_RAYS = 40            # Raycasting density

# Utility Weights for Planning
LAMBDA_INTEREST = 1.0    # Weight on seeing interesting things
LAMBDA_ENTROPY = 0.8     # Weight on exploring unknown
LAMBDA_GLOW = 1.5        # Weight on following continuity (HIGHEST)
SACCADE_COST = 0.05      # Cost of rotating (degrees)
```

**Key insight:** `LAMBDA_GLOW` is highest (1.5) - the robot prefers following visual trails over pure entropy exploration.

## State Management

Three global numpy arrays (64×64) track world understanding:

- **`interest_map`**: Semantic interest (0=boring, 1=interesting)
- **`seen_map`**: Observation coverage (0=unseen, 1=seen)
- **`glow_map`**: Transient curiosity predictions

Maps use moving average updates when re-observing cells: 70% old value + 30% new (scout_semantix.py:128).

## The "Glow" Mechanism (Visual Continuity)

This is the core innovation. When vision analysis detects off-screen continuity:

**Input:** `lead_direction='left'` (e.g., "cable runs off left edge")

**Process** (scout_semantix.py:132-175):
1. Decay existing glow by 50%
2. Calculate target angle: current_heading + FOV/2 + 20°
3. Project 40° cone into unseen areas to the left
4. Paint those cells with "white glow" (intensity fades with distance)
5. Only glow unseen cells (seen_map < 0.5)

**Planning** (scout_semantix.py:180-209):
- For each candidate angle, calculate: `utility = LAMBDA_ENTROPY × unseen_cells + LAMBDA_GLOW × glow_sum`
- Choose angle with highest utility
- Result: Robot follows predicted interest rather than systematically scanning

**Glow decay:** 50% per frame ensures predictions are transient and resolved into actual observations.

## DINOv2/v3 Technical Details

### DINOv2 (Default)

**Model:** `dinov2_vits14` via torch.hub
- Architecture: ViT-S/14 (Small variant, 14×14 patch size)
- Input: 224×224 images
- Patches: 224/14 = 16 → 16×16 = 256 patches
- Features: 384-dimensional embeddings per patch
- Loading: `torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')`

**Interest Score Calculation:**
```python
variance = torch.var(patch_tokens, dim=1).mean().item()

# Empirically calibrated for PyBullet renders:
if variance < 2.95:      interest_score = 0.1  # Empty walls
elif variance > 3.05:    interest_score = 1.0  # High clutter
else:                    # Linear interpolation
    interest_score = (variance - 2.95) / (3.05 - 2.95)
```

**Note:** PyBullet renders have much higher variance (~2.9-3.1) than simple test images (~0.008-0.03).

**Lead Direction Detection:**
```python
# Reshape to spatial grid
patches = patch_tokens.reshape(1, 16, 16, -1)
energy = torch.norm(patches, dim=-1).squeeze()  # L2 norm per patch

# Compare edges
left_edge = energy[:, :2].mean()    # Left 2 columns
right_edge = energy[:, -2:].mean()  # Right 2 columns
center = energy[:, 4:-4].mean()     # Center 8 columns

# Decision
if left_edge > center*1.15 AND left_edge > right_edge*1.2:
    return 'left'
```

### DINOv3 (Optional)

**Model:** `facebook/dinov3-vits16-pretrain-lvd1689m` via Hugging Face
- Architecture: ViT-S/16 (16×16 patch size)
- Input: 224×224 images
- Patches: 224/16 = 14 → 14×14 = 196 patches
- Total tokens: 1 CLS + 4 registers + 196 patches = 201
- Features: 384-dimensional embeddings per patch
- **Requires Hugging Face authentication** (gated model)

**Differences from DINOv2:**
- Improved training (released August 2025)
- **Lower variance** than DINOv2: empty ~0.027, clutter ~0.030
- Register tokens (4) that act as memory slots
- Loaded via `AutoModel.from_pretrained()` instead of torch.hub

**Thresholds recalibrated for PyBullet:**
```python
# Empirically calibrated from actual PyBullet renders:
if variance < 0.028:     interest_score = 0.1  # Empty walls
elif variance > 0.031:   interest_score = 1.0  # High clutter
else:  # Linear interpolation
    interest_score = (variance - 0.028) / (0.031 - 0.028)
```

**Observed values:**
- Empty wall: 0.0274
- Ducks visible: 0.0303 → Interest: 0.77

### Fallback Mode

If PyTorch/model loading fails, uses numpy gradient-based edge detection:
```python
gray = np.mean(image_rgb, axis=2)
gy, gx = np.gradient(gray)
edge_energy = np.sqrt(gx**2 + gy**2)
interest_score = min(max(edge_energy.mean() / 25.0, 0.1), 1.0)
```

## PyBullet Environment

Setup: `scout_semantix.py:237-278`

**Elements:**
- Plane ground
- 4 walls (10m cubes at z=5) forming 20m × 20m enclosure
  - East wall: [10, 0, 5]
  - West wall: [-10, 0, 5]
  - North/South walls: [0, ±10, 5]
- **Interesting Zone (East)**:
  - 5 boxes at X: 4-8m, Y: -2 to +1.2m, Z: 0.5m
  - 5 ducks at X: 1-5m, Y: -1 to +0.2m, Z: 0.5m (duck trail)
- **Boring Zone (West)**: Empty
- **Robot**: R2D2 URDF at [0,0,0.5], rotates on yaw axis only

**Expected observations:**
- 0° (East): Should see ducks → variance ~0.030 (DINOv3) or ~3.0 (DINOv2)
- 180° (West): Empty wall → variance ~0.027 (DINOv3) or ~2.9 (DINOv2)

**Camera:** `scout_semantix.py:263-278`
- Position: Robot head at z=0.8m
- Target: 2m ahead in current heading direction, looking slightly down
- Resolution: 320×320 pixels
- FOV: 60° (matches planning FOV_DEGREES)

## Extending the Vision System

To add a new vision model:

1. Create class in `vision_alternatives.py` inheriting from `BaseVisionClient`
2. Implement `analyze_scene(image_rgb)`:
   ```python
   def analyze_scene(self, image_rgb):
       # image_rgb: numpy array (H, W, 3), uint8, RGB order
       return {
           'interest_score': float,  # 0-1, visual complexity
           'lead_direction': str,    # 'left'/'right'/'center'/'none'
           'hazard_score': float     # 0-1, danger assessment
       }
   ```
3. Update `scout_semantix.py:52-64` to instantiate your client
4. Interest score: 0=blank wall, 1=dense clutter/objects
5. Lead direction: where visual features continue off-frame edges
6. Include robust error handling and fallback mechanisms

**Example:**
```python
class MyVisionClient(BaseVisionClient):
    def analyze_scene(self, image_rgb):
        # Your analysis here
        return {'interest_score': 0.5, 'lead_direction': 'none', 'hazard_score': 0.1}
```

Then in `scout_semantix.py`:
```python
from vision_alternatives import MyVisionClient
vlm_client = MyVisionClient()
```

## Error Handling & Robustness

**Implemented safeguards (scout_semantix.py:327-332, 399-409):**

1. **Vision analysis wrapper**: try/except around `analyze_scene()` continues simulation with safe defaults if vision fails
2. **Resource cleanup**: try/finally ensures `p.disconnect()` always runs, preventing orphaned processes
3. **Crash logging**: Exception handler with full traceback for debugging
4. **Comprehensive logging**: [INIT], [SETUP], [VIZ], [EXIT] tags track execution flow

**Log location**: `/tmp/scout_final_test.log` (or specify with `> your_log.txt`)

## Development Notes

**Environment:**
- macOS with Apple Silicon (MPS GPU acceleration)
- PyTorch 2.7.0 with MPS backend
- Python 3.11+
- Matplotlib backend: MacOSX

**Performance:**
- DINOv2 inference: ~50-100ms per frame on Apple Silicon
- Total cycle time: ~5 seconds (includes simulation + rendering + planning)
- Memory: ~500MB (model weights + simulation)

**Common Issues:**
- **SSL certificate errors**: Fixed with SSL workaround in vision_alternatives.py:74-81
- **GitHub rate limits**: Fixed with `skip_validation=True` in torch.hub.load
- **xFormers warnings**: Safe to ignore (optional optimization library)
- **Model cache**: `~/.cache/torch/hub/facebookresearch_dinov2_main`

**Git Branch:**
- Current: `feature/dinov2-integration`
- Main branch: `master`

**Modified Files (Latest):**
- scout_semantix.py: Vision mode selection
- vision_alternatives.py: DINOv2/v3 client with SSL fixes
- requirements.txt: Added torch, torchvision, transformers, huggingface_hub
- README.md: Updated quick start and structure
- Reorganized: docs/, tests/, assets/ directories

## DINOv3 Setup (Optional)

DINOv3 is gated on Hugging Face. To enable:

```bash
# 1. Install huggingface_hub (already in requirements.txt)
pip install huggingface_hub

# 2. Get token from https://huggingface.co/settings/tokens
# 3. Accept terms at https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m

# 4. Login
huggingface-cli login
# Paste your token

# 5. Run with DINOv3
USE_DINOV3=1 python3 scout_semantix.py
```

## Troubleshooting

**SSL Certificate Error (macOS)**
```bash
/Applications/Python\ 3.11/Install\ Certificates.command
```

**"xFormers not available" warnings**
- Safe to ignore - optional optimization library

**DINOv2 model download fails**
- Check internet connection
- Model cached at `~/.cache/torch/hub/facebookresearch_dinov2_main`
- Delete cache and retry if corrupted

**PyBullet window freezes**
- Normal during DINOv2 inference (~50-200ms per frame)
- Simulation pauses briefly every 5 seconds

**All interest scores are 1.0 or 0.1**
- Expected with synthetic test images
- Real PyBullet camera images will show proper variance

## Production Testing

**Test scenario:**
1. Place hazard behind obstacle (smoking power supply behind pallets)
2. Add visual trail (power cable leading to it)
3. Verify robot follows cable using glow predictions
4. Success: Finds hazard without exhaustive scanning

**Expected phases:**
- Initial entropy scan → detect trail → glow activates → follow glow → discover hazard

## File Organization

```
Vision_Test/
├── scout_semantix.py          # Main simulation (373 lines)
├── vision_alternatives.py     # DINOv2/v3 clients (320 lines)
├── vlm_client.py              # VLM interface (161 lines)
├── mock_vlm_server.py         # Mock VLM server
├── run_demo.sh                # Demo script
├── requirements.txt           # Dependencies
├── README.md                  # User docs
├── CLAUDE.md                  # Developer docs (this file)
├── tests/
│   ├── test_components.py     # Component validation
│   └── test_dinov3.py         # Vision client tests
├── assets/
│   └── *.stl                  # 3D models
└── pybullet-3.2.7/            # Local PyBullet installation
