#!/usr/bin/env python3
"""
Semantix: "The Panopticon" - Stationary Active Vision Scout

Implements Semantic Curiosity & Visual Continuity for a stationary agent.
The agent rotates its view (saccades) to maximize Information Gain.

Key Features:
- "Interest Score": VLM rates visual complexity/clutter.
- "Visual Continuity": VLM predicts where interesting stuff goes (Left/Right).
- "Glow Projection": Projects utility into unseen areas based on continuity.
- "Saccadic Planning": Chooses the next view angle to maximize (Interest + Glow + Entropy).

References:
[1] Language-as-Cost: Proactive Hazard Mapping using VLM
[2] Active Vision & Saccadic Exploration
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import os
import math
import threading
import asyncio
import base64
from io import BytesIO

# Use Agg backend for matplotlib (headless rendering)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge

from vlm_client import VLMClient

# ============================================================================
# GLOBAL PARAMETERS
# ============================================================================

# Grid world configuration (Polar-ish usage, but kept Cartesian for map ease)
GRID_SIZE = 64          # 64×64 grid
WORLD_SIZE = 20.0       # ±10m square world
CELL_SIZE = WORLD_SIZE / GRID_SIZE

# Vision system
FRAME_PERIOD = 5.0      # seconds between frames (0.2 Hz)
FOV_DEGREES = 60        # field of view
MAX_RANGE = 8.0         # max vision range (meters)
FOV_RAYS = 40           # ray density

# Utility weights
LAMBDA_INTEREST = 1.0   # weight on seeing interesting things
LAMBDA_ENTROPY = 0.8    # weight on exploring the unknown
LAMBDA_GLOW = 1.5       # weight on following the "lead" (continuity)
SACCADE_COST = 0.05     # cost of rotating (degrees)

# Vision model settings
print("[INIT] Starting vision system...", flush=True)
USE_REAL_VLM = os.getenv('USE_VLM', '0') == '1'
if USE_REAL_VLM:
    print("[INIT] Vision Mode: VLM (Gemini/OpenAI)", flush=True)
    from vlm_client import VLMClient
    vlm_client = VLMClient(provider="gemini")
else:
    # Use DINOv2 by default for fast, local vision analysis (no auth required)
    # Set USE_DINOV3=1 to use DINOv3 instead (requires Hugging Face auth)
    print("[INIT] Vision Mode: DINO (Local Model)", flush=True)
    from vision_alternatives import DinoV2Client
    print("[INIT] Initializing DinoV2Client...", flush=True)
    vlm_client = DinoV2Client()
    print("[INIT] Vision client initialized successfully", flush=True)

# Bayesian update parameters
CONFIDENCE_DECAY = 0.10

# WebSocket client for dashboard
ws_client = None

# ============================================================================
# STATE
# ============================================================================

# Maps
# 0=Boring, 1=Interesting
interest_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
# 0=Unseen, 1=Seen
seen_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
# Glow map (Transient prediction)
glow_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

# Robot State
current_yaw = 0.0 # radians

# ============================================================================
# WEBSOCKET CLIENT
# ============================================================================

class DashboardClient:
    """WebSocket client to send data to web dashboard"""
    def __init__(self, url="ws://localhost:8080/ws/data"):
        self.url = url
        self.websocket = None
        self.loop = None
        self.thread = None
        self.queue = asyncio.Queue()

    def start(self):
        """Start WebSocket client in background thread"""
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        print(f"[WS] Dashboard client started, connecting to {self.url}")

    def _run_loop(self):
        """Run asyncio event loop in thread"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._connect_and_send())

    async def _connect_and_send(self):
        """Connect to WebSocket and send queued data"""
        import websockets

        while True:
            try:
                async with websockets.connect(self.url) as websocket:
                    self.websocket = websocket
                    print("[WS] Connected to dashboard")

                    while True:
                        # Get data from queue
                        data = await self.queue.get()

                        # Send to dashboard
                        await websocket.send(data)

            except Exception as e:
                print(f"[WS] Connection error: {e}, retrying in 2s...")
                self.websocket = None
                await asyncio.sleep(2)

    def send(self, data: dict):
        """Queue data to send to dashboard"""
        if self.loop:
            import json
            # Convert numpy types to native Python types before JSON serialization
            data_clean = convert_numpy_types(data)
            asyncio.run_coroutine_threadsafe(
                self.queue.put(json.dumps(data_clean)),
                self.loop
            )

def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def numpy_to_base64_png(img_array):
    """Convert numpy array to base64 PNG string"""
    # Handle different input formats
    if img_array.dtype != np.uint8:
        # Normalize to 0-255 if needed
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)

    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=100)
    ax.axis('off')

    if len(img_array.shape) == 2:
        # Grayscale or heatmap
        ax.imshow(img_array, cmap='hot')
    else:
        # RGB
        ax.imshow(img_array)

    plt.tight_layout(pad=0)

    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # Encode as base64
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

def create_topdown_view(robot_pos, robot_yaw, interest_map, seen_map, glow_map):
    """
    Create top-down visualization of the robot's world understanding.
    Shows: robot position/orientation, FOV cone, interestingness heatmap
    """
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

    # Create composite visualization: interest map with glow overlay
    # Normalize maps for visualization
    interest_normalized = np.clip(interest_map, 0, 1)
    glow_normalized = np.clip(glow_map, 0, 1)

    # Create RGB composite: interest (red channel), glow (green channel), seen (blue channel)
    composite = np.zeros((GRID_SIZE, GRID_SIZE, 3))
    composite[:, :, 0] = interest_normalized  # Red = interest
    composite[:, :, 1] = glow_normalized * 0.7  # Green = glow (dimmer)
    composite[:, :, 2] = seen_map * 0.3  # Blue = coverage (subtle)

    # Display composite
    ax.imshow(composite, origin='lower', extent=[-WORLD_SIZE/2, WORLD_SIZE/2, -WORLD_SIZE/2, WORLD_SIZE/2])

    # Draw robot position (center)
    robot_x, robot_y = robot_pos
    ax.plot(robot_x, robot_y, 'wo', markersize=12, markeredgewidth=2, markeredgecolor='cyan')

    # Draw FOV cone
    cone_length = MAX_RANGE
    cone_angle = np.radians(FOV_DEGREES)

    # Cone edges
    left_angle = robot_yaw + cone_angle / 2
    right_angle = robot_yaw - cone_angle / 2

    # Cone vertices
    cone_x = [robot_x,
              robot_x + cone_length * np.cos(left_angle),
              robot_x + cone_length * np.cos(right_angle)]
    cone_y = [robot_y,
              robot_y + cone_length * np.sin(left_angle),
              robot_y + cone_length * np.sin(right_angle)]

    ax.fill(cone_x, cone_y, color='cyan', alpha=0.2, edgecolor='cyan', linewidth=2)

    # Draw heading direction arrow
    arrow_length = 2.0
    ax.arrow(robot_x, robot_y,
             arrow_length * np.cos(robot_yaw),
             arrow_length * np.sin(robot_yaw),
             head_width=0.5, head_length=0.3, fc='yellow', ec='yellow', linewidth=2)

    # Add grid lines
    ax.grid(True, alpha=0.3, color='white', linewidth=0.5)

    # Labels and title
    ax.set_xlim(-WORLD_SIZE/2, WORLD_SIZE/2)
    ax.set_ylim(-WORLD_SIZE/2, WORLD_SIZE/2)
    ax.set_xlabel('X (meters)', color='white', fontsize=10)
    ax.set_ylabel('Y (meters)', color='white', fontsize=10)
    ax.set_title('Top-Down View\nRed=Interest | Green=Glow | Blue=Seen',
                 color='white', fontsize=12, pad=10)

    # Dark background
    ax.set_facecolor('#1a1a1a')
    fig.patch.set_facecolor('#1a1a1a')
    ax.tick_params(colors='white', labelsize=8)

    # Add compass
    ax.text(WORLD_SIZE/2 - 1, WORLD_SIZE/2 - 1, 'N', color='white', fontsize=14,
            ha='center', va='center', weight='bold')

    plt.tight_layout()

    # Convert to base64
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#1a1a1a')
    plt.close(fig)

    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# ============================================================================
# MAPPING UTILITIES
# ============================================================================

def world_to_grid(world_xy):
    gx = int(world_xy[0] / CELL_SIZE + GRID_SIZE / 2)
    gy = int(world_xy[1] / CELL_SIZE + GRID_SIZE / 2)
    return np.clip([gy, gx], 0, GRID_SIZE - 1)

def grid_to_world(grid_yx):
    gy, gx = grid_yx
    wx = (gx - GRID_SIZE / 2) * CELL_SIZE
    wy = (gy - GRID_SIZE / 2) * CELL_SIZE
    return np.array([wx, wy])

def get_fov_mask(robot_pos, heading, fov_deg, max_range):
    """Returns a list of (gy, gx) cells in the current FOV."""
    cells = set()

    # Ray casting
    angles = np.linspace(-np.radians(fov_deg/2), np.radians(fov_deg/2), FOV_RAYS)
    for angle_offset in angles:
        ray_angle = heading + angle_offset
        for r in np.arange(0.5, max_range, CELL_SIZE/2):
            x = robot_pos[0] + r * np.cos(ray_angle)
            y = robot_pos[1] + r * np.sin(ray_angle)
            gy, gx = world_to_grid([x, y])
            if 0 <= gy < GRID_SIZE and 0 <= gx < GRID_SIZE:
                cells.add((gy, gx))

    return list(cells)

def update_map(fov_cells, interest_score):
    """Update the semantic map with the VLM's interest score."""
    global interest_map, seen_map

    for gy, gx in fov_cells:
        # Simple update: weighted average or max
        # Here we trust the latest VLM score but decay it slightly over distance?
        # For simplicity: Max aggregation (if we saw it was interesting once, it stays interesting)
        # But we also need to handle "Boring" updates overwriting "Unknown".

        # If previously unseen, take the value.
        # If seen, average it?
        if seen_map[gy, gx] == 0:
            interest_map[gy, gx] = interest_score
        else:
            # Moving average
            interest_map[gy, gx] = 0.7 * interest_map[gy, gx] + 0.3 * interest_score

        seen_map[gy, gx] = 1.0

def project_glow(robot_pos, current_heading, lead_direction):
    """
    Project 'White Glow' (Predicted Interest) based on Visual Continuity.

    If lead_direction is 'left', project a cone to the left of the current FOV.
    """
    global glow_map

    # Decay old glow
    glow_map *= 0.5

    if lead_direction == 'none' or lead_direction == 'center':
        return

    # Determine angle of projection
    # Left means: Current Heading + FOV/2 + Offset
    angle_offset = 0
    if lead_direction == 'left':
        angle_offset = np.radians(FOV_DEGREES/2 + 20) # Look 20 deg past the edge
    elif lead_direction == 'right':
        angle_offset = -np.radians(FOV_DEGREES/2 + 20)

    target_angle = current_heading + angle_offset

    # Project a "Cone of Curiosity"
    # We use the same raycasting logic but write to glow_map
    # The cone is wider and fuzzier
    cone_width = 40 # degrees
    angles = np.linspace(-np.radians(cone_width/2), np.radians(cone_width/2), 20)

    for ang in angles:
        ray_angle = target_angle + ang
        for r in np.arange(1.0, MAX_RANGE * 0.8, CELL_SIZE):
            x = robot_pos[0] + r * np.cos(ray_angle)
            y = robot_pos[1] + r * np.sin(ray_angle)
            gy, gx = world_to_grid([x, y])

            if 0 <= gy < GRID_SIZE and 0 <= gx < GRID_SIZE:
                # Only glow in UNSEEN areas
                if seen_map[gy, gx] < 0.5:
                    # Glow intensity fades with distance
                    intensity = 1.0 * (1 - r/MAX_RANGE)
                    glow_map[gy, gx] = max(glow_map[gy, gx], intensity)

# ============================================================================
# PLANNING: NEXT BEST VIEW
# ============================================================================

def calculate_view_utility(robot_pos, candidate_heading):
    """
    Evaluate how good looking in a specific direction would be.
    Returns: total_score, (entropy_score, glow_score)
    """
    # Get cells in this candidate view
    cells = get_fov_mask(robot_pos, candidate_heading, FOV_DEGREES, MAX_RANGE)

    u_entropy = 0
    u_glow = 0

    for gy, gx in cells:
        is_seen = seen_map[gy, gx]

        # Entropy: High if unseen
        if not is_seen:
            u_entropy += 1.0

        # Glow: High if we predicted something there
        u_glow += glow_map[gy, gx]

    # Normalize? No, we want total information gain.
    if len(cells) == 0: return 0, (0, 0)

    score_entropy = LAMBDA_ENTROPY * u_entropy
    score_glow = LAMBDA_GLOW * u_glow

    total_score = score_entropy + score_glow
    return total_score, (score_entropy, score_glow)

def choose_next_angle(robot_pos, current_heading):
    """Scan 360 degrees and pick the best angle."""
    best_angle = current_heading
    best_score = -1e9
    best_components = (0, 0)

    candidates = np.linspace(0, 2*np.pi, 16, endpoint=False) # 16 directions

    for ang in candidates:
        score, components = calculate_view_utility(robot_pos, ang)

        if score > best_score:
            best_score = score
            best_angle = ang
            best_components = components

    return best_angle, best_score, best_components

# ============================================================================
# PYBULLET ENV
# ============================================================================

def setup_env():
    print("[SETUP] Connecting to PyBullet GUI...")
    p.connect(p.GUI)
    print("[SETUP] PyBullet connected")

    # Set search path for default URDFs
    import pybullet_data
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    print("[SETUP] Search path configured")

    # Room
    print("[SETUP] Loading ground plane...")
    p.loadURDF("plane.urdf")
    print("[SETUP] Ground plane loaded")

    # Walls - REMOVED per user request for simplification
    # p.loadURDF("cube.urdf", [10, 0, 5], globalScaling=10)  # East wall
    # p.loadURDF("cube.urdf", [-10, 0, 5], globalScaling=10)  # West wall
    # p.loadURDF("cube.urdf", [0, 10, 5], globalScaling=10)  # North wall
    # p.loadURDF("cube.urdf", [0, -10, 5], globalScaling=10)  # South wall
    print("[SETUP] Walls removed (Open world)")

    # === MANUAL SCENE LAYOUT ===
    print("[SETUP] Spawning MANUAL complex scene with large objects...")

    # 1. The "Office Cluster" (North-East) - Corner case: desk and bench meeting
    p.loadURDF("table/table.urdf", [3, 5, 0], p.getQuaternionFromEuler([0, 0, 0.3]), useFixedBase=True)
    p.loadURDF("table/table.urdf", [1.5, 4.5, 0], p.getQuaternionFromEuler([0, 0, 1.8]), useFixedBase=True)
    # Add ducks on tables for visual interest
    p.loadURDF("duck_vhacd.urdf", [3, 5, 0.7], globalScaling=2.0, useFixedBase=True)
    p.loadURDF("duck_vhacd.urdf", [1.5, 4.5, 0.7], globalScaling=2.0, useFixedBase=True)
    print("  Office Cluster (North-East)")

    # 2. The "Warehouse Zone" (East) - Scattered barrels and crates
    # Large barrels (spheres) at different heights and positions
    p.loadURDF("sphere2.urdf", [6, 0.5, 0.8], globalScaling=1.5, useFixedBase=True)
    p.loadURDF("sphere2.urdf", [6.5, -1.2, 0.8], globalScaling=1.5, useFixedBase=True)
    p.loadURDF("sphere2.urdf", [5.2, 1.8, 0.8], globalScaling=1.5, useFixedBase=True)
    # Tray boxes as crates
    p.loadURDF("tray/traybox.urdf", [7, 0, 0], globalScaling=2.0, useFixedBase=True)
    p.loadURDF("tray/traybox.urdf", [5.5, -0.8, 0], globalScaling=1.8, useFixedBase=True)
    # Teddy bear on top of crate (visual interest)
    p.loadURDF("teddy_vhacd.urdf", [7, 0, 1.2], globalScaling=3.0, useFixedBase=True)
    print("  Warehouse Zone (East)")

    # 3. The "Reading Nook" (South-West) - Corner case: bench + bookshelf adjacent
    # Bench (long table rotated)
    p.loadURDF("table/table.urdf", [-2, -5, 0], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)
    # "Bookshelf" (vertical tray boxes stacked)
    p.loadURDF("tray/traybox.urdf", [-3.5, -4.8, 0], p.getQuaternionFromEuler([0, 0, 1.57]), globalScaling=2.5, useFixedBase=True)
    p.loadURDF("tray/traybox.urdf", [-3.5, -4.8, 1.5], p.getQuaternionFromEuler([0, 0, 1.57]), globalScaling=2.5, useFixedBase=True)
    # Ducks on bench
    p.loadURDF("duck_vhacd.urdf", [-1.8, -5, 0.7], globalScaling=1.8, useFixedBase=True)
    p.loadURDF("duck_vhacd.urdf", [-2.2, -5, 0.7], globalScaling=1.8, useFixedBase=True)
    print("  Reading Nook (South-West) - Bench + Bookshelf corner case")

    # 4. The "Storage Yard" (South) - Scattered large objects
    p.loadURDF("sphere2.urdf", [2, -6, 0.8], globalScaling=1.6, useFixedBase=True)
    p.loadURDF("tray/traybox.urdf", [0.5, -6.5, 0], globalScaling=2.2, useFixedBase=True)
    p.loadURDF("teddy_vhacd.urdf", [-1, -6, 0.5], globalScaling=2.5, useFixedBase=True)
    print("  Storage Yard (South)")

    # 5. The "Central Hub" - Mixed objects creating occlusion challenges
    p.loadURDF("table/table.urdf", [1, 0.5, 0], p.getQuaternionFromEuler([0, 0, 0.8]), useFixedBase=True)
    p.loadURDF("sphere2.urdf", [-1.5, 1, 0.8], globalScaling=1.4, useFixedBase=True)
    p.loadURDF("tray/traybox.urdf", [-0.5, -1.5, 0], globalScaling=1.9, useFixedBase=True)
    # Duck trail leading from center to east (creates visual continuity for glow)
    p.loadURDF("duck_vhacd.urdf", [2, 0.2, 0.5], globalScaling=1.5, useFixedBase=True)
    p.loadURDF("duck_vhacd.urdf", [3.5, 0.4, 0.5], globalScaling=1.5, useFixedBase=True)
    p.loadURDF("duck_vhacd.urdf", [4.8, 0.3, 0.5], globalScaling=1.5, useFixedBase=True)
    print("  Central Hub - Visual continuity trail")

    # 6. The "North-West Corner" - Table and barrel grouping
    p.loadURDF("table/table.urdf", [-4, 4, 0], p.getQuaternionFromEuler([0, 0, 2.3]), useFixedBase=True)
    p.loadURDF("sphere2.urdf", [-5.5, 3.5, 0.8], globalScaling=1.5, useFixedBase=True)
    p.loadURDF("sphere2.urdf", [-3.2, 5, 0.8], globalScaling=1.3, useFixedBase=True)
    print("  North-West Corner")

    # 7. The "West Corridor" - Linear arrangement with varying objects
    p.loadURDF("tray/traybox.urdf", [-6, 0.5, 0], globalScaling=2.0, useFixedBase=True)
    p.loadURDF("teddy_vhacd.urdf", [-5.5, -1, 0.5], globalScaling=2.8, useFixedBase=True)
    p.loadURDF("sphere2.urdf", [-6.2, -2.5, 0.8], globalScaling=1.5, useFixedBase=True)
    p.loadURDF("table/table.urdf", [-5, 2, 0], p.getQuaternionFromEuler([0, 0, 1.2]), useFixedBase=True)
    print("  West Corridor")

    # 8. The "North Zone" - Dense cluster for high visual complexity
    p.loadURDF("tray/traybox.urdf", [0, 6.5, 0], globalScaling=2.1, useFixedBase=True)
    p.loadURDF("duck_vhacd.urdf", [0.5, 7, 0.5], globalScaling=2.2, useFixedBase=True)
    p.loadURDF("sphere2.urdf", [-1, 6.8, 0.8], globalScaling=1.4, useFixedBase=True)
    print("  North Dense Zone")

    # === EMPTY ZONES for contrast ===
    # South-East corner intentionally empty
    # Far West intentionally sparse
    print("  Empty zones: South-East corner, Far West (for exploration contrast)")

    # === ROBOT (REMOVED) ===
    # We are now just a disembodied camera at the origin.
    print("[SETUP] Robot body removed. Using pure camera.")
    
    return None

def get_camera_image(robot_id, yaw):
    """Render camera view from fixed position, oriented according to yaw."""
    # Camera positioned at origin, slightly elevated
    cam_height = 1.5
    pos = [0, 0, cam_height]

    # Target point 2m away in the direction robot is facing
    target_distance = 2.0
    tx = pos[0] + target_distance * np.cos(yaw)
    ty = pos[1] + target_distance * np.sin(yaw)
    tz = cam_height - 0.2  # Look slightly down

    # Up vector - always pointing up in world frame
    up_vector = [0, 0, 1]

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=pos,
        cameraTargetPosition=[tx, ty, tz],
        cameraUpVector=up_vector
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=FOV_DEGREES,
        aspect=1.0,
        nearVal=0.1,
        farVal=20.0
    )

    w, h, rgb, depth, seg = p.getCameraImage(
        width=320,
        height=320,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )

    rgb = np.array(rgb, dtype=np.uint8).reshape((h, w, 4))
    return rgb[:, :, :3]  # Drop alpha channel

# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    global current_yaw, ws_client

    print("[MAIN] Starting main function...")

    # Start dashboard client
    print("[MAIN] Starting web dashboard client...")
    ws_client = DashboardClient()
    ws_client.start()
    time.sleep(1)  # Give WebSocket time to connect

    robot_id = setup_env()
    print("[MAIN] Environment setup complete")
    print("\n" + "="*60)
    print("PANOPTICON ACTIVATED - Web Dashboard Mode")
    print("="*60)
    print("Open browser to: http://localhost:8080")
    print("="*60 + "\n")

    # Set initial robot orientation - No robot to reset
    # quat = p.getQuaternionFromEuler([0, 0, current_yaw])
    # p.resetBasePositionAndOrientation(robot_id, [0, 0, 0.5], quat)

    step = 0
    last_yaw = current_yaw  # Track when rotation actually changes
    last_snapshot = None
    last_analysis = {}
    last_analysis_time = time.time()  # Track time-based triggering

    try:
        while True:
            # Step physics (affects ducks/boxes, but robot is fixed base)
            p.stepSimulation()

            # Full vision analysis (time-based: every FRAME_PERIOD seconds)
            current_time = time.time()
            if current_time - last_analysis_time >= FRAME_PERIOD:

                # 1. Capture snapshot for analysis
                rgb = get_camera_image(robot_id, current_yaw)
                last_snapshot = numpy_to_base64_png(rgb)

                # 2. Analyze (Vision - with error handling for robustness)
                print(f"Analyzing view at {math.degrees(current_yaw):.1f}°...")
                try:
                    analysis = vlm_client.analyze_scene(rgb)
                except Exception as e:
                    print(f"  WARNING: Vision analysis failed - {e}")
                    print(f"  Continuing with safe defaults...")
                    analysis = {'interest_score': 0.1, 'lead_direction': 'none', 'hazard_score': 0.1}

                i_score = analysis.get('interest_score', 0.1)
                lead = analysis.get('lead_direction', 'none')
                print(f"  Interest: {i_score:.2f} | Lead: {lead}")

                # 3. Update Map
                fov_cells = get_fov_mask([0,0], current_yaw, FOV_DEGREES, MAX_RANGE)
                update_map(fov_cells, i_score)

                # 4. Project Glow (Curiosity)
                project_glow([0,0], current_yaw, lead)

                # 5. Plan Next Saccade
                next_yaw, score, (s_ent, s_glow) = choose_next_angle([0,0], current_yaw)

                reason = "Exploration (Entropy)"
                if s_glow > s_ent:
                    reason = "Curiosity (Following Glow)"

                print(f"  EXPLORATION LOG: Decision: {reason}")
                print(f"  EXPLORATION LOG: Saccading to {math.degrees(next_yaw):.1f}° (Ent: {s_ent:.1f}, Glow: {s_glow:.1f})")

                # Store metrics for dashboard
                last_analysis = {
                    'interest_score': i_score,
                    'lead_direction': lead,
                    'decision': reason,
                    'entropy_score': s_ent,
                    'glow_score': s_glow
                }

                # Generate top-down view
                topdown_view = create_topdown_view([0, 0], current_yaw, interest_map, seen_map, glow_map)

                # Send complete update with visualizations
                dashboard_data = {
                    'timestamp': time.time(),
                    'topdown_view': topdown_view,  # Replace live_feed with top-down
                    'snapshot': last_snapshot,
                    'metrics': {
                        'current_heading': math.degrees(current_yaw),
                        **last_analysis
                    }
                }

                # Add DINOv3 visualizations if available
                if 'pca_vis' in analysis:
                    try:
                        from scipy.ndimage import zoom
                        # Upscale visualizations
                        pca_upscaled = zoom(analysis['pca_vis'], (20, 20, 1), order=1)
                        attention_upscaled = zoom(analysis['attention_map'], (20, 20), order=1)
                        similarity_upscaled = zoom(analysis['similarity_map'], (20, 20), order=1)

                        dashboard_data['pca_vis'] = numpy_to_base64_png(pca_upscaled)
                        dashboard_data['attention_map'] = numpy_to_base64_png(attention_upscaled)
                        dashboard_data['similarity_map'] = numpy_to_base64_png(similarity_upscaled)
                    except Exception as e:
                        print(f"  WARNING: Failed to upscale visualizations: {e}")
                else:
                    print("  WARNING: No DINOv3 visualizations available")

                if ws_client:
                    ws_client.send(dashboard_data)

                # Update robot yaw
                current_yaw = next_yaw

                # Only update robot rotation if yaw changed
                if abs(current_yaw - last_yaw) > 0.01:
                    # No robot body to rotate anymore
                    # quat = p.getQuaternionFromEuler([0, 0, current_yaw])
                    # p.resetBasePositionAndOrientation(robot_id, [0, 0, 0.5], quat)
                    last_yaw = current_yaw

                # Reset analysis timer for next cycle
                last_analysis_time = time.time()

            time.sleep(0.01)
            step += 1

    except KeyboardInterrupt:
        print("\n[EXIT] Keyboard interrupt - shutting down...", flush=True)
    except Exception as e:
        print(f"\n[CRASH] Simulation crashed with error: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        # Always cleanup PyBullet resources, even on crashes
        print("[EXIT] Disconnecting PyBullet...", flush=True)
        p.disconnect()
        print("[EXIT] Cleanup complete", flush=True)

if __name__ == "__main__":
    main()
