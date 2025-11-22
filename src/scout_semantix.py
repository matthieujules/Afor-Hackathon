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
import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
import time
import os
import math
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

    # Walls - proper 20x20m enclosure with grounded walls
    print("[SETUP] Loading walls...")
    # Walls are 10m cubes, centered at z=5 (bottom at z=0, top at z=10)
    # Positioned at ±10m to create 20x20m enclosure
    p.loadURDF("cube.urdf", [10, 0, 5], globalScaling=10)  # East wall
    p.loadURDF("cube.urdf", [-10, 0, 5], globalScaling=10)  # West wall
    p.loadURDF("cube.urdf", [0, 10, 5], globalScaling=10)  # North wall
    p.loadURDF("cube.urdf", [0, -10, 5], globalScaling=10)  # South wall
    print("[SETUP] Walls created (20m x 20m enclosure)")

    # === INTERESTING ZONE (Right/East) ===
    print("[SETUP] Creating interesting zone (East)...")
    # Cluttered area with boxes - visible from robot at origin
    for i in range(5):
        x = 4 + i  # X: 4 to 8 meters east
        y = -2 + i * 0.8  # Y: -2 to +1 meters
        box_id = p.loadURDF("cube.urdf", [x, y, 0.5], globalScaling=1.0)
        print(f"  Box {i+1} at ({x:.1f}, {y:.1f}, 0.5)")

    # Trail of ducks leading to clutter - robot should follow this
    print("[SETUP] Creating duck trail...")
    for i in range(5):
        x = 1 + i  # X: 1 to 5 meters east
        y = -1 + i * 0.3  # Y: -1 to +0.2 meters
        duck_id = p.loadURDF("duck_vhacd.urdf", [x, y, 0.5], globalScaling=2.0)
        print(f"  Duck {i+1} at ({x:.1f}, {y:.1f}, 0.5)")

    # === BORING ZONE (West) ===
    # Empty

    # === ROBOT (FIXED BASE PANOPTICON) ===
    # useFixedBase=True prevents physics from fighting manual rotation
    # Robot is anchored at origin, rotates on yaw axis only
    print("[SETUP] Spawning Panopticon Agent (Fixed Base)...")
    robot_id = p.loadURDF("r2d2.urdf", [0, 0, 0.5], useFixedBase=True)
    print(f"[SETUP] Robot ID: {robot_id} (Fixed at origin)")

    return robot_id

def get_camera_image(robot_id, yaw):
    """Render camera view from robot head, oriented according to yaw."""
    # Camera positioned at robot head
    cam_height = 0.8
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
    global current_yaw

    print("[MAIN] Starting main function...")
    robot_id = setup_env()
    print("[MAIN] Environment setup complete")

    # Dashboard - 2x3 grid: Live feed + Analysis snapshot + 3 semantic visualizations
    print("[VIZ] Setting up matplotlib visualization...")
    plt.ion()
    print("[VIZ] Creating figure...")
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    ax_live = fig.add_subplot(gs[0, 0])
    ax_snapshot = fig.add_subplot(gs[1, 0])
    ax_pca = fig.add_subplot(gs[0, 1])
    ax_attention = fig.add_subplot(gs[1, 1])
    ax_similarity = fig.add_subplot(gs[0, 2])

    print("[VIZ] Figure created")

    ax_live.set_title("LIVE FEED (~10 FPS)", fontweight='bold', fontsize=12, color='red')
    ax_snapshot.set_title("Analysis Snapshot (Every 5s)", fontweight='bold', fontsize=10)
    ax_pca.set_title("DINOv3 PCA Rainbow", fontsize=10)
    ax_attention.set_title("DINOv3 Attention Map", fontsize=10)
    ax_similarity.set_title("DINOv3 Patch Similarity", fontsize=10)

    # Initialize with dummy data (will be updated)
    img_live = ax_live.imshow(np.zeros((320, 320, 3), dtype=np.uint8))
    img_snapshot = ax_snapshot.imshow(np.zeros((320, 320, 3), dtype=np.uint8))
    img_pca = ax_pca.imshow(np.zeros((14, 14, 3)))
    img_attention = ax_attention.imshow(np.zeros((14, 14)), cmap='hot', vmin=0, vmax=1)
    img_similarity = ax_similarity.imshow(np.zeros((14, 14)), cmap='viridis', vmin=0, vmax=1)

    ax_live.axis('off')
    ax_snapshot.axis('off')
    ax_pca.axis('off')
    ax_attention.axis('off')
    ax_similarity.axis('off')
    
    print("PANOPTICON ACTIVATED. Scanning for Interest...")

    # Set initial robot orientation
    quat = p.getQuaternionFromEuler([0, 0, current_yaw])
    p.resetBasePositionAndOrientation(robot_id, [0, 0, 0.5], quat)

    step = 0
    last_yaw = current_yaw  # Track when rotation actually changes

    try:
        while True:
            # Step physics (affects ducks/boxes, but robot is fixed base)
            p.stepSimulation()

            # Live feed update (~10 FPS at 20Hz sim speed)
            if step % 2 == 0:
                live_rgb = get_camera_image(robot_id, current_yaw)
                img_live.set_data(live_rgb)
                # Draw live feed updates more frequently
                if step % 10 == 0:  # Update display every 10 steps to reduce overhead
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()

            # Full vision analysis (every ~5 seconds)
            if step % 100 == 0:

                # 1. Capture snapshot for analysis
                rgb = get_camera_image(robot_id, current_yaw)

                # Update analysis snapshot view
                img_snapshot.set_data(rgb)

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
                    
                print(f"  Decision: {reason}")
                print(f"  Saccading to {math.degrees(next_yaw):.1f}° (Ent: {s_ent:.1f}, Glow: {s_glow:.1f})")

                # Update robot yaw
                current_yaw = next_yaw

                # Only update robot rotation if yaw changed
                if abs(current_yaw - last_yaw) > 0.01:
                    quat = p.getQuaternionFromEuler([0, 0, current_yaw])
                    p.resetBasePositionAndOrientation(robot_id, [0, 0, 0.5], quat)
                    last_yaw = current_yaw

                # Update DINOv3 Semantic Visualizations
                if 'pca_vis' in analysis:
                    # Upscale to make visualizations clearer
                    from scipy.ndimage import zoom
                    pca_upscaled = zoom(analysis['pca_vis'], (20, 20, 1), order=1)
                    attention_upscaled = zoom(analysis['attention_map'], (20, 20), order=1)
                    similarity_upscaled = zoom(analysis['similarity_map'], (20, 20), order=1)

                    img_pca.set_data(pca_upscaled)
                    img_attention.set_data(attention_upscaled)
                    img_similarity.set_data(similarity_upscaled)
                else:
                    print("  WARNING: No DINOv3 visualizations available (using DINOv2 or fallback mode)")
                
                fig.canvas.draw()
                fig.canvas.flush_events()
                
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
