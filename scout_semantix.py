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
USE_REAL_VLM = os.getenv('USE_VLM', '0') == '1'
if USE_REAL_VLM:
    print("Vision Mode: VLM (Gemini/OpenAI)")
    from vlm_client import VLMClient
    vlm_client = VLMClient(provider="gemini")
else:
    # Use DINOv2 by default for fast, local vision analysis (no auth required)
    # Set USE_DINOV3=1 to use DINOv3 instead (requires Hugging Face auth)
    print("Vision Mode: DINO (Local Model)")
    from vision_alternatives import DinoV2Client
    vlm_client = DinoV2Client()
    print()

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
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    
    # Room
    p.loadURDF("plane.urdf")
    # Walls
    p.loadURDF("cube.urdf", [10, 0, 1], globalScaling=10)
    p.loadURDF("cube.urdf", [-10, 0, 1], globalScaling=10)
    p.loadURDF("cube.urdf", [0, 10, 1], globalScaling=10)
    p.loadURDF("cube.urdf", [0, -10, 1], globalScaling=10)
    
    # === INTERESTING ZONE (Right) ===
    # Cluttered shelves, boxes, red canisters
    for i in range(5):
        p.loadURDF("cube.urdf", [6 + np.random.uniform(-1,1), 
                                 -5 + i*2 + np.random.uniform(-0.5,0.5), 0.5], 
                   globalScaling=0.8)
        
    # A "Trail" of small objects leading to the clutter
    for i in range(5):
        p.loadURDF("duck_vhacd.urdf", [2 + i, -5 + i*0.5, 0.5], globalScaling=3.0)

    # === BORING ZONE (Left) ===
    # Empty
    
    # Robot (Fixed Base)
    robot_id = p.loadURDF("r2d2.urdf", [0, 0, 0.5])
    return robot_id

def get_camera_image(robot_id, yaw):
    """Render camera view from robot head."""
    # R2D2 head is roughly at z=0.8
    pos = [0, 0, 0.8]
    
    # Target
    tx = pos[0] + 2 * np.cos(yaw)
    ty = pos[1] + 2 * np.sin(yaw)
    tz = pos[2] - 0.2 # Look slightly down
    
    view_matrix = p.computeViewMatrix(pos, [tx, ty, tz], [0, 0, 1])
    proj_matrix = p.computeProjectionMatrixFOV(FOV_DEGREES, 1.0, 0.1, 20.0)
    
    w, h, rgb, _, _ = p.getCameraImage(320, 320, view_matrix, proj_matrix)
    rgb = np.array(rgb, dtype=np.uint8).reshape((h, w, 4))
    return rgb[:, :, :3] # Drop alpha

# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    global current_yaw
    
    robot_id = setup_env()
    
    # Dashboard
    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax_map = axes[0]
    ax_glow = axes[1]
    ax_cam = axes[2]
    
    ax_map.set_title("Semantic Interest Map")
    ax_glow.set_title("Predictive Glow (Curiosity)")
    ax_cam.set_title("Robot Vision")
    
    img_map = ax_map.imshow(interest_map, vmin=0, vmax=1, cmap='magma', origin='lower')
    img_glow = ax_glow.imshow(glow_map, vmin=0, vmax=1, cmap='Blues', origin='lower')
    img_cam = ax_cam.imshow(np.zeros((320, 320, 3)))
    
    print("PANOPTICON ACTIVATED. Scanning for Interest...")
    
    step = 0
    try:
        while True:
            p.stepSimulation()
            
            # Rotate robot visually
            quat = p.getQuaternionFromEuler([0, 0, current_yaw])
            p.resetBasePositionAndOrientation(robot_id, [0,0,0.5], quat)
            
            if step % 100 == 0: # Every ~5 seconds (at 20Hz sim speed for demo)
                
                # 1. Capture
                rgb = get_camera_image(robot_id, current_yaw)
                img_cam.set_data(rgb)
                
                # 2. Analyze (VLM)
                print(f"Analyzing view at {math.degrees(current_yaw):.1f}°...")
                analysis = vlm_client.analyze_scene(rgb)
                
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
                
                current_yaw = next_yaw
                
                # Update Viz
                # Overlay seen mask on interest map?
                # For now just raw maps
                img_map.set_data(interest_map)
                img_glow.set_data(glow_map)
                
                # Draw robot FOV on map
                for patch in ax_map.patches:
                    patch.remove()
                wedge = Wedge((GRID_SIZE/2, GRID_SIZE/2), MAX_RANGE/CELL_SIZE, 
                              math.degrees(current_yaw) - FOV_DEGREES/2,
                              math.degrees(current_yaw) + FOV_DEGREES/2,
                              color='white', alpha=0.3)
                ax_map.add_patch(wedge)
                
                fig.canvas.draw()
                fig.canvas.flush_events()
                
            time.sleep(0.01)
            step += 1
            
    except KeyboardInterrupt:
        p.disconnect()

if __name__ == "__main__":
    main()
