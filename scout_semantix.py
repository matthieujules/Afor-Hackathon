#!/usr/bin/env python3
"""
Semantix: Research-Grade Proactive Hazard Scout with Discrete Vision

Implements Language-as-Cost (LaC) perception + Semantic-Entropy guided exploration
for robotic hazard mapping under severe perception constraints (0.2 Hz vision).

Key Features:
- VLM-style hazard scoring → Bayesian Beta posterior per grid cell
- Semantic entropy quantification for epistemic uncertainty
- "White glow" prediction for unseen high-value regions (Gaussian spatial prior)
- Event-triggered replanning (5s frame period, simulating RasPi constraints)
- 2×2 Mission Control dashboard: visited | hazard | entropy | utility

References:
[1] Language-as-Cost: Proactive Hazard Mapping using VLM (arXiv:2508.03138)
[2] Active semantic exploration (ActiveSGM/ActiveGAMER)
"""

import pybullet as p
import pybullet_data
import numpy as np
import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow
import time
import os

# ============================================================================
# GLOBAL PARAMETERS
# ============================================================================

# Grid world configuration
GRID_SIZE = 64          # 64×64 grid
WORLD_SIZE = 20.0       # ±10m square world
CELL_SIZE = WORLD_SIZE / GRID_SIZE

# Vision system (discrete, RasPi-like)
FRAME_PERIOD = 5.0      # seconds between frames (0.2 Hz)
FOV_DEGREES = 70        # field of view
MAX_RANGE = 6.0         # max vision range (meters)
FOV_RAYS = 31           # ray density for FOV projection

# Utility weights (for Language-as-Cost planning)
LAMBDA_HAZARD = 1.0     # weight on expected hazard value
LAMBDA_ENTROPY = 0.5    # weight on epistemic uncertainty
LAMBDA_GLOW = 0.8       # weight on predicted unseen value (white glow)
LAMBDA_PATH = 0.1       # cost on path distance

# VLM settings
USE_REAL_VLM = os.getenv('USE_VLM', '0') == '1'  # set USE_VLM=1 for real VLM
VLM_ENDPOINT = os.getenv('VLM_ENDPOINT', 'http://localhost:8000/score')

# Bayesian update parameters
CONFIDENCE_DECAY = 0.15  # w = exp(-kappa * range)
STALENESS_THRESHOLD = 2.0  # seconds; reduce weight if updated too recently

# Ablation modes (toggle with keyboard)
ABLATION_MODE = 'full'  # 'hazard_only', 'entropy_only', 'full'

# ============================================================================
# BAYESIAN MAPPING STATE
# ============================================================================

# Beta posterior per cell: Beta(α, β) over P(hazard)
alpha = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.float32)  # success counts
beta = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.float32)   # failure counts

# Observation metadata
seen = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)     # 0/1 mask
last_obs_time = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)  # timestamp

# Gaussian kernel for spatial risk diffusion (white glow)
# 5×5 Gaussian (σ ≈ 1.0)
K_GLOW = np.array([
    [1,  4,  7,  4, 1],
    [4, 16, 26, 16, 4],
    [7, 26, 41, 26, 7],
    [4, 16, 26, 16, 4],
    [1,  4,  7,  4, 1]
], dtype=np.float32)
K_GLOW /= K_GLOW.sum()

# ============================================================================
# BAYESIAN MAPPING UTILITIES
# ============================================================================

def hazard_mean():
    """Posterior mean μ = α/(α+β) ∈ [0,1]"""
    return alpha / (alpha + beta)

def hazard_entropy(mu):
    """Shannon entropy H(μ) = -μ log μ - (1-μ) log(1-μ)"""
    eps = 1e-6
    mu_safe = np.clip(mu, eps, 1 - eps)
    return -(mu_safe * np.log(mu_safe) + (1 - mu_safe) * np.log(1 - mu_safe))

def conv2d_manual(a, kernel):
    """Fast 2D convolution without scipy (reflect padding)"""
    pad = kernel.shape[0] // 2
    a_padded = np.pad(a, pad, mode='reflect')
    out = np.zeros_like(a)
    kh, kw = kernel.shape
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            out[i, j] = np.sum(a_padded[i:i+kh, j:j+kw] * kernel)
    return out

def white_glow(mu, seen_mask):
    """
    Predicted hazard value in UNSEEN cells via Gaussian spatial prior.

    Intuition: Hazards have spatial extent. Convolve observed hazard mean
    with Gaussian kernel, then mask to only unseen cells → "white glow"
    showing where we predict value might be lurking.

    Returns: array same shape as mu, nonzero only in unseen cells
    """
    convolved = conv2d_manual(mu, K_GLOW)
    return convolved * (1 - seen_mask)  # zero out seen cells

def bayes_update(cell_scores, current_time):
    """
    Bayesian update: incorporate VLM-style observations into Beta posterior.

    Args:
        cell_scores: dict[(gy, gx) -> (s, w)]
            s: hazard score ∈ [0,1] (from VLM or stub)
            w: confidence weight (range-dependent)
        current_time: simulation time (seconds)

    Updates global arrays: alpha, beta, seen, last_obs_time
    """
    global alpha, beta, seen, last_obs_time

    for (gy, gx), (s, w) in cell_scores.items():
        # Staleness gating: reduce weight if cell updated very recently
        time_since_last = current_time - last_obs_time[gy, gx]
        if time_since_last < STALENESS_THRESHOLD:
            w *= 0.3  # debounce rapid updates

        # Beta-Bernoulli conjugate update
        alpha[gy, gx] += w * s
        beta[gy, gx] += w * (1 - s)

        # Mark as observed
        seen[gy, gx] = 1
        last_obs_time[gy, gx] = current_time

# ============================================================================
# PERCEPTION: VLM-STYLE HAZARD SCORING
# ============================================================================

def world_to_grid(world_xy):
    """Convert world coordinates (meters) to grid indices"""
    gx = int(world_xy[0] / CELL_SIZE + GRID_SIZE / 2)
    gy = int(world_xy[1] / CELL_SIZE + GRID_SIZE / 2)
    return np.clip([gy, gx], 0, GRID_SIZE - 1)

def grid_to_world(grid_yx):
    """Convert grid indices to world coordinates (meters)"""
    gy, gx = grid_yx
    wx = (gx - GRID_SIZE / 2) * CELL_SIZE
    wy = (gy - GRID_SIZE / 2) * CELL_SIZE
    return np.array([wx, wy])

def fov_cells(robot_pos, robot_heading_rad):
    """
    Project camera FOV onto grid cells.

    Returns: list of (gy, gx, range_meters) within robot's field of view
    """
    cells = []
    angles = np.linspace(-np.radians(FOV_DEGREES/2),
                         np.radians(FOV_DEGREES/2),
                         FOV_RAYS)

    for angle_offset in angles:
        ray_angle = robot_heading_rad + angle_offset
        # Cast ray from robot position
        r = CELL_SIZE  # start at first cell
        while r <= MAX_RANGE:
            x = robot_pos[0] + r * np.cos(ray_angle)
            y = robot_pos[1] + r * np.sin(ray_angle)
            gy, gx = world_to_grid([x, y])

            if 0 <= gy < GRID_SIZE and 0 <= gx < GRID_SIZE:
                cells.append((gy, gx, r))
            r += CELL_SIZE

    # Deduplicate cells (same cell hit by multiple rays)
    seen_cells = {}
    for gy, gx, r in cells:
        key = (gy, gx)
        if key not in seen_cells or r < seen_cells[key]:
            seen_cells[key] = r

    return [(gy, gx, r) for (gy, gx), r in seen_cells.items()]

def hazard_score_stub(world_xy, hazard_locations):
    """
    Fast deterministic hazard scorer (stub for VLM).

    Returns high score (1.0) near known hazard locations,
    low ambient score (0.1) elsewhere.
    """
    min_dist = float('inf')
    for hazard_pos in hazard_locations:
        dist = np.linalg.norm(world_xy - hazard_pos[:2])  # x, y only
        min_dist = min(min_dist, dist)

    if min_dist < 1.5:  # within 1.5m of hazard
        return 1.0
    return 0.1  # ambient/background risk

def hazard_score_vlm(image_crop):
    """
    Real VLM hazard scorer (HTTP API).

    Sends image to VLM endpoint, gets hazard score ∈ [0,1].
    Prompt: "Rate hazard (chemical canister OR liquid spill) from 0-1."
    """
    try:
        import requests
        from io import BytesIO
        from PIL import Image

        # Convert numpy array to JPEG
        img = Image.fromarray(image_crop)
        buf = BytesIO()
        img.save(buf, format='JPEG', quality=85)
        buf.seek(0)

        response = requests.post(
            VLM_ENDPOINT,
            files={'image': buf},
            data={'prompt': 'Rate hazard relevant to chemical canister OR liquid spill from 0 to 1. Return just a number.'},
            timeout=3.0
        )

        if response.status_code == 200:
            return float(response.json()['score'])
        else:
            print(f"VLM API error: {response.status_code}")
            return 0.1
    except Exception as e:
        print(f"VLM error: {e}, falling back to stub")
        return 0.1

def score_fov_cells(fov_cell_list, robot_pos, hazard_locations, camera_image=None):
    """
    Score all cells in FOV with hazard likelihood.

    Args:
        fov_cell_list: list of (gy, gx, range)
        robot_pos: robot world position
        hazard_locations: list of hazard positions (for stub)
        camera_image: optional RGB image (for real VLM)

    Returns: dict[(gy, gx) -> (s, w)]
        s: hazard score ∈ [0,1]
        w: confidence weight (range-decayed)
    """
    scores = {}

    # If using real VLM, score the whole image once
    if USE_REAL_VLM and camera_image is not None:
        global_score = hazard_score_vlm(camera_image)
    else:
        global_score = None

    for gy, gx, r in fov_cell_list:
        world_xy = grid_to_world((gy, gx))

        # Get hazard score
        if global_score is not None:
            s = global_score  # use VLM score for all visible cells
        else:
            s = hazard_score_stub(world_xy, hazard_locations)

        # Confidence decays with range
        w = np.exp(-CONFIDENCE_DECAY * r)

        scores[(gy, gx)] = (s, w)

    return scores

# ============================================================================
# PLANNING: UTILITY-BASED WAYPOINT SELECTION
# ============================================================================

def candidate_waypoints(robot_grid_yx, step_cells=2):
    """
    Generate candidate waypoints in 8 directions around robot.

    Args:
        robot_grid_yx: (gy, gx) current position
        step_cells: distance in grid cells

    Returns: list of (gy, gx) candidates
    """
    gy, gx = robot_grid_yx
    candidates = []

    for dy in [-step_cells, 0, step_cells]:
        for dx in [-step_cells, 0, step_cells]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = gy + dy, gx + dx
            if 0 <= ny < GRID_SIZE and 0 <= nx < GRID_SIZE:
                candidates.append((ny, nx))

    return candidates

def utility_at_waypoint(waypoint_yx, robot_yx, mu, entropy, glow):
    """
    Compute utility U(x) at candidate waypoint.

    U = λ₁·max_hazard + λ₂·max_entropy + λ₃·max_glow - λ₄·path_cost

    Intuition: high value if nearby cells have high hazard (exploitation),
    high uncertainty (exploration), or high predicted unseen value (active sensing).
    """
    wy, wx = waypoint_yx

    # Extract 3×3 neighborhood around waypoint
    y0, y1 = max(0, wy-1), min(GRID_SIZE, wy+2)
    x0, x1 = max(0, wx-1), min(GRID_SIZE, wx+2)

    patch_mu = mu[y0:y1, x0:x1]
    patch_H = entropy[y0:y1, x0:x1]
    patch_glow = glow[y0:y1, x0:x1]

    # Compute value terms
    val_hazard = LAMBDA_HAZARD * np.max(patch_mu) if patch_mu.size > 0 else 0
    val_entropy = LAMBDA_ENTROPY * np.max(patch_H) if patch_H.size > 0 else 0
    val_glow = LAMBDA_GLOW * np.max(patch_glow) if patch_glow.size > 0 else 0

    # Ablation support
    if ABLATION_MODE == 'hazard_only':
        val_entropy = 0
        val_glow = 0
    elif ABLATION_MODE == 'entropy_only':
        val_hazard = 0
        val_glow = 0

    # Path cost (Euclidean distance)
    path_cost = LAMBDA_PATH * np.linalg.norm(np.array(waypoint_yx) - np.array(robot_yx))

    return val_hazard + val_entropy + val_glow - path_cost

def choose_waypoint(robot_grid_yx, mu, entropy, glow):
    """
    Select best waypoint via utility maximization.

    Returns: (best_waypoint_yx, best_utility)
    """
    candidates = candidate_waypoints(robot_grid_yx, step_cells=2)

    best_wp = robot_grid_yx
    best_U = -1e9

    for wp in candidates:
        U = utility_at_waypoint(wp, robot_grid_yx, mu, entropy, glow)
        if U > best_U:
            best_U = U
            best_wp = wp

    return best_wp, best_U

# ============================================================================
# PYBULLET SIMULATION ENVIRONMENT
# ============================================================================

def setup_pybullet_env():
    """Initialize PyBullet simulation with warehouse-like environment"""
    # Connect to physics server
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)

    # Load ground plane with concrete texture
    plane_id = p.loadURDF("plane.urdf")
    p.changeVisualShape(plane_id, -1, rgbaColor=[0.5, 0.5, 0.5, 1])

    # Create warehouse walls with industrial look
    wall_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.15, 10, 2])
    wall_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.15, 10, 2],
                                      rgbaColor=[0.4, 0.4, 0.45, 1])

    p.createMultiBody(0, wall_shape, wall_visual, [10, 0, 2])   # +X wall
    p.createMultiBody(0, wall_shape, wall_visual, [-10, 0, 2])  # -X wall

    wall_shape2 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[10, 0.15, 2])
    wall_visual2 = p.createVisualShape(p.GEOM_BOX, halfExtents=[10, 0.15, 2],
                                       rgbaColor=[0.4, 0.4, 0.45, 1])

    p.createMultiBody(0, wall_shape2, wall_visual2, [0, 10, 2])   # +Y wall
    p.createMultiBody(0, wall_shape2, wall_visual2, [0, -10, 2])  # -Y wall

    # Add some warehouse props (shelves, pallets)
    create_warehouse_props()

    return physics_client

def create_warehouse_props():
    """Create warehouse environment props (shelves, pallets, crates)"""

    # Wooden pallets
    pallet_positions = [
        [-8, -7, 0.1],
        [8, -7, 0.1],
        [-8, 8, 0.1],
        [7, 8, 0.1]
    ]

    for pos in pallet_positions:
        # Pallet base
        pallet_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.8, 0.08])
        pallet_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.8, 0.08],
                                           rgbaColor=[0.6, 0.4, 0.2, 1])
        p.createMultiBody(10.0, pallet_shape, pallet_visual, pos)

        # Crate on top
        crate_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.4])
        crate_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.4],
                                          rgbaColor=[0.5, 0.35, 0.2, 1])
        crate_pos = [pos[0], pos[1], pos[2] + 0.5]
        p.createMultiBody(5.0, crate_shape, crate_visual, crate_pos)

    # Industrial shelving units
    shelf_positions = [
        [-9, 0, 1.0],
        [9, 0, 1.0]
    ]

    for pos in shelf_positions:
        # Vertical posts
        post_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 1.0])
        post_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 1.0],
                                         rgbaColor=[0.3, 0.3, 0.3, 1])

        for offset_y in [-1.0, 1.0]:
            post_pos = [pos[0], pos[1] + offset_y, pos[2]]
            p.createMultiBody(0, post_shape, post_visual, post_pos)

        # Horizontal shelves
        shelf_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 1.0, 0.02])
        shelf_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 1.0, 0.02],
                                          rgbaColor=[0.35, 0.35, 0.35, 1])

        for shelf_height in [0.5, 1.0, 1.5]:
            shelf_pos = [pos[0], pos[1], shelf_height]
            p.createMultiBody(0, shelf_shape, shelf_visual, shelf_pos)

def create_r2d2_robot(start_pos):
    """
    Create simple R2D2-style differential drive robot using MultiBody links.

    Returns: robot_id
    """
    # Body (cylinder)
    body_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.3, height=0.8)
    body_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.3, length=0.8,
                                      rgbaColor=[0.8, 0.8, 0.9, 1])

    # Wheel shapes
    wheel_radius = 0.15
    wheel_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=wheel_radius, height=0.05)
    wheel_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=wheel_radius, length=0.05,
                                       rgbaColor=[0.2, 0.2, 0.2, 1])

    # Link definitions for 2 wheels
    link_masses = [1.0, 1.0]
    link_collision_shapes = [wheel_shape, wheel_shape]
    link_visual_shapes = [wheel_visual, wheel_visual]
    
    # Position relative to parent (chassis)
    # Chassis center is at z=0.4. Wheel center at z=0.15. Diff z = -0.25.
    link_positions = [
        [0, 0.35, -0.25],  # Left
        [0, -0.35, -0.25]  # Right
    ]
    
    # Orientation: Rotate 90 deg around X to make cylinder roll
    quat = p.getQuaternionFromEuler([np.pi/2, 0, 0])
    link_orientations = [quat, quat]
    
    link_inertial_frame_pos = [[0,0,0], [0,0,0]]
    link_inertial_frame_orn = [[0,0,0,1], [0,0,0,1]]
    
    link_parent_indices = [0, 0] # Both attached to base
    link_joint_types = [p.JOINT_REVOLUTE, p.JOINT_REVOLUTE]
    link_joint_axis = [[0,0,1], [0,0,1]] # Rotate around Z of the link

    robot_id = p.createMultiBody(
        baseMass=5.0,
        baseCollisionShapeIndex=body_shape,
        baseVisualShapeIndex=body_visual,
        basePosition=[start_pos[0], start_pos[1], 0.4],
        linkMasses=link_masses,
        linkCollisionShapeIndices=link_collision_shapes,
        linkVisualShapeIndices=link_visual_shapes,
        linkPositions=link_positions,
        linkOrientations=link_orientations,
        linkInertialFramePositions=link_inertial_frame_pos,
        linkInertialFrameOrientations=link_inertial_frame_orn,
        linkParentIndices=link_parent_indices,
        linkJointTypes=link_joint_types,
        linkJointAxis=link_joint_axis
    )

    return robot_id

def create_hazard_objects():
    """
    Create realistic hazard objects (chemical barrels, spills, warning signs).

    Returns: list of (x, y, z) positions
    """
    hazards = []

    # === CHEMICAL BARRELS (55-gallon drums) ===
    # Single barrels with warning markings
    single_barrel_positions = [
        [5, 5, 0.45],
        [-3, 7, 0.45]
    ]

    for pos in single_barrel_positions:
        # Main barrel body (red with yellow stripe)
        barrel_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.3, height=0.9)
        barrel_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.3, length=0.9,
                                           rgbaColor=[0.85, 0.1, 0.05, 1])  # Bright red
        barrel_id = p.createMultiBody(15.0, barrel_shape, barrel_visual, pos)

        # Yellow warning stripe (top ring)
        stripe_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.32, length=0.15,
                                           rgbaColor=[0.95, 0.85, 0.1, 1])  # Warning yellow
        stripe_pos = [pos[0], pos[1], pos[2] + 0.35]
        p.createMultiBody(0, -1, stripe_visual, stripe_pos)

        # Black hazard symbol (small cylinder on top)
        symbol_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.08, length=0.02,
                                           rgbaColor=[0.1, 0.1, 0.1, 1])
        symbol_pos = [pos[0], pos[1], pos[2] + 0.46]
        p.createMultiBody(0, -1, symbol_visual, symbol_pos)

        hazards.append(pos)

    # Stacked barrels (more dangerous - multiple hazards)
    stacked_positions = [
        [-5, -5],
        [6, -4]
    ]

    for base_xy in stacked_positions:
        # Bottom barrel
        pos_bottom = [base_xy[0], base_xy[1], 0.45]
        barrel_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.3, height=0.9)
        barrel_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.3, length=0.9,
                                           rgbaColor=[0.9, 0.15, 0.05, 1])
        p.createMultiBody(15.0, barrel_shape, barrel_visual, pos_bottom)

        # Yellow stripe
        stripe_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.32, length=0.15,
                                           rgbaColor=[0.95, 0.85, 0.1, 1])
        p.createMultiBody(0, -1, stripe_visual, [pos_bottom[0], pos_bottom[1], pos_bottom[2] + 0.35])

        # Top barrel (offset for realism)
        offset_x = 0.1 if base_xy[0] > 0 else -0.1
        pos_top = [base_xy[0] + offset_x, base_xy[1], 0.45 + 0.9]
        barrel_visual2 = p.createVisualShape(p.GEOM_CYLINDER, radius=0.3, length=0.9,
                                            rgbaColor=[0.85, 0.12, 0.08, 1])
        p.createMultiBody(15.0, barrel_shape, barrel_visual2, pos_top)

        # Stripe on top barrel
        p.createMultiBody(0, -1, stripe_visual, [pos_top[0], pos_top[1], pos_top[2] + 0.35])

        hazards.append(pos_bottom)
        hazards.append(pos_top)

    # === LIQUID SPILLS ===
    # Large irregular spill (multiple overlapping puddles)
    spill_center_1 = [3, -6]
    create_realistic_spill(spill_center_1, num_puddles=8, color=[0.7, 0.05, 0.0, 0.85])
    hazards.append([spill_center_1[0], spill_center_1[1], 0.01])

    # Medium spill near leaking barrel
    spill_center_2 = [-7, 2]
    create_realistic_spill(spill_center_2, num_puddles=5, color=[0.8, 0.1, 0.0, 0.9])
    hazards.append([spill_center_2[0], spill_center_2[1], 0.01])

    # === WARNING SIGNS ===
    # Place warning signs near major hazards
    sign_positions = [
        [5.8, 5.8, 0.5],    # Near single barrel
        [-5.8, -4.5, 0.5],  # Near stacked barrels
        [3.5, -5.5, 0.5]    # Near large spill
    ]

    for pos in sign_positions:
        # Yellow warning sign (triangle)
        sign_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.3, 0.3])
        sign_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.3, 0.3],
                                         rgbaColor=[0.95, 0.8, 0.1, 1])
        p.createMultiBody(0, sign_shape, sign_visual, pos)

        # Red exclamation mark
        mark_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.03, 0.05, 0.15],
                                         rgbaColor=[0.9, 0.0, 0.0, 1])
        p.createMultiBody(0, -1, mark_visual, [pos[0], pos[1], pos[2] + 0.05])

    return hazards

def create_realistic_spill(center_xy, num_puddles=6, color=[0.7, 0.0, 0.0, 0.8]):
    """
    Create an irregular spill using multiple overlapping puddles.

    Args:
        center_xy: [x, y] center of spill
        num_puddles: number of overlapping puddles for irregular shape
        color: RGBA color
    """
    for i in range(num_puddles):
        # Random offset from center
        angle = (i / num_puddles) * 2 * np.pi + np.random.uniform(-0.3, 0.3)
        radius = np.random.uniform(0.3, 0.8)
        offset_x = radius * np.cos(angle)
        offset_y = radius * np.sin(angle)

        # Puddle position
        pos = [center_xy[0] + offset_x, center_xy[1] + offset_y, 0.005]

        # Elliptical puddle
        size_x = np.random.uniform(0.4, 0.7)
        size_y = np.random.uniform(0.4, 0.7)

        puddle_shape = p.createCollisionShape(p.GEOM_BOX,
                                             halfExtents=[size_x, size_y, 0.005])
        puddle_visual = p.createVisualShape(p.GEOM_BOX,
                                           halfExtents=[size_x, size_y, 0.005],
                                           rgbaColor=color)
        p.createMultiBody(0, puddle_shape, puddle_visual, pos)

def get_robot_state(robot_id):
    """
    Get robot position and heading.

    Returns: (pos_xy, heading_rad)
    """
    pos, orn = p.getBasePositionAndOrientation(robot_id)

    # Extract yaw from quaternion
    euler = p.getEulerFromQuaternion(orn)
    heading = euler[2]  # yaw

    return np.array([pos[0], pos[1]]), heading

def set_wheel_velocities(robot_id, v_left, v_right):
    """
    Control robot via differential drive (velocity-based).

    Simple approximation: apply forces to base to simulate wheel motion.
    """
    # Get current state
    pos, heading = get_robot_state(robot_id)

    # Compute average velocity and turn rate
    v = (v_left + v_right) / 2.0
    omega = (v_right - v_left) / 0.7  # 0.7m wheelbase

    # Compute velocity in world frame
    vx = v * np.cos(heading)
    vy = v * np.sin(heading)

    # Apply velocity (simplified: just set linear/angular velocity)
    p.resetBaseVelocity(robot_id, [vx, vy, 0], [0, 0, omega])

def navigate_to_waypoint(robot_id, goal_world_xy):
    """
    Simple proportional controller to navigate toward waypoint.

    Returns: (v_left, v_right) wheel velocities
    """
    pos, heading = get_robot_state(robot_id)

    # Vector to goal
    to_goal = goal_world_xy - pos
    dist = np.linalg.norm(to_goal)

    if dist < 0.5:  # close enough
        return 0, 0

    # Desired heading
    desired_heading = np.arctan2(to_goal[1], to_goal[0])

    # Heading error
    heading_error = desired_heading - heading
    # Normalize to [-π, π]
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

    # Proportional control
    base_speed = 3.0
    turn_gain = 2.0

    v_left = base_speed - turn_gain * heading_error
    v_right = base_speed + turn_gain * heading_error

    # Clamp
    v_left = np.clip(v_left, -5, 5)
    v_right = np.clip(v_right, -5, 5)

    return v_left, v_right

# ============================================================================
# VISUALIZATION: 2×2 MISSION CONTROL DASHBOARD
# ============================================================================

def setup_dashboard():
    """Create 2×2 matplotlib dashboard"""
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Semantix: Mission Control - Semantic Uncertainty Engine',
                 fontsize=16, fontweight='bold')

    # Configure each subplot
    titles = [
        'Visited Map',
        'Hazard Posterior μ',
        'Semantic Entropy H(μ)',
        'Utility & Action'
    ]

    for ax, title in zip(axes.flat, titles):
        ax.set_title(title, fontweight='bold')
        ax.set_aspect('equal')
        ax.set_xlim(0, GRID_SIZE)
        ax.set_ylim(0, GRID_SIZE)

    # Create image artists
    images = []
    for ax in axes.flat:
        im = ax.imshow(np.zeros((GRID_SIZE, GRID_SIZE)),
                       origin='lower', cmap='gray', vmin=0, vmax=1)
        images.append(im)

    # Custom colormaps
    images[1].set_cmap('RdYlBu_r')  # Hazard: blue → red
    images[2].set_cmap('Greys')      # Entropy: dark → white
    images[3].set_cmap('hot')        # Utility: black → white → yellow

    # Add colorbars
    for ax, im in zip(axes.flat, images):
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()

    return fig, axes, images

def update_dashboard(axes, images, mu, entropy, glow, robot_grid_yx, waypoint_grid_yx, current_time, metrics):
    """Update all 4 panels of the dashboard"""

    # Panel 1: Visited map
    images[0].set_data(seen)
    images[0].set_clim(0, 1)

    # Panel 2: Hazard posterior
    images[1].set_data(mu)
    images[1].set_clim(0, 1)

    # Panel 3: Entropy
    images[2].set_data(entropy)
    images[2].set_clim(0, np.log(2))  # max entropy of Bernoulli

    # Panel 4: Utility (show glow for visualization)
    # Combine mu and glow for dramatic effect
    utility_vis = mu + glow
    images[3].set_data(utility_vis)
    images[3].set_clim(0, 1.5)

    # Clear previous markers
    for ax in axes.flat:
        for artist in ax.patches[:]:
            artist.remove()
        for artist in ax.texts[:]:
            artist.remove()

    # Add robot position marker (all panels)
    for ax in axes.flat:
        circle = Circle((robot_grid_yx[1], robot_grid_yx[0]), radius=1.5,
                       color='lime', fill=False, linewidth=2, label='Scout')
        ax.add_patch(circle)

    # Add waypoint marker and arrow (panel 4 only)
    if waypoint_grid_yx is not None:
        ax_util = axes[1, 1]

        # Waypoint marker
        circle = Circle((waypoint_grid_yx[1], waypoint_grid_yx[0]), radius=1.0,
                       color='cyan', fill=True, alpha=0.7, label='Target')
        ax_util.add_patch(circle)

        # Arrow from robot to waypoint
        dy = waypoint_grid_yx[0] - robot_grid_yx[0]
        dx = waypoint_grid_yx[1] - robot_grid_yx[1]
        if abs(dx) > 0.1 or abs(dy) > 0.1:
            arrow = FancyArrow(robot_grid_yx[1], robot_grid_yx[0],
                              dx * 0.7, dy * 0.7,
                              width=0.5, head_width=2, head_length=1.5,
                              color='yellow', alpha=0.8)
            ax_util.add_patch(arrow)

    # Add metrics text overlay (top-left corner of panel 1)
    ax_visited = axes[0, 0]
    metrics_text = (
        f"Time: {current_time:.1f}s\n"
        f"Coverage: {metrics['coverage']:.1f}%\n"
        f"Frames: {metrics['frames']}\n"
        f"Avg Hazard: {metrics['avg_hazard']:.3f}\n"
        f"Avg Entropy: {metrics['avg_entropy']:.3f}\n"
        f"Mode: {ABLATION_MODE}"
    )
    ax_visited.text(2, GRID_SIZE - 2, metrics_text,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                   color='white', family='monospace')

    plt.pause(0.001)

# ============================================================================
# MAIN CONTROL LOOP
# ============================================================================

def main():
    global ABLATION_MODE

    print("=" * 70)
    print("Semantix: Research-Grade Proactive Hazard Scout")
    print("=" * 70)
    print("Features:")
    print("  • VLM-style hazard perception (Language-as-Cost)")
    print("  • Bayesian semantic mapping with Beta posteriors")
    print("  • Semantic entropy for epistemic uncertainty")
    print("  • White-glow prediction for unseen value")
    print("  • Event-triggered replanning (5s frame period, 0.2 Hz)")
    print("=" * 70)
    print(f"Using {'REAL VLM' if USE_REAL_VLM else 'STUB SCORER'} for hazard detection")
    print("=" * 70)
    print("\nKeyboard controls:")
    print("  1: Ablation mode = hazard_only")
    print("  2: Ablation mode = entropy_only")
    print("  3: Ablation mode = full (default)")
    print("  Q: Quit")
    print("=" * 70)

    # Setup PyBullet
    physics_client = setup_pybullet_env()

    # Create robot (start at origin)
    start_pos = [0, 0]
    robot_id = create_r2d2_robot(start_pos)

    # Create hazards
    hazard_locations = create_hazard_objects()
    hazard_array = np.array(hazard_locations)

    print(f"\nSpawned {len(hazard_locations)} hazard objects in warehouse")

    # Setup visualization
    fig, axes, images = setup_dashboard()

    # Initialize control state
    last_frame_time = -FRAME_PERIOD  # trigger immediately
    current_waypoint = None
    frame_count = 0

    # Metrics
    metrics = {
        'coverage': 0,
        'frames': 0,
        'avg_hazard': 0,
        'avg_entropy': 0
    }

    # Simulation loop
    sim_time = 0.0
    dt = 1.0 / 240.0  # PyBullet default timestep
    step = 0

    print("\nStarting simulation...")
    print("Watch the Mission Control dashboard for real-time updates!\n")

    try:
        while True:
            # Step physics
            p.stepSimulation()
            sim_time = step * dt
            step += 1

            # Get robot state
            robot_pos, robot_heading = get_robot_state(robot_id)
            robot_grid_yx = world_to_grid(robot_pos)

            # ================================================================
            # EVENT-TRIGGERED PERCEPTION & PLANNING (discrete vision)
            # ================================================================
            time_since_frame = sim_time - last_frame_time

            if time_since_frame >= FRAME_PERIOD or current_waypoint is None:
                # NEW FRAME ARRIVED! (0.2 Hz tick)
                frame_count += 1
                print(f"\n[Frame {frame_count}] t={sim_time:.1f}s | Robot @ {robot_pos}")

                # 1. Project FOV onto grid
                fov_cells_list = fov_cells(robot_pos, robot_heading)
                print(f"  FOV: {len(fov_cells_list)} cells visible")

                # 2. Score hazards (VLM or stub)
                cell_scores = score_fov_cells(fov_cells_list, robot_pos, hazard_array)

                # 3. Bayesian update
                bayes_update(cell_scores, sim_time)

                # 4. Compute derived maps
                mu = hazard_mean()
                H = hazard_entropy(mu)
                glow = white_glow(mu, seen)

                # 5. Choose next waypoint via utility maximization
                current_waypoint, utility = choose_waypoint(robot_grid_yx, mu, H, glow)
                waypoint_world = grid_to_world(current_waypoint)

                print(f"  New waypoint: grid {current_waypoint} → world {waypoint_world}")
                print(f"  Utility: {utility:.3f}")

                # 6. Update metrics
                metrics['coverage'] = 100 * np.sum(seen) / (GRID_SIZE ** 2)
                metrics['frames'] = frame_count
                metrics['avg_hazard'] = np.mean(mu[seen > 0]) if np.sum(seen) > 0 else 0
                metrics['avg_entropy'] = np.mean(H[seen > 0]) if np.sum(seen) > 0 else 0

                print(f"  Coverage: {metrics['coverage']:.1f}% | "
                      f"Avg Hazard: {metrics['avg_hazard']:.3f} | "
                      f"Avg Entropy: {metrics['avg_entropy']:.3f}")

                # 7. Update dashboard
                update_dashboard(axes, images, mu, H, glow,
                               robot_grid_yx, current_waypoint,
                               sim_time, metrics)

                last_frame_time = sim_time

            # ================================================================
            # CONTINUOUS CONTROL (between frames)
            # ================================================================
            if current_waypoint is not None:
                waypoint_world = grid_to_world(current_waypoint)
                v_left, v_right = navigate_to_waypoint(robot_id, waypoint_world)
                set_wheel_velocities(robot_id, v_left, v_right)

            # Slow down simulation for human viewing
            if step % 24 == 0:  # 10 Hz update
                time.sleep(0.01)

            # Check for stop condition (coverage or time limit)
            if metrics['coverage'] > 95:
                print("\n" + "=" * 70)
                print("MISSION COMPLETE: 95% coverage achieved!")
                print("=" * 70)
                break

            if sim_time > 300:  # 5 minute timeout
                print("\n" + "=" * 70)
                print("TIME LIMIT REACHED")
                print("=" * 70)
                break

    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")

    finally:
        print("\n" + "=" * 70)
        print("FINAL METRICS")
        print("=" * 70)
        print(f"  Runtime: {sim_time:.1f}s")
        print(f"  Frames processed: {frame_count}")
        print(f"  Coverage: {metrics['coverage']:.1f}%")
        print(f"  Avg hazard (visited): {metrics['avg_hazard']:.3f}")
        print(f"  Avg entropy (visited): {metrics['avg_entropy']:.3f}")
        print("=" * 70)

        # Keep dashboard open
        print("\nDashboard will remain open. Close the matplotlib window to exit.")
        plt.ioff()
        plt.show()

        # Disconnect PyBullet
        p.disconnect()

if __name__ == "__main__":
    main()
