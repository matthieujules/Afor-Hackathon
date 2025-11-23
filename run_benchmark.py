#!/usr/bin/env python3
"""
Run Benchmark: Compare Systematic vs Semantic Exploration

Runs both exploration strategies in lock-step and visualizes their
performance side-by-side in real-time.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pybullet as p
import pybullet_data
import time
import math

from discovery_tracker import DiscoveryTracker
from benchmark_runner import BenchmarkVisualizer, save_benchmark_results
from vision_alternatives import DinoV2Client


# Configuration (matches scout_semantix.py)
GRID_SIZE = 64
WORLD_SIZE = 20.0
CELL_SIZE = WORLD_SIZE / GRID_SIZE
FRAME_PERIOD = 1.0  # Faster for benchmarking (1 second per saccade)
FOV_DEGREES = 60
MAX_RANGE = 8.0
FOV_RAYS = 40
LAMBDA_ENTROPY = 0.8
LAMBDA_GLOW = 1.5


class ExplorationAgent:
    """Single agent instance for exploration (can run systematic or semantic mode)"""

    def __init__(self, mode='semantic', physics_client_id=None):
        """
        Args:
            mode: 'systematic' or 'semantic'
            physics_client_id: PyBullet client ID (for shared simulation)
        """
        self.mode = mode
        self.physics_client = physics_client_id

        # State
        self.current_yaw = np.pi / 2  # Start facing North
        self.interest_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self.seen_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self.glow_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

        # Vision client (shared across agents for efficiency)
        self.vision_client = None

        # Discovery tracker
        self.tracker = None

        # Last analysis for logging
        self.last_analysis = {}
        self.last_camera_image = None

    def set_vision_client(self, client):
        """Set shared vision client"""
        self.vision_client = client

    def set_tracker(self, tracker):
        """Set discovery tracker"""
        self.tracker = tracker

    def world_to_grid(self, world_xy):
        """Convert world coordinates to grid indices"""
        gx = int(world_xy[0] / CELL_SIZE + GRID_SIZE / 2)
        gy = int(world_xy[1] / CELL_SIZE + GRID_SIZE / 2)
        return np.clip([gy, gx], 0, GRID_SIZE - 1)

    def get_fov_mask(self, robot_pos, heading):
        """Get list of grid cells in current FOV"""
        cells = set()
        angles = np.linspace(-np.radians(FOV_DEGREES/2), np.radians(FOV_DEGREES/2), FOV_RAYS)

        for angle_offset in angles:
            ray_angle = heading + angle_offset
            for r in np.arange(0.5, MAX_RANGE, CELL_SIZE/2):
                x = robot_pos[0] + r * np.cos(ray_angle)
                y = robot_pos[1] + r * np.sin(ray_angle)
                gy, gx = self.world_to_grid([x, y])
                if 0 <= gy < GRID_SIZE and 0 <= gx < GRID_SIZE:
                    cells.add((gy, gx))

        return list(cells)

    def update_map(self, fov_cells, interest_score):
        """Update semantic maps with vision analysis"""
        for gy, gx in fov_cells:
            if self.seen_map[gy, gx] == 0:
                self.interest_map[gy, gx] = interest_score
            else:
                self.interest_map[gy, gx] = 0.7 * self.interest_map[gy, gx] + 0.3 * interest_score

            self.seen_map[gy, gx] = 1.0

    def project_glow(self, robot_pos, current_heading, lead_direction):
        """Project white glow based on visual continuity"""
        # Decay old glow
        self.glow_map *= 0.5

        if lead_direction == 'none' or lead_direction == 'center':
            return

        # Determine projection angle
        angle_offset = 0
        if lead_direction == 'left':
            angle_offset = np.radians(FOV_DEGREES/2 + 20)
        elif lead_direction == 'right':
            angle_offset = -np.radians(FOV_DEGREES/2 + 20)

        target_angle = current_heading + angle_offset

        # Project cone of curiosity
        cone_width = 40
        angles = np.linspace(-np.radians(cone_width/2), np.radians(cone_width/2), 20)

        for ang in angles:
            ray_angle = target_angle + ang
            for r in np.arange(1.0, MAX_RANGE * 0.8, CELL_SIZE):
                x = robot_pos[0] + r * np.cos(ray_angle)
                y = robot_pos[1] + r * np.sin(ray_angle)
                gy, gx = self.world_to_grid([x, y])

                if 0 <= gy < GRID_SIZE and 0 <= gx < GRID_SIZE:
                    if self.seen_map[gy, gx] < 0.5:
                        intensity = 1.0 * (1 - r/MAX_RANGE)
                        self.glow_map[gy, gx] = max(self.glow_map[gy, gx], intensity)

    def calculate_view_utility(self, robot_pos, candidate_heading):
        """Calculate utility of looking in a direction"""
        cells = self.get_fov_mask(robot_pos, candidate_heading)

        u_entropy = 0
        u_glow = 0

        for gy, gx in cells:
            is_seen = self.seen_map[gy, gx]
            if not is_seen:
                u_entropy += 1.0
            u_glow += self.glow_map[gy, gx]

        if len(cells) == 0:
            return 0, (0, 0)

        score_entropy = LAMBDA_ENTROPY * u_entropy
        score_glow = LAMBDA_GLOW * u_glow

        return score_entropy + score_glow, (score_entropy, score_glow)

    def choose_next_angle(self, robot_pos):
        """Choose next viewing angle based on mode"""
        if self.mode == 'systematic':
            # Systematic: just rotate by 22.5°
            angle_increment = 2 * np.pi / 16
            next_angle = (self.current_yaw + angle_increment) % (2 * np.pi)
            score, components = self.calculate_view_utility(robot_pos, next_angle)
            return next_angle, score, components

        else:
            # Semantic: choose best angle based on entropy + glow
            best_angle = self.current_yaw
            best_score = -1e9
            best_components = (0, 0)

            candidates = np.linspace(0, 2*np.pi, 16, endpoint=False)

            for ang in candidates:
                score, components = self.calculate_view_utility(robot_pos, ang)
                if score > best_score:
                    best_score = score
                    best_angle = ang
                    best_components = components

            return best_angle, best_score, best_components

    def step(self, rgb_image):
        """
        Execute one exploration step: analyze scene, update maps, plan next view.

        Args:
            rgb_image: Current camera view (numpy array)

        Returns:
            bool: True if step successful
        """
        self.last_camera_image = rgb_image

        # 1. Analyze scene
        try:
            analysis = self.vision_client.analyze_scene(rgb_image)
        except Exception as e:
            print(f"  [{self.mode.upper()}] Vision analysis failed: {e}")
            analysis = {'interest_score': 0.1, 'lead_direction': 'none', 'hazard_score': 0.1}

        i_score = analysis.get('interest_score', 0.1)
        lead = analysis.get('lead_direction', 'none')

        # 2. Update maps
        fov_cells = self.get_fov_mask([0, 0], self.current_yaw)
        self.update_map(fov_cells, i_score)

        # 3. Project glow (only for semantic mode)
        if self.mode == 'semantic':
            self.project_glow([0, 0], self.current_yaw, lead)

        # 4. Update discovery tracker
        if self.tracker:
            newly_discovered = self.tracker.update(self.seen_map, time.time())
            if newly_discovered:
                print(f"  [{self.mode.upper()}] Discovered {len(newly_discovered)} new objects!")

        # 5. Plan next saccade
        next_yaw, score, (s_ent, s_glow) = self.choose_next_angle([0, 0])

        reason = "Systematic Rotation" if self.mode == 'systematic' else \
                 ("Curiosity (Glow)" if s_glow > s_ent else "Exploration (Entropy)")

        # Store analysis
        self.last_analysis = {
            'interest_score': i_score,
            'lead_direction': lead,
            'decision': reason,
            'entropy_score': s_ent,
            'glow_score': s_glow
        }

        print(f"  [{self.mode.upper():11s}] Yaw: {math.degrees(self.current_yaw):6.1f}° → {math.degrees(next_yaw):6.1f}° | "
              f"Interest: {i_score:.2f} | Lead: {lead:6s} | {reason}")

        # Update yaw
        self.current_yaw = next_yaw

        # Increment saccade counter
        if self.tracker:
            self.tracker.increment_saccade()

        return True

    def get_state_dict(self):
        """Get current state for visualization"""
        return {
            'seen_map': self.seen_map.copy(),
            'interest_map': self.interest_map.copy(),
            'glow_map': self.glow_map.copy(),
            'heading': self.current_yaw,
            'camera_image': self.last_camera_image,
            'tracker': self.tracker,
            'last_analysis': self.last_analysis
        }


def setup_pybullet_env():
    """Setup PyBullet environment with 3 distinct object clusters"""
    print("[SETUP] Connecting to PyBullet...")
    client_id = p.connect(p.DIRECT)  # Headless mode for faster rendering
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    # Load environment
    p.loadURDF("plane.urdf")

    # Walls
    wall_height = 3.0
    wall_z = wall_height / 2
    walls = []

    p.loadURDF("cube.urdf", [0, 8, wall_z], globalScaling=16, useFixedBase=True)
    walls.append((-8, 8, 16, 0.5))

    p.loadURDF("cube.urdf", [8, 0, wall_z], globalScaling=16, useFixedBase=True)
    walls.append((8, -8, 0.5, 16))

    p.loadURDF("cube.urdf", [0, -8, wall_z], globalScaling=16, useFixedBase=True)
    walls.append((-8, -8, 16, 0.5))

    p.loadURDF("cube.urdf", [-8, 0, wall_z], globalScaling=16, useFixedBase=True)
    walls.append((-8, -8, 0.5, 16))

    # Objects - ONLY 3 CLUSTERS (no random boxes)
    object_positions = []

    print("[SETUP] Creating 3 object clusters...")

    # CLUSTER 1: NORTH (visible from start - agent faces North)
    # 5 objects arranged along north wall
    print("  Cluster 1 (NORTH): 5 objects at y=6-7m")
    north_cluster = [
        [0, 7, 0],      # Center table
        [-2, 6.5, 0.2], # Left table
        [2, 6.5, -0.2], # Right table
        [-1, 7, 0.3],   # Left-center box on shelf
        [1, 7, -0.1],   # Right-center box on shelf
    ]
    for i, pos in enumerate(north_cluster):
        if i < 3:  # First 3 are tables
            p.loadURDF("table/table.urdf", pos, p.getQuaternionFromEuler([0, 0, pos[2]]), useFixedBase=True)
            object_positions.append((pos[0], pos[1], 'desk'))
        else:  # Last 2 are boxes
            p.loadURDF("cube.urdf", pos, globalScaling=0.5, useFixedBase=True)
            object_positions.append((pos[0], pos[1], 'box'))

    # CLUSTER 2: WEST (NOT visible initially - agent must turn left ~90°)
    # 5 objects along west wall
    print("  Cluster 2 (WEST): 5 objects at x=-6 to -7m")
    west_cluster = [
        [-7, 2, 0],     # Top desk
        [-7, 0, 0.3],   # Center desk
        [-7, -2, -0.2], # Bottom desk
        [-6.5, 1, 0.5], # Box on top desk
        [-6.5, -1, 0.4],# Box on bottom desk
    ]
    for i, pos in enumerate(west_cluster):
        if i < 3:  # First 3 are desks
            p.loadURDF("table/table.urdf", pos, p.getQuaternionFromEuler([0, 0, pos[2]]), useFixedBase=True)
            object_positions.append((pos[0], pos[1], 'desk'))
        else:  # Last 2 are boxes
            p.loadURDF("cube.urdf", pos, globalScaling=0.4, useFixedBase=True)
            object_positions.append((pos[0], pos[1], 'box'))

    # CLUSTER 3: SOUTH (behind agent - must turn 180°)
    # 5 objects along south wall
    print("  Cluster 3 (SOUTH): 5 objects at y=-6 to -7m")
    south_cluster = [
        [0, -7, 0],       # Center desk
        [2, -7, 0.4],     # Right desk
        [-2, -7, -0.3],   # Left desk
        [1, -6.5, 0.6],   # Right box
        [-1, -6.5, -0.2], # Left box
    ]
    for i, pos in enumerate(south_cluster):
        if i < 3:  # First 3 are desks
            p.loadURDF("table/table.urdf", pos, p.getQuaternionFromEuler([0, 0, pos[2]]), useFixedBase=True)
            object_positions.append((pos[0], pos[1], 'desk'))
        else:  # Last 2 are boxes
            p.loadURDF("cube.urdf", pos, globalScaling=0.4, useFixedBase=True)
            object_positions.append((pos[0], pos[1], 'box'))

    print(f"[SETUP] Environment ready: {len(object_positions)} objects in 3 clusters")
    print(f"         - North cluster: 5 objects (VISIBLE at start)")
    print(f"         - West cluster:  5 objects (turn LEFT to see)")
    print(f"         - South cluster: 5 objects (turn AROUND to see)\n")

    return client_id, object_positions, walls


def get_camera_image(yaw):
    """Render camera view from fixed position"""
    cam_height = 1.5
    pos = [0, 0, cam_height]

    target_distance = 2.0
    tx = pos[0] + target_distance * np.cos(yaw)
    ty = pos[1] + target_distance * np.sin(yaw)
    tz = cam_height - 0.2

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=pos,
        cameraTargetPosition=[tx, ty, tz],
        cameraUpVector=[0, 0, 1]
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=FOV_DEGREES, aspect=1.0, nearVal=0.1, farVal=20.0
    )

    w, h, rgb, depth, seg = p.getCameraImage(
        width=320, height=320,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )

    rgb = np.array(rgb, dtype=np.uint8).reshape((h, w, 4))
    return rgb[:, :, :3]


def run_benchmark(max_saccades=20):
    """
    Run benchmark comparison between systematic and semantic exploration.

    Args:
        max_saccades: Maximum number of saccades before stopping
    """
    print("\n" + "="*60)
    print("SEMANTIX EXPLORATION BENCHMARK")
    print("="*60 + "\n")

    # Setup PyBullet
    client_id, object_positions, walls = setup_pybullet_env()

    # Initialize vision client (shared)
    print("[INIT] Loading vision model...")
    vision_client = DinoV2Client()
    print("[INIT] Vision model ready\n")

    # Create agents
    systematic_agent = ExplorationAgent(mode='systematic', physics_client_id=client_id)
    semantic_agent = ExplorationAgent(mode='semantic', physics_client_id=client_id)

    systematic_agent.set_vision_client(vision_client)
    semantic_agent.set_vision_client(vision_client)

    # Create discovery trackers
    systematic_tracker = DiscoveryTracker(object_positions)
    semantic_tracker = DiscoveryTracker(object_positions)

    systematic_agent.set_tracker(systematic_tracker)
    semantic_agent.set_tracker(semantic_tracker)

    # Create visualizer
    print("[VIZ] Starting visualizer...")
    viz = BenchmarkVisualizer(object_positions, walls)
    print("[VIZ] Visualizer ready\n")

    print("="*60)
    print("STARTING EXPLORATION (press Ctrl+C to stop early)")
    print("="*60 + "\n")

    # Start timers
    start_time = time.time()
    systematic_completion_time = None
    semantic_completion_time = None
    systematic_completed = False
    semantic_completed = False

    try:
        for saccade in range(max_saccades):
            print(f"\n[STEP {saccade+1}/{max_saccades}]")

            # Get camera images for both agents
            systematic_image = get_camera_image(systematic_agent.current_yaw)
            semantic_image = get_camera_image(semantic_agent.current_yaw)

            # Step both agents (only if not yet complete)
            if not systematic_completed:
                systematic_agent.step(systematic_image)
                if systematic_tracker.is_complete() and not systematic_completed:
                    systematic_completion_time = time.time() - start_time
                    systematic_completed = True
                    print(f"\n{'='*60}")
                    print(f"  SYSTEMATIC COMPLETE at {systematic_completion_time:.1f}s ({systematic_tracker.get_stats()['saccades']} saccades)")
                    print(f"{'='*60}\n")

            if not semantic_completed:
                semantic_agent.step(semantic_image)
                if semantic_tracker.is_complete() and not semantic_completed:
                    semantic_completion_time = time.time() - start_time
                    semantic_completed = True
                    print(f"\n{'='*60}")
                    print(f"  SEMANTIC COMPLETE at {semantic_completion_time:.1f}s ({semantic_tracker.get_stats()['saccades']} saccades)")
                    print(f"{'='*60}\n")

            # Update visualization
            viz.update(systematic_agent.get_state_dict(),
                      semantic_agent.get_state_dict())

            # Check if both complete
            if systematic_completed and semantic_completed:
                print("\n[COMPLETE] Both agents found all objects!")
                break

            time.sleep(FRAME_PERIOD)

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Benchmark stopped by user")

    finally:
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60 + "\n")

        # Print results
        stats_sys = systematic_tracker.get_stats()
        stats_sem = semantic_tracker.get_stats()

        improvement = ((stats_sys['saccades'] - stats_sem['saccades']) /
                      stats_sys['saccades'] * 100) if stats_sys['saccades'] > 0 else 0

        print(f"SYSTEMATIC SCAN:")
        print(f"  Saccades: {stats_sys['saccades']}")
        print(f"  Objects: {stats_sys['discovered']}/{stats_sys['total_objects']} ({stats_sys['coverage_percent']:.1f}%)")
        if systematic_completion_time:
            print(f"  Time to 100%: {systematic_completion_time:.1f}s")
        print()

        print(f"SEMANTIC CURIOSITY:")
        print(f"  Saccades: {stats_sem['saccades']}")
        print(f"  Objects: {stats_sem['discovered']}/{stats_sem['total_objects']} ({stats_sem['coverage_percent']:.1f}%)")
        if semantic_completion_time:
            print(f"  Time to 100%: {semantic_completion_time:.1f}s")
        print()

        print(f"IMPROVEMENT:")
        print(f"  Saccades: {improvement:+.1f}% ({abs(stats_sys['saccades'] - stats_sem['saccades'])} fewer)")

        if systematic_completion_time and semantic_completion_time:
            time_improvement = ((systematic_completion_time - semantic_completion_time) /
                              systematic_completion_time * 100)
            print(f"  Time: {time_improvement:+.1f}% ({abs(systematic_completion_time - semantic_completion_time):.1f}s faster)")
        print()

        # Save results
        save_benchmark_results(systematic_tracker, semantic_tracker)

        # Keep visualization open
        print("\n[VIZ] Close window to exit...")
        input("Press Enter to close...")

        viz.close()
        p.disconnect()


if __name__ == "__main__":
    run_benchmark(max_saccades=20)
