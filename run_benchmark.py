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
FRAME_PERIOD = 2.0  # 2 seconds per saccade for testing (change to 10.0 for demos)
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
                # Log detailed discovery info
                discovered_desc = []
                for idx in newly_discovered:
                    pos = self.tracker.object_positions[idx]
                    discovered_desc.append(f"#{idx+1}@({pos[0]:.1f},{pos[1]:.1f})")

                print(f"  [{self.mode.upper():11s}] ✓ DISCOVERED {len(newly_discovered)}: {', '.join(discovered_desc)}")

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

        # Log current state with glow info
        glow_cells = np.count_nonzero(self.glow_map > 0.1)
        glow_max = self.glow_map.max()

        print(f"  [{self.mode.upper():11s}] Yaw: {math.degrees(self.current_yaw):6.1f}° → {math.degrees(next_yaw):6.1f}°")
        print(f"  [{self.mode.upper():11s}]   Interest: {i_score:.2f} | Lead: {lead:6s} | Decision: {reason}")
        print(f"  [{self.mode.upper():11s}]   Glow: {glow_cells} cells active (max={glow_max:.2f}) | Entropy: {s_ent:.1f} | Glow Score: {s_glow:.1f}")

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

    # Objects - ONLY 3 CLUSTERS (scattered, non-uniform)
    object_positions = []

    print("[SETUP] Creating 3 scattered object clusters...")

    # SUPER CLEAR CLUSTERS: 3 dense, distinct clusters with HUGE empty gaps
    # All within 7m radius for visibility (MAX_RANGE=8m)
    # Agent at [0,0] starts facing North (90°), FOV=60° sees 60°-120°

    # CLUSTER 1: NORTHWEST DENSE PACK - 6 objects tightly grouped
    # Positioned at 120°-150° (just at/past left FOV edge) to trigger LEFT detection
    print("  Cluster 1 (NORTHWEST): 6 objects TIGHTLY PACKED at 120°-150°")
    print("    → Semantic agent sees edge → glow LEFT → locks onto entire cluster")
    northwest_cluster = [
        [-3, 5.2, 0.1],    # angle≈120°, dist=6.03m
        [-3.5, 5, 0.2],    # angle≈125°, dist=6.10m
        [-4, 4.8, -0.1],   # angle≈130°, dist=6.24m
        [-4.5, 4.5, 0.3],  # angle≈135°, dist=6.36m
        [-4.8, 4, 0.2],    # angle≈140°, dist=6.24m
        [-5, 3.5, -0.1],   # angle≈145°, dist=6.10m
    ]
    for i, pos in enumerate(northwest_cluster):
        dist = (pos[0]**2 + pos[1]**2)**0.5
        angle_deg = math.degrees(math.atan2(pos[1], pos[0]))
        print(f"    Object {i+1}: ({pos[0]:4.1f}, {pos[1]:4.1f}) @ {angle_deg:5.1f}°, dist={dist:.2f}m")
        if i % 2 == 0:
            p.loadURDF("table/table.urdf", pos, p.getQuaternionFromEuler([0, 0, pos[2]]), useFixedBase=True)
            object_positions.append((pos[0], pos[1], 'desk'))
        else:
            p.loadURDF("cube.urdf", pos, globalScaling=0.4, useFixedBase=True)
            object_positions.append((pos[0], pos[1], 'box'))

    # EMPTY ZONE: 150° to 210° - NOTHING HERE
    print("  [EMPTY ZONE: 150°-210° - forces systematic to waste saccades]")

    # CLUSTER 2: SOUTHWEST DENSE PACK - 5 objects tightly grouped
    # Far from cluster 1, requires deliberate turn
    print("  Cluster 2 (SOUTHWEST): 5 objects TIGHTLY PACKED at 210°-240°")
    print("    → Isolated cluster, semantic must explore to find")
    southwest_cluster = [
        [-4, -4.5, 0.1],   # angle≈228°, dist=6.02m
        [-4.5, -4, 0.2],   # angle≈222°, dist=6.02m
        [-5, -3.5, -0.1],  # angle≈215°, dist=6.10m
        [-5.2, -3, 0.3],   # angle≈210°, dist=6.03m
        [-4.8, -4.8, 0.2], # angle≈225°, dist=6.79m
    ]
    for i, pos in enumerate(southwest_cluster):
        dist = (pos[0]**2 + pos[1]**2)**0.5
        angle_deg = math.degrees(math.atan2(pos[1], pos[0]))
        print(f"    Object {i+7}: ({pos[0]:4.1f}, {pos[1]:4.1f}) @ {angle_deg:5.1f}°, dist={dist:.2f}m")
        if i < 3:
            p.loadURDF("table/table.urdf", pos, p.getQuaternionFromEuler([0, 0, pos[2]]), useFixedBase=True)
            object_positions.append((pos[0], pos[1], 'desk'))
        else:
            p.loadURDF("cube.urdf", pos, globalScaling=0.4, useFixedBase=True)
            object_positions.append((pos[0], pos[1], 'box'))

    # EMPTY ZONE: 240° to 300° - NOTHING HERE
    print("  [EMPTY ZONE: 240°-300° - more wasted systematic saccades]")

    # CLUSTER 3: SOUTHEAST DENSE PACK - 4 objects tightly grouped
    # Opposite side from start, maximum discovery challenge
    print("  Cluster 3 (SOUTHEAST): 4 objects TIGHTLY PACKED at 300°-330°")
    print("    → Opposite from start, semantic must hunt for it")
    southeast_cluster = [
        [3.5, -5, 0.1],    # angle≈305°, dist=6.10m
        [4, -4.5, -0.2],   # angle≈312°, dist=6.02m
        [4.5, -4, 0.3],    # angle≈318°, dist=6.02m
        [4.8, -3.5, 0.1],  # angle≈324°, dist=6.03m
    ]
    for i, pos in enumerate(southeast_cluster):
        dist = (pos[0]**2 + pos[1]**2)**0.5
        angle_deg = math.degrees(math.atan2(pos[1], pos[0]))
        print(f"    Object {i+12}: ({pos[0]:4.1f}, {pos[1]:4.1f}) @ {angle_deg:5.1f}°, dist={dist:.2f}m")
        if i < 2:
            p.loadURDF("table/table.urdf", pos, p.getQuaternionFromEuler([0, 0, pos[2]]), useFixedBase=True)
            object_positions.append((pos[0], pos[1], 'desk'))
        else:
            p.loadURDF("cube.urdf", pos, globalScaling=0.35, useFixedBase=True)
            object_positions.append((pos[0], pos[1], 'box'))

    print(f"\n[SETUP] Environment ready: {len(object_positions)} objects in 3 DENSE clusters")
    print(f"         - Cluster 1 (NW):  6 objects PACKED at 120°-150° (edge visible)")
    print(f"         - EMPTY ZONE:      150°-210° (NO OBJECTS)")
    print(f"         - Cluster 2 (SW):  5 objects PACKED at 210°-240°")
    print(f"         - EMPTY ZONE:      240°-300° (NO OBJECTS)")
    print(f"         - Cluster 3 (SE):  4 objects PACKED at 300°-330°")
    print(f"\n[SETUP] Expected behavior:")
    print(f"         ✓ Semantic: Sees Cluster 1 edge → glow LEFT → locks onto cluster → discovers 6 quickly")
    print(f"                     Then hunts for remaining clusters → faster discovery")
    print(f"         ✗ Systematic: Rotates 22.5° each step → wastes time scanning empty zones")
    print(f"                       Discovers objects only when rotation happens to hit clusters\n")

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


def run_benchmark(max_saccades=30):
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

    # Create agents - BOTH START AT IDENTICAL POSITION
    INITIAL_YAW = np.pi / 2  # 90° = North
    print(f"[INIT] Both agents starting at position [0, 0] facing {math.degrees(INITIAL_YAW):.1f}° (North)")
    print(f"[INIT] FOV = {FOV_DEGREES}° → agents see from {math.degrees(INITIAL_YAW - np.radians(FOV_DEGREES/2)):.1f}° to {math.degrees(INITIAL_YAW + np.radians(FOV_DEGREES/2)):.1f}°")

    systematic_agent = ExplorationAgent(mode='systematic', physics_client_id=client_id)
    semantic_agent = ExplorationAgent(mode='semantic', physics_client_id=client_id)

    # FORCE SAME INITIAL YAW
    systematic_agent.current_yaw = INITIAL_YAW
    semantic_agent.current_yaw = INITIAL_YAW

    systematic_agent.set_vision_client(vision_client)
    semantic_agent.set_vision_client(vision_client)

    # Create discovery trackers
    systematic_tracker = DiscoveryTracker(object_positions)
    semantic_tracker = DiscoveryTracker(object_positions)

    systematic_agent.set_tracker(systematic_tracker)
    semantic_agent.set_tracker(semantic_tracker)

    print(f"[INIT] Agents initialized successfully")
    print(f"       Systematic: yaw={math.degrees(systematic_agent.current_yaw):.1f}°")
    print(f"       Semantic:   yaw={math.degrees(semantic_agent.current_yaw):.1f}°\n")

    # Create visualizer
    print("[VIZ] Starting visualizer...")
    viz = BenchmarkVisualizer(object_positions, walls)
    print("[VIZ] Visualizer ready\n")

    print("="*60)
    print("STARTING EXPLORATION")
    print("Running until both agents discover all 15 objects...")
    print("(max limit: {} saccades as safety)".format(max_saccades))
    print("="*60 + "\n")

    # Start timers
    start_time = time.time()
    systematic_completion_time = None
    semantic_completion_time = None
    systematic_completed = False
    semantic_completed = False

    try:
        for saccade in range(max_saccades):
            # Progress indicator with current coverage
            sys_stats = systematic_tracker.get_stats()
            sem_stats = semantic_tracker.get_stats()

            print(f"\n{'='*60}")
            print(f"SACCADE #{saccade+1} - T={saccade*FRAME_PERIOD:.0f}s")
            print(f"  Systematic: {sys_stats['discovered']}/15 objects | Semantic: {sem_stats['discovered']}/15 objects")
            print(f"  Current yaw: Systematic={math.degrees(systematic_agent.current_yaw):.1f}° | Semantic={math.degrees(semantic_agent.current_yaw):.1f}°")
            print(f"{'='*60}")

            # BOTH AGENTS STEP IN PERFECT SYNCHRONIZATION
            # Get camera images for both agents AT SAME TIME
            systematic_image = get_camera_image(systematic_agent.current_yaw)
            semantic_image = get_camera_image(semantic_agent.current_yaw)

            # Verify both images are the same initially (on first step)
            if saccade == 0:
                print(f"[INIT-CHECK] First frame - both agents should see identical views")
                print(f"             Systematic camera: yaw={math.degrees(systematic_agent.current_yaw):.1f}°")
                print(f"             Semantic camera:   yaw={math.degrees(semantic_agent.current_yaw):.1f}°")
                images_match = np.array_equal(systematic_image, semantic_image)
                print(f"             Initial images match: {images_match}")

            # Step both agents TOGETHER (only if not yet complete)
            if not systematic_completed:
                systematic_agent.step(systematic_image)
                if systematic_tracker.is_complete() and not systematic_completed:
                    systematic_completion_time = time.time() - start_time
                    systematic_completed = True
                    print(f"\n{'🎯'*30}")
                    print(f"  SYSTEMATIC COMPLETE!")
                    print(f"  Time: {systematic_completion_time:.1f}s | Saccades: {systematic_tracker.get_stats()['saccades']}")
                    print(f"{'🎯'*30}\n")

            if not semantic_completed:
                semantic_agent.step(semantic_image)
                if semantic_tracker.is_complete() and not semantic_completed:
                    semantic_completion_time = time.time() - start_time
                    semantic_completed = True
                    print(f"\n{'🚀'*30}")
                    print(f"  SEMANTIC COMPLETE!")
                    print(f"  Time: {semantic_completion_time:.1f}s | Saccades: {semantic_tracker.get_stats()['saccades']}")
                    print(f"{'🚀'*30}\n")

            # Update visualization
            viz.update(systematic_agent.get_state_dict(),
                      semantic_agent.get_state_dict())

            # Check if both complete
            if systematic_completed and semantic_completed:
                print("\n[COMPLETE] Both agents found all 15 objects!")
                break

            # Wait exactly FRAME_PERIOD seconds before next synchronized step
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

        # Keep visualization open briefly
        print("\n[VIZ] Keeping visualization open for 5 seconds...")
        time.sleep(5)

        viz.close()
        p.disconnect()

        print("\n[BENCHMARK] Complete!")


if __name__ == "__main__":
    run_benchmark(max_saccades=50)  # Generous limit to ensure completion
