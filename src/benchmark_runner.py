#!/usr/bin/env python3
"""
Benchmark Runner: Side-by-side comparison of exploration strategies

Runs both systematic (baseline) and semantic (curiosity-driven) exploration
in parallel, showing real-time visual comparison of discovery efficiency.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for web
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle
import time
import math
from typing import Dict, Tuple, List
import base64
from io import BytesIO

from discovery_tracker import DiscoveryTracker


class BenchmarkVisualizer:
    """Real-time side-by-side visualization of two exploration strategies"""

    def __init__(self, object_positions, walls, grid_size=64, world_size=20.0, fov_degrees=25):
        self.object_positions = object_positions
        self.walls = walls
        self.grid_size = grid_size
        self.world_size = world_size
        self.fov_degrees = fov_degrees

        # Create figure with 2 columns (systematic | semantic) and 4 rows
        self.fig = plt.figure(figsize=(18, 14))

        # Row 1: Top-down views + metrics
        self.ax_systematic_map = plt.subplot(3, 3, 1)
        self.ax_semantic_map = plt.subplot(3, 3, 2)
        self.ax_metrics = plt.subplot(3, 3, 3)

        # Row 2: Camera views + stats
        self.ax_systematic_cam = plt.subplot(3, 3, 4)
        self.ax_semantic_cam = plt.subplot(3, 3, 5)
        self.ax_comparison = plt.subplot(3, 3, 6)

        # Row 3: Attention map (semantic only) and glow map
        self.ax_attention = plt.subplot(3, 3, 8)  # Under semantic camera
        self.ax_glow_detail = plt.subplot(3, 3, 7)  # Under systematic camera

        # Style
        for ax in [self.ax_systematic_map, self.ax_semantic_map,
                   self.ax_systematic_cam, self.ax_semantic_cam]:
            ax.set_facecolor('#1a1a1a')

        self.fig.patch.set_facecolor('#0a0a0a')

        # Data storage for metrics plotting
        self.systematic_history = {'saccades': [], 'coverage': []}
        self.semantic_history = {'saccades': [], 'coverage': []}

    def update(self, systematic_data: Dict, semantic_data: Dict):
        """
        Update all subplots with latest data from both agents.

        Args:
            systematic_data: {
                'seen_map': ndarray,
                'interest_map': ndarray,
                'glow_map': ndarray,
                'heading': float,
                'camera_image': ndarray,
                'tracker': DiscoveryTracker,
                'last_analysis': dict
            }
            semantic_data: Same structure as systematic_data
        """

        # Recreate figure
        self.fig = plt.figure(figsize=(18, 14))
        self.ax_systematic_map = plt.subplot(3, 3, 1)
        self.ax_semantic_map = plt.subplot(3, 3, 2)
        self.ax_metrics = plt.subplot(3, 3, 3)
        self.ax_systematic_cam = plt.subplot(3, 3, 4)
        self.ax_semantic_cam = plt.subplot(3, 3, 5)
        self.ax_comparison = plt.subplot(3, 3, 6)
        self.ax_glow_detail = plt.subplot(3, 3, 7)
        self.ax_attention = plt.subplot(3, 3, 8)

        for ax in [self.ax_systematic_map, self.ax_semantic_map,
                   self.ax_systematic_cam, self.ax_semantic_cam]:
            ax.set_facecolor('#1a1a1a')

        self.fig.patch.set_facecolor('#0a0a0a')

        # 1. Draw top-down maps
        self._draw_topdown_map(self.ax_systematic_map, systematic_data, "SYSTEMATIC SCAN")
        self._draw_topdown_map(self.ax_semantic_map, semantic_data, "SEMANTIC CURIOSITY")

        # 2. Draw camera views
        self._draw_camera_view(self.ax_systematic_cam, systematic_data['camera_image'], "Systematic View")
        self._draw_camera_view(self.ax_semantic_cam, semantic_data['camera_image'], "Semantic View")

        # 3. Update metrics plot
        self._draw_metrics_comparison(systematic_data['tracker'], semantic_data['tracker'])

        # 4. Draw current stats comparison
        self._draw_stats_panel(systematic_data, semantic_data)

        # 5. Draw glow map detail
        self._draw_glow_detail(self.ax_glow_detail, semantic_data)

        # 6. Draw attention map (semantic only)
        self._draw_attention_map(self.ax_attention, semantic_data)

        plt.tight_layout()

        # Instead of showing, render to base64 image
        buf = BytesIO()
        plt.savefig(buf, format='png', facecolor='#0a0a0a', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(self.fig)

        return img_base64

    def _draw_topdown_map(self, ax, data, title):
        """Draw top-down view with discoveries highlighted"""
        seen_map = data['seen_map']
        glow_map = data['glow_map']
        heading = data['heading']
        tracker = data['tracker']

        # Composite visualization
        composite = np.zeros((self.grid_size, self.grid_size, 3))
        composite[:, :, 0] = np.clip(seen_map, 0, 1) * 0.4  # Red = explored
        composite[:, :, 1] = np.clip(glow_map * 2.0, 0, 1)  # Green = glow predictions (boosted)

        ax.imshow(composite, origin='lower',
                 extent=[-self.world_size/2, self.world_size/2,
                        -self.world_size/2, self.world_size/2])

        # Draw objects (color-coded by discovery status)
        for idx, (x, y, obj_type) in enumerate(self.object_positions):
            if idx in tracker.discovered_objects:
                color = 'lime'  # Discovered
                alpha = 0.9
            else:
                color = 'yellow'  # Undiscovered
                alpha = 0.5

            if obj_type == 'desk':
                ax.add_patch(Rectangle((x-0.5, y-0.4), 1.0, 0.8,
                           facecolor=color, alpha=alpha, edgecolor='white', linewidth=1))
            elif obj_type == 'box':
                ax.add_patch(Circle((x, y), 0.25,
                           facecolor=color, alpha=alpha, edgecolor='white', linewidth=1))

        # Draw walls
        for x, y, w, h in self.walls:
            ax.add_patch(Rectangle((x, y), w, h,
                       color='gray', alpha=0.3, edgecolor='white', linewidth=1))

        # Draw agent position and FOV
        robot_x, robot_y = 0, 0
        ax.plot(robot_x, robot_y, 'wo', markersize=10, markeredgewidth=2, markeredgecolor='cyan')

        # FOV cone
        max_range = 8.0
        cone_angle = np.radians(self.fov_degrees)
        left_angle = heading + cone_angle / 2
        right_angle = heading - cone_angle / 2

        cone_x = [robot_x,
                  robot_x + max_range * np.cos(left_angle),
                  robot_x + max_range * np.cos(right_angle)]
        cone_y = [robot_y,
                  robot_y + max_range * np.sin(left_angle),
                  robot_y + max_range * np.sin(right_angle)]

        ax.fill(cone_x, cone_y, color='cyan', alpha=0.15, edgecolor='cyan', linewidth=2)

        # Heading arrow
        arrow_length = 2.0
        ax.arrow(robot_x, robot_y,
                arrow_length * np.cos(heading),
                arrow_length * np.sin(heading),
                head_width=0.4, head_length=0.3, fc='yellow', ec='yellow', linewidth=2)

        # Title with stats
        stats = tracker.get_stats()

        # Get angular coverage from data
        angular_coverage_pct = data.get('angular_coverage_pct', 0)

        ax.set_title(f"{title}\n{stats['discovered']}/{stats['total_objects']} objects ({stats['coverage_percent']:.0f}%) | "
                    f"{stats['saccades']} saccades | FOV Coverage: {angular_coverage_pct:.0f}%",
                    color='white', fontsize=11, fontweight='bold', pad=10)

        ax.set_xlim(-self.world_size/2, self.world_size/2)
        ax.set_ylim(-self.world_size/2, self.world_size/2)
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white', labelsize=8)
        ax.set_xlabel('X (meters)', color='white', fontsize=9)
        ax.set_ylabel('Y (meters)', color='white', fontsize=9)
        ax.grid(True, alpha=0.2, color='white', linewidth=0.5)

    def _draw_camera_view(self, ax, image, title):
        """Draw camera image"""
        if image is not None:
            ax.imshow(image)
        ax.set_title(title, color='white', fontsize=11, fontweight='bold')
        ax.axis('off')

    def _draw_metrics_comparison(self, tracker_systematic, tracker_semantic):
        """Plot coverage over time comparison"""
        ax = self.ax_metrics

        # Update history
        stats_sys = tracker_systematic.get_stats()
        stats_sem = tracker_semantic.get_stats()

        self.systematic_history['saccades'].append(stats_sys['saccades'])
        self.systematic_history['coverage'].append(stats_sys['coverage_percent'])

        self.semantic_history['saccades'].append(stats_sem['saccades'])
        self.semantic_history['coverage'].append(stats_sem['coverage_percent'])

        # Plot lines
        ax.plot(self.systematic_history['saccades'],
               self.systematic_history['coverage'],
               'r-o', linewidth=2, markersize=4, label='Systematic Scan')

        ax.plot(self.semantic_history['saccades'],
               self.semantic_history['coverage'],
               'g-o', linewidth=2, markersize=4, label='Semantic Curiosity')

        ax.set_xlabel('Saccades', color='white', fontsize=10)
        ax.set_ylabel('Coverage %', color='white', fontsize=10)
        ax.set_title('Discovery Efficiency Comparison', color='white', fontsize=11, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white', labelsize=8)
        ax.set_ylim(0, 105)

        # Add completion markers
        if stats_sys['complete']:
            ax.axvline(stats_sys['saccades'], color='red', linestyle='--', alpha=0.5)
            ax.text(stats_sys['saccades'], 50, f"Systematic\nComplete\n{stats_sys['saccades']} saccades",
                   rotation=90, va='center', ha='right', color='red', fontsize=8)

        if stats_sem['complete']:
            ax.axvline(stats_sem['saccades'], color='green', linestyle='--', alpha=0.5)
            ax.text(stats_sem['saccades'], 50, f"Semantic\nComplete\n{stats_sem['saccades']} saccades",
                   rotation=90, va='center', ha='left', color='green', fontsize=8)

    def _draw_stats_panel(self, systematic_data, semantic_data):
        """Draw detailed statistics comparison"""
        ax = self.ax_comparison
        ax.axis('off')

        stats_sys = systematic_data['tracker'].get_stats()
        stats_sem = semantic_data['tracker'].get_stats()

        # Calculate improvement
        if stats_sys['saccades'] > 0:
            improvement = ((stats_sys['saccades'] - stats_sem['saccades']) /
                          stats_sys['saccades'] * 100)
        else:
            improvement = 0

        # Text summary
        summary = f"""
╔═══════════════════════════════════╗
║   BENCHMARK COMPARISON            ║
╚═══════════════════════════════════╝

SYSTEMATIC SCAN (Baseline):
  • Saccades: {stats_sys['saccades']}
  • Discovered: {stats_sys['discovered']}/{stats_sys['total_objects']}
  • Coverage: {stats_sys['coverage_percent']:.1f}%
  • Strategy: Fixed 22.5° rotation

SEMANTIC CURIOSITY (Ours):
  • Saccades: {stats_sem['saccades']}
  • Discovered: {stats_sem['discovered']}/{stats_sem['total_objects']}
  • Coverage: {stats_sem['coverage_percent']:.1f}%
  • Strategy: Glow-guided exploration

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IMPROVEMENT: {improvement:+.1f}%
  ({abs(stats_sys['saccades'] - stats_sem['saccades'])} fewer saccades)

Last Decision:
  Sys: Rotation sequence
  Sem: {semantic_data.get('last_analysis', {}).get('decision', 'N/A')}
"""

        ax.text(0.05, 0.95, summary, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               color='white',
               bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.8))

        ax.set_facecolor('#0a0a0a')

    def _draw_glow_detail(self, ax, semantic_data):
        """Draw detailed glow map heatmap"""
        glow_map = semantic_data['glow_map']

        im = ax.imshow(glow_map, cmap='hot', origin='lower', vmin=0, vmax=1)
        ax.set_title('Glow Map (Predicted Interest)', color='white', fontsize=10, fontweight='bold')
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white', labelsize=7)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Intensity', color='white', fontsize=8)
        cbar.ax.tick_params(colors='white', labelsize=7)

    def _draw_attention_map(self, ax, semantic_data):
        """Draw DINOv2 attention map from current semantic agent POV"""
        attention_map = semantic_data.get('last_analysis', {}).get('attention_map', None)

        if attention_map is not None:
            # Attention map is 16x16 from DINOv2 patch energy
            im = ax.imshow(attention_map, cmap='hot', origin='upper', vmin=0, vmax=1)
            ax.set_title('DINOv2 Attention Map\n(Current Semantic POV)',
                        color='white', fontsize=10, fontweight='bold')

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Energy', color='white', fontsize=8)
            cbar.ax.tick_params(colors='white', labelsize=7)
        else:
            ax.text(0.5, 0.5, 'No Attention Data', transform=ax.transAxes,
                   ha='center', va='center', color='white', fontsize=12)
            ax.set_title('DINOv2 Attention Map\n(Current Semantic POV)',
                        color='white', fontsize=10, fontweight='bold')

        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white', labelsize=7)
        ax.axis('off')

    def close(self):
        """Close the visualization"""
        plt.close(self.fig)


def save_benchmark_results(systematic_tracker: DiscoveryTracker,
                          semantic_tracker: DiscoveryTracker,
                          filename: str = "benchmark_results.txt"):
    """Save detailed benchmark results to file"""

    stats_sys = systematic_tracker.get_stats()
    stats_sem = semantic_tracker.get_stats()

    improvement = ((stats_sys['saccades'] - stats_sem['saccades']) /
                   stats_sys['saccades'] * 100) if stats_sys['saccades'] > 0 else 0

    with open(filename, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("SEMANTIX EXPLORATION BENCHMARK RESULTS\n")
        f.write("=" * 60 + "\n\n")

        f.write("SYSTEMATIC SCAN (Baseline):\n")
        f.write(f"  Total Saccades: {stats_sys['saccades']}\n")
        f.write(f"  Objects Discovered: {stats_sys['discovered']}/{stats_sys['total_objects']}\n")
        f.write(f"  Final Coverage: {stats_sys['coverage_percent']:.1f}%\n")
        f.write(f"  Efficiency: {stats_sys['efficiency']:.2f} objects/saccade\n\n")

        f.write("SEMANTIC CURIOSITY (Ours):\n")
        f.write(f"  Total Saccades: {stats_sem['saccades']}\n")
        f.write(f"  Objects Discovered: {stats_sem['discovered']}/{stats_sem['total_objects']}\n")
        f.write(f"  Final Coverage: {stats_sem['coverage_percent']:.1f}%\n")
        f.write(f"  Efficiency: {stats_sem['efficiency']:.2f} objects/saccade\n\n")

        f.write("COMPARISON:\n")
        f.write(f"  Improvement: {improvement:+.1f}%\n")
        f.write(f"  Saccades Saved: {stats_sys['saccades'] - stats_sem['saccades']}\n\n")

        f.write("DISCOVERY ORDER (Semantic):\n")
        for entry in semantic_tracker.get_discovery_order()[:10]:
            f.write(f"  Saccade {entry['saccade']:2d}: {entry['type']:6s} at ({entry['position'][0]:5.1f}, {entry['position'][1]:5.1f})\n")

    print(f"\n[BENCHMARK] Results saved to {filename}")


if __name__ == "__main__":
    print("Benchmark runner should be imported and used by run_benchmark.py")
