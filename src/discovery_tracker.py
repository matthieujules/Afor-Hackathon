#!/usr/bin/env python3
"""
Object Discovery Tracker for Benchmarking

Tracks which objects have been discovered by the agent and measures
exploration efficiency.
"""

import numpy as np
import math
from typing import List, Tuple, Dict, Set


class DiscoveryTracker:
    """Tracks object discovery for benchmarking exploration strategies"""

    def __init__(self, object_positions: List[Tuple[float, float, str]],
                 grid_size: int = 64,
                 world_size: float = 20.0,
                 seen_threshold: float = 0.3):
        """
        Args:
            object_positions: List of (x, y, type) tuples for all objects in scene
            grid_size: Grid resolution
            world_size: World dimensions (square)
            seen_threshold: Minimum seen_map value to count as discovered
        """
        self.object_positions = object_positions
        self.grid_size = grid_size
        self.world_size = world_size
        self.cell_size = world_size / grid_size
        self.seen_threshold = seen_threshold

        # Track discovered objects by their index
        self.discovered_objects: Set[int] = set()

        # History for analysis
        self.discovery_history: List[Dict] = []  # {saccade: int, object_idx: int, timestamp: float}
        self.saccade_count = 0

        # Convert object positions to grid coordinates for faster checking
        self.object_grid_coords = []
        for x, y, obj_type in object_positions:
            gx, gy = self._world_to_grid(x, y)
            self.object_grid_coords.append((gx, gy, obj_type))

    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices"""
        gx = int(x / self.cell_size + self.grid_size / 2)
        gy = int(y / self.cell_size + self.grid_size / 2)
        gx = np.clip(gx, 0, self.grid_size - 1)
        gy = np.clip(gy, 0, self.grid_size - 1)
        return gx, gy

    def update(self, seen_map: np.ndarray, timestamp: float) -> List[int]:
        """
        Check for newly discovered objects based on seen_map coverage.

        Args:
            seen_map: Current seen map (grid_size x grid_size)
            timestamp: Current simulation time

        Returns:
            List of newly discovered object indices
        """
        newly_discovered = []

        for idx, (gx, gy, obj_type) in enumerate(self.object_grid_coords):
            if idx not in self.discovered_objects:
                # Check if object location has been seen
                if seen_map[gy, gx] >= self.seen_threshold:
                    self.discovered_objects.add(idx)
                    newly_discovered.append(idx)

                    # Log discovery
                    self.discovery_history.append({
                        'saccade': self.saccade_count,
                        'object_idx': idx,
                        'timestamp': timestamp,
                        'position': self.object_positions[idx][:2],
                        'type': self.object_positions[idx][2]
                    })

        return newly_discovered

    def increment_saccade(self):
        """Called when agent makes a saccade (changes view angle)"""
        self.saccade_count += 1

    def get_coverage_percent(self) -> float:
        """Return percentage of objects discovered"""
        if len(self.object_positions) == 0:
            return 100.0
        return 100.0 * len(self.discovered_objects) / len(self.object_positions)

    def is_complete(self) -> bool:
        """Check if all objects have been discovered"""
        return len(self.discovered_objects) == len(self.object_positions)

    def get_stats(self) -> Dict:
        """Return summary statistics"""
        return {
            'total_objects': len(self.object_positions),
            'discovered': len(self.discovered_objects),
            'coverage_percent': self.get_coverage_percent(),
            'saccades': self.saccade_count,
            'efficiency': len(self.discovered_objects) / max(self.saccade_count, 1),  # objects per saccade
            'complete': self.is_complete()
        }

    def get_discovery_order(self) -> List[Dict]:
        """Return objects in order they were discovered"""
        return sorted(self.discovery_history, key=lambda x: x['saccade'])

    def get_undiscovered_objects(self) -> List[Tuple[float, float, str]]:
        """Return list of objects not yet discovered"""
        undiscovered = []
        for idx, obj in enumerate(self.object_positions):
            if idx not in self.discovered_objects:
                undiscovered.append(obj)
        return undiscovered

    def reset(self):
        """Reset tracker for new run"""
        self.discovered_objects.clear()
        self.discovery_history.clear()
        self.saccade_count = 0


def compute_discovery_efficiency(tracker: DiscoveryTracker) -> Dict:
    """
    Compute detailed efficiency metrics from a completed run.

    Returns metrics like:
    - Discovery rate (objects/saccade)
    - Time to 50%, 75%, 100% coverage
    - Clustering efficiency (how well it exploits object clusters)
    """
    if tracker.saccade_count == 0:
        return {}

    history = tracker.get_discovery_order()
    total = len(tracker.object_positions)

    # Find saccades to reach coverage milestones
    milestones = {}
    for entry in history:
        coverage = (entry['object_idx'] + 1) / total * 100

        if coverage >= 50 and '50%' not in milestones:
            milestones['50%'] = entry['saccade']
        if coverage >= 75 and '75%' not in milestones:
            milestones['75%'] = entry['saccade']
        if coverage >= 100 and '100%' not in milestones:
            milestones['100%'] = entry['saccade']

    return {
        'discovery_rate': len(tracker.discovered_objects) / tracker.saccade_count,
        'total_saccades': tracker.saccade_count,
        'total_objects': total,
        'milestones': milestones,
        'final_coverage': tracker.get_coverage_percent()
    }
