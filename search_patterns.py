"""
Search pattern generators for ground robot patrol.

All coordinates in centimeters. Returns list of (x_cm, y_cm) waypoints.
Origin (0, 0) is the robot's starting position.
"""

import math
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


def generate_lawnmower(
    width_cm: float = 200.0,
    height_cm: float = 200.0,
    spacing_cm: float = 30.0,
) -> List[Tuple[float, float]]:
    """
    Lawnmower (boustrophedon) pattern.

    Robot sweeps back and forth in parallel lines.

        ┌──────────────────┐
        │  →→→→→→→→→→→→→→  │
        │  ←←←←←←←←←←←←←←  │
        │  →→→→→→→→→→→→→→  │
        └──────────────────┘
    """
    waypoints = []
    num_lines = int(height_cm / spacing_cm) + 1

    for i in range(num_lines):
        y = i * spacing_cm
        if y > height_cm:
            y = height_cm

        if i % 2 == 0:
            waypoints.append((width_cm, y))
        else:
            waypoints.append((0.0, y))

    logger.info(f"Lawnmower pattern: {len(waypoints)} waypoints, "
                f"{width_cm}x{height_cm}cm, spacing={spacing_cm}cm")
    return waypoints


def generate_spiral(
    radius_cm: float = 100.0,
    spacing_cm: float = 30.0,
) -> List[Tuple[float, float]]:
    """
    Outward spiral pattern from center.

    Robot spirals outward from origin.
    """
    waypoints = []
    points_per_loop = 12
    num_loops = int(radius_cm / spacing_cm)

    for loop in range(1, num_loops + 1):
        r = spacing_cm * loop
        for point in range(points_per_loop):
            angle = 2 * math.pi * point / points_per_loop
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            waypoints.append((x, y))

    logger.info(f"Spiral pattern: {len(waypoints)} waypoints, "
                f"radius={radius_cm}cm, spacing={spacing_cm}cm")
    return waypoints


def generate_expanding_square(
    size_cm: float = 200.0,
    spacing_cm: float = 30.0,
) -> List[Tuple[float, float]]:
    """
    Expanding square pattern from center.

    Robot moves in increasingly larger squares.
    """
    waypoints = [(0.0, 0.0)]
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # E, N, W, S
    x, y = 0.0, 0.0
    step = 1
    half = size_cm / 2

    while step * spacing_cm <= size_cm:
        for dir_idx in range(4):
            dx, dy = directions[dir_idx]
            moves = step if dir_idx < 2 else step + 1

            for _ in range(moves):
                x += dx * spacing_cm
                y += dy * spacing_cm

                if abs(x) <= half and abs(y) <= half:
                    waypoints.append((x, y))

        step += 2

    logger.info(f"Expanding square pattern: {len(waypoints)} waypoints, "
                f"size={size_cm}cm, spacing={spacing_cm}cm")
    return waypoints


def generate_detect_only(**kwargs) -> List[Tuple[float, float]]:
    """
    No movement — just detect at current position.
    Returns a single waypoint at origin.
    """
    logger.info("Detect-only pattern: 1 waypoint at origin (no movement)")
    return [(0.0, 0.0)]


PATTERNS = {
    "lawnmower": generate_lawnmower,
    "spiral": generate_spiral,
    "expanding_square": generate_expanding_square,
    "detect_only": generate_detect_only,
}


def generate_pattern(name: str, **kwargs) -> List[Tuple[float, float]]:
    """Generate waypoints for a named pattern."""
    gen = PATTERNS.get(name)
    if gen is None:
        raise ValueError(f"Unknown pattern: {name}. Available: {list(PATTERNS.keys())}")
    return gen(**kwargs)
