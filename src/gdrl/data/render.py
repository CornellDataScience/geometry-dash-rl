"""Synthetic frame renderer.

Real screen capture of a running Geometry Dash window is finicky on
macOS (needs screen-recording permission + a window bounds lookup) and
breaks completely when training against the mock env. For rapid
iteration on the student architecture we want a non-zero 84x84 input
that still carries the information a vision student needs to learn:
player position, ground line, and nearby obstacles.

This module rasterizes the privileged observation into that view. It is
deliberately simple — no anti-aliasing, no colors, just a schematic
"radar" image centered on the player with a configurable world window.
"""

from __future__ import annotations

import numpy as np

FRAME_SIZE = 84
PLAYER_PX = 6
OBSTACLE_PX = 4
GROUND_PX_OFFSET = 54  # ground line y-coordinate in the 84-row image

# Feature-obs layout (must match the real mod and MockIPCAdapter).
OBJ_OBS_START = 8
FLOATS_PER_OBJ = 6
MAX_NEARBY_OBJECTS = 100


def render_frame(
    obs: np.ndarray,
    *,
    world_width: float = 400.0,
    world_height: float = 200.0,
    frame_size: int = FRAME_SIZE,
) -> np.ndarray:
    """Rasterize a privileged obs into a grayscale 84x84 frame.

    The player sits at roughly 1/4 from the left. Positive x in the
    world extends to the right of the player. The world window is
    ``world_width`` units wide so obstacles at relX in [-world_width/4,
    3*world_width/4] are visible.
    """
    frame = np.zeros((frame_size, frame_size), dtype=np.uint8)

    # Ground line — a single row at GROUND_PX_OFFSET.
    ground_y = min(frame_size - 1, GROUND_PX_OFFSET * frame_size // FRAME_SIZE)
    frame[ground_y, :] = 80

    # Player position in world: read from obs[1]. The screen always
    # shows the player at a fixed screen x; only the player y moves.
    player_screen_x = frame_size // 4
    player_y_world = float(obs[1])
    # Map world y upward from ground line. 1 world unit = (frame_size/2)/world_height px.
    y_scale = (frame_size / 2) / max(world_height, 1e-6)
    player_pixel_y = int(ground_y - player_y_world * y_scale)
    player_pixel_y = max(0, min(frame_size - 1, player_pixel_y))

    _draw_box(frame, player_screen_x, player_pixel_y, PLAYER_PX, value=255)

    # Obstacles: read all MAX_NEARBY_OBJECTS slots, skip empty ones.
    visible_left = -world_width / 4.0
    visible_right = 3.0 * world_width / 4.0
    x_scale = frame_size / world_width
    for i in range(MAX_NEARBY_OBJECTS):
        base = OBJ_OBS_START + i * FLOATS_PER_OBJ
        if base + FLOATS_PER_OBJ > obs.shape[0]:
            break
        rel_x = float(obs[base + 0])
        rel_y = float(obs[base + 1])
        obj_id = float(obs[base + 3])
        # Empty slot: mod zero-fills trailing slots.
        if rel_x == 0.0 and rel_y == 0.0 and obj_id == 0.0:
            continue
        if rel_x < visible_left or rel_x > visible_right:
            continue
        screen_x = int(player_screen_x + rel_x * x_scale)
        # Obstacles sit on the ground (rel_y is relative to player y).
        screen_y = int(ground_y - (rel_y + player_y_world) * y_scale)
        screen_x = max(0, min(frame_size - 1, screen_x))
        screen_y = max(0, min(frame_size - 1, screen_y))
        _draw_box(frame, screen_x, screen_y, OBSTACLE_PX, value=200)

    return frame


def _draw_box(frame: np.ndarray, cx: int, cy: int, size: int, value: int) -> None:
    h, w = frame.shape
    half = size // 2
    x0 = max(0, cx - half)
    x1 = min(w, cx + half + 1)
    y0 = max(0, cy - half)
    y1 = min(h, cy + half + 1)
    if x0 < x1 and y0 < y1:
        frame[y0:y1, x0:x1] = value
