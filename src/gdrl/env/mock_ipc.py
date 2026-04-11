"""Mock IPC adapter — simulates a tiny Geometry Dash-style platformer in
pure Python so the teacher PPO loop, the collector, and the env tests can
run without a live Geometry Dash instance.

Matches the real ``GeodeSharedMemoryAdapter`` interface:
  - verify_version()
  - read_tick() / wait_next_tick()
  - read_obs() / read_next_obs()
  - send_action(int)
  - send_reset()
  - read_level_complete_flag()
  - read_player_input()
  - close()

The observation layout mirrors the real mod:
  obs[0] = x position
  obs[1] = y position
  obs[2] = y velocity
  obs[3] = x velocity per frame
  obs[4] = on_ground (0/1)
  obs[5] = is_dead  (0/1)
  obs[6] = speed multiplier (always 1 in mock)
  obs[7] = mode (always 0 = cube in mock)
  obs[8..607] = up to 100 nearest obstacles as (relX, relY, objType, objID, scaleX, scaleY)

Physics are deliberately simple so PPO can solve it quickly and any
regressions in the training loop show up fast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import List, Optional

import numpy as np


OBS_DIM = 608
OBJ_OBS_START = 8
FLOATS_PER_OBJ = 6
MAX_NEARBY_OBJECTS = 100


@dataclass
class MockConfig:
    level_length: float = 1500.0
    x_velocity: float = 10.0
    gravity: float = -0.8
    jump_velocity: float = 14.0
    ground_y: float = 0.0
    player_height: float = 2.0
    spike_height: float = 3.0
    spike_half_width: float = 1.2
    min_gap: float = 25.0
    max_gap: float = 55.0
    first_spike_x: float = 60.0
    seed: Optional[int] = None
    expected_version: int = 2

    # How many floats the adapter will write each frame.
    obs_dim: int = OBS_DIM


@dataclass
class _Spike:
    x: float
    obj_id: int = 8  # spikeUp in live_monitor's OBJ_ID_NAMES


@dataclass
class _State:
    x: float = 0.0
    y: float = 0.0
    y_vel: float = 0.0
    x_vel: float = 0.0
    on_ground: bool = True
    dead: bool = False
    level_done: bool = False
    tick: int = 0
    spikes: List[_Spike] = field(default_factory=list)


class MockIPCAdapter:
    """In-process GD-like platformer that quacks like ``GeodeSharedMemoryAdapter``.

    The simulation runs each time ``send_action`` is called — actions advance
    physics by one frame. ``read_obs`` returns the latest state.
    """

    def __init__(self, cfg: MockConfig | None = None):
        self.cfg = cfg or MockConfig()
        self._rng = random.Random(self.cfg.seed)
        self._state = _State()
        self._obs = np.zeros(OBS_DIM, dtype=np.float32)
        self._pending_reset = True
        self._level_start_x = 0.0
        self._last_tick = 0
        self._pending_action = 0
        self.shm_name = "mock"
        self._spawn_spikes()
        self._reset_state()

    # ------------------------------------------------------------------ API
    def verify_version(self) -> None:
        # Mock is always on the "expected" version.
        return

    def read_tick(self) -> int:
        return int(self._state.tick)

    def wait_next_tick(self, timeout_s: float = 0.2, poll_s: float = 0.001) -> bool:
        # Mock is synchronous — if send_action advanced the tick, caller
        # will see the new value immediately. Return True if the tick has
        # advanced since the last call.
        advanced = self._state.tick != self._last_tick
        self._last_tick = self._state.tick
        return advanced

    def read_obs(self) -> np.ndarray:
        return self._obs.copy()

    def read_next_obs(self, timeout_s: float = 0.2) -> np.ndarray:
        return self.read_obs()

    def send_action(self, action: int) -> None:
        self._pending_action = 1 if int(action) else 0
        self._step_physics(self._pending_action)
        self._pending_action = 0

    def send_reset(self) -> None:
        self._pending_reset = True
        self._reset_state()

    def read_level_complete_flag(self) -> bool:
        return bool(self._state.level_done)

    def read_player_input(self) -> bool:
        return False

    def read_obs_dim(self) -> int:
        return OBS_DIM

    def close(self) -> None:
        pass

    # -------------------------------------------------------------- internals
    def _spawn_spikes(self) -> None:
        """Generate a deterministic-ish sequence of spike obstacles."""
        spikes: List[_Spike] = []
        x = self.cfg.first_spike_x
        while x < self.cfg.level_length:
            spikes.append(_Spike(x=x))
            gap = self._rng.uniform(self.cfg.min_gap, self.cfg.max_gap)
            x += gap
        self._state.spikes = spikes

    def _reset_state(self) -> None:
        self._state.x = 0.0
        self._state.y = self.cfg.ground_y
        self._state.y_vel = 0.0
        self._state.x_vel = self.cfg.x_velocity
        self._state.on_ground = True
        self._state.dead = False
        self._state.level_done = False
        # tick monotonically increases so a waiter can detect the transition.
        self._state.tick += 1
        self._level_start_x = self._state.x
        self._pending_reset = False
        self._write_obs()

    def _step_physics(self, jump_action: int) -> None:
        if self._state.dead or self._state.level_done:
            # tick still advances so wait_next_tick returns True and
            # downstream code can observe the terminal frame.
            self._state.tick += 1
            self._write_obs()
            return

        # Horizontal: constant velocity forward.
        self._state.x += self._state.x_vel

        # Vertical: jump on press if on ground, apply gravity, clamp to ground.
        if jump_action and self._state.on_ground:
            self._state.y_vel = self.cfg.jump_velocity
            self._state.on_ground = False
        self._state.y_vel += self.cfg.gravity
        self._state.y += self._state.y_vel

        if self._state.y <= self.cfg.ground_y:
            self._state.y = self.cfg.ground_y
            self._state.y_vel = 0.0
            self._state.on_ground = True

        # Collision with spikes: player is "in" the spike's x span AND below
        # the spike's apex y (so a jump over the spike clears it).
        player_top = self._state.y + self.cfg.player_height
        for sp in self._state.spikes:
            dx = self._state.x - sp.x
            if -self.cfg.spike_half_width <= dx <= self.cfg.spike_half_width:
                if player_top < self.cfg.spike_height and self._state.on_ground:
                    self._state.dead = True
                    break

        # Level completion check.
        if self._state.x >= self.cfg.level_length:
            self._state.level_done = True

        self._state.tick += 1
        self._write_obs()

    def _write_obs(self) -> None:
        obs = self._obs
        obs.fill(0.0)
        obs[0] = self._state.x
        obs[1] = self._state.y
        obs[2] = self._state.y_vel
        obs[3] = self._state.x_vel
        obs[4] = 1.0 if self._state.on_ground else 0.0
        obs[5] = 1.0 if self._state.dead else 0.0
        obs[6] = 1.0
        obs[7] = 0.0  # cube mode

        # Nearest obstacles: spikes within [-100, +2000] window, sorted by
        # absolute distance. Matches the real mod's scan_nearby_objects.
        px = self._state.x
        py = self._state.y
        window_behind = -100.0
        window_ahead = 2000.0
        candidates: List[tuple[float, float, float]] = []
        for sp in self._state.spikes:
            rel_x = sp.x - px
            if rel_x < window_behind or rel_x > window_ahead:
                continue
            rel_y = 0.0 - py  # spikes sit on the ground
            candidates.append((abs(rel_x), rel_x, rel_y))
        candidates.sort(key=lambda c: c[0])

        count = min(len(candidates), MAX_NEARBY_OBJECTS)
        for i in range(count):
            _, rel_x, rel_y = candidates[i]
            base = OBJ_OBS_START + i * FLOATS_PER_OBJ
            obs[base + 0] = rel_x
            obs[base + 1] = rel_y
            obs[base + 2] = 1.0  # gameplay object type
            obs[base + 3] = 8.0  # spikeUp objID
            obs[base + 4] = 1.0  # scaleX
            obs[base + 5] = 1.0  # scaleY
