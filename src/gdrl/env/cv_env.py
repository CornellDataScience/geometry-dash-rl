"""CV-based gym env over the v4 SHM mod.

Observation: dict
    frame:   uint8 (STACK, 1, FRAME_H, FRAME_W) — most-recent-last
    mode_id: int   ∈ {0=cube, 1=ship, 2=ball, 3=ufo}

Action: Discrete(2) — 0=idle, 1=jump

Reward: max-x progress + cv-version constants. No reward clipping.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from gdrl.env.geode_ipc_v3 import (
    GeodeV4Adapter, FRAME_W, FRAME_H,
)


@dataclass
class CvEnvConfig:
    stack: int = 4
    max_steps: int = 5000
    # reward shaping (cv-version-style; no clipping; max-x progress)
    default_reward: float = 0.01
    jump_punishment: float = -0.2
    death_punishment: float = -10.0
    beating_level: float = 100.0
    progress_scale: float = 0.001     # reward per world-unit of new max-x
    # mode-flicker debounce: ignore single-frame mode changes
    mode_debounce: int = 2


class GDCvEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, ipc: GeodeV4Adapter | None = None, cfg: CvEnvConfig | None = None):
        self.cfg = cfg or CvEnvConfig()
        self.ipc = ipc if ipc is not None else GeodeV4Adapter()
        self.ipc.verify_version()

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Dict({
            "frame": gym.spaces.Box(0, 255, shape=(self.cfg.stack, 1, FRAME_H, FRAME_W), dtype=np.uint8),
            "mode_id": gym.spaces.Discrete(4),
        })

        self._frames: deque[np.ndarray] = deque(maxlen=self.cfg.stack)
        self._mode_history: deque[int] = deque(maxlen=self.cfg.mode_debounce)
        self._x_max = 0.0
        self._steps = 0
        self._level_id: int | None = None

    # --- helpers ---

    def _stacked_frames(self) -> np.ndarray:
        # pad with the oldest frame if we have fewer than stack frames
        first = self._frames[0]
        out = np.zeros((self.cfg.stack, 1, FRAME_H, FRAME_W), dtype=np.uint8)
        n = len(self._frames)
        for i in range(self.cfg.stack):
            src_idx = max(0, i - (self.cfg.stack - n))
            out[i, 0] = self._frames[src_idx] if src_idx < n else first
        return out

    def _smoothed_mode_id(self, raw: int) -> int:
        self._mode_history.append(int(raw))
        if len(self._mode_history) < self._mode_history.maxlen:
            return int(raw)
        # majority vote across the small history (2 frames by default)
        counts = np.bincount(np.fromiter(self._mode_history, dtype=np.int64), minlength=4)
        return int(counts.argmax())

    def _read_frame_with_retries(self, timeout_s: float = 0.5) -> np.ndarray:
        # if no frame published yet, spin until one is.
        for _ in range(int(timeout_s / 0.01)):
            try:
                return self.ipc.read_frame()
            except RuntimeError:
                self.ipc.wait_next_tick(timeout_s=0.05)
        raise RuntimeError("timeout waiting for first frame from mod")

    def _wait_alive(self, max_ticks: int = 200) -> np.ndarray:
        for _ in range(max_ticks):
            self.ipc.wait_next_tick(timeout_s=0.5)
            obs = self.ipc.read_obs()
            if obs[5] < 0.5:
                return obs
        raise RuntimeError("env reset: player never came alive")

    def _burn_in(self, n_ticks: int = 2) -> np.ndarray:
        # send no-op actions to clear stale mod globals (g_prevX, button state)
        last = None
        for _ in range(n_ticks):
            self.ipc.send_action(0)
            self.ipc.wait_next_tick(timeout_s=0.5)
            last = self.ipc.read_obs()
        return last if last is not None else self.ipc.read_obs()

    def _init_frame_stack(self) -> None:
        self._frames.clear()
        for _ in range(self.cfg.stack):
            self.ipc.wait_next_frame_seq(timeout_s=1.0)
            self._frames.append(self._read_frame_with_retries())

    # --- gym API ---

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        target_pct = int(options.get("percent", 0))
        new_level_id = options.get("level_id", None)

        if new_level_id is not None and new_level_id != self._level_id:
            current_ep = self.ipc.read_episode_id()
            self.ipc.send_load_level(int(new_level_id))
            # wait for episode_id to bump (signals new PlayLayer::init ran)
            import time as _t
            t0 = _t.time()
            while _t.time() - t0 < 10.0:
                if self.ipc.read_episode_id() != current_ep:
                    break
                _t.sleep(0.05)
            else:
                raise RuntimeError(f"timeout loading level_id={new_level_id}")
            self._level_id = int(new_level_id)

        self.ipc.send_reset(percent=target_pct)
        obs = self._wait_alive()
        obs = self._burn_in(2)
        self._init_frame_stack()

        self._x_max = float(obs[0])
        self._mode_history.clear()
        self._steps = 0

        mode_id = self._smoothed_mode_id(int(obs[7]))
        return {
            "frame": self._stacked_frames(),
            "mode_id": mode_id,
        }, self._info(obs, mode_id)

    def step(self, action):
        self.ipc.send_action(int(action))
        self.ipc.wait_next_tick(timeout_s=0.2)
        obs = self.ipc.read_obs()
        # capture latest pixel frame (publishes every 4th tick in mod)
        self.ipc.wait_next_frame_seq(timeout_s=0.05)
        try:
            frame = self.ipc.read_frame()
        except RuntimeError:
            frame = self._frames[-1]  # reuse last if mod hasn't published yet
        self._frames.append(frame)

        x_now = float(obs[0])
        is_dead = bool(obs[5] > 0.5)
        level_done = self.ipc.read_level_complete_flag()
        mode_id = self._smoothed_mode_id(int(obs[7]))

        # max-x progress shaping (level-agnostic, no double-counting on respawn)
        progress = max(0.0, x_now - self._x_max)
        if x_now > self._x_max:
            self._x_max = x_now

        reward = self.cfg.default_reward + progress * self.cfg.progress_scale
        if int(action) == 1:
            reward += self.cfg.jump_punishment

        terminated = False
        if is_dead:
            reward = self.cfg.death_punishment
            terminated = True
        elif level_done:
            reward += self.cfg.beating_level
            terminated = True

        self._steps += 1
        truncated = self._steps >= self.cfg.max_steps

        observation = {
            "frame": self._stacked_frames(),
            "mode_id": mode_id,
        }
        return observation, float(reward), terminated, truncated, self._info(obs, mode_id)

    def _info(self, obs: np.ndarray, mode_id: int) -> dict:
        level_length = self.ipc.read_level_length()
        x_now = float(obs[0])
        percent = (x_now / level_length * 100.0) if level_length > 0 else 0.0
        return {
            "x": x_now,
            "y": float(obs[1]),
            "vy": float(obs[2]),
            "dx": float(obs[3]),
            "on_ground": bool(obs[4] > 0.5),
            "is_dead": bool(obs[5] > 0.5),
            "raw_mode_id": int(obs[7]),
            "mode_id": int(mode_id),
            "percent": percent,
            "x_max": self._x_max,
            "level_id": int(self.ipc.read_current_level_id()),
            "raw_obs": obs,
        }

    def close(self):
        try:
            self.ipc.close()
        except Exception:
            pass
