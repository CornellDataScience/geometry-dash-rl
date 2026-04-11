from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import gymnasium as gym
import numpy as np

from gdrl.env.geode_ipc import GeodeIPCConfig, GeodeSharedMemoryAdapter

X_IDX = 0
Y_IDX = 1
VY_IDX = 2
VX_IDX = 3
ON_GROUND_IDX = 4
DEAD_IDX = 5
SPEED_IDX = 6
MODE_IDX = 7


class IPCAdapter(Protocol):
    """Interface for the Geode shared-memory bridge."""

    def read_obs(self) -> np.ndarray: ...
    def read_next_obs(self, timeout_s: float = 0.2) -> np.ndarray: ...
    def send_action(self, action: int) -> None: ...
    def send_reset(self) -> None: ...
    def read_level_complete_flag(self) -> bool: ...
    def read_player_input(self) -> bool: ...
    def close(self) -> None: ...


@dataclass
class RewardConfig:
    progress_scale: float = 0.10
    progress_clip: float = 30.0
    alive_bonus: float = 0.02
    jump_penalty: float = 0.001
    death_penalty: float = 10.0
    completion_bonus: float = 100.0
    stall_penalty: float = 0.5
    stall_epsilon: float = 1e-3


class GDPrivilegedEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        ipc: IPCAdapter | None = None,
        *,
        obs_dim: int = 608,
        max_steps: int = 10_000,
        action_repeat: int = 2,
        stall_steps: int = 240,
        tick_timeout_s: float = 0.2,
        reset_wait_ticks: int = 120,
        reward_config: RewardConfig | None = None,
    ):
        self.obs_dim = obs_dim
        self.max_steps = max_steps
        self.action_repeat = max(1, int(action_repeat))
        self.stall_steps = max(0, int(stall_steps))
        self.tick_timeout_s = float(tick_timeout_s)
        self.reset_wait_ticks = max(1, int(reset_wait_ticks))
        self.reward_config = reward_config or RewardConfig()
        self.ipc = ipc or GeodeSharedMemoryAdapter(GeodeIPCConfig(obs_dim=obs_dim))

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.prev_x = 0.0
        self.best_x = 0.0
        self.steps = 0
        self.stall_count = 0

    def _coerce_obs(self, obs: np.ndarray) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        if arr.size < self.obs_dim:
            padded = np.zeros(self.obs_dim, dtype=np.float32)
            padded[:arr.size] = arr
            arr = padded
        elif arr.size > self.obs_dim:
            arr = arr[:self.obs_dim]
        return np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)

    def _read_step_obs(self) -> np.ndarray:
        obs = self.ipc.read_next_obs(timeout_s=self.tick_timeout_s)
        return self._coerce_obs(obs)

    def _read_level_complete(self) -> bool:
        return bool(self.ipc.read_level_complete_flag())

    def _read_player_input(self) -> bool:
        return bool(self.ipc.read_player_input())

    def _reset_counters(self, obs: np.ndarray) -> None:
        start_x = float(obs[X_IDX])
        self.prev_x = start_x
        self.best_x = start_x
        self.steps = 0
        self.stall_count = 0

    def _step_reward(self, obs: np.ndarray, action: int) -> tuple[float, bool, dict]:
        x = float(obs[X_IDX])
        progress = x - self.prev_x
        self.prev_x = x
        self.best_x = max(self.best_x, x)
        if progress > self.reward_config.stall_epsilon:
            self.stall_count = 0
        else:
            self.stall_count += 1

        clipped_progress = float(
            np.clip(progress, -self.reward_config.progress_clip, self.reward_config.progress_clip)
        )
        reward = (
            clipped_progress * self.reward_config.progress_scale
            + self.reward_config.alive_bonus
            - self.reward_config.jump_penalty * int(action)
        )

        is_dead = bool(obs[DEAD_IDX] > 0.5)
        level_done = self._read_level_complete()
        terminated = False
        if is_dead:
            reward -= self.reward_config.death_penalty
            terminated = True
        elif level_done:
            reward += self.reward_config.completion_bonus
            terminated = True

        info = {
            "x": x,
            "y": float(obs[Y_IDX]),
            "progress": progress,
            "best_x": self.best_x,
            "mode": int(obs[MODE_IDX]),
            "speed": float(obs[SPEED_IDX]),
            "dead": is_dead,
            "level_complete": level_done,
            "player_input": self._read_player_input(),
            "stall_count": self.stall_count,
        }
        return reward, terminated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.ipc.send_reset()

        obs = None
        for _ in range(self.reset_wait_ticks):
            obs = self._read_step_obs()
            if obs[DEAD_IDX] < 0.5:
                break
        if obs is None:
            raise RuntimeError("Unable to read observation during reset.")

        self._reset_counters(obs)
        info = {
            "x": float(obs[X_IDX]),
            "dead": bool(obs[DEAD_IDX] > 0.5),
            "mode": int(obs[MODE_IDX]),
            "speed": float(obs[SPEED_IDX]),
            "stall_count": self.stall_count,
        }
        return obs, info

    def step(self, action):
        action = int(action)
        total_reward = 0.0
        terminated = False
        truncated = False
        frames = 0
        info: dict = {}
        obs = None

        for _ in range(self.action_repeat):
            self.ipc.send_action(action)
            obs = self._read_step_obs()
            reward, terminated, info = self._step_reward(obs, action)
            total_reward += reward
            self.steps += 1
            frames += 1

            if terminated:
                break
            if self.steps >= self.max_steps:
                truncated = True
                break
            if self.stall_steps > 0 and self.stall_count >= self.stall_steps:
                total_reward -= self.reward_config.stall_penalty
                info["stall_truncated"] = True
                truncated = True
                break

        assert obs is not None
        info["frames"] = frames
        return obs, float(total_reward), terminated, truncated, info

    def close(self) -> None:
        self.ipc.close()
