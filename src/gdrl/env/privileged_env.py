from __future__ import annotations
import gymnasium as gym
import numpy as np

class IPCAdapter:
    """Interface for Geode shared-memory bridge."""
    def reset_level(self) -> None: ...
    def read_obs(self) -> np.ndarray: ...
    def send_action(self, action: int) -> None: ...



class GDPrivilegedEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, ipc: IPCAdapter, obs_dim: int = 608, max_steps: int = 10_000):
        # TODO: ipc must be provided — e.g. GeodeSharedMemoryAdapter
        self.ipc = ipc
        self.obs_dim = obs_dim
        self.max_steps = max_steps
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.prev_x = 0.0
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.ipc.send_reset()
        # wait for game to actually reset (tick changes + not dead)
        for _ in range(100):
            if hasattr(self.ipc, 'wait_next_tick'):
                self.ipc.wait_next_tick(timeout_s=0.5)
            obs = self.ipc.read_obs()
            if obs[5] < 0.5:  # not dead
                break
        self.prev_x = float(obs[0])
        self.steps = 0
        return obs, {}

    def step(self, action):
        self.ipc.send_action(int(action))
        # If adapter supports frame-sync reads, use them.
        if hasattr(self.ipc, 'read_next_obs'):
            obs = self.ipc.read_next_obs(timeout_s=0.2)
        else:
            obs = self.ipc.read_obs()
        x = float(obs[0])
        is_dead = bool(obs[5] > 0.5)
        level_done = hasattr(self.ipc, 'read_level_complete_flag') and self.ipc.read_level_complete_flag()

        progress = x - self.prev_x
        self.prev_x = x
        reward = progress * 0.1 - 0.01
        terminated = False

        if is_dead:
            reward -= 10.0
            terminated = True
        elif level_done:
            reward += 100.0
            terminated = True

        self.steps += 1
        truncated = self.steps >= self.max_steps
        return obs, reward, terminated, truncated, {}
