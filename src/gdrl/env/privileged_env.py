from __future__ import annotations
import gymnasium as gym
import numpy as np

class IPCAdapter:
    """Interface for Geode shared-memory bridge."""
    def reset_level(self) -> None: ...
    def read_obs(self) -> np.ndarray: ...
    def send_action(self, action: int) -> None: ...


class MockIPCAdapter(IPCAdapter):
    def __init__(self, obs_dim: int = 108):
        self.obs_dim = obs_dim
        self.t = 0
        self.x = 0.0
        self.dead = False

    def reset_level(self) -> None:
        self.t = 0
        self.x = 0.0
        self.dead = False

    def read_obs(self) -> np.ndarray:
        obs = np.zeros((self.obs_dim,), dtype=np.float32)
        obs[0] = self.x
        obs[1] = np.sin(self.t / 10) * 2.0
        obs[2] = np.cos(self.t / 15)
        obs[5] = 1.0 if self.dead else 0.0
        obs[7] = (self.t // 250) % 4
        return obs

    def send_action(self, action: int) -> None:
        self.t += 1
        self.x += 1.0 + (0.2 if action == 1 else 0.0)
        if self.t % 400 == 0 and action == 0:
            self.dead = True


class GDPrivilegedEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, ipc: IPCAdapter | None = None, obs_dim: int = 108, max_steps: int = 10_000):
        self.ipc = ipc or MockIPCAdapter(obs_dim=obs_dim)
        self.obs_dim = obs_dim
        self.max_steps = max_steps
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.prev_x = 0.0
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.ipc.reset_level()
        obs = self.ipc.read_obs()
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

        progress = x - self.prev_x
        self.prev_x = x
        reward = progress * 0.1 - 0.01
        terminated = False

        if is_dead:
            reward -= 10.0
            terminated = True

        self.steps += 1
        truncated = self.steps >= self.max_steps
        return obs, reward, terminated, truncated, {}
