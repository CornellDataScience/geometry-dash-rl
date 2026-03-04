"""
Gymnasium environment wrapper for Geometry Dash RL.

Communicates with the Geode mod via shared memory for lockstep
frame-by-frame control of the game.

Observation: float32 vector of 161 elements
  - 11 player features
  - 150 object features (25 slots × 6 floats)

Action: Discrete(2) — 0=no jump, 1=jump
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from shm_reader import open_shm, read_state, write_action, wait_for_game


class GDEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(161,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)
        self.buf = open_shm()

    def _get_obs(self):
        state = read_state(self.buf)
        player = [
            state['playerX'],
            state['playerY'],
            state['velocityY'],
            state['rotation'],
            float(state['gameMode']),
            float(state['isOnGround']),
            float(state['isDead']),
            state['speedMultiplier'],
            float(state['gravityFlipped']),
            state['levelPercent'],
            float(state['numObjectsFound']),
        ]
        objects_flat = []
        for obj in state['objects']:
            objects_flat.extend(obj)
        return np.array(player + objects_flat, dtype=np.float32), state

    def step(self, action):
        write_action(self.buf, command=0, action=int(action))
        wait_for_game(self.buf)
        obs, state = self._get_obs()
        reward = state['reward']
        terminated = bool(state['done'])
        truncated = False
        info = {
            'attempt': state['attemptNumber'],
            'percent': state['levelPercent'],
        }
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        write_action(self.buf, command=1, action=0)
        wait_for_game(self.buf)
        obs, state = self._get_obs()
        info = {
            'attempt': state['attemptNumber'],
            'percent': state['levelPercent'],
        }
        return obs, info

    def close(self):
        write_action(self.buf, command=2, action=0)
