from __future__ import annotations

import unittest

import numpy as np

from gdrl.env.privileged_env import GDPrivilegedEnv, RewardConfig


def _obs(x: float, *, dead: bool = False, mode: int = 0, speed: float = 1.0) -> np.ndarray:
    obs = np.zeros(608, dtype=np.float32)
    obs[0] = x
    obs[5] = 1.0 if dead else 0.0
    obs[6] = speed
    obs[7] = float(mode)
    return obs


class FakeIPC:
    def __init__(self, observations: list[np.ndarray], level_complete_flags: list[bool] | None = None):
        self.observations = [np.asarray(obs, dtype=np.float32) for obs in observations]
        self.level_complete_flags = list(level_complete_flags or [False] * len(observations))
        self.actions: list[int] = []
        self.reset_calls = 0
        self.player_input = False
        self._last_obs = self.observations[0]
        self._last_level_complete = False

    def send_action(self, action: int) -> None:
        self.actions.append(int(action))

    def send_reset(self) -> None:
        self.reset_calls += 1

    def read_next_obs(self, timeout_s: float = 0.2) -> np.ndarray:
        if self.observations:
            self._last_obs = self.observations.pop(0)
            self._last_level_complete = self.level_complete_flags.pop(0)
        return self._last_obs

    def read_obs(self) -> np.ndarray:
        return self._last_obs

    def read_level_complete_flag(self) -> bool:
        return self._last_level_complete

    def read_player_input(self) -> bool:
        return self.player_input

    def close(self) -> None:
        return None


class GDPrivilegedEnvTests(unittest.TestCase):
    def test_reset_skips_dead_frames_and_step_aggregates_progress(self):
        ipc = FakeIPC(
            [
                _obs(0.0, dead=True),
                _obs(1.0, dead=False),
                _obs(4.0),
                _obs(8.0),
            ]
        )
        env = GDPrivilegedEnv(
            ipc=ipc,
            action_repeat=2,
            stall_steps=0,
            reward_config=RewardConfig(
                progress_scale=1.0,
                progress_clip=100.0,
                alive_bonus=0.0,
                jump_penalty=0.0,
                stall_penalty=0.0,
            ),
        )

        obs, info = env.reset()
        self.assertEqual(ipc.reset_calls, 1)
        self.assertEqual(ipc.actions, [])
        self.assertFalse(info["dead"])
        self.assertAlmostEqual(float(obs[0]), 1.0)

        obs, reward, terminated, truncated, info = env.step(1)
        self.assertEqual(ipc.actions, [1, 1])
        self.assertAlmostEqual(reward, 7.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["frames"], 2)
        self.assertAlmostEqual(float(obs[0]), 8.0)

    def test_stall_truncation_applies_penalty(self):
        ipc = FakeIPC([_obs(2.0), _obs(2.0), _obs(2.0)])
        env = GDPrivilegedEnv(
            ipc=ipc,
            action_repeat=1,
            stall_steps=2,
            reward_config=RewardConfig(
                progress_scale=1.0,
                progress_clip=100.0,
                alive_bonus=0.0,
                jump_penalty=0.0,
                stall_penalty=1.5,
            ),
        )

        env.reset()
        _, reward1, terminated1, truncated1, _ = env.step(0)
        self.assertEqual(reward1, 0.0)
        self.assertFalse(terminated1)
        self.assertFalse(truncated1)

        _, reward2, terminated2, truncated2, info2 = env.step(0)
        self.assertEqual(reward2, -1.5)
        self.assertFalse(terminated2)
        self.assertTrue(truncated2)
        self.assertTrue(info2["stall_truncated"])


if __name__ == "__main__":
    unittest.main()
