"""Offline tests that exercise GDPrivilegedEnv + MockIPCAdapter without a
live Geometry Dash instance.

These are runnable with `pytest src/gdrl/env/test_mock_env.py` or with a
plain `python -m gdrl.env.test_mock_env` (the module has an __main__
entry point so it also acts as a smoke script).
"""

from __future__ import annotations

import numpy as np

from gdrl.env.factory import EnvBuildConfig, build_env
from gdrl.env.mock_ipc import MockConfig, MockIPCAdapter
from gdrl.env.privileged_env import GDPrivilegedEnv, RewardConfig


def _make_env(seed: int = 0, max_steps: int = 500) -> GDPrivilegedEnv:
    cfg = EnvBuildConfig(
        mode="mock",
        max_steps=max_steps,
        action_repeat=1,
        stall_steps=0,
        reset_wait_ticks=5,
        mock_seed=seed,
    )
    return build_env(cfg)


def test_mock_adapter_runs_standalone():
    """MockIPCAdapter should advance ticks, track x, and respect resets."""
    ipc = MockIPCAdapter(MockConfig(seed=1, level_length=200.0))
    ipc.verify_version()

    first = ipc.read_obs()
    assert first.shape == (608,)
    assert first[5] < 0.5, "Mock should not spawn dead"

    # Forward motion when we do nothing.
    for _ in range(3):
        ipc.send_action(0)
    after = ipc.read_obs()
    assert after[0] > first[0], f"x should advance: before={first[0]} after={after[0]}"

    # Reset sends us back near the start line.
    ipc.send_reset()
    reset_obs = ipc.read_obs()
    assert reset_obs[0] < after[0], "send_reset should rewind x"
    assert reset_obs[5] < 0.5, "reset should clear the dead flag"


def test_privileged_env_reset_and_step_shapes():
    env = _make_env(seed=2)
    obs, info = env.reset()
    assert obs.shape == (608,)
    assert obs[5] < 0.5
    assert "x" in info

    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == (608,)
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        if terminated or truncated:
            break
    env.close()


def test_privileged_env_reports_episode_on_death():
    # Never jumping guarantees the agent eventually runs into a spike.
    env = _make_env(seed=3, max_steps=10_000)
    env.reset()
    terminated = False
    for _ in range(2_000):
        _, _, terminated, truncated, info = env.step(0)
        if terminated or truncated:
            break
    assert terminated, "No-jump policy should eventually die on a spike"
    env.close()


def test_privileged_env_truncates_on_max_steps():
    cfg = EnvBuildConfig(
        mode="mock",
        max_steps=5,
        action_repeat=1,
        stall_steps=0,
        reset_wait_ticks=5,
        mock_seed=4,
        mock_level_length=10_000.0,  # too far to hit a spike in 5 steps
        mock_min_gap=9_000.0,
        mock_max_gap=9_500.0,
    )
    env = build_env(cfg)
    env.reset()
    for _ in range(10):
        _, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            break
    assert truncated and not terminated
    env.close()


def test_ppo_smoke_training_runs():
    """End-to-end: PPO trains on the mock env for a few hundred steps
    without crashing. Catches obvious regressions in the training wiring."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    def _thunk():
        return Monitor(_make_env(seed=0, max_steps=200))

    vec = DummyVecEnv([_thunk])
    vec = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = PPO(
        "MlpPolicy",
        vec,
        n_steps=64,
        batch_size=32,
        n_epochs=2,
        learning_rate=1e-3,
        gamma=0.99,
        verbose=0,
        device="cpu",
    )
    model.learn(total_timesteps=256, progress_bar=False)
    vec.close()


if __name__ == "__main__":
    test_mock_adapter_runs_standalone()
    print("ok: test_mock_adapter_runs_standalone")
    test_privileged_env_reset_and_step_shapes()
    print("ok: test_privileged_env_reset_and_step_shapes")
    test_privileged_env_reports_episode_on_death()
    print("ok: test_privileged_env_reports_episode_on_death")
    test_privileged_env_truncates_on_max_steps()
    print("ok: test_privileged_env_truncates_on_max_steps")
    test_ppo_smoke_training_runs()
    print("ok: test_ppo_smoke_training_runs")
    print("all tests passed")
