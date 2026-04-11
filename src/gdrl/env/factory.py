"""Factory helpers for constructing GD privileged envs.

Centralizes the "real IPC vs mock IPC" decision so the teacher trainer,
the rollout collector, and future eval/inference scripts all share the
same plumbing. Without this, every caller has to duplicate the adapter
construction + GDPrivilegedEnv wiring and it's easy to get out of sync.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

from gdrl.env.privileged_env import GDPrivilegedEnv, RewardConfig


@dataclass
class EnvBuildConfig:
    """Everything needed to build a GDPrivilegedEnv at training time."""

    mode: str = "real"               # "real" | "mock"
    shm_name: str = "gdrl_ipc"
    obs_dim: int = 608
    max_steps: int = 10_000
    action_repeat: int = 2
    stall_steps: int = 240
    tick_timeout_s: float = 0.2
    reset_wait_ticks: int = 120
    reward_config: RewardConfig = field(default_factory=RewardConfig)

    # Mock-only fields — ignored when mode="real".
    mock_seed: Optional[int] = None
    mock_level_length: float = 1500.0
    mock_x_velocity: float = 10.0
    mock_min_gap: float = 25.0
    mock_max_gap: float = 55.0

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def build_ipc_adapter(cfg: EnvBuildConfig):
    """Create either a real ``GeodeSharedMemoryAdapter`` or a ``MockIPCAdapter``.

    Importing ``geode_ipc`` is safe — it only touches shared memory when
    the adapter is actually constructed.
    """
    if cfg.mode == "mock":
        from gdrl.env.mock_ipc import MockConfig, MockIPCAdapter

        mock_cfg = MockConfig(
            level_length=cfg.mock_level_length,
            x_velocity=cfg.mock_x_velocity,
            min_gap=cfg.mock_min_gap,
            max_gap=cfg.mock_max_gap,
            seed=cfg.mock_seed,
        )
        return MockIPCAdapter(mock_cfg)

    if cfg.mode != "real":
        raise ValueError(f"Unknown env mode: {cfg.mode!r} (expected 'real' or 'mock')")

    from gdrl.env.geode_ipc import GeodeIPCConfig, GeodeSharedMemoryAdapter

    return GeodeSharedMemoryAdapter(
        GeodeIPCConfig(shm_name=cfg.shm_name, obs_dim=cfg.obs_dim)
    )


def build_env(cfg: EnvBuildConfig) -> GDPrivilegedEnv:
    ipc = build_ipc_adapter(cfg)
    return GDPrivilegedEnv(
        ipc=ipc,
        obs_dim=cfg.obs_dim,
        max_steps=cfg.max_steps,
        action_repeat=cfg.action_repeat,
        stall_steps=cfg.stall_steps,
        tick_timeout_s=cfg.tick_timeout_s,
        reset_wait_ticks=cfg.reset_wait_ticks,
        reward_config=cfg.reward_config,
    )
