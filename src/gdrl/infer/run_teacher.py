"""Run a trained teacher PPO model on the privileged env.

Used for:
  * Manual verification that a checkpoint still plays the level.
  * Collecting quick eval stats without spinning up the full training script.
  * Driving a live Geometry Dash run (with ``--env-mode real``) so you can
    watch the agent play.

Usage:
  python -m gdrl.infer.run_teacher \\
      --model artifacts/smoke_teacher \\
      --env-mode mock \\
      --episodes 5

If the model was trained with VecNormalize, also pass
``--vecnorm-path artifacts/smoke_teacher.vecnorm.pkl``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from statistics import mean

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from gdrl.env.factory import EnvBuildConfig, build_env
from gdrl.env.privileged_env import RewardConfig


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run a trained teacher PPO model.")
    ap.add_argument("--model", type=str, required=True, help="Path to a saved PPO model (without .zip).")
    ap.add_argument("--vecnorm-path", type=str, default=None, help="Optional VecNormalize stats pickle.")
    ap.add_argument("--env-mode", choices=("real", "mock"), default="mock")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=10_000)
    ap.add_argument("--action-repeat", type=int, default=2)
    ap.add_argument("--stall-steps", type=int, default=240)
    ap.add_argument("--deterministic", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shm-name", type=str, default="gdrl_ipc")
    ap.add_argument("--mock-seed", type=int, default=None)
    ap.add_argument("--mock-level-length", type=float, default=1_500.0)
    ap.add_argument("--print-every", type=int, default=1)
    ap.add_argument("--render-delay", type=float, default=0.0, help="Sleep between env steps (seconds).")
    return ap.parse_args()


def build_inference_env(args: argparse.Namespace, vecnorm_path: Path | None):
    cfg = EnvBuildConfig(
        mode=args.env_mode,
        shm_name=args.shm_name,
        max_steps=args.max_steps,
        action_repeat=args.action_repeat,
        stall_steps=args.stall_steps,
        reward_config=RewardConfig(),
        mock_seed=args.mock_seed if args.mock_seed is not None else args.seed,
        mock_level_length=args.mock_level_length,
    )

    def _thunk():
        return Monitor(build_env(cfg))

    vec_env = DummyVecEnv([_thunk])
    if vecnorm_path is not None:
        vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    return vec_env


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    if model_path.suffix == ".zip":
        model_path = model_path.with_suffix("")

    vecnorm_path: Path | None = None
    if args.vecnorm_path:
        vecnorm_path = Path(args.vecnorm_path)
        if not vecnorm_path.exists():
            print(f"[run_teacher] --vecnorm-path {vecnorm_path} does not exist")
            return 2

    env = build_inference_env(args, vecnorm_path)
    model = PPO.load(str(model_path), device="cpu")

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    episode_max_x: list[float] = []
    completions: int = 0
    deaths: int = 0

    try:
        for ep in range(args.episodes):
            obs = env.reset()
            done = np.array([False])
            total_reward = 0.0
            length = 0
            while not done[0]:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, done, info = env.step(action)
                total_reward += float(reward[0])
                length += 1
                if args.render_delay > 0:
                    time.sleep(args.render_delay)
            # `info` is a list from VecEnv; Monitor-wrapped episode stats
            # live under info[0]["episode"].
            inner = info[0] if isinstance(info, (list, tuple)) else info
            ep_stats = inner.get("episode", {}) if isinstance(inner, dict) else {}
            best_x = float(inner.get("best_x", 0.0)) if isinstance(inner, dict) else 0.0
            dead = bool(inner.get("dead", False)) if isinstance(inner, dict) else False
            completed = bool(inner.get("level_complete", False)) if isinstance(inner, dict) else False
            episode_rewards.append(ep_stats.get("r", total_reward))
            episode_lengths.append(int(ep_stats.get("l", length)))
            episode_max_x.append(best_x)
            if completed:
                completions += 1
            if dead:
                deaths += 1
            if args.print_every > 0 and (ep + 1) % args.print_every == 0:
                print(
                    f"ep={ep + 1:3d}/{args.episodes} "
                    f"reward={episode_rewards[-1]:+8.2f} "
                    f"length={episode_lengths[-1]:5d} "
                    f"max_x={best_x:8.1f} "
                    f"{'DEAD' if dead else 'alive'} "
                    f"{'COMPLETED' if completed else ''}",
                    flush=True,
                )
    finally:
        env.close()

    if episode_rewards:
        print()
        print("=" * 60)
        print(f"episodes       : {len(episode_rewards)}")
        print(f"reward mean    : {mean(episode_rewards):+.2f}")
        print(f"reward min/max : {min(episode_rewards):+.2f} / {max(episode_rewards):+.2f}")
        print(f"length mean    : {mean(episode_lengths):.1f}")
        print(f"max_x mean     : {mean(episode_max_x):.1f}")
        print(f"completions    : {completions}")
        print(f"deaths         : {deaths}")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
