from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from gdrl.env.factory import EnvBuildConfig, build_env
from gdrl.env.privileged_env import RewardConfig


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train a privileged PPO teacher for Geometry Dash.")
    ap.add_argument("--total-timesteps", type=int, default=1_000_000)
    ap.add_argument("--out", type=str, default="artifacts/teacher_ppo")
    ap.add_argument("--checkpoint-dir", type=str, default="artifacts/checkpoints")
    ap.add_argument("--checkpoint-freq", type=int, default=25_000)
    ap.add_argument(
        "--tensorboard-dir",
        type=str,
        default="",
        help="Optional tensorboard log directory. Leave empty to disable "
        "(requires `pip install tensorboard`).",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--progress-bar", action="store_true", default=False)

    ap.add_argument(
        "--env-mode",
        choices=("real", "mock"),
        default="real",
        help="'real' = live Geometry Dash via Geode shared memory; "
        "'mock' = pure-Python platformer simulator for testing/iteration.",
    )
    ap.add_argument("--n-envs", type=int, default=1)
    ap.add_argument(
        "--vec-env",
        choices=("dummy", "subproc"),
        default="dummy",
        help="DummyVecEnv runs envs in-process; SubprocVecEnv uses worker processes "
        "(only safe with mock env - the real env needs exclusive shared memory).",
    )

    ap.add_argument("--norm-obs", action="store_true", default=False)
    ap.add_argument("--norm-reward", action="store_true", default=False)
    ap.add_argument("--clip-obs", type=float, default=10.0)
    ap.add_argument("--clip-reward", type=float, default=10.0)

    ap.add_argument("--eval-freq", type=int, default=0)
    ap.add_argument("--eval-episodes", type=int, default=5)

    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--n-steps", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-epochs", type=int, default=10)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-range", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)

    ap.add_argument("--max-steps", type=int, default=10_000)
    ap.add_argument("--action-repeat", type=int, default=2)
    ap.add_argument("--stall-steps", type=int, default=240)
    ap.add_argument("--tick-timeout", type=float, default=0.2)
    ap.add_argument("--reset-wait-ticks", type=int, default=120)
    ap.add_argument("--shm-name", type=str, default="gdrl_ipc")

    ap.add_argument("--progress-scale", type=float, default=0.10)
    ap.add_argument("--progress-clip", type=float, default=30.0)
    ap.add_argument("--alive-bonus", type=float, default=0.02)
    ap.add_argument("--jump-penalty", type=float, default=0.001)
    ap.add_argument("--death-penalty", type=float, default=10.0)
    ap.add_argument("--completion-bonus", type=float, default=100.0)
    ap.add_argument("--stall-penalty", type=float, default=0.5)

    ap.add_argument("--mock-level-length", type=float, default=1_500.0)
    ap.add_argument("--mock-x-velocity", type=float, default=10.0)
    ap.add_argument("--mock-min-gap", type=float, default=25.0)
    ap.add_argument("--mock-max-gap", type=float, default=55.0)

    return ap.parse_args()


def make_reward_config(args: argparse.Namespace) -> RewardConfig:
    return RewardConfig(
        progress_scale=args.progress_scale,
        progress_clip=args.progress_clip,
        alive_bonus=args.alive_bonus,
        jump_penalty=args.jump_penalty,
        death_penalty=args.death_penalty,
        completion_bonus=args.completion_bonus,
        stall_penalty=args.stall_penalty,
    )


def make_env_build_config(args: argparse.Namespace, seed_offset: int) -> EnvBuildConfig:
    return EnvBuildConfig(
        mode=args.env_mode,
        shm_name=args.shm_name,
        max_steps=args.max_steps,
        action_repeat=args.action_repeat,
        stall_steps=args.stall_steps,
        tick_timeout_s=args.tick_timeout,
        reset_wait_ticks=args.reset_wait_ticks,
        reward_config=make_reward_config(args),
        mock_seed=args.seed + seed_offset,
        mock_level_length=args.mock_level_length,
        mock_x_velocity=args.mock_x_velocity,
        mock_min_gap=args.mock_min_gap,
        mock_max_gap=args.mock_max_gap,
    )


def make_env_thunk(args: argparse.Namespace, seed_offset: int):
    def _factory():
        cfg = make_env_build_config(args, seed_offset)
        env = build_env(cfg)
        return Monitor(
            env,
            info_keywords=("x", "best_x", "progress", "mode", "speed", "stall_count"),
        )

    return _factory


def build_vec_env(args: argparse.Namespace, training: bool, n_envs: int | None = None):
    n = n_envs if n_envs is not None else args.n_envs
    thunks = [make_env_thunk(args, seed_offset=i) for i in range(n)]
    if args.vec_env == "subproc" and n > 1:
        vec_env = SubprocVecEnv(thunks)
    else:
        vec_env = DummyVecEnv(thunks)

    if args.norm_obs or args.norm_reward:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=args.norm_obs,
            norm_reward=args.norm_reward and training,
            clip_obs=args.clip_obs,
            clip_reward=args.clip_reward,
            training=training,
        )
    return vec_env


def write_run_config(args: argparse.Namespace, out_path: Path) -> None:
    payload = vars(args).copy()
    payload["reward_config"] = vars(make_reward_config(args))
    config_path = out_path.with_suffix(".json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = build_vec_env(args, training=True)

    policy_kwargs = {
        "activation_fn": nn.ReLU,
        "net_arch": {"pi": [256, 256, 128], "vf": [256, 256, 128]},
    }

    tensorboard_log = args.tensorboard_dir or None
    if tensorboard_log is not None:
        try:
            import tensorboard  # noqa: F401
        except ImportError:
            print(
                "[train_ppo] tensorboard not installed — skipping TB logging. "
                "Run `pip install tensorboard` to enable.",
            )
            tensorboard_log = None
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        seed=args.seed,
        verbose=1,
        device=args.device,
    )

    callbacks: list = []
    if args.checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(args.checkpoint_freq // max(args.n_envs, 1), 1),
                save_path=str(checkpoint_dir),
                name_prefix=out_path.stem,
                save_vecnormalize=isinstance(env, VecNormalize),
            )
        )

    eval_env = None
    if args.eval_freq > 0:
        if args.env_mode != "mock":
            print(
                "[train_ppo] --eval-freq > 0 ignored: real env can't be duplicated safely. "
                "Use --env-mode mock to enable eval.",
            )
        else:
            eval_env = build_vec_env(args, training=False, n_envs=1)
            if isinstance(env, VecNormalize) and isinstance(eval_env, VecNormalize):
                eval_env.obs_rms = env.obs_rms
                eval_env.ret_rms = env.ret_rms
            callbacks.append(
                EvalCallback(
                    eval_env,
                    best_model_save_path=str(checkpoint_dir / "best"),
                    log_path=str(checkpoint_dir / "eval"),
                    eval_freq=max(args.eval_freq // max(args.n_envs, 1), 1),
                    n_eval_episodes=args.eval_episodes,
                    deterministic=True,
                )
            )

    callback = CallbackList(callbacks) if callbacks else None

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            progress_bar=args.progress_bar,
        )
    finally:
        model.save(str(out_path))
        if isinstance(env, VecNormalize):
            env.save(str(out_path.with_suffix(".vecnorm.pkl")))
        write_run_config(args, out_path)
        env.close()
        if eval_env is not None:
            eval_env.close()

    print(f"[train_ppo] saved model -> {out_path}")


if __name__ == "__main__":
    main()
