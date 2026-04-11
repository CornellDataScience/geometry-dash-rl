from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from gdrl.data.collector import write_shard
from gdrl.data.render import render_frame
from gdrl.env.factory import EnvBuildConfig, build_env


def capture_frame(obs: np.ndarray, frame_idx: int, use_synthetic: bool) -> np.ndarray:
    """Return an 84x84 grayscale frame.

    When ``use_synthetic`` is set, rasterizes the privileged obs into a
    radar-style image so the student can train non-trivially without a
    live Geometry Dash screen. Otherwise returns a zero placeholder that
    will eventually be replaced by a real screen grab via `mss`.
    """
    if use_synthetic:
        return render_frame(obs)
    # TODO: replace with real screen capture of the GD window (84x84 grayscale)
    return np.zeros((84, 84), dtype=np.uint8)


def extract_teacher_probs(model: PPO, obs: np.ndarray) -> np.ndarray:
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.squeeze(0).detach().cpu().numpy()
    return np.asarray(probs, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Collect teacher rollouts into NPZ shards.")
    ap.add_argument("--teacher-path", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--out", type=str, default="artifacts/datasets/shard_000.npz")
    ap.add_argument("--level-id", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true", default=False)
    ap.add_argument("--max-steps", type=int, default=10_000)
    ap.add_argument("--action-repeat", type=int, default=2)
    ap.add_argument("--stall-steps", type=int, default=240)
    ap.add_argument("--mock", action="store_true", default=False)
    ap.add_argument("--mock-level-length", type=float, default=1_500.0)
    ap.add_argument("--mock-seed", type=int, default=0)
    ap.add_argument(
        "--synthetic-frames",
        action="store_true",
        default=False,
        help="Rasterize privileged obs into 84x84 radar frames instead of "
        "returning the zero placeholder. Useful for exercising student "
        "distillation without real screen capture.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = build_env(
        EnvBuildConfig(
            mode="mock" if args.mock else "real",
            max_steps=args.max_steps,
            action_repeat=args.action_repeat,
            stall_steps=args.stall_steps,
            mock_level_length=args.mock_level_length,
            mock_seed=args.mock_seed,
        )
    )
    model = PPO.load(args.teacher_path)

    frames, probs, modes, level_ids, frame_idxs = [], [], [], [], []
    idx = 0

    for _ in range(args.episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            teacher_prob = extract_teacher_probs(model, obs)
            action, _ = model.predict(obs, deterministic=args.deterministic)

            frame = capture_frame(obs, idx, use_synthetic=args.synthetic_frames)
            frames.append(frame)
            probs.append(teacher_prob)
            modes.append(int(obs[7]))
            level_ids.append(args.level_id)
            frame_idxs.append(idx)
            idx += 1

            obs, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated

    write_shard(out_path, frames, probs, modes, level_ids, frame_idxs)
    print(f"wrote {len(frames)} samples -> {out_path}")


if __name__ == "__main__":
    main()
