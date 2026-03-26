from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from gdrl.env.privileged_env import GDPrivilegedEnv
from gdrl.data.collector import write_shard


def capture_frame(obs: np.ndarray, frame_idx: int) -> np.ndarray:
    # TODO: replace with real screen capture of the GD window (84x84 grayscale)
    return np.zeros((84, 84), dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher-path', type=str, required=True)
    ap.add_argument('--episodes', type=int, default=5)
    ap.add_argument('--out', type=str, default='artifacts/datasets/mock_shard_000.npz')
    args = ap.parse_args()

    env = GDPrivilegedEnv()
    model = PPO.load(args.teacher_path)

    frames, probs, modes, level_ids, frame_idxs = [], [], [], [], []
    idx = 0

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            # TODO: extract real action probabilities from teacher policy distribution
            teacher_prob = np.array([1.0, 0.0], dtype=np.float32)
            if int(action) == 1:
                teacher_prob[:] = [0.0, 1.0]

            frame = capture_frame(obs, idx)
            frames.append(frame)
            probs.append(teacher_prob)
            modes.append(int(obs[7]) % 4)
            level_ids.append(0)
            frame_idxs.append(idx)
            idx += 1

            obs, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated

    write_shard(args.out, frames, probs, modes, level_ids, frame_idxs)
    print(f'wrote {len(frames)} samples -> {args.out}')


if __name__ == '__main__':
    main()
