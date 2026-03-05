from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from gdrl.env.privileged_env import GDPrivilegedEnv
from gdrl.data.collector import write_shard


def fake_frame(obs: np.ndarray, frame_idx: int) -> np.ndarray:
    # placeholder synthetic frame generator until real screen capture is wired
    img = np.zeros((84, 84), dtype=np.uint8)
    x = int((obs[0] % 100) / 100 * 83)
    y = int((obs[1] + 3) / 6 * 83)
    img[max(0, y-2):min(84, y+3), max(0, x-2):min(84, x+3)] = 255
    img[(frame_idx % 84), :] = 32
    return img


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
            # SB3 doesn't expose dist directly from predict, approximate with one-hot for now
            teacher_prob = np.array([1.0, 0.0], dtype=np.float32)
            if int(action) == 1:
                teacher_prob[:] = [0.0, 1.0]

            frame = fake_frame(obs, idx)
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
