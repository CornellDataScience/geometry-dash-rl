"""DAgger rollout recorder.

Loads a TeacherPolicy checkpoint, runs it live in the game via GDPrivilegedEnv,
and records every state the policy visits. The output is used by dagger_align.py
to produce human-labeled training data for the next DAgger iteration.

Usage:
    python -m gdrl.data.dagger_rollout \
        --policy artifacts/bc_warmup.pt \
        --episodes 50 \
        --out artifacts/rollouts/iter1.npz
"""
from __future__ import annotations
import argparse
import sys
import time
from collections import deque
from pathlib import Path
from multiprocessing import shared_memory

import numpy as np
import torch

from gdrl.env.geode_ipc_v3 import GeodeV3Adapter, GeodeIPCV3Config
from gdrl.env.privileged_env import GDPrivilegedEnv
from gdrl.teacher.model import TeacherPolicy, OBS_DIM, STACK_SIZE


def _wait_for_segment(name: str, timeout_s: float) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            shm = shared_memory.SharedMemory(name=name, create=False)
            shm.close()
            return True
        except FileNotFoundError:
            time.sleep(0.2)
    return False


def _make_stack(frame_buf: deque, device: str) -> torch.Tensor:
    """Stack frame buffer oldest→newest into a (1, STACK_SIZE*OBS_DIM) tensor."""
    stacked = np.concatenate(list(frame_buf), axis=0)  # (STACK_SIZE * OBS_DIM,)
    return torch.from_numpy(stacked).unsqueeze(0).to(device)


def run_rollout(policy: TeacherPolicy, env: GDPrivilegedEnv, episodes: int, device: str):
    """Roll out policy for a fixed number of episodes.

    Returns arrays of (obs, x_pos, policy_action, episode_id) across all steps.
    """
    all_obs = []
    all_x = []
    all_actions = []
    all_episode_ids = []

    episode_id = 0
    for ep in range(episodes):
        obs, _ = env.reset()
        episode_id += 1
        done = False
        ep_steps = 0

        # initialise frame buffer with the first obs repeated STACK_SIZE times
        frame_buf: deque = deque([obs.copy() for _ in range(STACK_SIZE)], maxlen=STACK_SIZE)

        while not done:
            obs_stacked = _make_stack(frame_buf, device)
            action = int(policy.predict(obs_stacked).item())

            all_obs.append(obs.copy())
            all_x.append(float(obs[0]))
            all_actions.append(action)
            all_episode_ids.append(episode_id)

            obs, _, terminated, truncated, _ = env.step(action)
            frame_buf.append(obs.copy())
            done = terminated or truncated
            ep_steps += 1

        x_reached = float(obs[0])
        print(f"ep={ep+1}/{episodes}  steps={ep_steps}  x={x_reached:.1f}", flush=True)

    return (
        np.array(all_obs, dtype=np.float32),
        np.array(all_x, dtype=np.float32),
        np.array(all_actions, dtype=np.uint8),
        np.array(all_episode_ids, dtype=np.uint32),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Roll out a policy and record visited states.")
    ap.add_argument("--policy", required=True, help="Path to TeacherPolicy checkpoint (.pt)")
    ap.add_argument("--episodes", type=int, default=50, help="Number of episodes to roll out")
    ap.add_argument("--out", required=True, help="Output .npz path for rollout data")
    ap.add_argument("--shm-name", default="gdrl_ipc_v3", help="Shared memory segment name")
    ap.add_argument("--connect-timeout", type=float, default=30.0)
    ap.add_argument("--hidden", type=int, default=256, help="Must match the trained model")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"loading policy from {args.policy} ...", flush=True)
    policy = TeacherPolicy(hidden=args.hidden).to(device)
    policy.load_state_dict(torch.load(args.policy, map_location=device))
    policy.eval()

    print(f"waiting for shared memory '{args.shm_name}' ...", flush=True)
    if not _wait_for_segment(args.shm_name, args.connect_timeout):
        print("timeout waiting for shared memory segment", file=sys.stderr)
        return 2

    adapter = GeodeV3Adapter(GeodeIPCV3Config(shm_name=args.shm_name))
    try:
        adapter.verify_version()
    except Exception as e:
        print(f"version check failed: {e}", file=sys.stderr)
        adapter.close()
        return 2

    env = GDPrivilegedEnv(ipc=adapter)

    print(f"rolling out {args.episodes} episodes ...", flush=True)
    obs, x_pos, actions, episode_ids = run_rollout(policy, env, args.episodes, device)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        obs=obs,
        x_pos=x_pos,
        policy_actions=actions,
        episode_ids=episode_ids,
    )
    print(f"saved {len(obs)} frames -> {out}", flush=True)

    adapter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
