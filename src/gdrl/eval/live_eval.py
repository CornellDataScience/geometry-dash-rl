"""Live in-game evaluation of a trained model.

Runs the model on the currently loaded GD level for N episodes,
measuring progress (X position), survival time, and completion rate.

Usage:
    python -m gdrl.eval.live_eval --model artifacts/bc_model.pt --episodes 10
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from gdrl.env.geode_ipc_v3 import GeodeV3Adapter, GeodeIPCV3Config
from gdrl.model.mlp_agent import GDPolicyMLP
from gdrl.model.obs_preprocess import (
    ObsPreprocessor,
    ObsNormalizer,
    PROCESSED_FRAME_DIM,
    RAW_OBS_DIM,
)


def run_eval(
    model: GDPolicyMLP,
    adapter: GeodeV3Adapter,
    preprocessor: ObsPreprocessor,
    n_episodes: int = 10,
    stack_size: int = 4,
    timeout_s: float = 0.2,
    verbose: bool = False,
) -> dict:
    """Run model for n_episodes, return aggregate metrics."""
    results = []

    for ep in range(n_episodes):
        # reset level
        adapter.send_reset()
        time.sleep(0.3)

        # wait for level to start
        if not adapter.wait_next_tick(timeout_s=5.0):
            print(f"  episode {ep+1}: timeout waiting for level start", flush=True)
            continue

        obs_stack = None
        steps = 0
        max_x = 0.0
        completed = False
        n_jumps_sent = 0

        while True:
            if not adapter.wait_next_tick(timeout_s=timeout_s):
                break  # level ended or game paused

            raw_obs = adapter.read_obs()
            x_pos = float(raw_obs[0])
            is_dead = bool(raw_obs[5])
            level_done = adapter.read_level_complete_flag()

            max_x = max(max_x, x_pos)

            if is_dead:
                break
            if level_done:
                completed = True
                break

            # initialize stack by replicating first frame (matches training)
            if obs_stack is None:
                obs_stack = np.tile(raw_obs, (stack_size, 1))
            else:
                # shift left, add new frame
                obs_stack[:-1] = obs_stack[1:]
                obs_stack[-1] = raw_obs

            # preprocess and get action
            stacked_flat = obs_stack.reshape(-1)
            processed = preprocessor.process_stacked(stacked_flat, stack_size=stack_size)
            x_tensor = torch.from_numpy(processed).unsqueeze(0)
            with torch.no_grad():
                logit, _ = model(x_tensor)
            logit_val = float(logit.squeeze().item())
            action = 1 if logit_val > 0.0 else 0
            adapter.send_action(action)
            if action == 1:
                n_jumps_sent += 1

            if verbose and (steps < 20 or steps % 30 == 0 or action == 1):
                print(
                    f"    step={steps:4d}  x={x_pos:7.0f}  y={raw_obs[1]:6.0f}  "
                    f"vy={raw_obs[2]:+6.2f}  on_ground={int(raw_obs[4])}  "
                    f"logit={logit_val:+7.3f}  action={action}",
                    flush=True,
                )

            steps += 1

        results.append({
            "episode": ep + 1,
            "steps": steps,
            "max_x": max_x,
            "completed": completed,
        })
        print(
            f"  episode {ep+1}/{n_episodes}: "
            f"steps={steps} max_x={max_x:.0f} jumps_sent={n_jumps_sent} "
            f"{'COMPLETED' if completed else 'died'}",
            flush=True,
        )

    if not results:
        return {"avg_x": 0, "best_x": 0, "avg_steps": 0, "completion_rate": 0, "episodes": 0}

    return {
        "avg_x": np.mean([r["max_x"] for r in results]),
        "best_x": max(r["max_x"] for r in results),
        "avg_steps": np.mean([r["steps"] for r in results]),
        "completion_rate": sum(r["completed"] for r in results) / len(results),
        "episodes": len(results),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate trained model in-game.")
    ap.add_argument("--model", required=True, help="Path to model checkpoint.")
    ap.add_argument("--norm", default=None, help="Path to normalizer .npz (auto-detected if not given).")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--shm-name", default="gdrl_ipc_v3")
    ap.add_argument("--stack", type=int, default=4)
    ap.add_argument("--verbose", action="store_true", help="Print per-frame debug info.")
    args = ap.parse_args()

    # load model
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    input_dim = checkpoint.get("input_dim", PROCESSED_FRAME_DIM * args.stack)
    stack_size = checkpoint.get("stack_size", args.stack)
    model = GDPolicyMLP(input_dim=input_dim, stack_size=stack_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"loaded model from {args.model} (epoch {checkpoint.get('epoch', '?')})", flush=True)

    # load normalizer
    norm_path = args.norm
    if norm_path is None:
        auto_path = Path(args.model).with_suffix(".norm.npz")
        if auto_path.exists():
            norm_path = str(auto_path)
    if norm_path:
        normalizer = ObsNormalizer.load(norm_path)
        print(f"loaded normalizer from {norm_path}", flush=True)
    else:
        normalizer = None
        print("no normalizer found, using raw observations", flush=True)

    preprocessor = ObsPreprocessor(normalizer=normalizer)

    # connect to game
    adapter = GeodeV3Adapter(GeodeIPCV3Config(shm_name=args.shm_name))
    print(f"connected to SHM '{args.shm_name}'", flush=True)

    try:
        metrics = run_eval(model, adapter, preprocessor, n_episodes=args.episodes,
                           stack_size=stack_size, verbose=args.verbose)
    finally:
        adapter.close()

    print(f"\nresults ({metrics['episodes']} episodes):")
    print(f"  avg X:       {metrics['avg_x']:.0f}")
    print(f"  best X:      {metrics['best_x']:.0f}")
    print(f"  avg steps:   {metrics['avg_steps']:.0f}")
    print(f"  completion:  {metrics['completion_rate']*100:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
