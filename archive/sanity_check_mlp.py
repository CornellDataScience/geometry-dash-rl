"""Offline sanity check: run a trained model on a recorded shard
and compare predictions against ground-truth labels.

If this works correctly, the model + preprocessor are fine and
any live-eval failure is in the live IPC loop.

Usage:
    python -m gdrl.eval.sanity_check --model artifacts/bc_stereo_only.pt \\
        --shard artifacts/recordings/stereo_madness/<session>/shard_00000.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from gdrl.model.mlp_agent import GDPolicyMLP
from gdrl.model.obs_preprocess import (
    ObsPreprocessor,
    ObsNormalizer,
    PROCESSED_FRAME_DIM,
    RAW_OBS_DIM,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--norm", default=None)
    ap.add_argument("--shard", required=True, help="Path to a single .npz shard")
    ap.add_argument("--stack", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=500)
    args = ap.parse_args()

    # load model
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    input_dim = ckpt.get("input_dim", PROCESSED_FRAME_DIM * args.stack)
    stack_size = ckpt.get("stack_size", args.stack)
    model = GDPolicyMLP(input_dim=input_dim, stack_size=stack_size)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # normalizer
    norm_path = args.norm or Path(args.model).with_suffix(".norm.npz")
    normalizer = ObsNormalizer.load(norm_path) if Path(norm_path).exists() else None
    preprocessor = ObsPreprocessor(normalizer=normalizer)
    print(f"loaded model + normalizer", flush=True)

    # load shard
    data = np.load(args.shard)
    obs = data["obs"]    # (N, 608)
    actions = data["actions"]  # (N,)
    n = min(len(obs), args.max_frames)
    print(f"running {n} frames from {args.shard}", flush=True)

    # build stacks the same way the dataset does (replicate first frame at start)
    K = stack_size
    correct = 0
    pred_jumps = 0
    label_jumps = 0
    matches = 0

    for i in range(n):
        # build stack ending at frame i
        stack = np.zeros((K, RAW_OBS_DIM), dtype=np.float32)
        for k in range(K):
            src = max(0, i - k)
            stack[K - 1 - k] = obs[src]
        stacked = stack.reshape(-1)
        processed = preprocessor.process_stacked(stacked, stack_size=K)
        x = torch.from_numpy(processed).unsqueeze(0)
        with torch.no_grad():
            logit, _ = model(x)
        logit_val = float(logit.squeeze().item())
        pred = 1 if logit_val > 0.0 else 0
        label = int(actions[i])

        if pred == label:
            correct += 1
        if pred == 1:
            pred_jumps += 1
        if label == 1:
            label_jumps += 1
            if pred == 1:
                matches += 1

        # print first 50 frames for inspection
        if i < 50 or (i % 50 == 0):
            print(f"  frame {i:4d}  logit={logit_val:+.3f}  pred={pred}  label={label}  "
                  f"{'✓' if pred == label else '✗'}", flush=True)

    print(f"\nsummary over {n} frames:")
    print(f"  accuracy:     {correct/n:.3f}")
    print(f"  pred jumps:   {pred_jumps} ({pred_jumps/n*100:.1f}%)")
    print(f"  label jumps:  {label_jumps} ({label_jumps/n*100:.1f}%)")
    print(f"  jump matches: {matches}/{label_jumps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
