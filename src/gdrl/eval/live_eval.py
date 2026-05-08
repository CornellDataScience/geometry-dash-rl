"""Greedy live in-game evaluation of a CV Rainbow checkpoint.

Loads a checkpoint saved by gdrl.train.dqn_cv and runs N greedy episodes per
level, reporting per-level success rate, mean / best max-percent, and per-mode
action distribution.

Usage:
    python -m gdrl.eval.live_eval --checkpoint artifacts/dqn_cv/latest.pt --episodes 5 \\
        --levels stereo_madness back_on_track polargeist
"""
from __future__ import annotations
import argparse
import os
from collections import defaultdict

import numpy as np
import torch

from gdrl.env.cv_env import GDCvEnv, CvEnvConfig
from gdrl.model.cv_moe_dqn import ModeGatedRainbow
from gdrl.train.dqn_cv import LEVELS


def _device(s: str) -> torch.device:
    if s == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--levels", nargs="+", default=list(LEVELS), choices=list(LEVELS))
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=8000)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    device = _device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    print(f"loaded {args.checkpoint} (env_steps={ckpt.get('env_steps','?')} "
          f"episode={ckpt.get('episode','?')} curriculum_idx={ckpt.get('curriculum_idx','?')})")

    aux_enabled = ckpt.get("cfg", {}).get("aux_enabled", True)
    model = ModeGatedRainbow(stack=4, aux=aux_enabled).to(device)
    model.load_state_dict(ckpt["online"])
    model.eval()

    env = GDCvEnv(cfg=CvEnvConfig(stack=4, max_steps=args.max_steps))

    summary = {}
    try:
        for name in args.levels:
            level_id = LEVELS[name]
            wins, max_pcts, rewards, mode_actions = 0, [], [], defaultdict(lambda: np.zeros(2, dtype=np.int64))
            for ep in range(args.episodes):
                obs, info = env.reset(options={"level_id": level_id, "percent": 0})
                ep_r = 0.0
                ep_max = info["percent"]
                while True:
                    f = torch.from_numpy(obs["frame"]).unsqueeze(0).to(device)
                    m = torch.tensor([obs["mode_id"]], dtype=torch.long, device=device)
                    with torch.no_grad():
                        q = model(f, m)
                    a = int(q.argmax(dim=-1).item())
                    mode_actions[obs["mode_id"]][a] += 1
                    obs, r, term, trunc, info = env.step(a)
                    ep_r += r
                    ep_max = max(ep_max, info["percent"])
                    if term or trunc:
                        break
                won = info["percent"] >= 99.0 and not info["is_dead"]
                wins += int(won)
                max_pcts.append(ep_max)
                rewards.append(ep_r)
                print(f"  [{name}] ep {ep+1}/{args.episodes}: max_pct={ep_max:.1f} won={won} reward={ep_r:.1f}")

            summary[name] = {
                "win_rate": wins / max(args.episodes, 1),
                "best_pct": float(max(max_pcts) if max_pcts else 0.0),
                "mean_pct": float(np.mean(max_pcts) if max_pcts else 0.0),
                "mean_reward": float(np.mean(rewards) if rewards else 0.0),
                "mode_actions": {int(k): v.tolist() for k, v in mode_actions.items()},
            }
    finally:
        env.close()

    print("\n=== summary ===")
    for name, m in summary.items():
        print(f"{name}: win_rate={m['win_rate']:.2f} best_pct={m['best_pct']:.1f} "
              f"mean_pct={m['mean_pct']:.1f} mean_reward={m['mean_reward']:.1f}")
        for mode_id, counts in m["mode_actions"].items():
            total = sum(counts)
            jump_frac = counts[1] / total if total else 0.0
            print(f"    mode={mode_id}: actions={counts} jump_frac={jump_frac:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
