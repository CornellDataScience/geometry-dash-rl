"""DAgger alignment: label policy-visited states with human actions.

Takes the rollout .npz produced by dagger_rollout.py and the human recordings
directory, then for each frame the policy visited looks up the nearest X position
in the human recordings and uses the human's action there as the label.

This avoids real-time labeling — the human just plays the level once normally
and the alignment is done offline by X-position matching.

Output .npz contains:
    obs         (N, 608) float32  — obs from the policy rollout
    actions     (N,)     uint8    — human action at nearest matching X position
    x_pos       (N,)     float32  — X position of each rollout frame
    episode_ids (N,)     uint32   — rollout episode ids

Usage:
    python -m gdrl.data.dagger_align \
        --rollout artifacts/rollouts/iter1.npz \
        --human-data artifacts/recordings/ \
        --out artifacts/dagger_labeled/iter1.npz
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np

from gdrl.data.obs_dataset import find_shards, ShardIndex


def build_human_index(shard_root: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load all human shards and return (x_positions, actions) sorted by x_pos.

    x_positions: (M,) float32 — obs[0] for every human frame
    actions:     (M,) uint8   — player_input for every human frame
    """
    sessions = find_shards(shard_root)
    if not sessions:
        raise FileNotFoundError(f"no human shards found under {shard_root}")

    index = ShardIndex(sessions)
    n = len(index)

    x_pos = np.empty(n, dtype=np.float32)
    actions = np.empty(n, dtype=np.uint8)

    for i in range(n):
        obs, action, _ = index.get(i)
        x_pos[i] = obs[0]
        actions[i] = action

    # sort by x position so we can binary search
    order = np.argsort(x_pos, kind="stable")
    return x_pos[order], actions[order]


def align(
    rollout_x: np.ndarray,
    human_x: np.ndarray,
    human_actions: np.ndarray,
) -> np.ndarray:
    """For each rollout X position find the nearest human X and return its action.

    Uses binary search (O(N log M)) — fast even for large rollouts.
    """
    # searchsorted gives insertion point; nearest is either idx-1 or idx
    idx = np.searchsorted(human_x, rollout_x)
    idx = np.clip(idx, 0, len(human_x) - 1)

    # compare with neighbour to the left and pick the closer one
    idx_left = np.maximum(idx - 1, 0)
    dist_right = np.abs(human_x[idx] - rollout_x)
    dist_left = np.abs(human_x[idx_left] - rollout_x)
    nearest = np.where(dist_left < dist_right, idx_left, idx)

    return human_actions[nearest]


def main() -> int:
    ap = argparse.ArgumentParser(description="Align rollout states to human labels by X position.")
    ap.add_argument("--rollout", required=True, help="Rollout .npz from dagger_rollout.py")
    ap.add_argument("--human-data", required=True, help="Human recordings shard root")
    ap.add_argument("--out", required=True, help="Output labeled .npz path")
    args = ap.parse_args()

    print(f"loading rollout from {args.rollout} ...", flush=True)
    rollout = np.load(args.rollout)
    obs = rollout["obs"]               # (N, 608)
    x_pos = rollout["x_pos"]           # (N,)
    episode_ids = rollout["episode_ids"]

    print(f"rollout: {len(obs)} frames  x=[{x_pos.min():.1f}, {x_pos.max():.1f}]", flush=True)

    print(f"building human index from {args.human_data} ...", flush=True)
    human_x, human_actions = build_human_index(args.human_data)
    print(
        f"human data: {len(human_x)} frames  x=[{human_x.min():.1f}, {human_x.max():.1f}]"
        f"  jump_rate={human_actions.mean():.3f}",
        flush=True,
    )

    # warn if the policy reached X positions the human never visited
    beyond = (x_pos > human_x.max()).sum()
    if beyond > 0:
        print(
            f"WARNING: {beyond} rollout frames are beyond the furthest human X "
            f"({human_x.max():.1f}) — labels for those frames use the last human frame.",
            flush=True,
        )

    print("aligning ...", flush=True)
    labeled_actions = align(x_pos, human_x, human_actions)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        obs=obs,
        actions=labeled_actions,
        x_pos=x_pos,
        episode_ids=episode_ids,
    )
    print(
        f"saved {len(obs)} labeled frames -> {out}  "
        f"jump_rate={labeled_actions.mean():.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
