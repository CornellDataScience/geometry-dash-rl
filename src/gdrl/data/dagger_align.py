"""DAgger alignment: label policy-visited states with grounded human jump presses.

Takes the rollout .npz produced by dagger_rollout.py and the human recordings
directory, then for each frame the policy visited looks up the nearest X position
in the human recordings and uses the human's grounded jump-press label there.

This avoids real-time labeling — the human just plays the level once normally
and the alignment is done offline by X-position matching.

Output .npz contains:
    obs         (N, 608) float32  — obs from the policy rollout
    actions     (N,)     uint8    — grounded jump press labels
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

from gdrl.data.obs_dataset import (
    TARGET_SEMANTICS,
    action_allowed,
    find_shards,
    ShardIndex,
    grounded_jump_labels,
)


def build_human_index(shard_root: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load all human shards and return (x_positions, grounded_labels) sorted by x_pos.

    x_positions: (M,) float32 — obs[0] for every human frame
    actions:     (M,) uint8   — action && on_ground for every human frame
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
        actions[i] = index.get_label(i)

    # sort by x position so we can binary search
    order = np.argsort(x_pos, kind="stable")
    return x_pos[order], actions[order]


def cluster_press_x(press_x: np.ndarray, gap: float) -> np.ndarray:
    """Cluster repeated human press samples into one representative X per event."""
    press_x = np.sort(np.asarray(press_x, dtype=np.float32))
    if len(press_x) == 0:
        return press_x
    clusters: list[list[float]] = [[float(press_x[0])]]
    for x in press_x[1:]:
        if float(x) - clusters[-1][-1] <= gap:
            clusters[-1].append(float(x))
        else:
            clusters.append([float(x)])
    return np.asarray([np.median(c) for c in clusters], dtype=np.float32)


def build_human_press_index(shard_root: str | Path, cluster_gap: float = 80.0) -> np.ndarray:
    """Return clustered X positions where the human made grounded jump presses."""
    human_x, human_actions = build_human_index(shard_root)
    return cluster_press_x(human_x[human_actions.astype(bool)], gap=cluster_gap)


def align(
    rollout_x: np.ndarray,
    human_x: np.ndarray,
    human_actions: np.ndarray,
) -> np.ndarray:
    """For each rollout X position find the nearest human X and return its label.

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


def align_to_press_events(
    rollout_x: np.ndarray,
    human_press_x: np.ndarray,
    x_tolerance: float,
    rollout_on_ground: np.ndarray | None = None,
    episode_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Label one nearest rollout frame per human press event.

    This preserves "how many times jump is pressed" semantics. A tolerance
    window is used only to choose the nearest matching frame; it does not turn
    the whole window into positive labels.
    """
    labels = np.zeros(len(rollout_x), dtype=np.uint8)
    if len(human_press_x) == 0 or len(rollout_x) == 0:
        return labels

    rollout_x = np.asarray(rollout_x)
    human_press_x = np.asarray(human_press_x)
    if rollout_on_ground is None:
        rollout_on_ground = np.ones(len(rollout_x), dtype=bool)
    else:
        rollout_on_ground = np.asarray(rollout_on_ground).astype(bool)
    if episode_ids is None:
        episode_ids = np.zeros(len(rollout_x), dtype=np.int64)
    else:
        episode_ids = np.asarray(episode_ids)

    for ep in np.unique(episode_ids):
        ep_idx = np.where((episode_ids == ep) & rollout_on_ground)[0]
        if len(ep_idx) == 0:
            continue
        ep_x = rollout_x[ep_idx]
        lo = ep_x.min() - x_tolerance
        hi = ep_x.max() + x_tolerance
        press_events = human_press_x[(human_press_x >= lo) & (human_press_x <= hi)]
        for press_x in press_events:
            nearest_local = int(np.argmin(np.abs(ep_x - press_x)))
            if abs(float(ep_x[nearest_local]) - float(press_x)) <= x_tolerance:
                labels[ep_idx[nearest_local]] = 1
    return labels


def main() -> int:
    ap = argparse.ArgumentParser(description="Align rollout states to human labels by X position.")
    ap.add_argument("--rollout", required=True, help="Rollout .npz from dagger_rollout.py")
    ap.add_argument("--human-data", required=True, help="Human recordings shard root")
    ap.add_argument("--out", required=True, help="Output labeled .npz path")
    ap.add_argument("--x-tolerance", type=float, default=20.0,
                    help="X-distance tolerance for matching rollout frames to grounded human press events.")
    ap.add_argument("--press-cluster-gap", type=float, default=80.0,
                    help="Cluster human press X positions within this distance into one expert event.")
    args = ap.parse_args()

    print(f"loading rollout from {args.rollout} ...", flush=True)
    rollout = np.load(args.rollout)
    obs = rollout["obs"]               # (N, 608)
    x_pos = rollout["x_pos"]           # (N,)
    episode_ids = rollout["episode_ids"]

    print(f"rollout: {len(obs)} frames  x=[{x_pos.min():.1f}, {x_pos.max():.1f}]", flush=True)

    print(f"building grounded human index from {args.human_data} ...", flush=True)
    human_x, human_actions = build_human_index(args.human_data)
    human_press_x = cluster_press_x(human_x[human_actions.astype(bool)], gap=args.press_cluster_gap)
    print(
        f"human data: {len(human_x)} frames  x=[{human_x.min():.1f}, {human_x.max():.1f}]"
        f"  grounded_jump_rate={human_actions.mean():.3f}"
        f"  grounded_press_events={len(human_press_x)}",
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
    labeled_actions = align_to_press_events(
        x_pos,
        human_press_x,
        args.x_tolerance,
        rollout_on_ground=np.asarray([action_allowed(frame) for frame in obs], dtype=bool),
        episode_ids=episode_ids,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        obs=obs,
        actions=labeled_actions,
        x_pos=x_pos,
        episode_ids=episode_ids,
        target_semantics=np.array(TARGET_SEMANTICS),
    )
    print(
        f"saved {len(obs)} labeled frames -> {out}  "
        f"grounded_jump_rate={labeled_actions.mean():.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
