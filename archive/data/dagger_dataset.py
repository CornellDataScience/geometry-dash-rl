"""DAgger dataset aggregator.

Combines the original human recordings with all DAgger-labeled datasets from
every past iteration into one growing dataset. Old data is never discarded —
the dataset only grows each iteration. Output is written as shards in the same
format HumanPlayDataset reads, so bc_train.py works unchanged.

Output directory layout:
    <out>/
        shard_00000.npz
        shard_00001.npz
        ...

Each shard contains: obs, actions, episode_ids (+ zero-filled ticks/is_dead/level_done).

Usage:
    python -m gdrl.data.dagger_dataset \
        --human-data artifacts/recordings/ \
        --labeled-dir artifacts/dagger_labeled/ \
        --out artifacts/dagger_dataset_iter1/ \
        --shard-size 20000
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np

from gdrl.data.obs_dataset import find_shards, ShardIndex, OBS_DIM


def _load_human(shard_root: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all human recording shards. Returns (obs, actions, episode_ids)."""
    sessions = find_shards(shard_root)
    if not sessions:
        raise FileNotFoundError(f"no human shards found under {shard_root}")
    index = ShardIndex(sessions)
    n = len(index)
    obs = np.empty((n, OBS_DIM), dtype=np.float32)
    actions = np.empty(n, dtype=np.uint8)
    episode_ids = np.empty(n, dtype=np.int64)
    for i in range(n):
        o, a, ep = index.get(i)
        obs[i] = o
        actions[i] = a
        episode_ids[i] = ep
    return obs, actions, episode_ids


def _load_labeled(labeled_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load all dagger_align output .npz files from labeled_dir.

    Returns (obs, actions, episode_ids) concatenated across all files,
    or None if the directory is empty / doesn't exist.
    """
    labeled_dir = Path(labeled_dir)
    files = sorted(labeled_dir.glob("*.npz")) if labeled_dir.exists() else []
    if not files:
        return None

    all_obs, all_actions, all_eps = [], [], []
    ep_offset = 0
    for f in files:
        data = np.load(f)
        o = data["obs"]
        a = data["actions"]
        # episode_ids may not be present in older labeled files — default to zeros
        eps = data["episode_ids"].astype(np.int64) if "episode_ids" in data else np.zeros(len(o), dtype=np.int64)
        # offset episode ids so they don't collide across files
        eps = eps + ep_offset
        ep_offset += int(eps.max()) + 1
        all_obs.append(o)
        all_actions.append(a)
        all_eps.append(eps)
        print(f"  loaded {f.name}: {len(o)} frames", flush=True)

    return (
        np.concatenate(all_obs, axis=0),
        np.concatenate(all_actions, axis=0).astype(np.uint8),
        np.concatenate(all_eps, axis=0),
    )


def _write_shards(
    out_dir: Path,
    obs: np.ndarray,
    actions: np.ndarray,
    episode_ids: np.ndarray,
    shard_size: int,
) -> int:
    """Write data as shard_*.npz files into out_dir. Returns number of shards written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(obs)
    shard_idx = 0
    for start in range(0, n, shard_size):
        end = min(start + shard_size, n)
        path = out_dir / f"shard_{shard_idx:05d}.npz"
        np.savez_compressed(
            path,
            obs=obs[start:end],
            actions=actions[start:end],
            episode_ids=episode_ids[start:end],
            ticks=np.zeros(end - start, dtype=np.uint32),
            is_dead=np.zeros(end - start, dtype=np.uint8),
            level_done=np.zeros(end - start, dtype=np.uint8),
        )
        shard_idx += 1
    return shard_idx


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate human + DAgger labeled data into one dataset.")
    ap.add_argument("--human-data", required=True, help="Human recordings shard root (artifacts/recordings/)")
    ap.add_argument("--labeled-dir", required=True, help="Directory of dagger_align output .npz files")
    ap.add_argument("--out", required=True, help="Output shard directory")
    ap.add_argument("--shard-size", type=int, default=20000, help="Frames per output shard")
    args = ap.parse_args()

    print("loading human recordings ...", flush=True)
    h_obs, h_actions, h_eps = _load_human(args.human_data)
    print(f"  human: {len(h_obs)} frames  jump_rate={h_actions.mean():.3f}", flush=True)

    print(f"loading DAgger labeled data from {args.labeled_dir} ...", flush=True)
    labeled = _load_labeled(args.labeled_dir)
    if labeled is None:
        print("  no labeled files found — using human data only", flush=True)
        obs = h_obs
        actions = h_actions
        episode_ids = h_eps
    else:
        d_obs, d_actions, d_eps = labeled
        # offset DAgger episode ids so they don't collide with human episode ids
        d_eps = d_eps + int(h_eps.max()) + 1
        obs = np.concatenate([h_obs, d_obs], axis=0)
        actions = np.concatenate([h_actions, d_actions], axis=0)
        episode_ids = np.concatenate([h_eps, d_eps], axis=0)
        print(
            f"  dagger labeled: {len(d_obs)} frames  jump_rate={d_actions.mean():.3f}",
            flush=True,
        )

    print(
        f"total: {len(obs)} frames  "
        f"human={len(h_obs)}  dagger={len(obs) - len(h_obs)}  "
        f"jump_rate={actions.mean():.3f}",
        flush=True,
    )

    out = Path(args.out)
    n_shards = _write_shards(out, obs, actions, episode_ids, args.shard_size)
    print(f"wrote {n_shards} shards -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
