"""PyTorch Dataset for recorded human gameplay shards.

Loads NPZ shards written by `gdrl.data.record_human`, builds a flat index
across all sessions, and returns frame-stacked observation tensors for
behavioral cloning.

Directory layout produced by the recorder:

    <shard_root>/
        20260408_143022/        <- session 0
            shard_00000.npz
            shard_00001.npz
        20260408_150511/        <- session 1
            shard_00000.npz
            ...

Each session's mod reload resets the mod-side `episode_id` counter to 0.
To keep frame stacking from crossing sessions when raw episode_ids collide,
we namespace episode_ids by session index:

    effective_episode_id = session_idx * SESSION_STRIDE + raw_episode_id

Frame stacking respects these effective episode ids: if the stack would
cross into a different episode, the oldest valid frame in the current
episode is replicated.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False
    Dataset = object  # type: ignore


OBS_DIM = 608
SESSION_STRIDE = 1_000_000  # max episodes per session before namespace wraps


def find_shards(shard_root: str | Path) -> list[list[Path]]:
    """Return a list of sessions; each session is a sorted list of shard paths.

    Accepts any nesting depth. Groups shards by their parent directory
    (each parent dir = one session). Handles:
    - shard_root/shard_*.npz  (single session)
    - shard_root/<session>/shard_*.npz  (flat sessions)
    - shard_root/<level>/<session>/shard_*.npz  (level folders)
    """
    root = Path(shard_root)
    # find all shards recursively, group by parent dir
    all_shards = sorted(root.rglob("shard_*.npz"))
    if not all_shards:
        return []
    groups: dict[Path, list[Path]] = {}
    for p in all_shards:
        groups.setdefault(p.parent, []).append(p)
    # sort sessions by directory name for deterministic ordering
    return [sorted(shards) for _, shards in sorted(groups.items())]


class ShardIndex:
    """Flat index over multiple sessions × shards.

    Accepts a list of sessions; each session is a list of shard paths (in
    chronological order). Loads every shard into memory and concatenates
    `obs`, `actions`, and session-namespaced `effective_episode_ids`.
    """

    def __init__(self, sessions: list[list[Path]]):
        if not sessions or not any(sessions):
            raise ValueError("ShardIndex needs at least one shard")
        self.sessions = sessions
        self._shards: List[dict] = []
        for session_idx, shard_paths in enumerate(sessions):
            for p in shard_paths:
                data = np.load(p)
                raw_eps = data["episode_ids"].astype(np.int64)
                effective_eps = session_idx * SESSION_STRIDE + raw_eps
                self._shards.append({
                    "obs": data["obs"],
                    "actions": data["actions"],
                    "episode_ids": effective_eps,
                    "session_idx": session_idx,
                    "path": p,
                })
        self.lengths = [s["obs"].shape[0] for s in self._shards]
        self.cum = np.concatenate([[0], np.cumsum(self.lengths)])
        self.total = int(self.cum[-1])

    def __len__(self) -> int:
        return self.total

    def get(self, global_idx: int) -> tuple[np.ndarray, int, int]:
        """Return (obs_vector, action, effective_episode_id) at flat index."""
        shard_i = int(np.searchsorted(self.cum, global_idx, side="right") - 1)
        local = global_idx - int(self.cum[shard_i])
        s = self._shards[shard_i]
        return s["obs"][local], int(s["actions"][local]), int(s["episode_ids"][local])

    def episode_ids_array(self) -> np.ndarray:
        return np.concatenate([s["episode_ids"] for s in self._shards])


class HumanPlayDataset(Dataset):
    """Dataset of (stacked_obs, action) for behavioral cloning.

    stacked_obs has shape (stack_size * 608,). Frames in the stack are ordered
    oldest → newest. If the stack would cross an episode boundary backward,
    the oldest available frame in the current episode is replicated.
    """

    def __init__(
        self,
        shard_dir: str | Path | None = None,
        stack_size: int = 4,
        indices: np.ndarray | None = None,
        preprocessor=None,
        sessions: list[list[Path]] | None = None,
    ):
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is required for HumanPlayDataset")
        if sessions is None:
            if shard_dir is None:
                raise ValueError("must provide shard_dir or sessions")
            shard_dir = Path(shard_dir)
            sessions = find_shards(shard_dir)
            if not sessions:
                raise FileNotFoundError(f"no shards found under {shard_dir}")
        self.index = ShardIndex(sessions)
        self.stack_size = stack_size
        self.preprocessor = preprocessor
        self.episode_ids = self.index.episode_ids_array()
        if indices is None:
            self.indices = np.arange(len(self.index), dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def _stacked_obs(self, idx: int) -> np.ndarray:
        """Build a (stack_size * OBS_DIM,) vector ending at idx."""
        K = self.stack_size
        ep = int(self.episode_ids[idx])
        stack = np.zeros((K, OBS_DIM), dtype=np.float32)
        # walk backward, stop at episode boundary
        for k in range(K):
            src = idx - k
            if src < 0 or int(self.episode_ids[src]) != ep:
                # replicate oldest valid frame in same episode
                src = idx - k + 1
                if src < 0:
                    src = 0
                obs, _, _ = self.index.get(src)
                # fill the remaining (older) slots with this same obs
                for kk in range(k, K):
                    stack[K - 1 - kk] = obs
                break
            obs, _, _ = self.index.get(src)
            stack[K - 1 - k] = obs
        return stack.reshape(-1)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        x = self._stacked_obs(idx)
        if self.preprocessor is not None:
            x = self.preprocessor.process_stacked(x, stack_size=self.stack_size)
        _, action, _ = self.index.get(idx)
        return torch.from_numpy(x), torch.tensor(action, dtype=torch.float32)


def train_val_split(
    shard_dir: str | Path,
    val_fraction: float = 0.1,
    seed: int = 0,
    stack_size: int = 4,
    preprocessor=None,
) -> Tuple["HumanPlayDataset", "HumanPlayDataset"]:
    """Split by frame index. Episodes may straddle the split — fine for BC."""
    shard_dir = Path(shard_dir)
    sessions = find_shards(shard_dir)
    if not sessions:
        raise FileNotFoundError(f"no shards under {shard_dir}")
    tmp_index = ShardIndex(sessions)
    n = len(tmp_index)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(n * val_fraction)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train = HumanPlayDataset(shard_dir, stack_size=stack_size, indices=train_idx, preprocessor=preprocessor)
    val = HumanPlayDataset(shard_dir, stack_size=stack_size, indices=val_idx, preprocessor=preprocessor)
    return train, val


def train_val_split_by_level(
    shard_dir: str | Path,
    val_level: str,
    stack_size: int = 4,
    preprocessor=None,
) -> Tuple["HumanPlayDataset", "HumanPlayDataset"]:
    """Hold out one level for validation, train on all others.

    Layout assumed: <shard_dir>/<level_name>/<session_dir>/shard_*.npz

    Val frames are returned in original temporal order (not shuffled), so
    event-level metrics computed on val are meaningful.
    """
    shard_dir = Path(shard_dir)
    sessions = find_shards(shard_dir)
    if not sessions:
        raise FileNotFoundError(f"no shards under {shard_dir}")
    val_path = (shard_dir / val_level).resolve()
    if not val_path.exists() or not val_path.is_dir():
        raise FileNotFoundError(f"val level dir not found: {val_path}")

    val_sessions: list[list[Path]] = []
    train_sessions: list[list[Path]] = []
    for shard_paths in sessions:
        if not shard_paths:
            continue
        # each session is shard_dir/<level>/<timestamp>/shard_*.npz, so
        # the session dir's parent is the level dir
        if shard_paths[0].resolve().parent.parent == val_path:
            val_sessions.append(shard_paths)
        else:
            train_sessions.append(shard_paths)

    if not val_sessions:
        raise ValueError(f"no sessions found under {val_path}")
    if not train_sessions:
        raise ValueError(f"all sessions are val; nothing left to train on")

    train = HumanPlayDataset(
        sessions=train_sessions, stack_size=stack_size, preprocessor=preprocessor
    )
    val = HumanPlayDataset(
        sessions=val_sessions, stack_size=stack_size, preprocessor=preprocessor
    )
    return train, val
