from __future__ import annotations

from glob import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class DistillNPZDataset(Dataset):
    def __init__(self, source: str | Path, stack_size: int = 4):
        self.stack_size = int(stack_size)
        self.paths = self._resolve_paths(source)
        self.frames_shards: list[np.ndarray] = []
        self.teacher_prob_shards: list[np.ndarray] = []
        self.mode_shards: list[np.ndarray] = []
        self.indices: list[tuple[int, int]] = []
        sample_modes: list[int] = []

        for shard_idx, path in enumerate(self.paths):
            with np.load(path) as data:
                frames = np.asarray(data["frames"], dtype=np.uint8)
                teacher_probs = np.asarray(data["teacher_probs"], dtype=np.float32)
                modes = np.asarray(data["modes"], dtype=np.int64)

            if not (len(frames) == len(teacher_probs) == len(modes)):
                raise ValueError(f"Mismatched shard lengths in {path}")

            self.frames_shards.append(frames)
            self.teacher_prob_shards.append(teacher_probs)
            self.mode_shards.append(modes)

            sample_count = max(0, len(frames) - self.stack_size + 1)
            for offset in range(sample_count):
                self.indices.append((shard_idx, offset))
                sample_modes.append(int(modes[offset + self.stack_size - 1]))

        self.sample_modes = np.asarray(sample_modes, dtype=np.int64)
        self.num_modes = int(self.sample_modes.max()) + 1 if len(self.sample_modes) else 0

    def _resolve_paths(self, source: str | Path) -> list[Path]:
        path = Path(source)
        if path.is_dir():
            paths = sorted(path.glob("*.npz"))
        elif any(ch in str(source) for ch in "*?[]"):
            paths = sorted(Path(p) for p in glob(str(source)))
        else:
            paths = [path]
        if not paths:
            raise FileNotFoundError(f"No NPZ shards found for {source!r}")
        return paths

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        shard_idx, offset = self.indices[idx]
        frames = self.frames_shards[shard_idx]
        teacher_probs = self.teacher_prob_shards[shard_idx]
        modes = self.mode_shards[shard_idx]

        stacked = frames[offset:offset + self.stack_size].astype(np.float32) / 255.0
        probs = teacher_probs[offset + self.stack_size - 1]
        mode = int(modes[offset + self.stack_size - 1])
        return (
            torch.from_numpy(stacked),
            torch.from_numpy(np.asarray(probs, dtype=np.float32)),
            torch.tensor(mode, dtype=torch.long),
        )
