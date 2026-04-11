from __future__ import annotations

"""Dataset schema draft for distillation data."""

from pathlib import Path
import numpy as np


def write_shard(path: str | Path, frames, teacher_probs, modes, level_ids, frame_idxs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frames=np.asarray(frames, dtype=np.uint8),
        teacher_probs=np.asarray(teacher_probs, dtype=np.float32),
        modes=np.asarray(modes, dtype=np.int32),
        level_ids=np.asarray(level_ids, dtype=np.int32),
        frame_idxs=np.asarray(frame_idxs, dtype=np.int64),
    )
