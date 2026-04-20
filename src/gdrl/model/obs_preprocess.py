"""Observation preprocessor: raw 608-float obs → 277-float processed obs.

Handles:
1. Player feature extraction (8 raw → 7 processed)
2. Object subsampling (100 → 30 via density-aware selection)
3. Semantic object encoding (6 raw floats → 9 processed per object)
4. Optional normalization of continuous features
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Set

import numpy as np

# ---------- constants ----------

RAW_OBS_DIM = 608
RAW_OBJ_START = 8
RAW_FLOATS_PER_OBJ = 6
RAW_MAX_OBJECTS = 100

N_SELECTED_OBJECTS = 30
N_NEAREST = 15
N_EXPONENTIAL = 15

PROCESSED_PLAYER_DIM = 7      # y, vy, dx, on_ground, mode one-hot(4)
PROCESSED_OBJ_DIM = 9         # relX, relY, category one-hot(5), scaleX, scaleY
PROCESSED_FRAME_DIM = PROCESSED_PLAYER_DIM + N_SELECTED_OBJECTS * PROCESSED_OBJ_DIM  # 277

N_MODES = 4  # cube, ship, ball, ufo

# ---------- semantic category mapping ----------

CAT_HAZARD = 0
CAT_ORB = 1
CAT_PAD = 2
CAT_PORTAL = 3
CAT_BLOCK = 4
N_CATEGORIES = 5

HAZARD_IDS: Set[int] = {8, 9, 39, 103, 392, 421, 88, 89, 98}

ORB_IDS: Set[int] = {36, 84, 141, 1022, 1330, 1594, 1704, 3005}

PAD_IDS: Set[int] = {35, 67, 140, 1332, 3027}

PORTAL_IDS: Set[int] = {
    10, 11, 12, 13, 45, 46, 47, 99, 101, 286, 660, 745, 1331, 1933,
    200, 201, 202, 203, 1334,
}

_OBJ_ID_TO_CAT: Dict[int, int] = {}
for _id in HAZARD_IDS:
    _OBJ_ID_TO_CAT[_id] = CAT_HAZARD
for _id in ORB_IDS:
    _OBJ_ID_TO_CAT[_id] = CAT_ORB
for _id in PAD_IDS:
    _OBJ_ID_TO_CAT[_id] = CAT_PAD
for _id in PORTAL_IDS:
    _OBJ_ID_TO_CAT[_id] = CAT_PORTAL


def obj_category(obj_id: int) -> int:
    return _OBJ_ID_TO_CAT.get(obj_id, CAT_BLOCK)


# ---------- normalization ----------

@dataclass
class ObsNormalizer:
    """Per-feature mean/std for continuous features only.

    Binary features (on_ground, one-hots, is_hazard) are NOT normalized.
    Stores arrays of shape (PROCESSED_FRAME_DIM,).
    """
    mean: np.ndarray   # (277,)
    std: np.ndarray    # (277,)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + 1e-8)

    def save(self, path: str | Path) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    @staticmethod
    def load(path: str | Path) -> "ObsNormalizer":
        d = np.load(path)
        return ObsNormalizer(mean=d["mean"], std=d["std"])

    @staticmethod
    def identity() -> "ObsNormalizer":
        return ObsNormalizer(
            mean=np.zeros(PROCESSED_FRAME_DIM, dtype=np.float32),
            std=np.ones(PROCESSED_FRAME_DIM, dtype=np.float32),
        )


def _continuous_mask() -> np.ndarray:
    """Boolean mask: True for continuous features that should be normalized."""
    mask = np.zeros(PROCESSED_FRAME_DIM, dtype=bool)
    # player: y(0), vy(1), dx(2) are continuous; on_ground(3) + mode one-hot(4-6) are binary
    mask[0:3] = True
    # objects: relX(0), relY(1) continuous; cat one-hot(2-6) binary; scaleX(7), scaleY(8) continuous
    for i in range(N_SELECTED_OBJECTS):
        base = PROCESSED_PLAYER_DIM + i * PROCESSED_OBJ_DIM
        mask[base + 0] = True   # relX
        mask[base + 1] = True   # relY
        mask[base + 7] = True   # scaleX
        mask[base + 8] = True   # scaleY
    return mask


CONTINUOUS_MASK = _continuous_mask()


# ---------- subsampling ----------

def _subsample_objects(raw_objects: np.ndarray) -> np.ndarray:
    """Select 30 objects from up to 100 via density-aware sampling.

    raw_objects: (100, 6) — sorted by |relX| ascending (mod guarantees this).
    Returns: (30, 6) — 15 nearest + 15 exponentially spaced.
    """
    # find how many valid objects exist (non-zero relX or relY)
    valid_mask = (raw_objects[:, 0] != 0.0) | (raw_objects[:, 1] != 0.0)
    n_valid = int(valid_mask.sum())

    result = np.zeros((N_SELECTED_OBJECTS, RAW_FLOATS_PER_OBJ), dtype=np.float32)

    if n_valid == 0:
        return result

    valid = raw_objects[:n_valid]

    # nearest 15 (or all if fewer)
    n_near = min(N_NEAREST, n_valid)
    result[:n_near] = valid[:n_near]

    if n_valid <= N_NEAREST:
        return result

    # exponentially spaced: starting from the distance of the 15th object
    d_min = abs(valid[N_NEAREST - 1, 0])
    if d_min < 1.0:
        d_min = 1.0

    remaining = valid[N_NEAREST:]
    remaining_dists = np.abs(remaining[:, 0])
    filled = n_near

    for k in range(N_EXPONENTIAL):
        if filled >= N_SELECTED_OBJECTS:
            break
        threshold = d_min * (2.0 ** k)
        candidates = np.where(remaining_dists >= threshold)[0]
        if len(candidates) == 0:
            break
        # pick the nearest object beyond the threshold
        best = candidates[0]
        result[filled] = remaining[best]
        filled += 1
        # exclude this object from future picks
        remaining_dists[best] = -1.0

    return result


# ---------- encoding ----------

def _encode_objects(selected: np.ndarray) -> np.ndarray:
    """Encode 30 objects from raw (30, 6) → processed (30, 9).

    Raw per object: relX, relY, objType, objID, scaleX, scaleY
    Processed: relX, relY, cat_onehot(5), scaleX, scaleY
    """
    n = selected.shape[0]
    out = np.zeros((n, PROCESSED_OBJ_DIM), dtype=np.float32)

    for i in range(n):
        relX, relY, _objType, objID, scaleX, scaleY = selected[i]
        if relX == 0.0 and relY == 0.0 and objID == 0.0:
            continue  # empty slot, leave as zeros
        out[i, 0] = relX
        out[i, 1] = relY
        cat = obj_category(int(objID))
        out[i, 2 + cat] = 1.0  # one-hot
        out[i, 7] = scaleX
        out[i, 8] = scaleY

    return out


def _encode_player(raw_obs: np.ndarray) -> np.ndarray:
    """Extract and encode player features from raw obs[0..7] → (7,).

    Raw: x(0), y(1), vy(2), dx(3), on_ground(4), is_dead(5), always_1(6), mode(7)
    Processed: y, vy, dx, on_ground, mode_onehot(4)
    """
    out = np.zeros(PROCESSED_PLAYER_DIM, dtype=np.float32)
    out[0] = raw_obs[1]    # y
    out[1] = raw_obs[2]    # vy
    out[2] = raw_obs[3]    # dx
    out[3] = raw_obs[4]    # on_ground
    mode = int(raw_obs[7])
    if 0 <= mode < N_MODES:
        out[4 + mode] = 1.0
    return out


# ---------- main preprocessor ----------

class ObsPreprocessor:
    """Converts raw 608-float observations to 277-float processed vectors.

    Usage:
        prep = ObsPreprocessor()
        processed = prep.process_frame(raw_obs)           # (608,) → (277,)
        stacked = prep.process_stacked(raw_stacked, K=4)  # (2432,) → (1108,)
    """

    def __init__(self, normalizer: ObsNormalizer | None = None):
        self.normalizer = normalizer

    def process_frame(self, raw_obs: np.ndarray) -> np.ndarray:
        """Process a single raw observation. (608,) → (277,)."""
        player = _encode_player(raw_obs)

        raw_objects = raw_obs[RAW_OBJ_START:].reshape(RAW_MAX_OBJECTS, RAW_FLOATS_PER_OBJ)
        selected = _subsample_objects(raw_objects)
        encoded_objects = _encode_objects(selected)

        frame = np.concatenate([player, encoded_objects.reshape(-1)])

        if self.normalizer is not None:
            continuous = CONTINUOUS_MASK
            frame[continuous] = self.normalizer.normalize(frame)[continuous]

        return frame

    def process_stacked(self, raw_stacked: np.ndarray, stack_size: int = 4) -> np.ndarray:
        """Process a stacked raw observation. (K*608,) → (K*277,)."""
        frames = raw_stacked.reshape(stack_size, RAW_OBS_DIM)
        processed = np.stack([self.process_frame(f) for f in frames])
        return processed.reshape(-1)


# ---------- normalization computation ----------

def compute_normalizer(shard_dir: str | Path, stack_size: int = 4) -> ObsNormalizer:
    """Compute per-feature mean/std from all shards in shard_dir.

    Only computes stats over continuous features; binary features get
    mean=0, std=1 so normalization is a no-op for them.
    """
    from gdrl.data.obs_dataset import find_shards, ShardIndex

    sessions = find_shards(shard_dir)
    if not sessions:
        raise FileNotFoundError(f"no shards under {shard_dir}")
    index = ShardIndex(sessions)

    prep = ObsPreprocessor(normalizer=None)

    # online Welford's algorithm for numerical stability
    n = 0
    mean = np.zeros(PROCESSED_FRAME_DIM, dtype=np.float64)
    m2 = np.zeros(PROCESSED_FRAME_DIM, dtype=np.float64)

    for i in range(len(index)):
        raw_obs, _, _ = index.get(i)
        processed = prep.process_frame(raw_obs).astype(np.float64)
        n += 1
        delta = processed - mean
        mean += delta / n
        delta2 = processed - mean
        m2 += delta * delta2

    std = np.sqrt(m2 / max(n - 1, 1))

    # force binary features to mean=0, std=1
    mean_f32 = mean.astype(np.float32)
    std_f32 = std.astype(np.float32)
    binary_mask = ~CONTINUOUS_MASK
    mean_f32[binary_mask] = 0.0
    std_f32[binary_mask] = 1.0
    # clamp std to avoid div-by-zero on near-constant features
    std_f32 = np.maximum(std_f32, 1e-6)

    return ObsNormalizer(mean=mean_f32, std=std_f32)
