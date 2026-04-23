"""Observation preprocessor: raw 608-float obs → compact processed obs.

Handles:
1. Player feature extraction (8 raw → 7 processed)
2. Object subsampling (100 → 30 via density-aware selection)
3. Object encoding with type ID for learned embedding (6 raw → 5 processed per object)
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
PROCESSED_OBJ_DIM = 5         # relX, relY, type_id, scaleX, scaleY
PROCESSED_FRAME_DIM = PROCESSED_PLAYER_DIM + N_SELECTED_OBJECTS * PROCESSED_OBJ_DIM  # 157

N_MODES = 4  # cube, ship, ball, ufo

# ---------- object type mapping ----------

EMBED_DIM = 8
N_OBJ_TYPES = 42
BLOCK_TYPE_IDX = 41

# Maps GD object ID → type index (0-41)
_OBJ_ID_TO_TYPE: Dict[int, int] = {}

# Hazards: spikes (0-5), saws (6-8)
_SPIKE_MAP = {8: 0, 9: 1, 39: 2, 103: 3, 392: 4, 421: 5}
_SAW_MAP = {88: 6, 89: 7, 98: 8}

# Orbs (9-16)
_ORB_MAP = {36: 9, 84: 10, 141: 11, 1022: 12, 1330: 13, 1594: 14, 1704: 15, 3005: 16}

# Pads (17-21)
_PAD_MAP = {35: 17, 67: 18, 140: 19, 1332: 20, 3027: 21}

# Portals (22-40)
_PORTAL_MAP = {
    10: 22, 11: 23,           # gravity
    12: 24, 13: 25,           # ship, cube
    99: 26, 286: 27,          # ball, ufo
    660: 28, 745: 29,         # wave, robot
    1331: 30, 1933: 31,       # spider, swing
    45: 32, 46: 33,           # mirror on/off
    47: 34, 101: 35,          # big, mini
    200: 36, 201: 37,         # speed slow, norm
    202: 38, 203: 39,         # speed fast, vfast
    1334: 40,                 # speed vslow
}

# Block IDs (all map to BLOCK_TYPE_IDX = 41)
_BLOCK_IDS = {1, 2, 3, 4, 5, 6, 7, 40, 83, 289, 291}

for _src in (_SPIKE_MAP, _SAW_MAP, _ORB_MAP, _PAD_MAP, _PORTAL_MAP):
    _OBJ_ID_TO_TYPE.update(_src)
for _id in _BLOCK_IDS:
    _OBJ_ID_TO_TYPE[_id] = BLOCK_TYPE_IDX


def obj_type_index(obj_id: int) -> int:
    """Return type index (0-41) for a GD object ID. Unknown → block (41)."""
    return _OBJ_ID_TO_TYPE.get(obj_id, BLOCK_TYPE_IDX)


# ---------- normalization ----------

@dataclass
class ObsNormalizer:
    """Per-feature mean/std for continuous features only.

    Binary/categorical features are NOT normalized.
    Stores arrays of shape (PROCESSED_FRAME_DIM,).
    """
    mean: np.ndarray   # (157,)
    std: np.ndarray    # (157,)

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
    # objects: relX(0) continuous, relY(1) continuous, type_id(2) NOT continuous, scaleX(3) continuous, scaleY(4) continuous
    for i in range(N_SELECTED_OBJECTS):
        base = PROCESSED_PLAYER_DIM + i * PROCESSED_OBJ_DIM
        mask[base + 0] = True   # relX
        mask[base + 1] = True   # relY
        # mask[base + 2] = False — type_id (integer, not continuous)
        mask[base + 3] = True   # scaleX
        mask[base + 4] = True   # scaleY
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
    """Encode 30 objects from raw (30, 6) → processed (30, 5).

    Raw per object: relX, relY, objType, objID, scaleX, scaleY
    Processed: relX, relY, type_id (float), scaleX, scaleY
    """
    n = selected.shape[0]
    out = np.zeros((n, PROCESSED_OBJ_DIM), dtype=np.float32)

    for i in range(n):
        relX, relY, _objType, objID, scaleX, scaleY = selected[i]
        if relX == 0.0 and relY == 0.0 and objID == 0.0:
            continue  # empty slot, leave as zeros
        out[i, 0] = relX
        out[i, 1] = relY
        out[i, 2] = float(obj_type_index(int(objID)))
        out[i, 3] = scaleX
        out[i, 4] = scaleY

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
    """Converts raw 608-float observations to 157-float processed vectors.

    Usage:
        prep = ObsPreprocessor()
        processed = prep.process_frame(raw_obs)           # (608,) → (157,)
        stacked = prep.process_stacked(raw_stacked, K=4)  # (2432,) → (628,)
    """

    def __init__(self, normalizer: ObsNormalizer | None = None):
        self.normalizer = normalizer

    def process_frame(self, raw_obs: np.ndarray) -> np.ndarray:
        """Process a single raw observation. (608,) → (157,)."""
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
        """Process a stacked raw observation. (K*608,) → (K*157,)."""
        frames = raw_stacked.reshape(stack_size, RAW_OBS_DIM)
        processed = np.stack([self.process_frame(f) for f in frames])
        return processed.reshape(-1)


# ---------- normalization computation ----------

def compute_normalizer(shard_dir: str | Path, stack_size: int = 4) -> ObsNormalizer:
    """Compute per-feature mean/std from all shards in shard_dir.

    Only computes stats over continuous features; binary/categorical features get
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

    # force binary/categorical features to mean=0, std=1
    mean_f32 = mean.astype(np.float32)
    std_f32 = std.astype(np.float32)
    binary_mask = ~CONTINUOUS_MASK
    mean_f32[binary_mask] = 0.0
    std_f32[binary_mask] = 1.0
    # clamp std to avoid div-by-zero on near-constant features
    std_f32 = np.maximum(std_f32, 1e-6)

    return ObsNormalizer(mean=mean_f32, std=std_f32)
