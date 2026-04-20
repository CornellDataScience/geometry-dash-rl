"""Tests for observation preprocessor."""
from __future__ import annotations

import numpy as np
import pytest

from gdrl.model.obs_preprocess import (
    ObsPreprocessor,
    ObsNormalizer,
    PROCESSED_FRAME_DIM,
    PROCESSED_PLAYER_DIM,
    PROCESSED_OBJ_DIM,
    N_SELECTED_OBJECTS,
    RAW_OBS_DIM,
    RAW_OBJ_START,
    RAW_FLOATS_PER_OBJ,
    RAW_MAX_OBJECTS,
    N_CATEGORIES,
    CAT_HAZARD,
    CAT_ORB,
    CAT_PAD,
    CAT_PORTAL,
    CAT_BLOCK,
    obj_category,
    _subsample_objects,
    _encode_objects,
    _encode_player,
    CONTINUOUS_MASK,
)


def _make_raw_obs(
    x=100.0, y=200.0, vy=5.0, dx=2.5, on_ground=1.0, is_dead=0.0, mode=0,
    objects=None,
) -> np.ndarray:
    obs = np.zeros(RAW_OBS_DIM, dtype=np.float32)
    obs[0] = x
    obs[1] = y
    obs[2] = vy
    obs[3] = dx
    obs[4] = on_ground
    obs[5] = is_dead
    obs[6] = 1.0
    obs[7] = float(mode)
    if objects is not None:
        for i, obj in enumerate(objects):
            if i >= RAW_MAX_OBJECTS:
                break
            base = RAW_OBJ_START + i * RAW_FLOATS_PER_OBJ
            obs[base:base + RAW_FLOATS_PER_OBJ] = obj
    return obs


# --- dimension constants ---

def test_processed_frame_dim():
    assert PROCESSED_FRAME_DIM == 277
    assert PROCESSED_FRAME_DIM == PROCESSED_PLAYER_DIM + N_SELECTED_OBJECTS * PROCESSED_OBJ_DIM


def test_processed_player_dim():
    assert PROCESSED_PLAYER_DIM == 7  # y, vy, dx, on_ground, mode_onehot(4)


def test_processed_obj_dim():
    assert PROCESSED_OBJ_DIM == 9  # relX, relY, cat_onehot(5), scaleX, scaleY


# --- category mapping ---

def test_spike_is_hazard():
    for spike_id in [8, 9, 39, 103, 392, 421]:
        assert obj_category(spike_id) == CAT_HAZARD

def test_saw_is_hazard():
    for saw_id in [88, 89, 98]:
        assert obj_category(saw_id) == CAT_HAZARD

def test_orb_category():
    for orb_id in [36, 84, 141, 1022, 1330, 1594, 1704, 3005]:
        assert obj_category(orb_id) == CAT_ORB

def test_pad_category():
    for pad_id in [35, 67, 140, 1332, 3027]:
        assert obj_category(pad_id) == CAT_PAD

def test_portal_category():
    for portal_id in [12, 13, 200, 201, 660, 1331]:
        assert obj_category(portal_id) == CAT_PORTAL

def test_block_category():
    for block_id in [1, 2, 3, 40, 83, 289]:
        assert obj_category(block_id) == CAT_BLOCK

def test_unknown_id_defaults_to_block():
    assert obj_category(99999) == CAT_BLOCK


# --- player encoding ---

def test_encode_player_basic():
    raw = _make_raw_obs(x=500, y=200, vy=5, dx=2.5, on_ground=1, mode=0)
    p = _encode_player(raw)
    assert p.shape == (PROCESSED_PLAYER_DIM,)
    assert p[0] == 200.0    # y
    assert p[1] == 5.0      # vy
    assert p[2] == 2.5      # dx
    assert p[3] == 1.0      # on_ground
    assert p[4] == 1.0      # mode cube
    assert p[5] == 0.0
    assert p[6] == 0.0


def test_encode_player_ship_mode():
    raw = _make_raw_obs(mode=1)
    p = _encode_player(raw)
    assert p[4] == 0.0
    assert p[5] == 1.0  # ship
    assert p[6] == 0.0


def test_encode_player_drops_absolute_x():
    raw1 = _make_raw_obs(x=100)
    raw2 = _make_raw_obs(x=50000)
    p1 = _encode_player(raw1)
    p2 = _encode_player(raw2)
    np.testing.assert_array_equal(p1, p2)


# --- object subsampling ---

def test_subsample_empty():
    raw = np.zeros((RAW_MAX_OBJECTS, RAW_FLOATS_PER_OBJ), dtype=np.float32)
    result = _subsample_objects(raw)
    assert result.shape == (N_SELECTED_OBJECTS, RAW_FLOATS_PER_OBJ)
    assert np.all(result == 0.0)


def test_subsample_fewer_than_15():
    raw = np.zeros((RAW_MAX_OBJECTS, RAW_FLOATS_PER_OBJ), dtype=np.float32)
    for i in range(5):
        raw[i] = [float(i + 1) * 10, 0, 0, 8, 1, 1]  # relX, relY, type, id, sx, sy
    result = _subsample_objects(raw)
    assert result.shape == (N_SELECTED_OBJECTS, RAW_FLOATS_PER_OBJ)
    # first 5 filled, rest zero
    for i in range(5):
        assert result[i, 0] == float(i + 1) * 10
    assert np.all(result[5:] == 0.0)


def test_subsample_30_objects():
    raw = np.zeros((RAW_MAX_OBJECTS, RAW_FLOATS_PER_OBJ), dtype=np.float32)
    for i in range(50):
        raw[i] = [float(i + 1) * 10, 5.0, 0, 1, 1, 1]
    result = _subsample_objects(raw)
    assert result.shape == (N_SELECTED_OBJECTS, RAW_FLOATS_PER_OBJ)
    # first 15 should be the 15 nearest
    for i in range(15):
        assert result[i, 0] == float(i + 1) * 10
    # remaining 15 should be exponentially spaced (non-zero, increasing)
    for i in range(15, N_SELECTED_OBJECTS):
        if result[i, 0] == 0.0:
            break  # some might not be filled if not enough candidates
        if i > 15:
            assert result[i, 0] > result[i - 1, 0]


# --- object encoding ---

def test_encode_spike():
    selected = np.array([[50.0, 10.0, 0.0, 8.0, 1.0, 1.0]], dtype=np.float32)
    selected = np.vstack([selected, np.zeros((29, 6), dtype=np.float32)])
    encoded = _encode_objects(selected)
    assert encoded.shape == (30, PROCESSED_OBJ_DIM)
    obj = encoded[0]
    assert obj[0] == 50.0   # relX
    assert obj[1] == 10.0   # relY
    assert obj[2 + CAT_HAZARD] == 1.0  # hazard one-hot
    assert obj[2 + CAT_ORB] == 0.0
    assert obj[7] == 1.0    # scaleX
    assert obj[8] == 1.0    # scaleY


def test_encode_orb():
    selected = np.zeros((30, 6), dtype=np.float32)
    selected[0] = [100.0, 20.0, 0.0, 36.0, 1.5, 1.5]  # yellow orb
    encoded = _encode_objects(selected)
    assert encoded[0, 2 + CAT_ORB] == 1.0


def test_encode_empty_slot_stays_zero():
    selected = np.zeros((30, 6), dtype=np.float32)
    encoded = _encode_objects(selected)
    assert np.all(encoded == 0.0)


# --- full preprocessor ---

def test_process_frame_output_shape():
    prep = ObsPreprocessor()
    raw = _make_raw_obs(objects=[[50, 10, 0, 8, 1, 1]])
    result = prep.process_frame(raw)
    assert result.shape == (PROCESSED_FRAME_DIM,)


def test_process_stacked_output_shape():
    prep = ObsPreprocessor()
    raw = np.zeros(RAW_OBS_DIM * 4, dtype=np.float32)
    result = prep.process_stacked(raw, stack_size=4)
    assert result.shape == (PROCESSED_FRAME_DIM * 4,)


def test_process_frame_with_normalization():
    normalizer = ObsNormalizer(
        mean=np.zeros(PROCESSED_FRAME_DIM, dtype=np.float32),
        std=np.ones(PROCESSED_FRAME_DIM, dtype=np.float32) * 100.0,
    )
    prep = ObsPreprocessor(normalizer=normalizer)
    raw = _make_raw_obs(y=200.0, vy=5.0, dx=2.5)
    result = prep.process_frame(raw)
    # continuous features should be scaled by 1/100
    assert abs(result[0] - 200.0 / 100.0) < 0.01  # y normalized
    # binary features should be unchanged
    assert result[3] == 1.0  # on_ground stays 1


def test_normalizer_save_load(tmp_path):
    mean = np.random.randn(PROCESSED_FRAME_DIM).astype(np.float32)
    std = np.abs(np.random.randn(PROCESSED_FRAME_DIM)).astype(np.float32) + 0.1
    norm = ObsNormalizer(mean=mean, std=std)
    path = tmp_path / "test_norm.npz"
    norm.save(path)
    loaded = ObsNormalizer.load(path)
    np.testing.assert_array_almost_equal(loaded.mean, mean)
    np.testing.assert_array_almost_equal(loaded.std, std)


def test_continuous_mask_shape():
    assert CONTINUOUS_MASK.shape == (PROCESSED_FRAME_DIM,)
    # player: first 3 continuous, next 4 binary
    assert CONTINUOUS_MASK[0] == True   # y
    assert CONTINUOUS_MASK[1] == True   # vy
    assert CONTINUOUS_MASK[2] == True   # dx
    assert CONTINUOUS_MASK[3] == False  # on_ground
    assert CONTINUOUS_MASK[4] == False  # mode one-hot
