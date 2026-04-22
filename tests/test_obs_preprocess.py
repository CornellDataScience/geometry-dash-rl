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
    N_OBJ_TYPES,
    EMBED_DIM,
    BLOCK_TYPE_IDX,
    obj_type_index,
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
    assert PROCESSED_FRAME_DIM == 157
    assert PROCESSED_FRAME_DIM == PROCESSED_PLAYER_DIM + N_SELECTED_OBJECTS * PROCESSED_OBJ_DIM


def test_processed_player_dim():
    assert PROCESSED_PLAYER_DIM == 7  # y, vy, dx, on_ground, mode_onehot(4)


def test_processed_obj_dim():
    assert PROCESSED_OBJ_DIM == 5  # relX, relY, type_id, scaleX, scaleY


def test_embedding_constants():
    assert N_OBJ_TYPES == 42
    assert EMBED_DIM == 8
    assert BLOCK_TYPE_IDX == 41


# --- type index mapping ---

def test_spike_types():
    assert obj_type_index(8) == 0   # spikeUp
    assert obj_type_index(9) == 1   # spikeDown
    assert obj_type_index(39) == 2  # spike2
    assert obj_type_index(103) == 3 # spike3
    assert obj_type_index(392) == 4 # spikeSmall
    assert obj_type_index(421) == 5 # spikeTiny


def test_saw_types():
    assert obj_type_index(88) == 6  # sawblade
    assert obj_type_index(89) == 7  # sawbladeLg
    assert obj_type_index(98) == 8  # sawbladeMed


def test_orb_types():
    assert obj_type_index(36) == 9   # yellowOrb
    assert obj_type_index(84) == 10  # pinkOrb
    assert obj_type_index(141) == 11 # gravOrb
    assert obj_type_index(1022) == 12 # greenOrb
    assert obj_type_index(1330) == 13 # redOrb
    assert obj_type_index(1594) == 14 # dashOrb
    assert obj_type_index(1704) == 15 # dropOrb
    assert obj_type_index(3005) == 16 # spiderOrb


def test_pad_types():
    assert obj_type_index(35) == 17  # yellowPad
    assert obj_type_index(67) == 18  # pinkPad
    assert obj_type_index(140) == 19 # gravPad
    assert obj_type_index(1332) == 20 # redPad
    assert obj_type_index(3027) == 21 # spiderPad


def test_portal_types():
    assert obj_type_index(10) == 22  # gravPortalDown
    assert obj_type_index(11) == 23  # gravPortalUp
    assert obj_type_index(12) == 24  # shipPortal
    assert obj_type_index(13) == 25  # cubePortal
    assert obj_type_index(99) == 26  # ballPortal
    assert obj_type_index(286) == 27 # ufoPortal
    assert obj_type_index(660) == 28 # wavePortal
    assert obj_type_index(745) == 29 # robotPortal
    assert obj_type_index(1331) == 30 # spiderPortal
    assert obj_type_index(1933) == 31 # swingPortal
    assert obj_type_index(45) == 32  # mirrorOn
    assert obj_type_index(46) == 33  # mirrorOff
    assert obj_type_index(47) == 34  # bigPortal
    assert obj_type_index(101) == 35 # miniPortal
    assert obj_type_index(200) == 36 # speedSlow
    assert obj_type_index(201) == 37 # speedNorm
    assert obj_type_index(202) == 38 # speedFast
    assert obj_type_index(203) == 39 # speedVFast
    assert obj_type_index(1334) == 40 # speedVSlow


def test_block_types():
    for block_id in [1, 2, 3, 4, 5, 6, 7, 40, 83, 289, 291]:
        assert obj_type_index(block_id) == BLOCK_TYPE_IDX


def test_unknown_id_defaults_to_block():
    assert obj_type_index(99999) == BLOCK_TYPE_IDX


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
    assert obj[2] == 0.0    # type_id for spikeUp
    assert obj[3] == 1.0    # scaleX
    assert obj[4] == 1.0    # scaleY


def test_encode_yellow_orb():
    selected = np.zeros((30, 6), dtype=np.float32)
    selected[0] = [100.0, 20.0, 0.0, 36.0, 1.5, 1.5]  # yellow orb
    encoded = _encode_objects(selected)
    assert encoded[0, 2] == 9.0  # type_id for yellowOrb


def test_encode_gravity_orb():
    selected = np.zeros((30, 6), dtype=np.float32)
    selected[0] = [100.0, 20.0, 0.0, 141.0, 1.0, 1.0]  # gravity orb
    encoded = _encode_objects(selected)
    assert encoded[0, 2] == 11.0  # type_id for gravOrb


def test_encode_ship_portal():
    selected = np.zeros((30, 6), dtype=np.float32)
    selected[0] = [200.0, 0.0, 0.0, 12.0, 1.0, 1.0]  # ship portal
    encoded = _encode_objects(selected)
    assert encoded[0, 2] == 24.0  # type_id for shipPortal


def test_encode_unknown_id():
    selected = np.zeros((30, 6), dtype=np.float32)
    selected[0] = [50.0, 10.0, 0.0, 99999.0, 1.0, 1.0]
    encoded = _encode_objects(selected)
    assert encoded[0, 2] == float(BLOCK_TYPE_IDX)


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


def test_continuous_mask_objects():
    # first object starts at index 7
    base = PROCESSED_PLAYER_DIM  # 7
    assert CONTINUOUS_MASK[base + 0] == True   # relX
    assert CONTINUOUS_MASK[base + 1] == True   # relY
    assert CONTINUOUS_MASK[base + 2] == False  # type_id (not continuous)
    assert CONTINUOUS_MASK[base + 3] == True   # scaleX
    assert CONTINUOUS_MASK[base + 4] == True   # scaleY
