"""Pure-Python tests for V3 ring buffer adapter.

Creates a fake SHM segment, writes FrameSlots in the same layout the C++ mod
would, then verifies the adapter reads them back correctly. Does NOT require
the game to be running.
"""
from __future__ import annotations
import struct
import uuid

import numpy as np
import pytest
from multiprocessing import shared_memory

from gdrl.env.geode_ipc_v3 import (
    GeodeV3Adapter,
    GeodeIPCV3Config,
    OBS_DIM,
    RING_CAPACITY,
    FRAME_SLOT_SIZE,
    total_shm_size,
    _HEADER_FMT,
    _HEADER_SIZE,
    _MIRROR_OBS_OFFSET,
    _MIRROR_ACTION_OFFSET,
    _MIRROR_CTRL_OFFSET,
    _MIRROR_PLAYER_INPUT_OFFSET,
    _MIRROR_LEVEL_DONE_OFFSET,
    _RING_OFFSET,
    _FRAME_TICK_OFFSET,
    _FRAME_OBS_OFFSET,
    _FRAME_PLAYER_INPUT_OFFSET,
    _FRAME_LEVEL_DONE_OFFSET,
    _FRAME_IS_DEAD_OFFSET,
    _FRAME_EPISODE_OFFSET,
)


@pytest.fixture
def fake_shm():
    """Create a fresh SHM segment sized for V3, yield (name, buf, shm)."""
    name = f"gdrl_test_{uuid.uuid4().hex[:8]}"
    size = total_shm_size()
    shm = shared_memory.SharedMemory(name=name, create=True, size=size)
    # zero it
    shm.buf[:size] = b"\x00" * size
    try:
        yield name, shm.buf, shm
    finally:
        shm.close()
        try:
            shm.unlink()
        except Exception:
            pass


def _write_header(buf, version=3, tick=0, obs_dim=OBS_DIM,
                  ring_cap=RING_CAPACITY, write_idx=0, dropped=0, ep=0):
    struct.pack_into(_HEADER_FMT, buf, 0,
                     version, tick, obs_dim, ring_cap, write_idx, dropped, ep)


def _write_slot(buf, slot_idx, tick, obs, player_input=0, level_done=0,
                is_dead=0, episode_id=0):
    base = _RING_OFFSET + (slot_idx % RING_CAPACITY) * FRAME_SLOT_SIZE
    struct.pack_into('<I', buf, base + _FRAME_TICK_OFFSET, tick)
    obs_arr = np.asarray(obs, dtype=np.float32)
    assert obs_arr.shape == (OBS_DIM,)
    buf[base + _FRAME_OBS_OFFSET:base + _FRAME_OBS_OFFSET + OBS_DIM * 4] = obs_arr.tobytes()
    buf[base + _FRAME_PLAYER_INPUT_OFFSET] = player_input
    buf[base + _FRAME_LEVEL_DONE_OFFSET] = level_done
    buf[base + _FRAME_IS_DEAD_OFFSET] = is_dead
    struct.pack_into('<I', buf, base + _FRAME_EPISODE_OFFSET, episode_id)


def _bump_write_idx(buf, new_idx):
    # write_idx is at offset 4+4+2+2 = 12
    struct.pack_into('<I', buf, 12, new_idx)


def test_total_size_matches_layout():
    expected = _RING_OFFSET + RING_CAPACITY * FRAME_SLOT_SIZE
    assert total_shm_size() == expected


def test_header_size_is_24():
    assert _HEADER_SIZE == 24


def test_frame_slot_size():
    # tick(4) + obs(2432) + 4 u8 + ep(4) + 4 u8 pad = 2448
    assert FRAME_SLOT_SIZE == 2448


def test_verify_version_ok(fake_shm):
    name, buf, _ = fake_shm
    _write_header(buf)
    ad = GeodeV3Adapter(GeodeIPCV3Config(shm_name=name))
    ad.verify_version()  # should not raise
    ad.close()


def test_verify_version_mismatch(fake_shm):
    name, buf, _ = fake_shm
    _write_header(buf, version=2)
    ad = GeodeV3Adapter(GeodeIPCV3Config(shm_name=name))
    with pytest.raises(RuntimeError, match="version mismatch"):
        ad.verify_version()
    ad.close()


def test_drain_ring_no_backfill_first_call(fake_shm):
    name, buf, _ = fake_shm
    _write_header(buf, write_idx=10)
    ad = GeodeV3Adapter(GeodeIPCV3Config(shm_name=name))
    frames, dropped = ad.drain_ring()
    assert frames == []
    assert dropped == 0
    ad.close()


def test_drain_ring_reads_new_frames(fake_shm):
    name, buf, _ = fake_shm
    _write_header(buf, write_idx=0)
    ad = GeodeV3Adapter(GeodeIPCV3Config(shm_name=name))
    frames, _ = ad.drain_ring()  # initialize
    assert frames == []

    # write 5 frames
    for i in range(5):
        obs = np.full(OBS_DIM, float(i), dtype=np.float32)
        _write_slot(buf, i, tick=100 + i, obs=obs,
                    player_input=i % 2, episode_id=1)
    _bump_write_idx(buf, 5)

    frames, dropped = ad.drain_ring()
    assert dropped == 0
    assert len(frames) == 5
    for i, f in enumerate(frames):
        assert f.tick == 100 + i
        assert f.player_input == i % 2
        assert f.episode_id == 1
        assert np.allclose(f.obs, float(i))
    ad.close()


def test_drain_ring_drops_when_overflowing(fake_shm):
    name, buf, _ = fake_shm
    _write_header(buf, write_idx=0)
    ad = GeodeV3Adapter(GeodeIPCV3Config(shm_name=name))
    ad.drain_ring()

    # write 2x ring capacity worth of frames
    n = RING_CAPACITY * 2
    for i in range(n):
        obs = np.full(OBS_DIM, float(i), dtype=np.float32)
        _write_slot(buf, i, tick=i, obs=obs)
    _bump_write_idx(buf, n)

    frames, dropped = ad.drain_ring()
    assert dropped == n - RING_CAPACITY
    assert len(frames) == RING_CAPACITY
    # the frames returned should be the most recent RING_CAPACITY
    expected_first_tick = n - RING_CAPACITY
    assert frames[0].tick == expected_first_tick
    assert frames[-1].tick == n - 1
    ad.close()


def test_drain_ring_incremental(fake_shm):
    name, buf, _ = fake_shm
    _write_header(buf, write_idx=0)
    ad = GeodeV3Adapter(GeodeIPCV3Config(shm_name=name))
    ad.drain_ring()

    # batch 1
    for i in range(3):
        _write_slot(buf, i, tick=i, obs=np.zeros(OBS_DIM, dtype=np.float32))
    _bump_write_idx(buf, 3)
    f1, _ = ad.drain_ring()
    assert len(f1) == 3

    # batch 2: more frames
    for i in range(3, 7):
        _write_slot(buf, i, tick=i, obs=np.zeros(OBS_DIM, dtype=np.float32))
    _bump_write_idx(buf, 7)
    f2, _ = ad.drain_ring()
    assert len(f2) == 4
    assert [f.tick for f in f2] == [3, 4, 5, 6]
    ad.close()


def test_send_action_writes_mirror(fake_shm):
    name, buf, _ = fake_shm
    _write_header(buf)
    ad = GeodeV3Adapter(GeodeIPCV3Config(shm_name=name))
    ad.send_action(1)
    assert buf[_MIRROR_ACTION_OFFSET] == 1
    ad.send_action(0)
    assert buf[_MIRROR_ACTION_OFFSET] == 0
    ad.close()


def test_send_reset_sets_ctrl_bit(fake_shm):
    name, buf, _ = fake_shm
    _write_header(buf)
    ad = GeodeV3Adapter(GeodeIPCV3Config(shm_name=name))
    ad.send_reset()
    assert buf[_MIRROR_CTRL_OFFSET] & 0x01
    ad.close()


def test_read_obs_mirror(fake_shm):
    name, buf, _ = fake_shm
    _write_header(buf)
    obs = np.arange(OBS_DIM, dtype=np.float32)
    buf[_MIRROR_OBS_OFFSET:_MIRROR_OBS_OFFSET + OBS_DIM * 4] = obs.tobytes()
    ad = GeodeV3Adapter(GeodeIPCV3Config(shm_name=name))
    out = ad.read_obs()
    assert np.array_equal(out, obs)
    ad.close()
