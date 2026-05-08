"""Pure-Python tests for V4 SHM adapter (header + mirror + frame seq + ring).

Fakes the C++ mod by creating a SHM segment with the v4 layout and verifies
the adapter reads/writes correctly. Does NOT require the game to be running.
"""
from __future__ import annotations
import struct
import threading
import time
import uuid

import numpy as np
import pytest
from multiprocessing import shared_memory

from gdrl.env.geode_ipc_v3 import (
    GeodeV4Adapter,
    GeodeIPCV4Config,
    OBS_DIM,
    RING_CAPACITY,
    FRAME_W, FRAME_H, FRAME_BYTES,
    FRAME_SLOT_SIZE,
    total_shm_size,
    _HEADER_FMT, _HEADER_SIZE,
    _MIRROR_OBS_OFFSET,
    _MIRROR_ACTION_OFFSET,
    _MIRROR_CTRL_OFFSET,
    _MIRROR_PLAYER_INPUT_OFFSET,
    _MIRROR_LEVEL_DONE_OFFSET,
    _MIRROR_REQ_LEVEL_ID_OFFSET,
    _MIRROR_RESET_PCT_OFFSET,
    _MIRROR_FRAME_SEQ_OFFSET,
    _MIRROR_FRAME_OFFSET,
    _RING_OFFSET,
    _FRAME_TICK_OFFSET, _FRAME_OBS_OFFSET, _FRAME_PLAYER_INPUT_OFFSET,
    _FRAME_LEVEL_DONE_OFFSET, _FRAME_IS_DEAD_OFFSET, _FRAME_EPISODE_OFFSET,
)


@pytest.fixture
def fake_shm():
    name = f"gdrl_test_{uuid.uuid4().hex[:8]}"
    shm = shared_memory.SharedMemory(name=name, create=True, size=total_shm_size())
    yield name, shm
    try:
        shm.close()
        shm.unlink()
    except Exception:
        pass


def _write_header(shm, *, version=4, tick=0, obs_dim=OBS_DIM, ring_cap=RING_CAPACITY,
                  write_idx=0, dropped=0, ep=0, frame_w=FRAME_W, frame_h=FRAME_H,
                  frame_c=1, level_length=2000.0, current_level_id=1):
    struct.pack_into(_HEADER_FMT, shm.buf, 0,
                     version, tick, obs_dim, ring_cap, write_idx, dropped, ep,
                     frame_w, frame_h, frame_c, 0, level_length, current_level_id)


def test_header_size_and_total_size():
    assert _HEADER_SIZE == 40
    assert total_shm_size() == _RING_OFFSET + RING_CAPACITY * FRAME_SLOT_SIZE


def test_offsets_consistent():
    assert _MIRROR_OBS_OFFSET == _HEADER_SIZE
    assert _MIRROR_ACTION_OFFSET == _HEADER_SIZE + OBS_DIM * 4
    assert _MIRROR_CTRL_OFFSET == _MIRROR_ACTION_OFFSET + 1
    assert _MIRROR_REQ_LEVEL_ID_OFFSET == _MIRROR_LEVEL_DONE_OFFSET + 1
    assert _MIRROR_RESET_PCT_OFFSET == _MIRROR_REQ_LEVEL_ID_OFFSET + 4
    # frame mirror after 15-byte reserved padding
    assert _MIRROR_FRAME_SEQ_OFFSET == _MIRROR_RESET_PCT_OFFSET + 1 + 15
    assert _MIRROR_FRAME_OFFSET == _MIRROR_FRAME_SEQ_OFFSET + 4


def test_version_check(fake_shm):
    name, shm = fake_shm
    _write_header(shm, version=3)
    ad = GeodeV4Adapter(GeodeIPCV4Config(shm_name=name))
    with pytest.raises(RuntimeError, match="version mismatch"):
        ad.verify_version()
    ad.close()


def test_basic_reads_and_action_write(fake_shm):
    name, shm = fake_shm
    _write_header(shm, tick=42, ep=7, level_length=1500.0, current_level_id=2)
    obs = np.linspace(0.0, 100.0, OBS_DIM, dtype=np.float32)
    shm.buf[_MIRROR_OBS_OFFSET:_MIRROR_OBS_OFFSET + OBS_DIM * 4] = obs.tobytes()

    ad = GeodeV4Adapter(GeodeIPCV4Config(shm_name=name))
    ad.verify_version()
    assert ad.read_tick() == 42
    assert ad.read_episode_id() == 7
    assert ad.read_level_length() == pytest.approx(1500.0)
    assert ad.read_current_level_id() == 2
    np.testing.assert_array_equal(ad.read_obs(), obs)

    ad.send_action(1)
    assert shm.buf[_MIRROR_ACTION_OFFSET] == 1
    ad.close()


def test_reset_with_percent(fake_shm):
    name, shm = fake_shm
    _write_header(shm)
    ad = GeodeV4Adapter(GeodeIPCV4Config(shm_name=name))
    ad.send_reset(percent=23)
    assert shm.buf[_MIRROR_RESET_PCT_OFFSET] == 23
    assert shm.buf[_MIRROR_CTRL_OFFSET] & 0x01
    ad.close()


def test_load_level_sets_bit_and_id(fake_shm):
    name, shm = fake_shm
    _write_header(shm)
    ad = GeodeV4Adapter(GeodeIPCV4Config(shm_name=name))
    ad.send_load_level(3)
    assert struct.unpack_from('<I', shm.buf, _MIRROR_REQ_LEVEL_ID_OFFSET)[0] == 3
    assert shm.buf[_MIRROR_CTRL_OFFSET] & 0x02
    ad.close()


def test_frame_seq_lock_free_read(fake_shm):
    name, shm = fake_shm
    _write_header(shm)
    pixels = np.arange(FRAME_BYTES, dtype=np.uint8) % 256
    shm.buf[_MIRROR_FRAME_OFFSET:_MIRROR_FRAME_OFFSET + FRAME_BYTES] = pixels.tobytes()
    struct.pack_into('<I', shm.buf, _MIRROR_FRAME_SEQ_OFFSET, 1)

    ad = GeodeV4Adapter(GeodeIPCV4Config(shm_name=name))
    f = ad.read_frame()
    assert f.shape == (FRAME_H, FRAME_W)
    assert f.dtype == np.uint8
    np.testing.assert_array_equal(f.ravel(), pixels)
    ad.close()


def test_frame_seq_zero_raises(fake_shm):
    name, shm = fake_shm
    _write_header(shm)
    # leave seq=0 => no frame published
    ad = GeodeV4Adapter(GeodeIPCV4Config(shm_name=name))
    with pytest.raises(RuntimeError, match="seq=0"):
        ad.read_frame()
    ad.close()


def test_frame_seq_concurrent_write(fake_shm):
    """Stress: writer flips pixels and seq concurrently. Reader must never
    return a torn frame (mixed pixels from two seqs)."""
    name, shm = fake_shm
    _write_header(shm)
    # initialize seq=1 so first read can succeed
    p0 = np.full(FRAME_BYTES, 100, dtype=np.uint8)
    shm.buf[_MIRROR_FRAME_OFFSET:_MIRROR_FRAME_OFFSET + FRAME_BYTES] = p0.tobytes()
    struct.pack_into('<I', shm.buf, _MIRROR_FRAME_SEQ_OFFSET, 1)

    stop = threading.Event()

    def writer():
        seq = 2
        while not stop.is_set():
            v = np.uint8(seq & 0xff)
            arr = np.full(FRAME_BYTES, v, dtype=np.uint8)
            shm.buf[_MIRROR_FRAME_OFFSET:_MIRROR_FRAME_OFFSET + FRAME_BYTES] = arr.tobytes()
            struct.pack_into('<I', shm.buf, _MIRROR_FRAME_SEQ_OFFSET, seq)
            seq += 1

    ad = GeodeV4Adapter(GeodeIPCV4Config(shm_name=name))
    t = threading.Thread(target=writer)
    t.start()
    try:
        t0 = time.time()
        torn = 0
        reads = 0
        while time.time() - t0 < 0.3:
            try:
                f = ad.read_frame()
            except RuntimeError:
                continue
            reads += 1
            # all pixels in a valid frame must be identical (writer always
            # fills with one value matching its seq mod 256)
            if not (f == f[0, 0]).all():
                torn += 1
    finally:
        stop.set()
        t.join()
        ad.close()
    assert reads > 0, "writer/reader didn't run"
    assert torn == 0, f"detected {torn}/{reads} torn frames"


def test_ring_drain(fake_shm):
    name, shm = fake_shm
    _write_header(shm, write_idx=0)
    ad = GeodeV4Adapter(GeodeIPCV4Config(shm_name=name))
    # first call returns nothing but seeds last_consumed
    frames, dropped = ad.drain_ring()
    assert frames == [] and dropped == 0

    # write 3 slots
    for i in range(3):
        base = _RING_OFFSET + i * FRAME_SLOT_SIZE
        struct.pack_into('<I', shm.buf, base + _FRAME_TICK_OFFSET, 100 + i)
        obs = np.full(OBS_DIM, float(i), dtype=np.float32)
        shm.buf[base + _FRAME_OBS_OFFSET:base + _FRAME_OBS_OFFSET + OBS_DIM * 4] = obs.tobytes()
        shm.buf[base + _FRAME_PLAYER_INPUT_OFFSET] = i % 2
        shm.buf[base + _FRAME_LEVEL_DONE_OFFSET] = 0
        shm.buf[base + _FRAME_IS_DEAD_OFFSET] = 0
        struct.pack_into('<I', shm.buf, base + _FRAME_EPISODE_OFFSET, 1)
    struct.pack_into(_HEADER_FMT, shm.buf, 0,
                     4, 102, OBS_DIM, RING_CAPACITY, 3, 0, 1,
                     FRAME_W, FRAME_H, 1, 0, 2000.0, 1)

    frames, dropped = ad.drain_ring()
    assert dropped == 0
    assert [f.tick for f in frames] == [100, 101, 102]
    assert frames[1].player_input == 1
    ad.close()
