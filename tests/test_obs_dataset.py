"""Tests for record_human shard format + obs_dataset frame stacking."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pytest

from gdrl.data.obs_dataset import (
    HumanPlayDataset,
    ShardIndex,
    train_val_split,
    find_shards,
    OBS_DIM,
    SESSION_STRIDE,
)


def _make_shard(path: Path, n: int, episode_id: int = 1, start_tick: int = 0):
    obs = np.zeros((n, OBS_DIM), dtype=np.float32)
    for i in range(n):
        obs[i, 0] = float(start_tick + i)  # encode tick in obs[0] for verification
    np.savez_compressed(
        path,
        obs=obs,
        actions=np.array([i % 3 == 0 for i in range(n)], dtype=np.uint8),
        ticks=np.arange(start_tick, start_tick + n, dtype=np.uint32),
        episode_ids=np.full(n, episode_id, dtype=np.uint32),
        is_dead=np.zeros(n, dtype=np.uint8),
        level_done=np.zeros(n, dtype=np.uint8),
    )


def _make_multi_episode_shard(path: Path, episode_lengths: list[int]):
    total = sum(episode_lengths)
    obs = np.zeros((total, OBS_DIM), dtype=np.float32)
    eps = np.zeros(total, dtype=np.uint32)
    actions = np.zeros(total, dtype=np.uint8)
    ticks = np.zeros(total, dtype=np.uint32)
    pos = 0
    for ep_id, length in enumerate(episode_lengths, start=1):
        for k in range(length):
            obs[pos, 0] = float(pos)
            eps[pos] = ep_id
            ticks[pos] = pos
            pos += 1
    np.savez_compressed(
        path, obs=obs, actions=actions, ticks=ticks,
        episode_ids=eps,
        is_dead=np.zeros(total, dtype=np.uint8),
        level_done=np.zeros(total, dtype=np.uint8),
    )


def test_shard_index_concatenates(tmp_path):
    _make_shard(tmp_path / "shard_00000.npz", 10, episode_id=1, start_tick=0)
    _make_shard(tmp_path / "shard_00001.npz", 7, episode_id=2, start_tick=10)
    # single "session" = all shards in tmp_path
    sessions = find_shards(tmp_path)
    idx = ShardIndex(sessions)
    assert len(idx) == 17
    obs0, _, ep0 = idx.get(0)
    # session_idx=0 so effective == raw
    assert obs0[0] == 0.0 and ep0 == 1
    obs10, _, ep10 = idx.get(10)
    assert obs10[0] == 10.0 and ep10 == 2
    obs16, _, ep16 = idx.get(16)
    assert obs16[0] == 16.0 and ep16 == 2


def test_find_shards_legacy_single_dir(tmp_path):
    _make_shard(tmp_path / "shard_00000.npz", 5)
    _make_shard(tmp_path / "shard_00001.npz", 5)
    sessions = find_shards(tmp_path)
    assert len(sessions) == 1
    assert len(sessions[0]) == 2


def test_find_shards_multiple_session_subdirs(tmp_path):
    (tmp_path / "20260101_000000").mkdir()
    (tmp_path / "20260102_000000").mkdir()
    _make_shard(tmp_path / "20260101_000000" / "shard_00000.npz", 5)
    _make_shard(tmp_path / "20260102_000000" / "shard_00000.npz", 5)
    sessions = find_shards(tmp_path)
    assert len(sessions) == 2
    # sorted lexicographically so earlier timestamp first
    assert "20260101" in str(sessions[0][0])
    assert "20260102" in str(sessions[1][0])


def test_shard_index_session_namespacing(tmp_path):
    """Two sessions with colliding raw episode_ids get different effective ids."""
    (tmp_path / "sess_a").mkdir()
    (tmp_path / "sess_b").mkdir()
    _make_shard(tmp_path / "sess_a" / "shard_00000.npz", 5, episode_id=1)
    _make_shard(tmp_path / "sess_b" / "shard_00000.npz", 5, episode_id=1)
    sessions = find_shards(tmp_path)
    idx = ShardIndex(sessions)
    assert len(idx) == 10
    _, _, ep_a = idx.get(0)
    _, _, ep_b = idx.get(5)
    assert ep_a == 1  # session 0: 0*STRIDE + 1
    assert ep_b == SESSION_STRIDE + 1  # session 1: 1*STRIDE + 1
    assert ep_a != ep_b


def test_frame_stacking_does_not_cross_session_boundary(tmp_path):
    """Even if raw episode_ids match, frames from different sessions must not mix."""
    (tmp_path / "sess_a").mkdir()
    (tmp_path / "sess_b").mkdir()
    # sess_a frames have obs[0] in [0..4], sess_b frames have obs[0] in [100..104]
    _make_shard(tmp_path / "sess_a" / "shard_00000.npz", 5, episode_id=1, start_tick=0)
    _make_shard(tmp_path / "sess_b" / "shard_00000.npz", 5, episode_id=1, start_tick=100)
    ds = HumanPlayDataset(tmp_path, stack_size=4)
    # idx 5 is the first frame of session b
    x, _ = ds[5]
    x = x.numpy().reshape(4, OBS_DIM)
    # newest = 5 -> sess_b frame 0 -> obs[0] = 100
    assert x[3, 0] == 100.0
    # older slots must NOT pull sess_a frames 4,3,2 (values 4,3,2) —
    # they must replicate sess_b frame 0 (value 100)
    assert x[2, 0] == 100.0
    assert x[1, 0] == 100.0
    assert x[0, 0] == 100.0


def test_dataset_basic_load(tmp_path):
    _make_shard(tmp_path / "shard_00000.npz", 20)
    ds = HumanPlayDataset(tmp_path, stack_size=4)
    assert len(ds) == 20
    x, y = ds[5]
    assert x.shape == (4 * OBS_DIM,)
    assert y.dtype.is_floating_point


def test_frame_stacking_order_oldest_to_newest(tmp_path):
    _make_shard(tmp_path / "shard_00000.npz", 10)
    ds = HumanPlayDataset(tmp_path, stack_size=4)
    x, _ = ds[5]
    x = x.numpy().reshape(4, OBS_DIM)
    # frames 2,3,4,5 in oldest→newest order
    assert x[0, 0] == 2.0
    assert x[1, 0] == 3.0
    assert x[2, 0] == 4.0
    assert x[3, 0] == 5.0


def test_frame_stacking_replicates_at_episode_start(tmp_path):
    _make_shard(tmp_path / "shard_00000.npz", 10)
    ds = HumanPlayDataset(tmp_path, stack_size=4)
    x, _ = ds[1]
    x = x.numpy().reshape(4, OBS_DIM)
    # newest is frame 1, frame 0 exists, frames -1/-2 should replicate
    assert x[3, 0] == 1.0
    assert x[2, 0] == 0.0
    # older slots should be replicated to oldest valid in episode (frame 0)
    assert x[1, 0] == 0.0
    assert x[0, 0] == 0.0


def test_frame_stacking_does_not_cross_episode_boundary(tmp_path):
    _make_multi_episode_shard(tmp_path / "shard_00000.npz", [5, 5])
    ds = HumanPlayDataset(tmp_path, stack_size=4)
    # idx 5 is start of episode 2 (first frame of new episode)
    x, _ = ds[5]
    x = x.numpy().reshape(4, OBS_DIM)
    # newest = 5
    assert x[3, 0] == 5.0
    # all older slots should NOT pull from frame 4 (different episode)
    # they should replicate frame 5 itself
    assert x[2, 0] == 5.0
    assert x[1, 0] == 5.0
    assert x[0, 0] == 5.0


def test_frame_stacking_partial_episode_lookback(tmp_path):
    _make_multi_episode_shard(tmp_path / "shard_00000.npz", [5, 5])
    ds = HumanPlayDataset(tmp_path, stack_size=4)
    # idx 6 is second frame of episode 2; can look back to idx 5 only
    x, _ = ds[6]
    x = x.numpy().reshape(4, OBS_DIM)
    assert x[3, 0] == 6.0
    assert x[2, 0] == 5.0
    # frame 4 is in episode 1 — should NOT be used; replicate 5 instead
    assert x[1, 0] == 5.0
    assert x[0, 0] == 5.0


def test_train_val_split_disjoint(tmp_path):
    _make_shard(tmp_path / "shard_00000.npz", 100)
    train, val = train_val_split(tmp_path, val_fraction=0.2, seed=42, stack_size=2)
    assert len(train) == 80
    assert len(val) == 20
    overlap = set(train.indices.tolist()) & set(val.indices.tolist())
    assert len(overlap) == 0


def test_action_label_matches_index(tmp_path):
    _make_shard(tmp_path / "shard_00000.npz", 12)
    ds = HumanPlayDataset(tmp_path, stack_size=1)
    for i in range(12):
        _, y = ds[i]
        expected = float(i % 3 == 0)
        assert float(y) == expected


def test_shard_writer_creates_session_subdir(tmp_path):
    """ShardWriter must place shards inside a named subdir, not the root."""
    from gdrl.data.record_human import ShardWriter

    writer = ShardWriter(tmp_path, shard_size=5, session_name="test_session")
    assert writer.out_dir == tmp_path / "test_session"
    assert writer.out_dir.is_dir()
    assert list(tmp_path.glob("shard_*.npz")) == []  # nothing in root


def test_shard_writer_does_not_clobber_other_sessions(tmp_path):
    """Two ShardWriters with different session names coexist cleanly."""
    from gdrl.data.record_human import ShardWriter

    class _F:
        def __init__(self, v):
            self.obs = np.full(OBS_DIM, float(v), dtype=np.float32)
            self.tick = v
            self.episode_id = 1
            self.is_dead = 0
            self.level_done = 0
            self.player_input = 0

    w1 = ShardWriter(tmp_path, shard_size=3, session_name="sess_a")
    for i in range(3):
        w1.append(_F(i))
    w2 = ShardWriter(tmp_path, shard_size=3, session_name="sess_b")
    for i in range(100, 103):
        w2.append(_F(i))

    assert (tmp_path / "sess_a" / "shard_00000.npz").exists()
    assert (tmp_path / "sess_b" / "shard_00000.npz").exists()
    a = np.load(tmp_path / "sess_a" / "shard_00000.npz")
    b = np.load(tmp_path / "sess_b" / "shard_00000.npz")
    assert a["obs"][0, 0] == 0.0
    assert b["obs"][0, 0] == 100.0
