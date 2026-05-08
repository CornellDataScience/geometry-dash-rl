from __future__ import annotations

import numpy as np

from gdrl.data.dagger_align import align, align_to_press_events, build_human_index, cluster_press_x
from gdrl.data.obs_dataset import OBS_DIM, grounded_jump_labels


def test_build_human_index_uses_grounded_press_labels(tmp_path):
    obs = np.zeros((4, OBS_DIM), dtype=np.float32)
    obs[:, 0] = np.array([10, 20, 30, 40], dtype=np.float32)
    obs[:, 4] = 1.0
    actions = np.array([0, 1, 0, 1], dtype=np.uint8)
    np.savez_compressed(
        tmp_path / "shard_00000.npz",
        obs=obs,
        actions=actions,
        ticks=np.arange(4, dtype=np.uint32),
        episode_ids=np.ones(4, dtype=np.uint32),
        is_dead=np.zeros(4, dtype=np.uint8),
        level_done=np.zeros(4, dtype=np.uint8),
    )

    human_x, human_labels = build_human_index(tmp_path)

    assert human_x.tolist() == [10.0, 20.0, 30.0, 40.0]
    assert human_labels.tolist() == [1, 0, 1, 0]


def test_aligned_labels_are_gated_by_rollout_grounded_state():
    human_x = np.array([10, 20, 30], dtype=np.float32)
    human_labels = np.array([1, 1, 1], dtype=np.uint8)
    rollout_x = np.array([11, 21, 29], dtype=np.float32)
    rollout_obs = np.zeros((3, OBS_DIM), dtype=np.float32)
    rollout_obs[:, 4] = np.array([1, 0, 1], dtype=np.float32)

    aligned = align(rollout_x, human_x, human_labels)
    labels = grounded_jump_labels(rollout_obs, aligned)

    assert aligned.tolist() == [1, 1, 1]
    assert labels.tolist() == [1, 0, 1]


def test_align_to_press_events_uses_x_tolerance():
    human_press_x = np.array([100, 200], dtype=np.float32)
    rollout_x = np.array([81, 95, 120, 150, 181, 199, 221], dtype=np.float32)

    labels = align_to_press_events(rollout_x, human_press_x, x_tolerance=20.0)

    assert labels.tolist() == [0, 1, 0, 0, 0, 1, 0]


def test_align_to_press_events_respects_episode_boundaries():
    human_press_x = np.array([100], dtype=np.float32)
    rollout_x = np.array([95, 105, 94, 106], dtype=np.float32)
    episode_ids = np.array([1, 1, 2, 2], dtype=np.uint32)

    labels = align_to_press_events(
        rollout_x,
        human_press_x,
        x_tolerance=10.0,
        episode_ids=episode_ids,
    )

    assert labels.tolist() == [1, 0, 1, 0]


def test_cluster_press_x_collapses_repeated_playthrough_timings():
    press_x = np.array([462, 467, 483, 929, 935, 1381], dtype=np.float32)

    clustered = cluster_press_x(press_x, gap=80.0)

    assert clustered.tolist() == [467.0, 932.0, 1381.0]
