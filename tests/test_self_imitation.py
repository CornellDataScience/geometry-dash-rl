from __future__ import annotations

import unittest

import numpy as np

from gdrl.teacher.curriculum_reset import CurriculumResetConfig, CurriculumResetPolicy
from gdrl.teacher.self_imitation import SegmentCurriculum, SegmentEliteReplay, SelfImitationConfig


class SegmentEliteReplayTests(unittest.TestCase):
    def test_add_episode_builds_prefixes_for_reached_segments(self):
        cfg = SelfImitationConfig(segment_size=100.0, elite_prefixes=3)
        replay = SegmentEliteReplay(cfg)

        obs = np.zeros((5, 608), dtype=np.float32)
        obs[:, 0] = [0.0, 30.0, 90.0, 150.0, 210.0]
        actions = np.asarray([0, 1, 0, 1, 0], dtype=np.int64)
        post_x = np.asarray([30.0, 90.0, 150.0, 210.0, 240.0], dtype=np.float32)

        added = replay.add_episode(
            obs,
            actions,
            post_x,
            episode_max_x=240.0,
            training_step=1234,
        )

        self.assertEqual(added, 2)
        self.assertEqual(replay.max_segment_index, 1)
        self.assertEqual(replay._prefixes[0][0].prefix_length, 3)
        self.assertEqual(replay._prefixes[1][0].prefix_length, 4)
        np.testing.assert_array_equal(replay._prefixes[1][0].actions, np.asarray([0, 1, 0, 1]))

    def test_keeps_farthest_elites_per_segment(self):
        cfg = SelfImitationConfig(segment_size=100.0, elite_prefixes=2)
        replay = SegmentEliteReplay(cfg)

        obs = np.zeros((4, 608), dtype=np.float32)
        actions = np.asarray([0, 0, 1, 1], dtype=np.int64)
        post_x = np.asarray([40.0, 120.0, 180.0, 240.0], dtype=np.float32)

        replay.add_episode(obs, actions, post_x, episode_max_x=180.0, training_step=1)
        replay.add_episode(obs, actions, post_x, episode_max_x=260.0, training_step=2)
        replay.add_episode(obs, actions, post_x, episode_max_x=220.0, training_step=3)

        stored = replay._prefixes[0]
        self.assertEqual(len(stored), 2)
        self.assertEqual([prefix.episode_max_x for prefix in stored], [260.0, 220.0])


class SegmentCurriculumTests(unittest.TestCase):
    def test_advances_and_backtracks_with_hysteresis(self):
        cfg = SelfImitationConfig(
            segment_size=100.0,
            recent_episodes=5,
            min_mastery_episodes=5,
            promote_threshold=0.8,
            regress_threshold=0.4,
        )
        curriculum = SegmentCurriculum(cfg)

        for _ in range(5):
            curriculum.record_episode(250.0)
        self.assertEqual(curriculum.mastered_segments, 2)
        self.assertGreaterEqual(curriculum.success_rate(1), 0.8)

        for _ in range(5):
            curriculum.record_episode(20.0)
        self.assertEqual(curriculum.mastered_segments, 0)
        self.assertLess(curriculum.success_rate(0), 0.4)

    def test_active_bc_segments_uses_first_segment_before_mastery(self):
        cfg = SelfImitationConfig(segment_size=100.0)
        curriculum = SegmentCurriculum(cfg)

        self.assertEqual(curriculum.active_bc_segments(-1), [])
        self.assertEqual(curriculum.active_bc_segments(0), [0])


class CurriculumResetPolicyTests(unittest.TestCase):
    def test_policy_warms_up_then_samples_mastered_segments(self):
        cfg = CurriculumResetConfig(
            enabled=True,
            segment_size=75.0,
            checkpoint_reset_prob=1.0,
            backtrack_segments=1,
            warmup_episodes=3,
        )
        policy = CurriculumResetPolicy(cfg, seed=0)

        policy.update(mastered_segments=2, episodes_completed=2)
        self.assertIsNone(policy.choose_checkpoint_x())

        policy.update(mastered_segments=2, episodes_completed=3)
        checkpoint_x = policy.choose_checkpoint_x()
        self.assertIn(checkpoint_x, {75.0, 150.0})
        self.assertEqual(policy.last_mode, "checkpoint")


if __name__ == "__main__":
    unittest.main()
