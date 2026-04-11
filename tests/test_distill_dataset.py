from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from gdrl.student.dataset import DistillNPZDataset


def _write_shard(path: Path, frame_values: list[int], modes: list[int], probs: list[list[float]]) -> None:
    frame_count = len(frame_values)
    frames = np.asarray(
        [np.full((2, 2), fill_value=value, dtype=np.uint8) for value in frame_values],
        dtype=np.uint8,
    )
    np.savez_compressed(
        path,
        frames=frames,
        teacher_probs=np.asarray(probs, dtype=np.float32),
        modes=np.asarray(modes, dtype=np.int64),
        level_ids=np.zeros(frame_count, dtype=np.int32),
        frame_idxs=np.arange(frame_count, dtype=np.int64),
    )


class DistillDatasetTests(unittest.TestCase):
    def test_directory_loading_preserves_shard_boundaries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_shard(
                root / "shard_a.npz",
                frame_values=[1, 2, 3, 4],
                modes=[0, 1, 2, 3],
                probs=[[0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.1, 0.9]],
            )
            _write_shard(
                root / "shard_b.npz",
                frame_values=[10, 11, 12, 13, 14],
                modes=[4, 4, 5, 5, 5],
                probs=[[0.4, 0.6], [0.3, 0.7], [0.2, 0.8], [0.25, 0.75], [0.1, 0.9]],
            )

            ds = DistillNPZDataset(root, stack_size=4)

            self.assertEqual(len(ds), 3)
            self.assertEqual(ds.num_modes, 6)

            frames, probs, mode = ds[1]
            self.assertEqual(tuple(frames.shape), (4, 2, 2))
            self.assertAlmostEqual(float(frames[0, 0, 0]), 10.0 / 255.0)
            self.assertAlmostEqual(float(frames[-1, 0, 0]), 13.0 / 255.0)
            self.assertEqual(int(mode), 5)
            self.assertTrue(np.allclose(probs.numpy(), np.asarray([0.25, 0.75], dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
