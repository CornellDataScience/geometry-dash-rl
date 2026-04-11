from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class CurriculumResetConfig:
    enabled: bool = False
    segment_size: float = 100.0
    checkpoint_reset_prob: float = 0.7
    backtrack_segments: int = 1
    warmup_episodes: int = 25


class CurriculumResetPolicy:
    """Choose between full resets and checkpoint resets near mastered segments."""

    def __init__(self, cfg: CurriculumResetConfig, *, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.mastered_segments = 0
        self.episodes_completed = 0
        self.last_selected_segment = -1
        self.last_selected_x = 0.0
        self.last_mode = "full"

    def update(self, *, mastered_segments: int, episodes_completed: int) -> None:
        self.mastered_segments = max(0, int(mastered_segments))
        self.episodes_completed = max(0, int(episodes_completed))

    def choose_checkpoint_x(self) -> float | None:
        self.last_selected_segment = -1
        self.last_selected_x = 0.0
        self.last_mode = "full"

        if not self.cfg.enabled:
            return None
        if self.cfg.segment_size <= 0.0:
            return None
        if self.episodes_completed < max(0, self.cfg.warmup_episodes):
            return None
        if self.mastered_segments <= 0:
            return None
        if float(self.rng.random()) >= float(self.cfg.checkpoint_reset_prob):
            return None

        upper_segment = max(self.mastered_segments - 1, 0)
        lower_segment = max(0, upper_segment - max(0, self.cfg.backtrack_segments))
        segment = int(self.rng.integers(lower_segment, upper_segment + 1))
        checkpoint_x = float((segment + 1) * self.cfg.segment_size)

        self.last_selected_segment = segment
        self.last_selected_x = checkpoint_x
        self.last_mode = "checkpoint"
        return checkpoint_x

    def export_state(self) -> dict:
        return {
            "config": asdict(self.cfg),
            "mastered_segments": self.mastered_segments,
            "episodes_completed": self.episodes_completed,
            "last_selected_segment": self.last_selected_segment,
            "last_selected_x": self.last_selected_x,
            "last_mode": self.last_mode,
        }
