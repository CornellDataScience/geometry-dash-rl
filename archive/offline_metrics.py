"""Jump-event evaluation metrics with temporal tolerance.

Instead of per-frame accuracy (misleading due to timing tolerance),
extract contiguous jump events and match them between human labels
and model predictions within a ±tolerance window.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class JumpEvent:
    """A contiguous run of action=1."""
    start: int        # first frame index with action=1
    end: int          # last frame index with action=1 (inclusive)
    episode_id: int

    @property
    def center(self) -> int:
        return (self.start + self.end) // 2

    @property
    def duration(self) -> int:
        return self.end - self.start + 1


def extract_jump_events(
    actions: np.ndarray,
    episode_ids: np.ndarray,
) -> List[JumpEvent]:
    """Extract contiguous jump events, respecting episode boundaries.

    Args:
        actions: (N,) binary array of jump labels
        episode_ids: (N,) episode ID per frame

    Returns:
        List of JumpEvent, one per contiguous run of action=1 within an episode.
    """
    events = []
    n = len(actions)
    i = 0
    while i < n:
        if actions[i] == 1:
            start = i
            ep = int(episode_ids[i])
            while i < n and actions[i] == 1 and int(episode_ids[i]) == ep:
                i += 1
            events.append(JumpEvent(start=start, end=i - 1, episode_id=ep))
        else:
            i += 1
    return events


def match_events(
    human_events: List[JumpEvent],
    model_events: List[JumpEvent],
    tolerance: int = 15,
) -> dict:
    """Match jump events between human and model with temporal tolerance.

    A human event is matched if any model event's center is within
    ±tolerance frames of the human event's center AND they share the
    same episode_id.

    Returns:
        dict with keys: tp, fp, fn, event_precision, event_recall, event_f1
    """
    # group model events by episode for faster lookup
    model_by_ep: dict[int, List[JumpEvent]] = {}
    for e in model_events:
        model_by_ep.setdefault(e.episode_id, []).append(e)

    matched_model = set()  # indices into model_events
    tp = 0
    fn = 0

    for he in human_events:
        ep_events = model_by_ep.get(he.episode_id, [])
        best_dist = float("inf")
        best_idx = -1
        for idx, me in enumerate(ep_events):
            # use global index for tracking
            global_idx = id(me)
            if global_idx in matched_model:
                continue
            dist = abs(he.center - me.center)
            if dist <= tolerance and dist < best_dist:
                best_dist = dist
                best_idx = idx
                best_global = global_idx

        if best_idx >= 0:
            tp += 1
            matched_model.add(best_global)
        else:
            fn += 1

    fp = len(model_events) - len(matched_model)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
        "human_events": len(human_events),
        "model_events": len(model_events),
    }
