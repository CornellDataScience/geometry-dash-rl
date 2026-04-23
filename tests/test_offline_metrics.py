"""Tests for jump-event evaluation metrics."""
from __future__ import annotations

import numpy as np
import pytest

from gdrl.eval.offline_metrics import extract_jump_events, match_events, JumpEvent


# --- extract_jump_events ---

def test_extract_no_jumps():
    actions = np.array([0, 0, 0, 0, 0])
    episodes = np.array([0, 0, 0, 0, 0])
    events = extract_jump_events(actions, episodes)
    assert events == []


def test_extract_single_jump():
    actions = np.array([0, 0, 1, 1, 1, 0, 0])
    episodes = np.array([0, 0, 0, 0, 0, 0, 0])
    events = extract_jump_events(actions, episodes)
    assert len(events) == 1
    assert events[0].start == 2
    assert events[0].end == 4
    assert events[0].center == 3
    assert events[0].duration == 3


def test_extract_two_jumps():
    actions = np.array([1, 1, 0, 0, 1, 0])
    episodes = np.array([0, 0, 0, 0, 0, 0])
    events = extract_jump_events(actions, episodes)
    assert len(events) == 2
    assert events[0].start == 0
    assert events[0].end == 1
    assert events[1].start == 4
    assert events[1].end == 4


def test_extract_respects_episode_boundary():
    actions = np.array([1, 1, 1, 1, 1])
    episodes = np.array([0, 0, 0, 1, 1])
    events = extract_jump_events(actions, episodes)
    assert len(events) == 2
    assert events[0].start == 0
    assert events[0].end == 2
    assert events[0].episode_id == 0
    assert events[1].start == 3
    assert events[1].end == 4
    assert events[1].episode_id == 1


def test_extract_single_frame_jump():
    actions = np.array([0, 1, 0])
    episodes = np.array([0, 0, 0])
    events = extract_jump_events(actions, episodes)
    assert len(events) == 1
    assert events[0].duration == 1


# --- match_events ---

def test_match_perfect():
    human = [JumpEvent(10, 12, 0), JumpEvent(50, 55, 0)]
    model = [JumpEvent(10, 12, 0), JumpEvent(50, 55, 0)]
    result = match_events(human, model, tolerance=15)
    assert result["tp"] == 2
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["event_f1"] == 1.0


def test_match_with_tolerance():
    human = [JumpEvent(10, 12, 0)]     # center=11
    model = [JumpEvent(20, 22, 0)]     # center=21, diff=10
    result = match_events(human, model, tolerance=15)
    assert result["tp"] == 1  # within tolerance


def test_match_outside_tolerance():
    human = [JumpEvent(10, 12, 0)]     # center=11
    model = [JumpEvent(40, 42, 0)]     # center=41, diff=30
    result = match_events(human, model, tolerance=15)
    assert result["tp"] == 0
    assert result["fn"] == 1
    assert result["fp"] == 1


def test_match_no_cross_episode():
    human = [JumpEvent(10, 12, 0)]
    model = [JumpEvent(10, 12, 1)]     # same timing, different episode
    result = match_events(human, model, tolerance=15)
    assert result["tp"] == 0
    assert result["fn"] == 1
    assert result["fp"] == 1


def test_match_extra_model_jumps():
    human = [JumpEvent(10, 12, 0)]
    model = [JumpEvent(10, 12, 0), JumpEvent(30, 32, 0), JumpEvent(60, 62, 0)]
    result = match_events(human, model, tolerance=15)
    assert result["tp"] == 1
    assert result["fp"] == 2  # two extra model jumps
    assert result["fn"] == 0


def test_match_missed_human_jumps():
    human = [JumpEvent(10, 12, 0), JumpEvent(50, 55, 0), JumpEvent(100, 102, 0)]
    model = [JumpEvent(10, 12, 0)]
    result = match_events(human, model, tolerance=15)
    assert result["tp"] == 1
    assert result["fn"] == 2  # two missed jumps
    assert result["fp"] == 0


def test_match_empty():
    result = match_events([], [], tolerance=15)
    assert result["tp"] == 0
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["event_f1"] == 0.0


def test_match_precision_recall():
    human = [JumpEvent(10, 12, 0), JumpEvent(50, 52, 0)]
    model = [JumpEvent(11, 13, 0), JumpEvent(80, 82, 0)]  # matches first, misses second, extra at 80
    result = match_events(human, model, tolerance=15)
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert abs(result["event_precision"] - 0.5) < 0.01
    assert abs(result["event_recall"] - 0.5) < 0.01
