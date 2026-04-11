from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from gdrl.teacher.curriculum_reset import CurriculumResetPolicy


@dataclass
class SelfImitationConfig:
    enabled: bool = True
    segment_size: float = 100.0
    elite_prefixes: int = 5
    recent_episodes: int = 50
    min_mastery_episodes: int = 10
    promote_threshold: float = 0.80
    regress_threshold: float = 0.50
    bc_batch_size: int = 128
    bc_batches: int = 1
    bc_interval_rollouts: int = 1
    bc_coef: float = 0.10


@dataclass
class ElitePrefix:
    segment_index: int
    episode_max_x: float
    prefix_length: int
    training_step: int
    obs: np.ndarray
    actions: np.ndarray

    def rank_key(self) -> tuple[float, int]:
        # Prefer farther-reaching episodes; if tied, prefer shorter prefixes.
        return (self.episode_max_x, -self.prefix_length)


@dataclass
class EpisodeTrace:
    obs: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    post_x: list[float] = field(default_factory=list)


class SegmentEliteReplay:
    def __init__(self, cfg: SelfImitationConfig):
        self.cfg = cfg
        self._prefixes: dict[int, list[ElitePrefix]] = defaultdict(list)

    @property
    def max_segment_index(self) -> int:
        if not self._prefixes:
            return -1
        return max(self._prefixes)

    def elite_counts(self) -> dict[int, int]:
        return {segment: len(prefixes) for segment, prefixes in self._prefixes.items()}

    def add_episode(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        post_x: np.ndarray,
        *,
        episode_max_x: float,
        training_step: int,
    ) -> int:
        if obs.ndim != 2 or actions.ndim != 1 or post_x.ndim != 1:
            raise ValueError("Episode arrays must be shaped as obs[T, D], actions[T], post_x[T]")
        if len(obs) == 0 or len(obs) != len(actions) or len(obs) != len(post_x):
            return 0

        max_segment = int(float(episode_max_x) // self.cfg.segment_size)
        if max_segment < 0:
            return 0

        targets = (np.arange(max_segment + 1, dtype=np.float32) + 1.0) * self.cfg.segment_size
        reach_indices = np.searchsorted(post_x, targets, side="left")

        added = 0
        for segment_index, prefix_end in enumerate(reach_indices):
            if prefix_end >= len(post_x):
                break

            prefix = ElitePrefix(
                segment_index=segment_index,
                episode_max_x=float(episode_max_x),
                prefix_length=int(prefix_end + 1),
                training_step=int(training_step),
                obs=np.array(obs[: prefix_end + 1], copy=True, dtype=np.float32),
                actions=np.array(actions[: prefix_end + 1], copy=True, dtype=np.int64),
            )
            self._insert(prefix)
            added += 1
        return added

    def _insert(self, prefix: ElitePrefix) -> None:
        entries = self._prefixes[prefix.segment_index]
        entries.append(prefix)
        entries.sort(key=lambda item: item.rank_key(), reverse=True)
        del entries[self.cfg.elite_prefixes :]

    def sample_batch(
        self,
        active_segments: list[int],
        *,
        batch_size: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        available_segments = [segment for segment in active_segments if self._prefixes.get(segment)]
        if not available_segments:
            return None

        segment_weights = np.asarray([segment + 1 for segment in available_segments], dtype=np.float64)
        segment_weights /= segment_weights.sum()

        obs_batch: list[np.ndarray] = []
        action_batch: list[int] = []
        for _ in range(batch_size):
            seg_offset = int(rng.choice(len(available_segments), p=segment_weights))
            segment = available_segments[seg_offset]
            elite = self._prefixes[segment][int(rng.integers(len(self._prefixes[segment])))]
            step_index = int(rng.integers(elite.prefix_length))
            obs_batch.append(elite.obs[step_index])
            action_batch.append(int(elite.actions[step_index]))

        return (
            np.asarray(obs_batch, dtype=np.float32),
            np.asarray(action_batch, dtype=np.int64),
        )


class SegmentCurriculum:
    def __init__(self, cfg: SelfImitationConfig):
        self.cfg = cfg
        self.mastered_segments = 0
        self._recent_max_x: deque[float] = deque(maxlen=cfg.recent_episodes)

    @property
    def frontier_segment(self) -> int:
        return self.mastered_segments

    def record_episode(self, episode_max_x: float) -> tuple[int, int]:
        before = self.mastered_segments
        self._recent_max_x.append(float(episode_max_x))
        self._refresh_mastery()
        return before, self.mastered_segments

    def success_rate(self, segment_index: int) -> float:
        if segment_index < 0 or not self._recent_max_x:
            return 0.0
        target_x = (segment_index + 1) * self.cfg.segment_size
        passed = sum(max_x >= target_x for max_x in self._recent_max_x)
        return passed / len(self._recent_max_x)

    def active_bc_segments(self, max_available_segment: int) -> list[int]:
        if max_available_segment < 0:
            return []
        if self.mastered_segments > 0:
            upper = min(self.mastered_segments - 1, max_available_segment)
        else:
            upper = 0
        return list(range(upper + 1))

    def snapshot(self, max_available_segment: int) -> dict[str, float | int]:
        return {
            "mastered_segments": self.mastered_segments,
            "frontier_segment": self.frontier_segment,
            "frontier_success": self.success_rate(self.frontier_segment),
            "latest_mastered_success": self.success_rate(self.mastered_segments - 1),
            "max_elite_segment": max_available_segment,
            "recent_episode_window": len(self._recent_max_x),
        }

    def _refresh_mastery(self) -> None:
        if len(self._recent_max_x) < self.cfg.min_mastery_episodes:
            return

        promotable = 0
        while self.success_rate(promotable) >= self.cfg.promote_threshold:
            promotable += 1

        if promotable > self.mastered_segments:
            self.mastered_segments = promotable
            return

        while self.mastered_segments > 0:
            current = self.mastered_segments - 1
            if self.success_rate(current) >= self.cfg.regress_threshold:
                break
            self.mastered_segments -= 1


class SelfImitationController(BaseCallback):
    def __init__(
        self,
        cfg: SelfImitationConfig,
        *,
        reset_policy: CurriculumResetPolicy | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.cfg = cfg
        self.replay = SegmentEliteReplay(cfg)
        self.curriculum = SegmentCurriculum(cfg)
        self.reset_policy = reset_policy
        self.rng = np.random.default_rng(0)
        self._episodes: dict[int, EpisodeTrace] = {}
        self.episodes_completed = 0
        self.rollout_count = 0
        self.pending_bc = False
        self.latest_episode_max_x = 0.0
        self.last_bc_loss = 0.0
        self.last_bc_samples = 0
        self.last_bc_segments = 0

    def _init_callback(self) -> None:
        setattr(self.model, "self_imitation_controller", self)

    def _on_step(self) -> bool:
        if not self.cfg.enabled:
            return True

        prev_obs = np.asarray(self.model._last_obs)
        actions = np.asarray(self.locals["actions"]).reshape(-1)
        dones = np.asarray(self.locals["dones"]).reshape(-1)
        infos = self.locals["infos"]

        for env_index, action in enumerate(actions):
            trace = self._episodes.setdefault(env_index, EpisodeTrace())
            trace.obs.append(np.array(prev_obs[env_index], copy=True, dtype=np.float32))
            trace.actions.append(int(action))

            info = infos[env_index] if env_index < len(infos) else {}
            post_x = float(info.get("x", prev_obs[env_index][0]))
            trace.post_x.append(post_x)

            if not bool(dones[env_index]):
                continue

            obs = np.asarray(trace.obs, dtype=np.float32)
            action_arr = np.asarray(trace.actions, dtype=np.int64)
            post_x_arr = np.asarray(trace.post_x, dtype=np.float32)
            episode_max_x = float(info.get("best_x", post_x_arr.max(initial=0.0)))
            self.latest_episode_max_x = episode_max_x

            self.replay.add_episode(
                obs,
                action_arr,
                post_x_arr,
                episode_max_x=episode_max_x,
                training_step=self.model.num_timesteps,
            )
            before, after = self.curriculum.record_episode(episode_max_x)
            self.episodes_completed += 1
            if self.reset_policy is not None:
                self.reset_policy.update(
                    mastered_segments=self.curriculum.mastered_segments,
                    episodes_completed=self.episodes_completed,
                )
            if self.verbose >= 1 and before != after:
                direction = "advanced" if after > before else "backtracked"
                print(
                    f"[self_imitation] {direction}: mastered_segments={before} -> {after} "
                    f"(latest max_x={episode_max_x:.1f})",
                    flush=True,
                )

            self._episodes[env_index] = EpisodeTrace()

        return True

    def _on_rollout_end(self) -> None:
        if not self.cfg.enabled:
            return
        self.rollout_count += 1
        self.pending_bc = True

        snapshot = self.curriculum.snapshot(self.replay.max_segment_index)
        self.logger.record("self_imitation/mastered_segments", snapshot["mastered_segments"])
        self.logger.record("self_imitation/frontier_segment", snapshot["frontier_segment"])
        self.logger.record("self_imitation/frontier_success", snapshot["frontier_success"])
        self.logger.record(
            "self_imitation/latest_mastered_success",
            snapshot["latest_mastered_success"],
        )
        self.logger.record("self_imitation/max_elite_segment", snapshot["max_elite_segment"])
        self.logger.record("self_imitation/latest_episode_max_x", self.latest_episode_max_x)
        self.logger.record("self_imitation/elite_segment_count", len(self.replay.elite_counts()))
        self.logger.record("self_imitation/last_bc_loss", self.last_bc_loss)
        self.logger.record("self_imitation/last_bc_samples", self.last_bc_samples)
        self.logger.record("self_imitation/last_bc_segments", self.last_bc_segments)
        self.logger.record("self_imitation/episodes_completed", self.episodes_completed)
        if self.reset_policy is not None:
            reset_state = self.reset_policy.export_state()
            self.logger.record("self_imitation/reset_last_segment", reset_state["last_selected_segment"])
            self.logger.record("self_imitation/reset_last_x", reset_state["last_selected_x"])

    def maybe_behavior_clone(self, model: PPO) -> None:
        if not self.cfg.enabled or not self.pending_bc:
            return
        if self.rollout_count % self.cfg.bc_interval_rollouts != 0:
            return

        self.pending_bc = False
        active_segments = self.curriculum.active_bc_segments(self.replay.max_segment_index)
        if not active_segments:
            self.last_bc_loss = 0.0
            self.last_bc_samples = 0
            self.last_bc_segments = 0
            return

        losses: list[float] = []
        samples = 0
        for _ in range(self.cfg.bc_batches):
            batch = self.replay.sample_batch(
                active_segments,
                batch_size=self.cfg.bc_batch_size,
                rng=self.rng,
            )
            if batch is None:
                break

            obs_np, actions_np = batch
            obs_tensor, _ = model.policy.obs_to_tensor(obs_np)
            actions_tensor = th.as_tensor(actions_np, device=model.device).long()
            _, log_prob, _ = model.policy.evaluate_actions(obs_tensor, actions_tensor)
            neg_log_prob = -log_prob.mean()
            loss = self.cfg.bc_coef * neg_log_prob

            model.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.policy.parameters(), model.max_grad_norm)
            model.policy.optimizer.step()

            losses.append(float(neg_log_prob.detach().cpu()))
            samples += int(actions_np.shape[0])

        self.last_bc_loss = float(np.mean(losses)) if losses else 0.0
        self.last_bc_samples = samples
        self.last_bc_segments = len(active_segments)

        if losses:
            model.logger.record("self_imitation/bc_loss", self.last_bc_loss)
            model.logger.record("self_imitation/bc_samples", self.last_bc_samples)
            model.logger.record("self_imitation/bc_segments", self.last_bc_segments)

    def export_state(self) -> dict[str, Any]:
        return {
            "config": asdict(self.cfg),
            "curriculum": self.curriculum.snapshot(self.replay.max_segment_index),
            "episodes_completed": self.episodes_completed,
            "latest_episode_max_x": self.latest_episode_max_x,
            "elite_counts": self.replay.elite_counts(),
            "last_bc_loss": self.last_bc_loss,
            "last_bc_samples": self.last_bc_samples,
            "last_bc_segments": self.last_bc_segments,
        }

    def save_state(self, path: Path) -> None:
        path.write_text(json.dumps(self.export_state(), indent=2, sort_keys=True), encoding="utf-8")


class SelfImitatingPPO(PPO):
    def __init__(self, *args, self_imitation_controller: SelfImitationController | None = None, **kwargs):
        self.self_imitation_controller = self_imitation_controller
        super().__init__(*args, **kwargs)

    def train(self) -> None:
        super().train()
        controller = getattr(self, "self_imitation_controller", None)
        if controller is not None:
            controller.maybe_behavior_clone(self)
