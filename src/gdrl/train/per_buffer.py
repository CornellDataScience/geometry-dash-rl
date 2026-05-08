"""Prioritized Experience Replay with n-step returns + atari-style frame ring.

Memory layout
-------------
Frames are stored once in a single ring buffer (capacity F frames, uint8).
A transition stores 4 stack indices for state and 4 for next_state, plus
scalar fields (action, accumulated n-step reward, gamma^n, done, mode_ids,
level_id, episode_id). This keeps memory close to ~8KB/transition for a
100k-transition buffer at 128x128x1 uint8.

Episode boundaries are respected by checking episode_id across all 4 stack
indices when sampling — transitions whose stack would span a boundary are
not emitted.

Priorities use a min/sum tree (proportional PER). New transitions get
priority = current max priority (or 1.0 if buffer empty).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from typing import Optional

import numpy as np


# ---------- frame ring ----------

class FrameRing:
    def __init__(self, capacity: int, h: int = 128, w: int = 128, c: int = 1):
        self.capacity = capacity
        self.h, self.w, self.c = h, w, c
        self.frames = np.zeros((capacity, c, h, w), dtype=np.uint8)
        # parallel array tagging which episode each frame belongs to (so a stack
        # query can detect cross-episode windows).
        self.frame_ep = np.full(capacity, -1, dtype=np.int64)
        self.write_idx = 0  # monotonic; modulo capacity for slot

    def add(self, frame: np.ndarray, episode_id: int) -> int:
        """Store a (c, h, w) or (h, w) frame, return its monotonic index."""
        if frame.ndim == 2:
            frame = frame[None, :, :]
        idx = self.write_idx
        slot = idx % self.capacity
        self.frames[slot] = frame
        self.frame_ep[slot] = episode_id
        self.write_idx = idx + 1
        return idx

    def get_stack(self, idxs: np.ndarray) -> np.ndarray:
        """idxs: shape (B, T) of monotonic frame indices → (B, T, C, H, W) uint8."""
        slots = idxs % self.capacity
        return self.frames[slots]

    def episodes_of(self, idxs: np.ndarray) -> np.ndarray:
        return self.frame_ep[idxs % self.capacity]

    def is_valid_window(self, frame_idx_latest: int, stack: int) -> bool:
        """Check 4 consecutive frames ending at `frame_idx_latest` are same episode."""
        if frame_idx_latest - (stack - 1) < max(0, self.write_idx - self.capacity):
            # frames have been overwritten
            return False
        eps = self.frame_ep[(np.arange(frame_idx_latest - stack + 1, frame_idx_latest + 1)) % self.capacity]
        return bool(np.all(eps == eps[0]) and eps[0] >= 0)


# ---------- sum-tree (proportional PER) ----------

class SumTree:
    """Power-of-two leaf sum tree for proportional PER sampling."""

    def __init__(self, capacity: int):
        size = 1
        while size < capacity:
            size *= 2
        self.leaf_count = size
        self.tree = np.zeros(2 * size, dtype=np.float64)
        self.capacity = capacity

    def update(self, i: int, p: float) -> None:
        idx = i + self.leaf_count
        diff = p - self.tree[idx]
        self.tree[idx] = p
        idx //= 2
        while idx:
            self.tree[idx] += diff
            idx //= 2

    def total(self) -> float:
        return float(self.tree[1])

    def find(self, s: float) -> int:
        idx = 1
        while idx < self.leaf_count:
            left = 2 * idx
            if self.tree[left] >= s:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1
        return idx - self.leaf_count


# ---------- transition store ----------

@dataclass
class _Transition:
    state_idxs: np.ndarray         # (T,) int64 monotonic frame indices
    next_state_idxs: np.ndarray    # (T,) int64 monotonic frame indices
    action: int
    reward: float                  # n-step accumulated
    gamma_n: float                 # gamma^n_actual (0.0 if terminal early-done)
    done: bool
    mode_id: int
    next_mode_id: int
    level_id: int
    episode_id: int
    aux_target: Optional[np.ndarray] = None   # (AUX_DIM,) float32 of state's privileged labels


# ---------- PER buffer ----------

class PERBuffer:
    """Proportional PER over n-step transitions sharing a frame ring."""

    def __init__(
        self,
        capacity: int,
        frame_capacity: int,
        stack: int = 4,
        n_step: int = 3,
        gamma: float = 0.99,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_anneal_steps: int = 1_000_000,
        eps_priority: float = 1e-6,
        h: int = 128,
        w: int = 128,
        c: int = 1,
    ):
        self.capacity = capacity
        self.stack = stack
        self.n_step = n_step
        self.gamma = gamma
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_anneal_steps = beta_anneal_steps
        self.eps_priority = eps_priority

        self.frames = FrameRing(frame_capacity, h=h, w=w, c=c)
        self.tree = SumTree(capacity)
        self.transitions: list[Optional[_Transition]] = [None] * capacity
        self.write = 0
        self.size = 0
        self.max_priority = 1.0
        self.beta_step = 0

        # n-step aggregator (one per env; current scope = single env)
        self._nstep: deque[tuple] = deque(maxlen=n_step)
        self._frame_buffer: deque[int] = deque(maxlen=stack + n_step)

    # --- ingestion ---

    def begin_episode(self, episode_id: int, initial_frames: list[np.ndarray]) -> None:
        """Seed the frame ring with the initial stack at reset.

        `initial_frames` should be a list of `stack` frames, oldest first.
        """
        self._nstep.clear()
        self._frame_buffer.clear()
        for f in initial_frames:
            idx = self.frames.add(f, episode_id)
            self._frame_buffer.append(idx)

    def add(
        self,
        new_frame: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        mode_id: int,
        next_mode_id: int,
        level_id: int,
        episode_id: int,
        aux_target: Optional[np.ndarray] = None,
    ) -> int:
        """Add one step. Returns number of n-step transitions emitted (0 or 1).

        `aux_target` is the privileged-state target measured *for the post-action
        state* (i.e. the state observed after this step), used as a free
        representation-learning signal in the trainer's aux head. Pass None to
        disable aux storage (aux features become zeros at sample time).
        """
        new_idx = self.frames.add(new_frame, episode_id)
        self._frame_buffer.append(new_idx)

        # need at least `stack` frames before we can produce a state vector
        if len(self._frame_buffer) < self.stack:
            return 0

        # state stack = last `stack` frames *before* this new one
        # next_state stack = last `stack` frames including this new one
        fb = list(self._frame_buffer)
        next_state_idxs = np.array(fb[-self.stack:], dtype=np.int64)
        state_idxs = np.array(fb[-self.stack - 1: -1] if len(fb) > self.stack
                              else fb[:self.stack], dtype=np.int64)

        # episode-coherent windows only
        if not (self.frames.is_valid_window(state_idxs[-1], self.stack)
                and self.frames.is_valid_window(next_state_idxs[-1], self.stack)):
            if done:
                self._nstep.clear()
            return 0

        self._nstep.append((state_idxs, action, reward, next_state_idxs, done,
                            mode_id, next_mode_id, level_id, episode_id, aux_target))

        emitted = 0
        # emit when nstep is full or terminal
        while len(self._nstep) == self.n_step or (self._nstep and self._nstep[-1][4]):
            r_acc = 0.0
            done_at_end = False
            n_actual = 0
            for k, (_, _, r, _, d, _, _, _, _, _) in enumerate(self._nstep):
                r_acc += (self.gamma ** k) * r
                n_actual = k + 1
                if d:
                    done_at_end = True
                    break
            head = self._nstep[0]
            tail = self._nstep[n_actual - 1]
            tr = _Transition(
                state_idxs=head[0],
                next_state_idxs=tail[3],
                action=int(head[1]),
                reward=float(r_acc),
                gamma_n=0.0 if done_at_end else (self.gamma ** n_actual),
                done=bool(done_at_end),
                mode_id=int(head[5]),
                next_mode_id=int(tail[6]),
                level_id=int(head[7]),
                episode_id=int(head[8]),
                aux_target=head[9],
            )
            self._store(tr)
            emitted += 1

            self._nstep.popleft()
            if done_at_end:
                self._nstep.clear()
                break
            if len(self._nstep) < self.n_step:
                break
        return emitted

    def _store(self, tr: _Transition) -> None:
        slot = self.write
        self.transitions[slot] = tr
        self.tree.update(slot, self.max_priority ** self.alpha)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # --- sampling ---

    def beta(self) -> float:
        if self.beta_anneal_steps <= 0:
            return self.beta_end
        t = min(1.0, self.beta_step / self.beta_anneal_steps)
        return self.beta_start + (self.beta_end - self.beta_start) * t

    def sample(self, batch_size: int) -> dict:
        assert self.size >= batch_size, f"buffer too small: {self.size} < {batch_size}"
        total = self.tree.total()
        seg = total / batch_size
        slots = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)
        for b in range(batch_size):
            s = np.random.uniform(b * seg, (b + 1) * seg)
            slot = self.tree.find(s)
            if self.transitions[slot] is None:
                slot = (slot - 1) % self.capacity
            slots[b] = slot
            priorities[b] = self.tree.tree[slot + self.tree.leaf_count]

        beta = self.beta()
        self.beta_step += 1
        probs = priorities / max(total, 1e-12)
        weights = (self.size * probs) ** (-beta)
        weights /= max(weights.max(), 1e-12)

        # gather
        states = np.empty((batch_size, self.stack, self.frames.c, self.frames.h, self.frames.w), dtype=np.uint8)
        next_states = np.empty_like(states)
        actions = np.empty(batch_size, dtype=np.int64)
        rewards = np.empty(batch_size, dtype=np.float32)
        gamma_n = np.empty(batch_size, dtype=np.float32)
        dones = np.empty(batch_size, dtype=np.float32)
        mode_ids = np.empty(batch_size, dtype=np.int64)
        next_mode_ids = np.empty(batch_size, dtype=np.int64)
        level_ids = np.empty(batch_size, dtype=np.int64)
        aux_targets = None
        aux_mask = np.zeros(batch_size, dtype=np.float32)
        for b, slot in enumerate(slots):
            tr = self.transitions[slot]
            states[b] = self.frames.get_stack(tr.state_idxs[None, :])[0]
            next_states[b] = self.frames.get_stack(tr.next_state_idxs[None, :])[0]
            actions[b] = tr.action
            rewards[b] = tr.reward
            gamma_n[b] = tr.gamma_n
            dones[b] = 1.0 if tr.done else 0.0
            mode_ids[b] = tr.mode_id
            next_mode_ids[b] = tr.next_mode_id
            level_ids[b] = tr.level_id
            if tr.aux_target is not None:
                if aux_targets is None:
                    aux_targets = np.zeros((batch_size, tr.aux_target.shape[0]), dtype=np.float32)
                aux_targets[b] = tr.aux_target
                aux_mask[b] = 1.0

        return {
            "slots": slots,
            "weights": weights.astype(np.float32),
            "states": states,
            "next_states": next_states,
            "actions": actions,
            "rewards": rewards,
            "gamma_n": gamma_n,
            "dones": dones,
            "mode_ids": mode_ids,
            "next_mode_ids": next_mode_ids,
            "level_ids": level_ids,
            "aux_targets": aux_targets,           # None or (B, AUX_DIM)
            "aux_mask": aux_mask,                  # (B,) 1=valid, 0=missing
        }

    def update_priorities(self, slots: np.ndarray, td_errors: np.ndarray) -> None:
        td = np.abs(td_errors) + self.eps_priority
        self.max_priority = max(self.max_priority, float(td.max()))
        for slot, p in zip(slots.tolist(), td.tolist()):
            self.tree.update(int(slot), p ** self.alpha)
