"""Snippet-PPO trainer for GD privileged policy.

Truncated n-step PPO with cross-rollout snippet replay buffer and
progress-based reward + localized death penalty.

Each rollout is sliced into non-overlapping K-frame snippets. Snippets are
stored in a FIFO buffer across multiple rollouts and shuffled when sampled
for updates. This decouples credit assignment (per-snippet) from the long
episode horizon and breaks temporal correlation between consecutive samples.

Sparse reward (recomputed in trainer; env reward ignored):
  r_t = progress_t * progress_scale  while alive
  r_T = -death_penalty               on death frame
  no level-complete bonus            (lands only in last snippet, indistinguishable
                                      from "lived through K frames" otherwise)

Usage:
    python -m gdrl.train.ppo \\
        --out artifacts/ppo_run1 \\
        [--bc-checkpoint artifacts/bc_model.pt] \\
        [--norm artifacts/bc_model.norm.npz]

Checkpoints are saved every --save-every updates to <out>/checkpoints/.
Tensorboard logs are written to <out>/logs/.
The final model is saved to <out>/ppo_final.pt in the same format as BC
checkpoints, so live_eval.py works without any changes.
"""
from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from gdrl.env.geode_ipc_v3 import GeodeV3Adapter, GeodeIPCV3Config
from gdrl.env.privileged_env import GDPrivilegedEnv
from gdrl.model.mlp_agent import GDPolicyMLP
from gdrl.model.obs_preprocess import ObsPreprocessor, ObsNormalizer, PROCESSED_FRAME_DIM


class _StackedEnv:
    """GDPrivilegedEnv wrapper that preprocesses obs and maintains a frame stack.

    Observations returned are torch.Tensor of shape (stack_size * PROCESSED_FRAME_DIM,).
    Augments info dict with 'progress' (delta-x since last step) and 'is_dead'
    so the trainer can recompute its own sparse reward without touching the env.
    """

    def __init__(self, env: GDPrivilegedEnv, preprocessor: ObsPreprocessor, stack_size: int = 4):
        self.env = env
        self.preprocessor = preprocessor
        self.stack_size = stack_size
        self._stack = np.zeros((stack_size, PROCESSED_FRAME_DIM), dtype=np.float32)
        self._prev_x = 0.0

    def reset(self) -> tuple[torch.Tensor, dict]:
        raw_obs, info = self.env.reset()
        self._prev_x = float(raw_obs[0])
        processed = self.preprocessor.process_frame(raw_obs)
        self._stack[:] = processed
        return torch.from_numpy(self._stack.reshape(-1).copy()), info

    def step(self, action: int) -> tuple[torch.Tensor, float, bool, bool, dict]:
        raw_obs, reward, terminated, truncated, info = self.env.step(action)
        new_x = float(raw_obs[0])
        progress = new_x - self._prev_x
        self._prev_x = new_x
        is_dead = bool(raw_obs[5] > 0.5)
        processed = self.preprocessor.process_frame(raw_obs)
        self._stack[:-1] = self._stack[1:]
        self._stack[-1] = processed
        info = dict(info) if info else {}
        info["progress"] = progress
        info["is_dead"] = is_dead
        info["x"] = new_x
        return torch.from_numpy(self._stack.reshape(-1).copy()), reward, terminated, truncated, info


@dataclass
class Snippet:
    obs: torch.Tensor          # (K, obs_dim)
    actions: torch.Tensor      # (K,) float
    log_probs: torch.Tensor    # (K,) float
    values: torch.Tensor       # (K,) float — collection-time V(s_t)
    returns: torch.Tensor      # (K,) float — n-step truncated return at each frame
    valid_mask: torch.Tensor   # (K,) float — 1 for real frames, 0 for post-death pad
    died_within: bool
    age: int = 0


def _episode_to_snippets(
    ep_obs: list[torch.Tensor],
    ep_actions: list[int],
    ep_logprobs: list[float],
    ep_values: list[float],
    ep_rewards: list[float],
    K: int,
    gamma: float,
    obs_dim: int,
    died: bool,
    final_value: float = 0.0,
) -> list[Snippet]:
    """Slice a finished episode (or rollout-ended partial) into K-stride snippets.

    died: True iff the episode terminated by death (last frame is the death frame).
    final_value: V(s_{L}) for the obs *after* the last collected step. Only used
        when the trailing chunk is a full K frames in a non-died episode (i.e.
        truncated by env or by rollout-end).
    """
    L = len(ep_obs)
    snippets: list[Snippet] = []
    for start in range(0, L, K):
        end = min(start + K, L)
        actual_len = end - start
        is_last = (end == L)
        died_within = died and is_last  # only the trailing chunk of a died ep contains the death

        # Drop trailing partial chunks that aren't death snippets.
        if not died_within and actual_len < K:
            continue

        # Bootstrap V(s_{end}).
        if died_within:
            bootstrap = 0.0
        elif end < L:
            bootstrap = ep_values[end]
        else:
            # full chunk at episode end without death (truncated or rollout-bound)
            bootstrap = final_value

        # Backward discounted return.
        returns_local = [0.0] * actual_len
        G = bootstrap
        for i in reversed(range(actual_len)):
            G = ep_rewards[start + i] + gamma * G
            returns_local[i] = G

        obs_t = torch.zeros(K, obs_dim)
        act_t = torch.zeros(K)
        lp_t = torch.zeros(K)
        val_t = torch.zeros(K)
        ret_t = torch.zeros(K)
        mask_t = torch.zeros(K)
        for i in range(actual_len):
            obs_t[i] = ep_obs[start + i]
            act_t[i] = float(ep_actions[start + i])
            lp_t[i] = ep_logprobs[start + i]
            val_t[i] = ep_values[start + i]
            ret_t[i] = returns_local[i]
            mask_t[i] = 1.0

        snippets.append(Snippet(
            obs=obs_t,
            actions=act_t,
            log_probs=lp_t,
            values=val_t,
            returns=ret_t,
            valid_mask=mask_t,
            died_within=died_within,
            age=0,
        ))

    return snippets


class SnippetBuffer:
    """FIFO buffer of Snippets with age-based eviction and stratified sampling."""

    def __init__(self, capacity: int, max_age: int):
        self.capacity = capacity
        self.max_age = max_age
        self._items: deque[Snippet] = deque()

    def __len__(self) -> int:
        return len(self._items)

    def add_many(self, snippets: list[Snippet]) -> None:
        for s in snippets:
            self._items.append(s)
        while len(self._items) > self.capacity:
            self._items.popleft()

    def age(self) -> None:
        """Increment age on all snippets; evict any whose age exceeds max_age."""
        for s in self._items:
            s.age += 1
        while self._items and self._items[0].age > self.max_age:
            self._items.popleft()
        # robust: rebuild deque dropping any older-than-max_age survivors
        # (in case popleft above missed non-FIFO-aged items — they're all
        # in collection order so popleft from front is sufficient, but keep
        # this as defense in depth).
        kept = [s for s in self._items if s.age <= self.max_age]
        if len(kept) != len(self._items):
            self._items = deque(kept)

    def num_died(self) -> int:
        return sum(1 for s in self._items if s.died_within)

    def sample_indices(self, n_total: int, death_floor_frac: float) -> np.ndarray:
        """Return n_total indices into self._items with stratified death-floor.

        At least ceil(n_total * death_floor_frac) indices come from died snippets
        (capped by available died snippets). Remainder uniform from non-died.
        With replacement if a pool is too small.
        """
        if len(self._items) == 0:
            return np.array([], dtype=np.int64)

        died_idx = np.array([i for i, s in enumerate(self._items) if s.died_within], dtype=np.int64)
        non_died_idx = np.array([i for i, s in enumerate(self._items) if not s.died_within], dtype=np.int64)

        target_death = int(np.ceil(n_total * death_floor_frac))
        n_death = min(target_death, len(died_idx)) if len(died_idx) > 0 else 0
        n_other = n_total - n_death

        chosen = []
        if n_death > 0:
            # without replacement up to pool size; with if requested > pool
            replace = n_death > len(died_idx)
            chosen.append(np.random.choice(died_idx, size=n_death, replace=replace))
        if n_other > 0 and len(non_died_idx) > 0:
            replace = n_other > len(non_died_idx)
            chosen.append(np.random.choice(non_died_idx, size=n_other, replace=replace))
        elif n_other > 0 and len(died_idx) > 0:
            # fallback if we have no non-died snippets at all
            chosen.append(np.random.choice(died_idx, size=n_other, replace=True))

        if not chosen:
            return np.array([], dtype=np.int64)

        idx = np.concatenate(chosen)
        np.random.shuffle(idx)
        return idx

    def get(self, idx: np.ndarray) -> list[Snippet]:
        return [self._items[i] for i in idx]


def _collect_rollout(
    env: _StackedEnv,
    model: GDPolicyMLP,
    n_steps: int,
    snippet_len: int,
    gamma: float,
    death_penalty: float,
    progress_scale: float,
    obs_dim: int,
    device: torch.device,
    obs: torch.Tensor,
) -> tuple[torch.Tensor, list[Snippet], dict]:
    """Run env for n_steps frames, recompute sparse reward, slice into snippets.

    Returns: (next_obs, list of snippets, stats).
    """
    model.eval()

    new_snippets: list[Snippet] = []
    ep_obs: list[torch.Tensor] = []
    ep_actions: list[int] = []
    ep_logprobs: list[float] = []
    ep_values: list[float] = []
    ep_rewards: list[float] = []
    cur_progress = 0.0
    cur_len = 0

    ep_lengths: list[int] = []
    ep_progress_totals: list[float] = []
    ep_died_count = 0

    for _ in range(n_steps):
        with torch.no_grad():
            logit, value = model(obs.unsqueeze(0).to(device))
            logit = logit.squeeze()
            value = float(value.squeeze())
            dist = torch.distributions.Bernoulli(logits=logit)
            action_t = dist.sample()
            log_prob = float(dist.log_prob(action_t))
            action = int(action_t.item())

        next_obs, _env_reward, terminated, truncated, info = env.step(action)
        progress = float(info.get("progress", 0.0))
        is_dead = bool(info.get("is_dead", False))

        # sparse reward
        if is_dead:
            reward = -death_penalty
        else:
            reward = progress * progress_scale

        ep_obs.append(obs.cpu().clone())
        ep_actions.append(action)
        ep_logprobs.append(log_prob)
        ep_values.append(value)
        ep_rewards.append(reward)
        cur_progress += progress
        cur_len += 1

        if terminated or truncated:
            died = is_dead and terminated
            snippets = _episode_to_snippets(
                ep_obs, ep_actions, ep_logprobs, ep_values, ep_rewards,
                K=snippet_len, gamma=gamma, obs_dim=obs_dim,
                died=died, final_value=0.0,
            )
            new_snippets.extend(snippets)
            ep_lengths.append(cur_len)
            ep_progress_totals.append(cur_progress)
            if died:
                ep_died_count += 1
            ep_obs, ep_actions, ep_logprobs, ep_values, ep_rewards = [], [], [], [], []
            cur_progress = 0.0
            cur_len = 0
            obs, _ = env.reset()
        else:
            obs = next_obs

    # Flush partial trailing episode (rollout boundary mid-episode).
    if len(ep_obs) > 0:
        with torch.no_grad():
            _, final_v = model(obs.unsqueeze(0).to(device))
            final_v = float(final_v.squeeze())
        snippets = _episode_to_snippets(
            ep_obs, ep_actions, ep_logprobs, ep_values, ep_rewards,
            K=snippet_len, gamma=gamma, obs_dim=obs_dim,
            died=False, final_value=final_v,
        )
        new_snippets.extend(snippets)

    stats = {
        "ep_len_mean": float(np.mean(ep_lengths)) if ep_lengths else None,
        "ep_progress_mean": float(np.mean(ep_progress_totals)) if ep_progress_totals else None,
        "n_episodes": len(ep_lengths),
        "n_died": ep_died_count,
        "n_new_snippets": len(new_snippets),
    }
    return obs, new_snippets, stats


def _flatten_batch(batch: list[Snippet], device: torch.device):
    obs = torch.stack([s.obs for s in batch]).to(device)               # (N, K, D)
    actions = torch.stack([s.actions for s in batch]).to(device)       # (N, K)
    log_probs = torch.stack([s.log_probs for s in batch]).to(device)   # (N, K)
    values = torch.stack([s.values for s in batch]).to(device)         # (N, K)
    returns = torch.stack([s.returns for s in batch]).to(device)       # (N, K)
    mask = torch.stack([s.valid_mask for s in batch]).to(device)       # (N, K)
    N, K, D = obs.shape
    return (
        obs.reshape(N * K, D),
        actions.reshape(N * K),
        log_probs.reshape(N * K),
        values.reshape(N * K),
        returns.reshape(N * K),
        mask.reshape(N * K),
    )


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp(min=1.0)
    return (x * mask).sum() / denom


def _update(
    model: GDPolicyMLP,
    optimizer: torch.optim.Optimizer,
    buffer: SnippetBuffer,
    n_epochs: int,
    value_epochs: int,
    batch_snippets: int,
    death_floor_frac: float,
    clip_range: float,
    vf_coef: float,
    ent_coef: float,
    max_grad_norm: float,
    device: torch.device,
    bc_anchor: GDPolicyMLP | None = None,
    kl_coef: float = 0.0,
) -> dict:
    if len(buffer) == 0:
        return {"pg_loss": 0.0, "vf_loss": 0.0, "entropy": 0.0,
                "kl_to_bc": 0.0, "ratio_mean": 1.0, "n_updates": 0}

    model.train()
    pg_losses, vf_losses, entropies, kl_losses, ratios = [], [], [], [], []
    n_minibatches = max(1, len(buffer) // batch_snippets)

    for _ in range(n_epochs):
        for _ in range(n_minibatches):
            idx = buffer.sample_indices(batch_snippets, death_floor_frac)
            batch = buffer.get(idx)
            obs_f, act_f, old_lp_f, old_v_f, ret_f, mask_f = _flatten_batch(batch, device)

            # advantage = return - V_collect, normalized over valid frames in this minibatch
            adv = ret_f - old_v_f
            valid = mask_f > 0.5
            if valid.any():
                adv_v = adv[valid]
                adv = (adv - adv_v.mean()) / (adv_v.std() + 1e-8)

            logit, value = model(obs_f)
            logit = logit.squeeze(-1)
            value = value.squeeze(-1)

            dist = torch.distributions.Bernoulli(logits=logit)
            new_lp = dist.log_prob(act_f)
            entropy = dist.entropy()

            ratio = torch.exp(new_lp - old_lp_f)
            pg_unclipped = -adv * ratio
            pg_clipped = -adv * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
            pg_per = torch.max(pg_unclipped, pg_clipped)
            pg_loss = _masked_mean(pg_per, mask_f)

            vf_per = 0.5 * (ret_f - value).pow(2)
            vf_loss = _masked_mean(vf_per, mask_f)

            ent_mean = _masked_mean(entropy, mask_f)

            loss = pg_loss + vf_coef * vf_loss - ent_coef * ent_mean

            kl_value = 0.0
            if bc_anchor is not None and kl_coef > 0:
                with torch.no_grad():
                    bc_logit, _ = bc_anchor(obs_f)
                    bc_logit = bc_logit.squeeze(-1)
                bc_dist = torch.distributions.Bernoulli(logits=bc_logit)
                kl_per = torch.distributions.kl_divergence(dist, bc_dist)
                kl = _masked_mean(kl_per, mask_f)
                loss = loss + kl_coef * kl
                kl_value = float(kl.item())

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            pg_losses.append(pg_loss.item())
            vf_losses.append(vf_loss.item())
            entropies.append(ent_mean.item())
            kl_losses.append(kl_value)
            ratios.append(ratio.detach().mean().item())

    # Value-only epochs (PPG-lite). Updates value head + shared backbone;
    # action head receives no gradient (no path through vf_loss).
    for _ in range(value_epochs):
        for _ in range(n_minibatches):
            idx = buffer.sample_indices(batch_snippets, death_floor_frac)
            batch = buffer.get(idx)
            obs_f, _act, _lp, _v, ret_f, mask_f = _flatten_batch(batch, device)

            _, value = model(obs_f)
            value = value.squeeze(-1)
            vf_per = 0.5 * (ret_f - value).pow(2)
            vf_loss = _masked_mean(vf_per, mask_f)

            optimizer.zero_grad()
            vf_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            vf_losses.append(vf_loss.item())

    return {
        "pg_loss": float(np.mean(pg_losses)) if pg_losses else 0.0,
        "vf_loss": float(np.mean(vf_losses)) if vf_losses else 0.0,
        "entropy": float(np.mean(entropies)) if entropies else 0.0,
        "kl_to_bc": float(np.mean(kl_losses)) if kl_losses else 0.0,
        "ratio_mean": float(np.mean(ratios)) if ratios else 1.0,
        "n_updates": len(pg_losses),
    }


def _kl_coef_at(step: int, kl_init: float, kl_final: float, anneal_steps: int) -> float:
    if anneal_steps <= 0:
        return kl_init
    t = min(1.0, step / float(anneal_steps))
    return kl_init + (kl_final - kl_init) * t


def main() -> int:
    ap = argparse.ArgumentParser(description="Snippet-PPO training for GD privileged policy.")
    ap.add_argument("--out", default="artifacts/ppo_run")
    ap.add_argument("--bc-checkpoint", default=None, help="BC checkpoint to warm-start from.")
    ap.add_argument("--norm", default=None, help=".norm.npz normalizer (auto-detected from --bc-checkpoint).")
    ap.add_argument("--total-steps", type=int, default=1_000_000)
    ap.add_argument("--n-steps", type=int, default=4096, help="Env steps per rollout.")
    ap.add_argument("--snippet-len", type=int, default=20, help="Frames per snippet (K).")
    ap.add_argument("--buffer-size", type=int, default=4096, help="Cross-rollout snippet buffer capacity.")
    ap.add_argument("--max-snippet-age", type=int, default=4, help="Drop snippets older than this many updates.")
    ap.add_argument("--death-floor-frac", type=float, default=0.1,
                    help="Min fraction of each minibatch sourced from died-snippets pool.")
    ap.add_argument("--death-penalty", type=float, default=3.0,
                    help="Negative reward applied on the death frame.")
    ap.add_argument("--progress-scale", type=float, default=0.1,
                    help="Per-frame reward = progress_scale * delta-x.")
    ap.add_argument("--n-epochs", type=int, default=4, help="Policy+value epochs per update.")
    ap.add_argument("--value-epochs", type=int, default=1, help="Extra value-only epochs after policy.")
    ap.add_argument("--batch-snippets", type=int, default=64,
                    help="Snippets per minibatch. Frame-batch ≈ batch_snippets * snippet_len.")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--clip-range", type=float, default=0.1)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--kl-coef", type=float, default=0.3,
                    help="Initial KL anchor coefficient to BC. Annealed to --kl-coef-final.")
    ap.add_argument("--kl-coef-final", type=float, default=0.05)
    ap.add_argument("--kl-anneal-steps", type=int, default=200_000)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--stack", type=int, default=4)
    ap.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N updates.")
    ap.add_argument("--shm-name", default="gdrl_ipc_v3")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.out)
    ckpt_dir = out / "checkpoints"
    log_dir = out / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # Load normalizer (auto-detect .norm.npz next to BC checkpoint).
    norm_path = args.norm
    if norm_path is None and args.bc_checkpoint:
        auto = Path(args.bc_checkpoint).with_suffix(".norm.npz")
        if auto.exists():
            norm_path = str(auto)
    if norm_path:
        normalizer = ObsNormalizer.load(norm_path)
        print(f"loaded normalizer from {norm_path}", flush=True)
    else:
        normalizer = None
        print("no normalizer — using unscaled observations", flush=True)

    preprocessor = ObsPreprocessor(normalizer=normalizer)
    input_dim = PROCESSED_FRAME_DIM * args.stack

    if args.bc_checkpoint:
        ckpt = torch.load(args.bc_checkpoint, map_location="cpu", weights_only=False)
        input_dim = ckpt.get("input_dim", input_dim)
        stack_size = ckpt.get("stack_size", args.stack)
        model = GDPolicyMLP(input_dim=input_dim, stack_size=stack_size).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"warm-started from {args.bc_checkpoint} (epoch {ckpt.get('epoch', '?')})", flush=True)
    else:
        stack_size = args.stack
        model = GDPolicyMLP(input_dim=input_dim, stack_size=stack_size).to(device)
        print("starting from random weights", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Optional BC anchor: keep PPO policy close to BC via KL penalty.
    bc_anchor = None
    if args.kl_coef > 0 and args.bc_checkpoint:
        bc_anchor = GDPolicyMLP(input_dim=input_dim, stack_size=stack_size).to(device)
        bc_ckpt = torch.load(args.bc_checkpoint, map_location=device, weights_only=False)
        bc_anchor.load_state_dict(bc_ckpt["model_state_dict"])
        bc_anchor.eval()
        for p in bc_anchor.parameters():
            p.requires_grad = False
        print(f"BC anchor enabled (kl_coef init={args.kl_coef} final={args.kl_coef_final} "
              f"over {args.kl_anneal_steps} steps)", flush=True)

    adapter = GeodeV3Adapter(GeodeIPCV3Config(shm_name=args.shm_name))
    adapter.verify_version()
    env = _StackedEnv(GDPrivilegedEnv(ipc=adapter), preprocessor, stack_size=stack_size)
    buffer = SnippetBuffer(capacity=args.buffer_size, max_age=args.max_snippet_age)

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(log_dir))
        print(f"tensorboard -> tensorboard --logdir {log_dir}", flush=True)
    except ImportError:
        writer = None
        print("tensorboard not available (pip install tensorboard)", flush=True)

    n_updates = args.total_steps // args.n_steps
    minutes = args.total_steps / 60 / 60
    print(f"training: {args.total_steps:,} steps = {n_updates} updates", flush=True)
    print(f"snippet K={args.snippet_len}, buffer={args.buffer_size}, max_age={args.max_snippet_age}", flush=True)
    print(f"reward: progress*{args.progress_scale}, death=-{args.death_penalty}, no level bonus", flush=True)
    print(f"at 60fps live, ≈ {minutes:.0f} min ({minutes/60:.1f} hr) of game time", flush=True)
    print(f"checkpoints -> {ckpt_dir}", flush=True)

    obs, _ = env.reset()
    total_steps = 0
    t_start = time.time()

    for update in range(1, n_updates + 1):
        # Age (and evict stale) BEFORE adding new snippets so fresh ones stay age=0.
        buffer.age()

        obs, new_snippets, rollout_stats = _collect_rollout(
            env=env, model=model, n_steps=args.n_steps,
            snippet_len=args.snippet_len, gamma=args.gamma,
            death_penalty=args.death_penalty, progress_scale=args.progress_scale,
            obs_dim=input_dim, device=device, obs=obs,
        )
        buffer.add_many(new_snippets)
        total_steps += args.n_steps

        kl_coef_now = _kl_coef_at(total_steps, args.kl_coef, args.kl_coef_final, args.kl_anneal_steps)

        update_stats = _update(
            model=model, optimizer=optimizer, buffer=buffer,
            n_epochs=args.n_epochs, value_epochs=args.value_epochs,
            batch_snippets=args.batch_snippets,
            death_floor_frac=args.death_floor_frac,
            clip_range=args.clip_range, vf_coef=args.vf_coef, ent_coef=args.ent_coef,
            max_grad_norm=args.max_grad_norm, device=device,
            bc_anchor=bc_anchor, kl_coef=kl_coef_now,
        )

        fps = total_steps / (time.time() - t_start)
        n_died_buf = buffer.num_died()
        death_frac = n_died_buf / max(1, len(buffer))
        len_str = f"{rollout_stats['ep_len_mean']:.0f}" if rollout_stats["ep_len_mean"] is not None else "—"
        prog_str = f"{rollout_stats['ep_progress_mean']:.1f}" if rollout_stats["ep_progress_mean"] is not None else "—"
        kl_str = f" kl={update_stats['kl_to_bc']:.4f} kc={kl_coef_now:.3f}" if bc_anchor is not None else ""
        print(
            f"update={update}/{n_updates} steps={total_steps:,} fps={fps:.0f} "
            f"ep_len={len_str} ep_prog={prog_str} "
            f"buf={len(buffer)}({n_died_buf}d {death_frac:.0%}) "
            f"pg={update_stats['pg_loss']:.4f} "
            f"vf={update_stats['vf_loss']:.4f} "
            f"ent={update_stats['entropy']:.4f} "
            f"r={update_stats['ratio_mean']:.3f}"
            f"{kl_str}",
            flush=True,
        )

        if writer:
            if rollout_stats["ep_len_mean"] is not None:
                writer.add_scalar("rollout/ep_len_mean", rollout_stats["ep_len_mean"], total_steps)
            if rollout_stats["ep_progress_mean"] is not None:
                writer.add_scalar("rollout/ep_progress_mean", rollout_stats["ep_progress_mean"], total_steps)
            writer.add_scalar("rollout/n_episodes", rollout_stats["n_episodes"], total_steps)
            writer.add_scalar("rollout/n_died", rollout_stats["n_died"], total_steps)
            writer.add_scalar("rollout/n_new_snippets", rollout_stats["n_new_snippets"], total_steps)
            writer.add_scalar("buffer/size", len(buffer), total_steps)
            writer.add_scalar("buffer/death_fraction", death_frac, total_steps)
            writer.add_scalar("train/pg_loss", update_stats["pg_loss"], total_steps)
            writer.add_scalar("train/vf_loss", update_stats["vf_loss"], total_steps)
            writer.add_scalar("train/entropy", update_stats["entropy"], total_steps)
            writer.add_scalar("train/kl_to_bc", update_stats["kl_to_bc"], total_steps)
            writer.add_scalar("train/kl_coef", kl_coef_now, total_steps)
            writer.add_scalar("train/ratio_mean", update_stats["ratio_mean"], total_steps)
            writer.add_scalar("train/fps", fps, total_steps)

        if update % args.save_every == 0:
            ckpt_path = ckpt_dir / f"ppo_update{update:05d}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "stack_size": stack_size,
                "update": update,
                "total_steps": total_steps,
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_path)
            print(f"  checkpoint -> {ckpt_path}", flush=True)

    final_path = out / "ppo_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "stack_size": stack_size,
        "update": n_updates,
        "total_steps": total_steps,
    }, final_path)
    print(f"saved -> {final_path}", flush=True)

    if writer:
        writer.close()
    adapter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
