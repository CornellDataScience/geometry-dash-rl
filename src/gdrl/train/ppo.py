"""PPO training using GDPolicyMLP with optional warm-start from a BC checkpoint.

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
    """

    def __init__(self, env: GDPrivilegedEnv, preprocessor: ObsPreprocessor, stack_size: int = 4):
        self.env = env
        self.preprocessor = preprocessor
        self.stack_size = stack_size
        self._stack = np.zeros((stack_size, PROCESSED_FRAME_DIM), dtype=np.float32)

    def reset(self) -> tuple[torch.Tensor, dict]:
        raw_obs, info = self.env.reset()
        processed = self.preprocessor.process_frame(raw_obs)
        self._stack[:] = processed  # replicate first frame across all stack slots
        return torch.from_numpy(self._stack.reshape(-1).copy()), info

    def step(self, action: int) -> tuple[torch.Tensor, float, bool, bool, dict]:
        raw_obs, reward, terminated, truncated, info = self.env.step(action)
        processed = self.preprocessor.process_frame(raw_obs)
        self._stack[:-1] = self._stack[1:]
        self._stack[-1] = processed
        return torch.from_numpy(self._stack.reshape(-1).copy()), reward, terminated, truncated, info


class RolloutBuffer:
    def __init__(self, n_steps: int, obs_dim: int):
        self.n_steps = n_steps
        self.obs = torch.zeros(n_steps, obs_dim)
        self.actions = torch.zeros(n_steps)
        self.rewards = torch.zeros(n_steps)
        self.dones = torch.zeros(n_steps)
        self.values = torch.zeros(n_steps)
        self.log_probs = torch.zeros(n_steps)
        self.pos = 0
        self._returns: torch.Tensor | None = None
        self._advantages: torch.Tensor | None = None

    def add(self, obs: torch.Tensor, action: int, reward: float, done: bool, value: float, log_prob: float):
        i = self.pos
        self.obs[i] = obs
        self.actions[i] = float(action)
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.values[i] = value
        self.log_probs[i] = log_prob
        self.pos += 1

    def full(self) -> bool:
        return self.pos >= self.n_steps

    def reset(self):
        self.pos = 0
        self._returns = None
        self._advantages = None

    def prepare(self, last_value: float, gamma: float, gae_lambda: float):
        """Compute GAE advantages and returns, stored for use in batches()."""
        adv = torch.zeros(self.n_steps)
        last_gae = 0.0
        for t in reversed(range(self.n_steps)):
            next_nterm = 1.0 - float(self.dones[t])
            next_val = last_value if t == self.n_steps - 1 else float(self.values[t + 1])
            delta = float(self.rewards[t]) + gamma * next_val * next_nterm - float(self.values[t])
            last_gae = delta + gamma * gae_lambda * next_nterm * last_gae
            adv[t] = last_gae
        self._advantages = adv
        self._returns = adv + self.values

    def batches(self, batch_size: int, device: torch.device):
        assert self._advantages is not None, "call prepare() before batches()"
        adv = (self._advantages - self._advantages.mean()) / (self._advantages.std() + 1e-8)
        perm = torch.randperm(self.n_steps)
        for start in range(0, self.n_steps, batch_size):
            idx = perm[start:start + batch_size]
            yield (
                self.obs[idx].to(device),
                self.actions[idx].to(device),
                self.log_probs[idx].to(device),
                self._returns[idx].to(device),
                adv[idx].to(device),
            )


def _collect_rollout(
    env: _StackedEnv,
    model: GDPolicyMLP,
    buffer: RolloutBuffer,
    obs: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """Fill buffer with n_steps of experience. Returns (obs after rollout, episode stats)."""
    model.eval()
    ep_rewards: list[float] = []
    ep_lengths: list[int] = []
    cur_reward = 0.0
    cur_len = 0

    while not buffer.full():
        with torch.no_grad():
            logit, value = model(obs.unsqueeze(0).to(device))
            logit = logit.squeeze()
            value = float(value.squeeze())
            dist = torch.distributions.Bernoulli(logits=logit)
            action_t = dist.sample()
            log_prob = float(dist.log_prob(action_t))
            action = int(action_t.item())

        next_obs, reward, terminated, truncated, _ = env.step(action)
        episode_end = terminated or truncated

        # use terminated (not truncated) for GAE: truncated episodes still have future value
        buffer.add(obs, action, float(reward), terminated, value, log_prob)
        cur_reward += float(reward)
        cur_len += 1

        if episode_end:
            ep_rewards.append(cur_reward)
            ep_lengths.append(cur_len)
            cur_reward = 0.0
            cur_len = 0
            obs, _ = env.reset()
        else:
            obs = next_obs

    return obs, {
        "ep_rew_mean": float(np.mean(ep_rewards)) if ep_rewards else None,
        "ep_len_mean": float(np.mean(ep_lengths)) if ep_lengths else None,
        "n_episodes": len(ep_rewards),
    }


def _update(
    model: GDPolicyMLP,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    n_epochs: int,
    batch_size: int,
    clip_range: float,
    vf_coef: float,
    ent_coef: float,
    max_grad_norm: float,
    device: torch.device,
) -> dict:
    model.train()
    pg_losses, vf_losses, entropies = [], [], []

    for _ in range(n_epochs):
        for obs_b, act_b, old_lp_b, ret_b, adv_b in buffer.batches(batch_size, device):
            logit, value = model(obs_b)
            logit = logit.squeeze(-1)
            value = value.squeeze(-1)

            dist = torch.distributions.Bernoulli(logits=logit)
            new_log_prob = dist.log_prob(act_b)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_prob - old_lp_b)
            pg_loss = torch.max(
                -adv_b * ratio,
                -adv_b * torch.clamp(ratio, 1 - clip_range, 1 + clip_range),
            ).mean()

            vf_loss = 0.5 * (ret_b - value).pow(2).mean()

            loss = pg_loss + vf_coef * vf_loss - ent_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            pg_losses.append(pg_loss.item())
            vf_losses.append(vf_loss.item())
            entropies.append(entropy.item())

    return {
        "pg_loss": float(np.mean(pg_losses)),
        "vf_loss": float(np.mean(vf_losses)),
        "entropy": float(np.mean(entropies)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="PPO training for GD privileged policy.")
    ap.add_argument("--out", default="artifacts/ppo_run")
    ap.add_argument("--bc-checkpoint", default=None, help="BC checkpoint to warm-start from.")
    ap.add_argument("--norm", default=None, help=".norm.npz normalizer (auto-detected from --bc-checkpoint).")
    ap.add_argument("--total-steps", type=int, default=1_000_000)
    ap.add_argument("--n-steps", type=int, default=2048, help="Env steps per rollout.")
    ap.add_argument("--n-epochs", type=int, default=10, help="PPO update epochs per rollout.")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-range", type=float, default=0.2)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--stack", type=int, default=4)
    ap.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N updates.")
    ap.add_argument("--shm-name", default="gdrl_ipc_v3")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out = Path(args.out)
    ckpt_dir = out / "checkpoints"
    log_dir = out / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # load normalizer (auto-detect .norm.npz next to BC checkpoint)
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
    input_dim = PROCESSED_FRAME_DIM * args.stack  # 628

    # build model, optionally warm-starting backbone + action head from BC
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

    adapter = GeodeV3Adapter(GeodeIPCV3Config(shm_name=args.shm_name))
    adapter.verify_version()
    env = _StackedEnv(GDPrivilegedEnv(ipc=adapter), preprocessor, stack_size=stack_size)
    buffer = RolloutBuffer(n_steps=args.n_steps, obs_dim=input_dim)

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(log_dir))
        print(f"tensorboard -> tensorboard --logdir {log_dir}", flush=True)
    except ImportError:
        writer = None
        print("tensorboard not available (pip install tensorboard)", flush=True)

    n_updates = args.total_steps // args.n_steps
    print(f"training: {args.total_steps:,} steps = {n_updates} updates", flush=True)
    print(f"checkpoints -> {ckpt_dir}", flush=True)

    obs, _ = env.reset()
    total_steps = 0
    t_start = time.time()

    for update in range(1, n_updates + 1):
        buffer.reset()
        obs, rollout_stats = _collect_rollout(env, model, buffer, obs, device)
        total_steps += args.n_steps

        with torch.no_grad():
            _, last_val = model(obs.unsqueeze(0).to(device))
        buffer.prepare(float(last_val.squeeze()), args.gamma, args.gae_lambda)

        update_stats = _update(
            model, optimizer, buffer,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            clip_range=args.clip_range,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            max_grad_norm=args.max_grad_norm,
            device=device,
        )

        fps = total_steps / (time.time() - t_start)
        rew_str = f"{rollout_stats['ep_rew_mean']:.2f}" if rollout_stats["ep_rew_mean"] is not None else "—"
        len_str = f"{rollout_stats['ep_len_mean']:.0f}" if rollout_stats["ep_len_mean"] is not None else "—"
        print(
            f"update={update}/{n_updates} steps={total_steps:,} fps={fps:.0f} "
            f"ep_rew={rew_str} ep_len={len_str} "
            f"pg={update_stats['pg_loss']:.4f} "
            f"vf={update_stats['vf_loss']:.4f} "
            f"ent={update_stats['entropy']:.4f}",
            flush=True,
        )

        if writer and rollout_stats["ep_rew_mean"] is not None:
            writer.add_scalar("rollout/ep_rew_mean", rollout_stats["ep_rew_mean"], total_steps)
            writer.add_scalar("rollout/ep_len_mean", rollout_stats["ep_len_mean"], total_steps)
            writer.add_scalar("rollout/n_episodes", rollout_stats["n_episodes"], total_steps)
            writer.add_scalar("train/pg_loss", update_stats["pg_loss"], total_steps)
            writer.add_scalar("train/vf_loss", update_stats["vf_loss"], total_steps)
            writer.add_scalar("train/entropy", update_stats["entropy"], total_steps)
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
