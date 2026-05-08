"""Rainbow DQN trainer over CV input + mode-gated MoE.

Loop:
  - Curriculum samples a level → env.reset(level_id, percent)
  - Roll out one episode (action via NoisyNet train-mode), store transitions in PER
  - Every `train_freq` env steps, sample a PER batch and run one Double-DQN update
  - Every `target_update` grad steps, copy online → target
  - Every `eval_every` env steps, pause and run a greedy rollout per level

CLI examples:
    python -m gdrl.train.dqn_cv --levels stereo_madness --no-curriculum
    python -m gdrl.train.dqn_cv --levels stereo_madness back_on_track polargeist
"""
from __future__ import annotations
import argparse
import os
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from gdrl.env.cv_env import GDCvEnv, CvEnvConfig
from gdrl.env.geode_ipc_v3 import GeodeV4Adapter, FRAME_W, FRAME_H
from gdrl.model.cv_moe_dqn import ModeGatedRainbow, AUX_DIM
from gdrl.train.per_buffer import PERBuffer


LEVELS = {
    "stereo_madness": 1,
    "back_on_track":  2,
    "polargeist":     3,
}


@dataclass
class TrainCfg:
    levels: tuple[str, ...] = ("stereo_madness",)
    use_curriculum: bool = False
    win_threshold: float = 0.6
    win_window: int = 20
    earlier_level_prob: float = 0.30

    # optim / replay
    lr: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 32
    buffer_capacity: int = 100_000
    frame_capacity: int = 200_000        # ~3x trans capacity to allow stack reuse
    n_step: int = 3
    target_update: int = 1000            # grad steps
    train_freq: int = 4                  # env steps per grad step
    learning_start: int = 2000           # min transitions before training

    # PER
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_anneal: int = 1_000_000

    # NoisyNet
    noisy_std: float = 0.5

    # aux
    aux_weight: float = 0.1
    aux_enabled: bool = True

    # eval
    eval_every_env_steps: int = 10_000
    eval_episodes: int = 1
    eval_max_steps: int = 5000

    # episode
    max_steps_per_episode: int = 5000
    random_spawn_prob: float = 0.30      # mid-level random spawn probability
    random_spawn_max_pct: int = 80

    # checkpointing / logging
    artifacts_dir: str = "artifacts/dqn_cv"
    log_every: int = 100                 # env steps
    save_every_episodes: int = 50

    # device
    device: str = "auto"

    # safety
    max_env_steps: int = 5_000_000


# ---------- helpers ----------

def _device(s: str) -> torch.device:
    if s == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(s)


def _aux_targets_from_obs(raw_obs: np.ndarray) -> np.ndarray:
    """Pick a small stable subset of privileged state for free supervision.

    Returns AUX_DIM features: [y, vy, dx, on_ground, nearestX, nearestY].
    Coarse normalization to keep MSE losses well-scaled.
    """
    y = raw_obs[1] / 500.0
    vy = raw_obs[2] / 30.0
    dx = raw_obs[3] / 30.0
    on_ground = raw_obs[4]
    # nearest object is at obs[8..9] (relX, relY); divide by their typical scale
    nearestX = raw_obs[8] / 500.0
    nearestY = raw_obs[9] / 500.0
    return np.array([y, vy, dx, on_ground, nearestX, nearestY], dtype=np.float32)


class Curriculum:
    def __init__(self, levels: list[tuple[str, int]], cfg: TrainCfg):
        self.levels = levels
        self.cfg = cfg
        self.idx = 0
        self.recent: deque[int] = deque(maxlen=cfg.win_window)
        self.per_level_winrates: dict[int, deque[int]] = {
            lid: deque(maxlen=cfg.win_window) for _, lid in levels
        }
        self.per_level_max_pct: dict[int, float] = {lid: 0.0 for _, lid in levels}

    def sample(self) -> tuple[str, int]:
        if not self.cfg.use_curriculum:
            return self.levels[self.idx]
        if self.idx > 0 and np.random.rand() < self.cfg.earlier_level_prob:
            i = np.random.randint(0, self.idx)
            return self.levels[i]
        return self.levels[self.idx]

    def report(self, level_id: int, won: bool, max_pct: float) -> None:
        self.per_level_winrates[level_id].append(1 if won else 0)
        if max_pct > self.per_level_max_pct[level_id]:
            self.per_level_max_pct[level_id] = max_pct
        if not self.cfg.use_curriculum:
            return
        cur_lid = self.levels[self.idx][1]
        if level_id == cur_lid:
            self.recent.append(1 if won else 0)
            if len(self.recent) >= self.cfg.win_window:
                wr = sum(self.recent) / len(self.recent)
                if wr >= self.cfg.win_threshold and self.idx + 1 < len(self.levels):
                    self.idx += 1
                    self.recent.clear()


# ---------- trainer ----------

class Trainer:
    def __init__(self, cfg: TrainCfg):
        self.cfg = cfg
        self.device = _device(cfg.device)

        os.makedirs(cfg.artifacts_dir, exist_ok=True)
        os.makedirs(os.path.join(cfg.artifacts_dir, "logs"), exist_ok=True)

        self.env = GDCvEnv(cfg=CvEnvConfig(
            stack=4,
            max_steps=cfg.max_steps_per_episode,
        ))

        self.online = ModeGatedRainbow(
            in_channels=1, stack=4, num_modes=4, num_actions=2,
            noisy_std=cfg.noisy_std, aux=cfg.aux_enabled,
        ).to(self.device)
        self.target = ModeGatedRainbow(
            in_channels=1, stack=4, num_modes=4, num_actions=2,
            noisy_std=cfg.noisy_std, aux=cfg.aux_enabled,
        ).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad = False

        self.optim = torch.optim.AdamW(self.online.parameters(), lr=cfg.lr)

        self.buf = PERBuffer(
            capacity=cfg.buffer_capacity,
            frame_capacity=cfg.frame_capacity,
            stack=4, n_step=cfg.n_step, gamma=cfg.gamma,
            alpha=cfg.per_alpha,
            beta_start=cfg.per_beta_start, beta_end=cfg.per_beta_end,
            beta_anneal_steps=cfg.per_beta_anneal,
            h=FRAME_H, w=FRAME_W, c=1,
        )

        levels = [(name, LEVELS[name]) for name in cfg.levels if name in LEVELS]
        if not levels:
            raise ValueError(f"no valid levels in {cfg.levels}; known={list(LEVELS)}")
        self.curriculum = Curriculum(levels, cfg)

        self.tb = SummaryWriter(os.path.join(cfg.artifacts_dir, "logs"))
        self.env_steps = 0
        self.grad_steps = 0
        self.episode = 0
        self.last_eval_step = 0

    # --- action / acting ---

    @torch.no_grad()
    def _act(self, frames: np.ndarray, mode_id: int, eval_mode: bool = False) -> int:
        f = torch.from_numpy(frames).unsqueeze(0).to(self.device)
        m = torch.tensor([mode_id], dtype=torch.long, device=self.device)
        was_train = self.online.training
        if eval_mode:
            self.online.eval()
        q = self.online(f, m)
        if eval_mode and was_train:
            self.online.train()
        return int(q.argmax(dim=-1).item())

    # --- training step ---

    def _train_step(self) -> dict:
        batch = self.buf.sample(self.cfg.batch_size)
        states = torch.from_numpy(batch["states"]).to(self.device)
        next_states = torch.from_numpy(batch["next_states"]).to(self.device)
        actions = torch.from_numpy(batch["actions"]).to(self.device)
        rewards = torch.from_numpy(batch["rewards"]).to(self.device)
        gamma_n = torch.from_numpy(batch["gamma_n"]).to(self.device)
        dones = torch.from_numpy(batch["dones"]).to(self.device)
        modes = torch.from_numpy(batch["mode_ids"]).to(self.device)
        next_modes = torch.from_numpy(batch["next_mode_ids"]).to(self.device)
        weights = torch.from_numpy(batch["weights"]).to(self.device)

        # noisy nets: resample noise per update so each batch has fresh exploration noise
        self.online.reset_noise()
        self.target.reset_noise()

        if self.cfg.aux_enabled:
            q, aux_pred = self.online(states, modes, return_aux=True)
        else:
            q = self.online(states, modes)
            aux_pred = None
        curr_q = q.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_online = self.online(next_states, next_modes)
            next_a = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target = self.target(next_states, next_modes).gather(1, next_a).squeeze(1)
            target = rewards + gamma_n * (1.0 - dones) * next_q_target

        td = curr_q - target
        loss_q = (weights * F.smooth_l1_loss(curr_q, target, reduction="none")).mean()

        loss_aux_val = 0.0
        if self.cfg.aux_enabled and aux_pred is not None and batch["aux_targets"] is not None:
            aux_targets = torch.from_numpy(batch["aux_targets"]).to(self.device)
            aux_mask = torch.from_numpy(batch["aux_mask"]).to(self.device).unsqueeze(1)
            denom = max(float(aux_mask.detach().sum().item()), 1.0)
            loss_aux = (aux_mask * (aux_pred - aux_targets).pow(2)).sum() / denom
            loss_aux_val = float(loss_aux.item())
            loss = loss_q + self.cfg.aux_weight * loss_aux
        else:
            loss = loss_q

        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optim.step()

        self.buf.update_priorities(batch["slots"], td.detach().abs().cpu().numpy())
        self.grad_steps += 1

        if self.grad_steps % self.cfg.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())

        return {
            "loss_q": float(loss_q.item()),
            "loss_aux": loss_aux_val,
            "td_mean": float(td.detach().abs().mean().item()),
            "q_mean": float(curr_q.detach().mean().item()),
            "weights_mean": float(weights.mean().item()),
            "beta": self.buf.beta(),
        }

    # --- episode driver ---

    def _run_episode(self, level_name: str, level_id: int, eval_mode: bool = False) -> dict:
        # spawn percent: random ≥1% during training, always 0% during eval
        if eval_mode or np.random.rand() >= self.cfg.random_spawn_prob:
            spawn_pct = 0
        else:
            spawn_pct = int(np.random.randint(1, max(2, self.cfg.random_spawn_max_pct)))

        obs, info = self.env.reset(options={"level_id": level_id, "percent": spawn_pct})
        ep_id = self.env.ipc.read_episode_id()

        # seed PER: ensure the frame ring has the initial stack tagged with this episode
        if not eval_mode:
            stacked: np.ndarray = obs["frame"]    # (T, 1, H, W) uint8
            initial_frames = [stacked[i, 0] for i in range(stacked.shape[0])]
            self.buf.begin_episode(ep_id, initial_frames)

        ep_reward = 0.0
        ep_steps = 0
        max_pct = info.get("percent", 0.0)
        won = False

        prev_mode = info["mode_id"]
        while True:
            action = self._act(obs["frame"], obs["mode_id"], eval_mode=eval_mode)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            ep_reward += reward
            ep_steps += 1
            if not eval_mode:
                self.env_steps += 1

            if not eval_mode:
                # store transition: only the latest captured frame is the "new" one
                new_frame: np.ndarray = next_obs["frame"][-1, 0]   # (H, W)
                aux = _aux_targets_from_obs(info["raw_obs"]) if self.cfg.aux_enabled else None
                self.buf.add(
                    new_frame=new_frame,
                    action=action,
                    reward=float(reward),
                    done=bool(terminated),
                    mode_id=int(prev_mode),
                    next_mode_id=int(next_obs["mode_id"]),
                    level_id=int(level_id),
                    episode_id=int(ep_id),
                    aux_target=aux,
                )
                prev_mode = int(next_obs["mode_id"])

                if (self.buf.size >= max(self.cfg.batch_size, self.cfg.learning_start)
                        and self.env_steps % self.cfg.train_freq == 0):
                    metrics = self._train_step()
                    if self.grad_steps % self.cfg.log_every == 0:
                        for k, v in metrics.items():
                            self.tb.add_scalar(f"train/{k}", v, self.env_steps)

                # eval cadence
                if self.env_steps - self.last_eval_step >= self.cfg.eval_every_env_steps:
                    self.last_eval_step = self.env_steps
                    self._eval_all_levels()

            obs = next_obs
            max_pct = max(max_pct, info.get("percent", 0.0))

            if terminated:
                won = bool(info.get("percent", 0.0) >= 99.0 and not info.get("is_dead"))
                break
            if truncated:
                break
            if not eval_mode and self.env_steps >= self.cfg.max_env_steps:
                break

        return {
            "level_name": level_name,
            "level_id": level_id,
            "spawn_pct": spawn_pct,
            "reward": ep_reward,
            "steps": ep_steps,
            "max_pct": max_pct,
            "won": won,
        }

    def _eval_all_levels(self) -> None:
        for level_name, level_id in self.curriculum.levels:
            for _ in range(self.cfg.eval_episodes):
                stat = self._run_episode(level_name, level_id, eval_mode=True)
                self.tb.add_scalar(f"eval/{level_name}/max_pct", stat["max_pct"], self.env_steps)
                self.tb.add_scalar(f"eval/{level_name}/won", float(stat["won"]), self.env_steps)
                self.tb.add_scalar(f"eval/{level_name}/reward", stat["reward"], self.env_steps)
                print(f"[eval][{level_name}] max_pct={stat['max_pct']:.1f} won={stat['won']} reward={stat['reward']:.1f}")

    # --- top-level ---

    def run(self) -> None:
        print(f"device={self.device} levels={[(n,i) for n,i in self.curriculum.levels]} curriculum={self.cfg.use_curriculum}")
        try:
            while self.env_steps < self.cfg.max_env_steps:
                level_name, level_id = self.curriculum.sample()
                stat = self._run_episode(level_name, level_id, eval_mode=False)
                self.episode += 1
                self.curriculum.report(level_id, stat["won"], stat["max_pct"])

                self.tb.add_scalar("episode/reward", stat["reward"], self.env_steps)
                self.tb.add_scalar("episode/steps", stat["steps"], self.env_steps)
                self.tb.add_scalar("episode/max_pct", stat["max_pct"], self.env_steps)
                self.tb.add_scalar(f"episode/{level_name}/max_pct", stat["max_pct"], self.env_steps)
                self.tb.add_scalar(f"episode/{level_name}/won", float(stat["won"]), self.env_steps)
                self.tb.add_scalar("episode/curriculum_idx", self.curriculum.idx, self.env_steps)
                self.tb.add_scalar("buffer/size", self.buf.size, self.env_steps)

                print(
                    f"ep={self.episode} env_steps={self.env_steps} "
                    f"level={level_name} spawn={stat['spawn_pct']} max_pct={stat['max_pct']:.1f} "
                    f"reward={stat['reward']:.1f} won={stat['won']} buf={self.buf.size}"
                )

                if self.episode % self.cfg.save_every_episodes == 0:
                    self._save("latest.pt")
        finally:
            self._save("final.pt")
            self.tb.flush()
            self.tb.close()
            self.env.close()

    def _save(self, name: str) -> None:
        path = os.path.join(self.cfg.artifacts_dir, name)
        torch.save({
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optim": self.optim.state_dict(),
            "env_steps": self.env_steps,
            "grad_steps": self.grad_steps,
            "episode": self.episode,
            "curriculum_idx": self.curriculum.idx,
            "per_level_winrates": {k: list(v) for k, v in self.curriculum.per_level_winrates.items()},
            "per_level_max_pct": dict(self.curriculum.per_level_max_pct),
            "cfg": self.cfg.__dict__,
        }, path)
        print(f"saved checkpoint -> {path}")


# ---------- CLI ----------

def parse_args(argv: list[str] | None = None) -> TrainCfg:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="+", default=["stereo_madness"], choices=list(LEVELS))
    ap.add_argument("--no-curriculum", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--buffer-capacity", type=int, default=100_000)
    ap.add_argument("--frame-capacity", type=int, default=200_000)
    ap.add_argument("--n-step", type=int, default=3)
    ap.add_argument("--target-update", type=int, default=1000)
    ap.add_argument("--train-freq", type=int, default=4)
    ap.add_argument("--learning-start", type=int, default=2000)
    ap.add_argument("--eval-every-env-steps", type=int, default=10_000)
    ap.add_argument("--max-env-steps", type=int, default=5_000_000)
    ap.add_argument("--max-steps-per-episode", type=int, default=5000)
    ap.add_argument("--random-spawn-prob", type=float, default=0.30)
    ap.add_argument("--aux-weight", type=float, default=0.1)
    ap.add_argument("--no-aux", action="store_true")
    ap.add_argument("--noisy-std", type=float, default=0.5)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--artifacts-dir", type=str, default="artifacts/dqn_cv")
    a = ap.parse_args(argv)
    return TrainCfg(
        levels=tuple(a.levels),
        use_curriculum=not a.no_curriculum,
        lr=a.lr, batch_size=a.batch_size,
        buffer_capacity=a.buffer_capacity, frame_capacity=a.frame_capacity,
        n_step=a.n_step, target_update=a.target_update, train_freq=a.train_freq,
        learning_start=a.learning_start,
        eval_every_env_steps=a.eval_every_env_steps,
        max_env_steps=a.max_env_steps, max_steps_per_episode=a.max_steps_per_episode,
        random_spawn_prob=a.random_spawn_prob,
        aux_weight=a.aux_weight, aux_enabled=not a.no_aux,
        noisy_std=a.noisy_std, device=a.device, artifacts_dir=a.artifacts_dir,
    )


def main() -> int:
    cfg = parse_args()
    Trainer(cfg).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
