"""PPO fine-tuning for the privileged teacher policy.

Loads a DAgger-trained TeacherPolicy checkpoint and fine-tunes it with PPO
via the live game environment. The DAgger weights are transferred into the
PPO policy network so training starts from a strong imitation-learned baseline
rather than random initialization.

Usage:
    python -m gdrl.teacher.train_ppo \
        --checkpoint artifacts/dagger_iter8.pt \
        --timesteps 500000 \
        --out artifacts/teacher_final
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from gdrl.env.geode_ipc_v3 import GeodeV3Adapter, GeodeIPCV3Config
from gdrl.env.privileged_env import GDPrivilegedEnv
from gdrl.teacher.model import TeacherPolicy, INPUT_DIM


def _transfer_weights(teacher: TeacherPolicy, ppo: PPO) -> None:
    """Copy TeacherPolicy weights into the PPO actor network.

    TeacherPolicy layout:     net[0] Linear(2432,h) net[2] Linear(h,h) net[4] Linear(h,2)
    SB3 MlpPolicy layout:     mlp_extractor.policy_net[0] Linear(2432,h)
                               mlp_extractor.policy_net[2] Linear(h,h)
                               action_net Linear(h,2)
    """
    sd = teacher.state_dict()
    ppo_policy = ppo.policy

    with torch.no_grad():
        ppo_policy.mlp_extractor.policy_net[0].weight.copy_(sd["net.0.weight"])
        ppo_policy.mlp_extractor.policy_net[0].bias.copy_(sd["net.0.bias"])
        ppo_policy.mlp_extractor.policy_net[2].weight.copy_(sd["net.2.weight"])
        ppo_policy.mlp_extractor.policy_net[2].bias.copy_(sd["net.2.bias"])
        ppo_policy.action_net.weight.copy_(sd["net.4.weight"])
        ppo_policy.action_net.bias.copy_(sd["net.4.bias"])

    print("transferred DAgger weights into PPO actor network", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="PPO fine-tuning from a DAgger checkpoint.")
    ap.add_argument("--checkpoint", default=None,
                    help="DAgger TeacherPolicy checkpoint to warm-start from")
    ap.add_argument("--timesteps", type=int, default=500_000,
                    help="Total environment steps to train for")
    ap.add_argument("--out", default="artifacts/teacher_final",
                    help="Output path for the saved PPO model (no extension)")
    ap.add_argument("--hidden", type=int, default=256,
                    help="Hidden size — must match the checkpoint")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-steps", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--shm-name", default="gdrl_ipc_v3")
    args = ap.parse_args()

    adapter = GeodeV3Adapter(GeodeIPCV3Config(shm_name=args.shm_name))
    adapter.verify_version()

    def make_env():
        return GDPrivilegedEnv(ipc=adapter)

    env = DummyVecEnv([make_env])

    # net_arch must match TeacherPolicy hidden size so weights transfer cleanly
    policy_kwargs = dict(net_arch=[args.hidden, args.hidden])

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        device="cpu",
    )

    if args.checkpoint:
        print(f"loading DAgger checkpoint from {args.checkpoint} ...", flush=True)
        teacher = TeacherPolicy(hidden=args.hidden)
        teacher.load_state_dict(
            torch.load(args.checkpoint, map_location="cpu")
        )
        _transfer_weights(teacher, model)
    else:
        print("no checkpoint provided — starting PPO from random weights", flush=True)

    print(f"training for {args.timesteps:,} timesteps ...", flush=True)
    model.learn(total_timesteps=args.timesteps)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    print(f"saved -> {out}.zip", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
