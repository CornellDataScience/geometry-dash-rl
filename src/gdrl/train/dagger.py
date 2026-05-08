"""DAgger for GDPolicyMLP — one full iteration end-to-end.

The pipeline:
  1. Load the current BC checkpoint (GDPolicyMLP + its normalizer).
  2. Roll out the policy in the live game for N episodes, recording every
     visited state as raw 608-dim obs.
  3. Label each visited state with the macro's action at the nearest
     X-position (re-using src/gdrl/data/dagger_align.py).
  4. Save the labeled rollout as a single shard in the same format as
     record_human shards, so imitation.py can read it directly.
  5. Set up a combined data directory (symlinks to the human recording
     sessions + the new DAgger session) and call imitation.py via subprocess
     with `--pretrained` pointing at the current checkpoint to fine-tune.

Run iteratively: feed the new checkpoint into the next invocation as `--policy`.

Usage:
    python -m gdrl.train.dagger \\
        --policy artifacts/bc_stereo_only.pt \\
        --human-data artifacts/recordings/stereo_madness \\
        --out-dir artifacts/dagger/iter_1 \\
        --episodes 20 \\
        --bc-epochs 30
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from gdrl.data.dagger_align import align_to_press_events, build_human_index, cluster_press_x
from gdrl.data.obs_dataset import TARGET_SEMANTICS, action_allowed, grounded_jump_labels
from gdrl.env.geode_ipc_v3 import GeodeIPCV3Config, GeodeV3Adapter
from gdrl.env.privileged_env import GDPrivilegedEnv
from gdrl.model.mlp_agent import GDPolicyMLP
from gdrl.model.obs_preprocess import (
    PROCESSED_FRAME_DIM,
    ObsNormalizer,
    ObsPreprocessor,
)


def _rollout_policy(
    policy: GDPolicyMLP,
    env: GDPrivilegedEnv,
    preprocessor: ObsPreprocessor,
    stack_size: int,
    episodes: int,
    device: torch.device,
    deterministic: bool,
    action_threshold: float,
):
    """Run policy for N episodes. Returns parallel arrays:
    - obs_raw  (T, 608)   raw per-frame obs straight from the IPC
    - x_pos    (T,)       per-frame x = obs[0]
    - actions  (T,)       action the policy actually took (uint8)
    - ep_ids   (T,)       1-indexed episode id

    `deterministic=True` selects logit > action_threshold (greedy).
    `deterministic=False` samples from Bernoulli(logits=logit - threshold)
    for diversity around the same live-eval decision boundary.
    """
    all_obs: list[np.ndarray] = []
    all_x: list[float] = []
    all_actions: list[int] = []
    all_ep_ids: list[int] = []

    stack = np.zeros((stack_size, PROCESSED_FRAME_DIM), dtype=np.float32)

    for ep_id in range(1, episodes + 1):
        raw_obs, _ = env.reset()
        x = float(raw_obs[0])
        processed = preprocessor.process_frame(raw_obs)
        stack[:] = processed

        ep_steps = 0
        ep_max_x = x
        done = False
        while not done:
            obs_t = torch.from_numpy(stack.reshape(-1).copy()).unsqueeze(0).to(device)
            with torch.no_grad():
                logit, _ = policy(obs_t)
                logit_v = float(logit.squeeze().item())
            if deterministic:
                action = int(logit_v > action_threshold)
            else:
                p = 1.0 / (1.0 + np.exp(-(logit_v - action_threshold)))
                action = int(np.random.random() < p)
            if not action_allowed(raw_obs):
                action = 0

            all_obs.append(raw_obs.copy())
            all_x.append(x)
            all_actions.append(action)
            all_ep_ids.append(ep_id)

            raw_obs, _, terminated, truncated, _ = env.step(action)
            x = float(raw_obs[0])
            if x > ep_max_x:
                ep_max_x = x
            processed = preprocessor.process_frame(raw_obs)
            stack[:-1] = stack[1:]
            stack[-1] = processed
            done = terminated or truncated
            ep_steps += 1

        print(f"  ep {ep_id}/{episodes}: steps={ep_steps} max_x={ep_max_x:.0f}", flush=True)

    return (
        np.asarray(all_obs, dtype=np.float32),
        np.asarray(all_x, dtype=np.float32),
        np.asarray(all_actions, dtype=np.uint8),
        np.asarray(all_ep_ids, dtype=np.uint32),
    )


def _setup_combined_data_dir(human_data_root: Path, out_dir: Path,
                             dagger_session_name: str) -> Path:
    """Create out_dir/data/ that imitation.py can consume.

    Creates real session directories in the combined dir and symlinks shard
    files inside them. pathlib's rglob does not recurse into symlinked
    directories on all platforms, so directory-level symlinks can silently
    exclude the human data from BC retraining.
    """
    combined = out_dir / "data"
    combined.mkdir(parents=True, exist_ok=True)

    for sess in human_data_root.iterdir():
        if not sess.is_dir():
            continue
        session_dir = combined / sess.name
        if session_dir.is_symlink():
            session_dir.unlink()
        session_dir.mkdir(exist_ok=True)
        for shard in sorted(sess.glob("shard_*.npz")):
            link = session_dir / shard.name
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(shard.resolve())

    dagger_session_dir = combined / dagger_session_name
    dagger_session_dir.mkdir(exist_ok=True)
    return dagger_session_dir


def main() -> int:
    ap = argparse.ArgumentParser(description="One DAgger iteration: rollout, label, retrain.")
    ap.add_argument("--policy", required=True,
                    help="GDPolicyMLP checkpoint (.pt). Its sibling .norm.npz must exist.")
    ap.add_argument("--human-data", required=True,
                    help="Directory of human macro recording sessions (each session a subdir of shard_*.npz).")
    ap.add_argument("--out-dir", required=True,
                    help="Output dir for this iteration's artifacts.")
    ap.add_argument("--episodes", type=int, default=20,
                    help="Episodes to roll out for state collection.")
    ap.add_argument("--bc-epochs", type=int, default=30, help="Epochs for the BC retrain.")
    ap.add_argument("--bc-patience", type=int, default=5)
    ap.add_argument("--bc-lr", type=float, default=3e-4)
    ap.add_argument("--bc-batch-size", type=int, default=256)
    ap.add_argument("--label-x-tolerance", type=float, default=20.0,
                    help="X-distance tolerance for matching rollout frames to grounded human press events.")
    ap.add_argument("--press-cluster-gap", type=float, default=80.0,
                    help="Cluster human press X positions within this distance into one expert event.")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Action logit threshold for rollout. Defaults to checkpoint action_threshold if present, else 0.")
    ap.add_argument("--deterministic", action="store_true",
                    help="Greedy action selection during rollouts. Default is stochastic Bernoulli "
                         "sampling for state diversity.")
    ap.add_argument("--shm-name", default="gdrl_ipc_v3")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # === Phase 1: load policy + normalizer ===
    print(f"[1/5] loading policy from {args.policy}...", flush=True)
    ckpt = torch.load(args.policy, map_location=device, weights_only=False)
    input_dim = ckpt.get("input_dim", PROCESSED_FRAME_DIM * 4)
    stack_size = ckpt.get("stack_size", 4)
    action_threshold = args.threshold
    if action_threshold is None:
        action_threshold = float(ckpt.get("action_threshold", 0.0))

    policy = GDPolicyMLP(input_dim=input_dim, stack_size=stack_size).to(device)
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()

    norm_path = Path(args.policy).with_suffix(".norm.npz")
    if not norm_path.exists():
        print(f"ERROR: normalizer not found at {norm_path}", file=sys.stderr)
        return 1
    normalizer = ObsNormalizer.load(str(norm_path))
    preprocessor = ObsPreprocessor(normalizer=normalizer)
    print(f"  input_dim={input_dim} stack={stack_size} mode={'greedy' if args.deterministic else 'stochastic'}",
          flush=True)
    print(f"  action_threshold={action_threshold:+.3f}", flush=True)

    # === Phase 2: roll out in live game ===
    print(f"[2/5] rolling out {args.episodes} episodes...", flush=True)
    adapter = GeodeV3Adapter(GeodeIPCV3Config(shm_name=args.shm_name))
    try:
        adapter.verify_version()
        env = GDPrivilegedEnv(ipc=adapter)
        obs_raw, x_pos, policy_actions, ep_ids = _rollout_policy(
            policy=policy, env=env, preprocessor=preprocessor,
            stack_size=stack_size, episodes=args.episodes,
            device=device, deterministic=args.deterministic,
            action_threshold=action_threshold,
        )
    finally:
        adapter.close()

    n_frames = len(obs_raw)
    print(f"  recorded {n_frames} frames across {args.episodes} episodes  "
          f"x range=[{x_pos.min():.0f}, {x_pos.max():.0f}]",
          flush=True)

    # === Phase 3: label each visited state with grounded human jump presses via x-position ===
    print(f"[3/5] aligning to grounded human jump presses from {args.human_data}...", flush=True)
    human_x, human_actions = build_human_index(args.human_data)
    raw_human_press_x = human_x[human_actions.astype(bool)]
    human_press_x = cluster_press_x(raw_human_press_x, gap=args.press_cluster_gap)
    print(f"  human grounded press samples={len(raw_human_press_x)}  "
          f"clustered_events={len(human_press_x)}  "
          f"x range=[{human_press_x.min():.0f}, {human_press_x.max():.0f}]"
          if len(human_press_x) else "  human grounded press events=0",
          flush=True)
    labeled_actions = align_to_press_events(
        x_pos,
        human_press_x,
        args.label_x_tolerance,
        rollout_on_ground=np.asarray([action_allowed(frame) for frame in obs_raw], dtype=bool),
        episode_ids=ep_ids,
    )

    grounded_policy_actions = grounded_jump_labels(obs_raw, policy_actions)
    grounded_agreement = float((grounded_policy_actions == labeled_actions).mean())
    print(f"  policy grounded_jump_rate={grounded_policy_actions.mean():.3f}  "
          f"label grounded_jump_rate={labeled_actions.mean():.3f}  "
          f"grounded_agreement={grounded_agreement:.3f}",
          flush=True)
    print(f"  -> {n_frames - int(grounded_agreement * n_frames)} grounded disagreement frames "
          f"(these are the new DAgger training samples)",
          flush=True)

    # === Phase 4: save shard + set up combined data dir ===
    print("[4/5] writing labeled shard and setting up combined data dir...", flush=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dagger_session_name = f"dagger_{timestamp}"

    dagger_session_dir = _setup_combined_data_dir(
        human_data_root=Path(args.human_data),
        out_dir=out_dir,
        dagger_session_name=dagger_session_name,
    )

    shard_path = dagger_session_dir / "shard_00000.npz"
    np.savez_compressed(
        shard_path,
        obs=obs_raw,
        actions=labeled_actions.astype(np.uint8),
        ticks=np.arange(n_frames, dtype=np.uint32),
        episode_ids=ep_ids,
        is_dead=np.zeros(n_frames, dtype=np.uint8),
        level_done=np.zeros(n_frames, dtype=np.uint8),
        target_semantics=np.array(TARGET_SEMANTICS),
        action_threshold=np.array(action_threshold, dtype=np.float32),
    )
    print(f"  shard -> {shard_path}", flush=True)
    print(f"  combined data dir -> {dagger_session_dir.parent}", flush=True)

    # Also save raw rollout artifacts for diagnostics
    raw_diag_path = out_dir / "rollout_diag.npz"
    np.savez_compressed(
        raw_diag_path,
        obs=obs_raw,
        x_pos=x_pos,
        policy_actions=policy_actions,
        grounded_policy_actions=grounded_policy_actions,
        labeled_actions=labeled_actions,
        episode_ids=ep_ids,
        action_threshold=np.array(action_threshold, dtype=np.float32),
    )
    print(f"  diagnostics -> {raw_diag_path}", flush=True)

    # === Phase 5: retrain via subprocess to imitation.py ===
    new_ckpt_path = out_dir / "bc_dagger.pt"
    cmd = [
        sys.executable, "-m", "gdrl.train.imitation",
        "--data-dir", str(dagger_session_dir.parent),
        "--out", str(new_ckpt_path),
        "--epochs", str(args.bc_epochs),
        "--patience", str(args.bc_patience),
        "--lr", str(args.bc_lr),
        "--batch-size", str(args.bc_batch_size),
        "--stack", str(stack_size),
        "--pretrained", args.policy,
        "--device", args.device,
    ]
    print(f"[5/5] retraining: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"BC retrain failed (exit {result.returncode})", file=sys.stderr)
        return result.returncode

    print("", flush=True)
    print("DAgger iteration complete.", flush=True)
    print(f"  new checkpoint: {new_ckpt_path}", flush=True)
    print(f"  next iteration: --policy {new_ckpt_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
