"""DAgger training loop orchestrator.

Runs the full DAgger pipeline iteratively:
  1. Roll out the current policy in the game (automated)
  2. Prompt the human to play the level once for labels
  3. Align rollout states to human labels by X position
  4. Aggregate all data into one growing dataset
  5. Retrain the policy on the aggregated dataset
  6. Repeat

After all iterations the final checkpoint is ready for PPO fine-tuning.

Usage:
    python -m gdrl.teacher.dagger_loop \
        --policy artifacts/bc_warmup.pt \
        --recordings artifacts/recordings/ \
        --out-dir artifacts/ \
        --iterations 8

Artifacts produced each iteration N:
    artifacts/rollouts/iterN.npz
    artifacts/dagger_labeled/iterN.npz
    artifacts/dagger_dataset_iterN/
    artifacts/dagger_iterN.pt          <- policy checkpoint for next iteration
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    """Run a subprocess command, raising on failure."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"command failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def prompt_human_recording(recordings_dir: str, iteration: int) -> None:
    """Pause and instruct the human to record a gameplay session."""
    print(
        f"\n{'='*60}\n"
        f"ITERATION {iteration} — HUMAN RECORDING NEEDED\n"
        f"{'='*60}\n"
        f"Open Geometry Dash, enter the level, then run this command\n"
        f"in a separate terminal:\n\n"
        f"  python -m gdrl.data.record_human --out {recordings_dir}\n\n"
        f"Play the level for 5-10 minutes (die and retry naturally).\n"
        f"Stop the recorder with Ctrl+C when done.\n"
        f"{'='*60}",
        flush=True,
    )
    input("\nPress Enter here once you have finished recording... ")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the full DAgger training loop.")
    ap.add_argument("--policy", required=True,
                    help="Starting policy checkpoint (bc_warmup.pt for iteration 1)")
    ap.add_argument("--recordings", default="artifacts/recordings/",
                    help="Human recordings directory (appended to each iteration)")
    ap.add_argument("--out-dir", default="artifacts/",
                    help="Root output directory for all artifacts")
    ap.add_argument("--iterations", type=int, default=8,
                    help="Number of DAgger iterations to run")
    ap.add_argument("--rollout-episodes", type=int, default=50,
                    help="Episodes per rollout session")
    ap.add_argument("--train-epochs", type=int, default=10,
                    help="BC training epochs per iteration")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=256,
                    help="Must match the starting policy's hidden size")
    ap.add_argument("--shm-name", default="gdrl_ipc_v3")
    ap.add_argument("--start-iter", type=int, default=1,
                    help="Resume from this iteration (1-indexed)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    rollout_dir = out / "rollouts"
    labeled_dir = out / "dagger_labeled"
    policy = args.policy

    print(f"DAgger loop: {args.iterations} iterations  start={args.start_iter}", flush=True)
    print(f"starting policy: {policy}", flush=True)

    for iteration in range(args.start_iter, args.iterations + 1):
        print(f"\n{'='*60}", flush=True)
        print(f"DAGGER ITERATION {iteration}/{args.iterations}", flush=True)
        print(f"{'='*60}", flush=True)

        rollout_path = rollout_dir / f"iter{iteration}.npz"
        labeled_path = labeled_dir / f"iter{iteration}.npz"
        dataset_dir = out / f"dagger_dataset_iter{iteration}"
        checkpoint_path = out / f"dagger_iter{iteration}.pt"

        # --- step 1: roll out the current policy ---
        print(f"\n[{iteration}/3] rolling out policy ...", flush=True)
        run([
            sys.executable, "-m", "gdrl.data.dagger_rollout",
            "--policy", policy,
            "--episodes", str(args.rollout_episodes),
            "--out", str(rollout_path),
            "--shm-name", args.shm_name,
            "--hidden", str(args.hidden),
        ])

        # --- step 2: human records the level ---
        prompt_human_recording(args.recordings, iteration)

        # --- step 3a: align rollout to human labels ---
        print(f"\n[{iteration}/3] aligning rollout to human labels ...", flush=True)
        run([
            sys.executable, "-m", "gdrl.data.dagger_align",
            "--rollout", str(rollout_path),
            "--human-data", args.recordings,
            "--out", str(labeled_path),
        ])

        # --- step 3b: aggregate all data ---
        print(f"\n[{iteration}/3] aggregating dataset ...", flush=True)
        run([
            sys.executable, "-m", "gdrl.data.dagger_dataset",
            "--human-data", args.recordings,
            "--labeled-dir", str(labeled_dir),
            "--out", str(dataset_dir),
        ])

        # --- step 3c: retrain ---
        print(f"\n[{iteration}/3] retraining policy ...", flush=True)
        run([
            sys.executable, "-m", "gdrl.teacher.bc_train",
            "--data", str(dataset_dir),
            "--out", str(checkpoint_path),
            "--epochs", str(args.train_epochs),
            "--batch-size", str(args.batch_size),
            "--hidden", str(args.hidden),
        ])

        policy = str(checkpoint_path)
        print(f"\niteration {iteration} complete  checkpoint -> {checkpoint_path}", flush=True)

    print(
        f"\n{'='*60}\n"
        f"DAgger complete. Final checkpoint: {policy}\n"
        f"\nNext step — PPO fine-tuning:\n"
        f"  python -m gdrl.teacher.train_ppo --checkpoint {policy}\n"
        f"{'='*60}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
