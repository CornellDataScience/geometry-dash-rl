"""Record human gameplay from V3 ring buffer to NPZ shards.

Usage:
    python -m gdrl.data.record_human --out artifacts/recordings/ \\
        --shard-size 10000 --shm-name gdrl_ipc_v3

The recorder consumes the mod's ring buffer and writes shards. Each shard:
    obs:         (N, 608) float32
    actions:     (N,)     uint8       # 1 if human jumped this frame
    ticks:       (N,)     uint32
    episode_ids: (N,)     uint32
    is_dead:     (N,)     uint8
    level_done:  (N,)     uint8
"""
from __future__ import annotations
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from multiprocessing import shared_memory

import numpy as np

from gdrl.env.geode_ipc_v3 import GeodeV3Adapter, GeodeIPCV3Config, OBS_DIM


def _wait_for_segment(name: str, timeout_s: float) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            shm = shared_memory.SharedMemory(name=name, create=False)
            shm.close()
            return True
        except FileNotFoundError:
            time.sleep(0.2)
    return False


class ShardWriter:
    def __init__(self, out_root: Path, shard_size: int, session_name: str | None = None):
        """Create a session subdir under out_root and write shards into it.

        Each recording run gets its own timestamped subdir so sessions never
        clobber each other. Example: artifacts/recordings/20260408_143022/
        """
        if session_name is None:
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_dir = Path(out_root) / session_name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.shard_idx = 0
        self._reset_buffers()

    def _reset_buffers(self):
        n = self.shard_size
        self.obs = np.zeros((n, OBS_DIM), dtype=np.float32)
        self.actions = np.zeros(n, dtype=np.uint8)
        self.ticks = np.zeros(n, dtype=np.uint32)
        self.episode_ids = np.zeros(n, dtype=np.uint32)
        self.is_dead = np.zeros(n, dtype=np.uint8)
        self.level_done = np.zeros(n, dtype=np.uint8)
        self.fill = 0

    def append(self, frame) -> None:
        i = self.fill
        self.obs[i] = frame.obs
        self.actions[i] = frame.player_input
        self.ticks[i] = frame.tick
        self.episode_ids[i] = frame.episode_id
        self.is_dead[i] = frame.is_dead
        self.level_done[i] = frame.level_done
        self.fill += 1
        if self.fill >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if self.fill == 0:
            return
        path = self.out_dir / f"shard_{self.shard_idx:05d}.npz"
        np.savez_compressed(
            path,
            obs=self.obs[:self.fill],
            actions=self.actions[:self.fill],
            ticks=self.ticks[:self.fill],
            episode_ids=self.episode_ids[:self.fill],
            is_dead=self.is_dead[:self.fill],
            level_done=self.level_done[:self.fill],
        )
        print(
            f"wrote {path.name}: {self.fill} frames "
            f"jumps={int(self.actions[:self.fill].sum())}",
            flush=True,
        )
        self.shard_idx += 1
        self._reset_buffers()


def main() -> int:
    ap = argparse.ArgumentParser(description="Record human gameplay to NPZ shards.")
    ap.add_argument("--shm-name", default="gdrl_ipc_v3")
    ap.add_argument("--out", default="artifacts/recordings/")
    ap.add_argument("--shard-size", type=int, default=10000)
    ap.add_argument("--connect-timeout", type=float, default=30.0)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="Stop after N frames (0 = run until Ctrl+C)")
    ap.add_argument("--status-every", type=float, default=2.0,
                    help="Print status every N seconds.")
    args = ap.parse_args()

    print(f"waiting for shared memory '{args.shm_name}'...", flush=True)
    if not _wait_for_segment(args.shm_name, args.connect_timeout):
        print("timeout waiting for V3 SHM segment", file=sys.stderr)
        return 2

    ad = GeodeV3Adapter(GeodeIPCV3Config(shm_name=args.shm_name))
    try:
        ad.verify_version()
    except Exception as e:
        print(f"version check failed: {e}", file=sys.stderr)
        ad.close()
        return 2

    writer = ShardWriter(Path(args.out), args.shard_size)
    print(f"writing shards to: {writer.out_dir}", flush=True)
    total_frames = 0
    total_dropped = 0
    last_status = time.time()
    last_status_frames = 0

    print("recording. press Ctrl+C to stop.", flush=True)
    try:
        while True:
            if not ad.wait_for_frames(timeout_s=1.0):
                # no frames in last second — game probably paused
                continue
            frames, dropped = ad.drain_ring()
            if dropped:
                total_dropped += dropped
                print(f"WARNING: dropped {dropped} frames (Python too slow)", flush=True)
            for f in frames:
                writer.append(f)
                total_frames += 1
                if args.max_frames > 0 and total_frames >= args.max_frames:
                    break

            now = time.time()
            if now - last_status >= args.status_every:
                fps = (total_frames - last_status_frames) / (now - last_status)
                print(
                    f"recorded={total_frames} fps={fps:.1f} "
                    f"dropped={total_dropped} shard_fill={writer.fill}",
                    flush=True,
                )
                last_status = now
                last_status_frames = total_frames

            if args.max_frames > 0 and total_frames >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("\nstopped by user", flush=True)
    finally:
        writer.flush()
        ad.close()
        print(f"total: {total_frames} frames, {total_dropped} dropped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
