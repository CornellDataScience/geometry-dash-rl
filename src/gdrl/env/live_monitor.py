from __future__ import annotations

import argparse
import sys

import time
from multiprocessing import shared_memory

from gdrl.env.geode_ipc import GeodeIPCConfig, GeodeSharedMemoryAdapter


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


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Monitor live Geometry Dash shared-memory telemetry."
    )
    ap.add_argument("--shm-name", default="gdrl_ipc", help="Shared memory segment name.")
    ap.add_argument(
        "--connect-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for Geode segment to appear.",
    )
    ap.add_argument(
        "--tick-timeout",
        type=float,
        default=1.0,
        help="Seconds to wait for each tick update before warning.",
    )
    ap.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames (0 means run until Ctrl+C).",
    )
    ap.add_argument(
        "--print-every",
        type=int,
        default=50,
        help="Print telemetry once every N ticks.",
    )
    args = ap.parse_args()
    if args.print_every <= 0:
        print("--print-every must be > 0", file=sys.stderr)
        return 2

    print(
        f"waiting for shared memory '{args.shm_name}' (timeout={args.connect_timeout:.1f}s)...",
        flush=True,
    )
    if not _wait_for_segment(name=args.shm_name, timeout_s=args.connect_timeout):
        print("timeout waiting for Geode shared memory segment", file=sys.stderr)
        return 2

    ad = GeodeSharedMemoryAdapter(GeodeIPCConfig(shm_name=args.shm_name))
    try:
        ad.verify_version()
        version, tick0 = ad._read_header()
        print("connected. press Ctrl+C to stop.", flush=True)
        print(f"header: version={version} tick={tick0}", flush=True)
        print(
            "columns: tick x y vy ground dead mode complete",
            flush=True,
        )

        frames = 0
        last_print_bucket = -1
        beat_announced = False
        while True:
            updated = ad.wait_next_tick(timeout_s=args.tick_timeout)
            if not updated:
                tick_now = ad.read_tick()
                print(
                    f"no tick update for {args.tick_timeout:.1f}s "
                    f"(current tick={tick_now}; open a level and keep it running)",
                    flush=True,
                )
                continue

            obs = ad.read_obs()
            tick = ad.read_tick()
            level_complete = ad.read_level_complete_flag()
            if level_complete and not beat_announced:
                print(
                    f"event: level_complete tick={tick}",
                    flush=True,
                )
                beat_announced = True
            elif not level_complete and beat_announced:
                beat_announced = False

            bucket = tick // args.print_every
            if bucket != last_print_bucket:
                print(
                    f"tick={tick:7d} x={obs[0]:8.2f} y={obs[1]:8.2f} vy={obs[2]:7.2f} "
                    f"ground={int(obs[4])} dead={int(obs[5])} mode={int(obs[7])} "
                    f"complete={int(level_complete)}",
                    flush=True,
                )
                last_print_bucket = bucket

            frames += 1
            if args.max_frames > 0 and frames >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("\nstopped by user", flush=True)
    finally:
        ad.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
