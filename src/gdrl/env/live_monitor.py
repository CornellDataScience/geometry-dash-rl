from __future__ import annotations

import argparse
import sys

import time
from multiprocessing import shared_memory

from gdrl.env.geode_ipc_v3 import GeodeIPCV3Config, GeodeV3Adapter

OBJ_OBS_START = 8
FLOATS_PER_OBJ = 6  # relX, relY, objType, objID, scaleX, scaleY
MAX_OBJECTS = 100

OBJ_ID_NAMES = {
    8: "spikeUp", 9: "spikeDown", 39: "spike2", 103: "spike3",
    392: "spikeSmall", 421: "spikeTiny",
    88: "sawblade", 89: "sawbladeLg", 98: "sawbladeMed",
    1: "block", 2: "block2", 3: "block3", 4: "block4",
    5: "block5", 6: "block6", 7: "block7",
    40: "blockLong", 83: "blockTall",
    35: "yellowPad", 67: "pinkPad", 140: "gravPad", 1332: "redPad",
    3027: "spiderPad",
    36: "yellowOrb", 84: "pinkOrb", 141: "gravOrb",
    1022: "greenOrb", 1330: "redOrb", 1594: "dashOrb",
    1704: "dropOrb", 3005: "spiderOrb",
    10: "gravPortalDown", 11: "gravPortalUp",
    12: "shipPortal", 13: "cubePortal",
    45: "mirrorOn", 46: "mirrorOff",
    47: "bigPortal", 101: "miniPortal",
    99: "ballPortal", 286: "ufoPortal",
    660: "wavePortal", 745: "robotPortal",
    1331: "spiderPortal", 1933: "swingPortal",
    200: "speedSlow", 201: "speedNorm", 202: "speedFast",
    203: "speedVFast", 1334: "speedVSlow",
    289: "slope45", 291: "slope22",
}

MODE_NAMES = {
    0: "cube", 1: "ship", 2: "ball", 3: "ufo",
    4: "wave", 5: "robot", 6: "spider", 7: "swing",
}


def obj_name(obj_id: int) -> str:
    return OBJ_ID_NAMES.get(obj_id, f"obj#{obj_id}")


def format_nearby_objects(obs, num_objects: int) -> str:
    lines = []
    count = min(num_objects, MAX_OBJECTS)
    for i in range(count):
        base = OBJ_OBS_START + i * FLOATS_PER_OBJ
        relX, relY, objType, objID, scaleX, scaleY = obs[base:base + FLOATS_PER_OBJ]
        if relX == 0.0 and relY == 0.0 and objType == 0.0 and objID == 0.0:
            break
        name = obj_name(int(objID))
        lines.append(f"  {i+1:2d}. {name:<16s} dx={relX:+8.1f}  dy={relY:+8.1f}  sx={scaleX:.1f} sy={scaleY:.1f}")
    if not lines:
        return "  [0 nearby objects]"
    header = f"  [{len(lines)} nearby objects]"
    return header + "\n" + "\n".join(lines)


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
    ap.add_argument("--shm-name", default="gdrl_ipc_v3", help="Shared memory segment name.")
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
    ap.add_argument(
        "--show-objects",
        action="store_true",
        default=False,
        help="Print nearby obstacle data.",
    )
    ap.add_argument(
        "--num-objects",
        type=int,
        default=20,
        help="Number of nearest objects to print (max 100).",
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

    ad = GeodeV3Adapter(GeodeIPCV3Config(shm_name=args.shm_name))
    try:
        ad.verify_version()
        version, tick0, obs_dim, _, _, _, _ = ad._read_header()
        print("connected. press Ctrl+C to stop.", flush=True)
        print(f"header: version={version} tick={tick0} obs_dim={obs_dim}", flush=True)

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
            mode_text = MODE_NAMES.get(int(obs[7]), f"?{int(obs[7])}")
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
                line = (
                    f"tick={tick:7d} x={obs[0]:8.2f} y={obs[1]:8.2f} "
                    f"vx={obs[3]:7.2f} vy={obs[2]:7.2f} "
                    f"ground={int(obs[4])} dead={int(obs[5])} mode={mode_text} "
                    f"complete={int(level_complete)}"
                )
                print(line, flush=True)
                if args.show_objects:
                    print(format_nearby_objects(obs, args.num_objects), flush=True)
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
