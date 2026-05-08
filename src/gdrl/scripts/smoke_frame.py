"""Smoke test: dump 30 frames from the mod's SHM frame mirror to /tmp.

Run this after launching GD with the TrainingPipeline mod loaded and a level
playing. Expects upright 128x128 grayscale PNGs in /tmp/gdrl_smoke_*.png.

Usage: python -m gdrl.scripts.smoke_frame
"""
from __future__ import annotations
import os
import time

import numpy as np

from gdrl.env.geode_ipc_v3 import GeodeV4Adapter, FRAME_W, FRAME_H


def main() -> int:
    out_dir = "/tmp"
    n_frames = 30

    try:
        ad = GeodeV4Adapter()
    except FileNotFoundError:
        print("ERROR: SHM '/gdrl_ipc_v4' not found. Is GD running with the mod loaded?")
        return 1

    ad.verify_version()
    print(f"adapter ready. tick={ad.read_tick()} ep={ad.read_episode_id()} "
          f"level_id={ad.read_current_level_id()} length={ad.read_level_length():.1f}")

    # Wait for the mod to publish at least one frame.
    t0 = time.time()
    while time.time() - t0 < 5.0:
        try:
            _ = ad.read_frame()
            break
        except RuntimeError:
            time.sleep(0.05)
    else:
        print("ERROR: mod did not publish any frame within 5s")
        return 2

    # PNG writer (avoid mandatory cv2/PIL dep). Tiny stdlib png writer.
    import zlib
    import struct

    def write_png_gray(path: str, arr: np.ndarray) -> None:
        h, w = arr.shape
        raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))
        comp = zlib.compress(raw, 9)

        def chunk(tag: bytes, data: bytes) -> bytes:
            return (struct.pack(">I", len(data)) + tag + data
                    + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff))

        with open(path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")
            f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)))
            f.write(chunk(b"IDAT", comp))
            f.write(chunk(b"IEND", b""))

    saved = 0
    for i in range(n_frames):
        ad.wait_next_frame_seq(timeout_s=1.0)
        try:
            f = ad.read_frame()
        except RuntimeError as e:
            print(f"frame {i}: {e}")
            continue
        path = os.path.join(out_dir, f"gdrl_smoke_{i:02d}.png")
        write_png_gray(path, f)
        saved += 1
        if i == 0:
            print(f"frame[0] shape={f.shape} dtype={f.dtype} "
                  f"min={int(f.min())} max={int(f.max())} mean={float(f.mean()):.1f}")

    print(f"saved {saved}/{n_frames} frames to {out_dir}/gdrl_smoke_*.png")
    print(f"final tick={ad.read_tick()} ep={ad.read_episode_id()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
