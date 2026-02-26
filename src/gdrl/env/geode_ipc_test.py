from __future__ import annotations
import argparse
import time
import struct
from multiprocessing import shared_memory
import numpy as np

from gdrl.env.geode_ipc import (
    GeodeSharedMemoryAdapter,
    GeodeIPCConfig,
    _HEADER_FMT,
    _HEADER_SIZE,
    _OBS_DIM,
    _OBS_FMT,
    _ACTION_OFFSET,
    _TOTAL_SIZE,
)


def writer(name: str = 'gdrl_ipc', frames: int = 50, dt: float = 0.01):
    shm = shared_memory.SharedMemory(name=name, create=True, size=_TOTAL_SIZE)
    try:
        buf = shm.buf
        version = 1
        tick = 0
        struct.pack_into(_HEADER_FMT, buf, 0, version, tick)
        for i in range(frames):
            tick += 1
            obs = np.zeros((_OBS_DIM,), dtype=np.float32)
            obs[0] = float(i)
            obs[1] = float(np.sin(i / 10.0))
            obs[7] = float((i // 10) % 4)
            struct.pack_into(_HEADER_FMT, buf, 0, version, tick)
            struct.pack_into(_OBS_FMT, buf, _HEADER_SIZE, *obs.tolist())
            time.sleep(dt)
        print('writer done')
    finally:
        shm.close()
        shm.unlink()


def reader(name: str = 'gdrl_ipc', frames: int = 20):
    ad = GeodeSharedMemoryAdapter(GeodeIPCConfig(shm_name=name))
    try:
        ad.verify_version()
        for _ in range(frames):
            obs = ad.read_next_obs(timeout_s=1.0)
            tick = ad.read_tick()
            print(f"tick={tick} x={obs[0]:.1f} y={obs[1]:.2f} mode={int(obs[7])}")
            ad.send_action(1 if int(obs[0]) % 2 == 0 else 0)
            # demonstrate action byte was written
            action = ad.buf[_ACTION_OFFSET]
            print(f"  action_in={action}")
    finally:
        ad.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('role', choices=['writer', 'reader'])
    ap.add_argument('--name', default='gdrl_ipc')
    args = ap.parse_args()
    if args.role == 'writer':
        writer(args.name)
    else:
        reader(args.name)


if __name__ == '__main__':
    main()
