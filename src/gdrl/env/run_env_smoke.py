from __future__ import annotations
import argparse
from gdrl.env.privileged_env import GDPrivilegedEnv, MockIPCAdapter
from gdrl.env.geode_ipc import GeodeSharedMemoryAdapter, GeodeIPCConfig


def make_env(mode: str, shm_name: str = 'gdrl_ipc'):
    if mode == 'mock':
        return GDPrivilegedEnv(ipc=MockIPCAdapter())
    if mode == 'geode':
        ipc = GeodeSharedMemoryAdapter(GeodeIPCConfig(shm_name=shm_name))
        return GDPrivilegedEnv(ipc=ipc)
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['mock', 'geode'], default='mock')
    ap.add_argument('--steps', type=int, default=10)
    ap.add_argument('--shm-name', default='gdrl_ipc')
    args = ap.parse_args()

    env = make_env(args.mode, args.shm_name)
    obs, _ = env.reset()
    print(f'start x={obs[0]:.2f} mode={int(obs[7])}')
    for i in range(args.steps):
        a = i % 2
        obs, r, term, trunc, _ = env.step(a)
        print(f'i={i:02d} a={a} x={obs[0]:.2f} dead={int(obs[5])} r={r:.3f}')
        if term or trunc:
            print('episode ended, resetting')
            obs, _ = env.reset()


if __name__ == '__main__':
    main()
