from __future__ import annotations

import argparse
import sys
from multiprocessing import shared_memory


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Unlink Geometry Dash RL shared-memory segment."
    )
    ap.add_argument("--shm-name", default="gdrl_ipc", help="Shared memory segment name.")
    args = ap.parse_args()

    try:
        shm = shared_memory.SharedMemory(name=args.shm_name, create=False)
    except FileNotFoundError:
        print(f"segment '{args.shm_name}' does not exist")
        return 0
    except PermissionError as e:
        print(f"permission error opening '{args.shm_name}': {e}", file=sys.stderr)
        return 2

    try:
        shm.close()
        shm.unlink()
        print(f"unlinked shared memory segment '{args.shm_name}'")
        return 0
    except FileNotFoundError:
        print(f"segment '{args.shm_name}' already removed")
        return 0
    except PermissionError as e:
        print(f"permission error unlinking '{args.shm_name}': {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
