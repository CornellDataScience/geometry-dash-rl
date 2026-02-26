from __future__ import annotations
import time
from multiprocessing import shared_memory


def wait_for_geode_segment(name: str = 'gdrl_ipc', timeout_s: float = 10.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            shm = shared_memory.SharedMemory(name=name, create=False)
            shm.close()
            return True
        except FileNotFoundError:
            time.sleep(0.2)
    return False


if __name__ == '__main__':
    ok = wait_for_geode_segment()
    print('ready' if ok else 'timeout')
