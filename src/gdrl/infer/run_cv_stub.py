from __future__ import annotations
import time
import numpy as np


def main():
    print('CV runtime stub: waiting for real capture/injection wiring...')
    for i in range(5):
        frame = np.zeros((84, 84), dtype=np.uint8)
        _ = frame.mean()
        print(f'tick {i}')
        time.sleep(0.1)


if __name__ == '__main__':
    main()
