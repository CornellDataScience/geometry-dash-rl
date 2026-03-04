"""
Shared memory interface for the GD RL mod.

Opens the POSIX shared memory region created by the Geode mod and provides
read/write helpers. Used by all test scripts and the Gymnasium environment.
"""

import ctypes
import ctypes.util
import mmap
import struct
import time
import sys

SHM_NAME = b"/gd_rl_shm"
SHM_SIZE = 672

# Byte offsets matching the C++ #pragma pack(1) struct
# See mod/src/shared_memory.hpp for authoritative layout
OFFSET_TURN_FLAG = 0
OFFSET_COMMAND = 4
OFFSET_ACTION = 8
OFFSET_PLAYER = 12   # 10 fields: 4f + 2i + 1i + 1f + 1i + 1f = 40 bytes
OFFSET_OBJECTS = 52   # 25 * 6 floats = 600 bytes
OFFSET_META = 652     # numObjectsFound(i) reward(f) done(i) attemptNumber(i) prevPlayerX(f)

# Struct format strings (little-endian)
PLAYER_FMT = '<ffff ii i f i f'   # playerX, playerY, velocityY, rotation, gameMode, isOnGround, isDead, speedMultiplier, gravityFlipped, levelPercent
OBJECTS_FMT = '<' + 'f' * 150     # 25 slots * 6 floats
META_FMT = '<ifiif'               # numObjectsFound, reward, done, attemptNumber, prevPlayerX

# Load libc for shm_open
_libc = ctypes.CDLL(ctypes.util.find_library('c'))

O_RDWR = 0x0002


def open_shm(timeout=30.0):
    """Open existing shared memory. Retries until available or timeout."""
    start = time.time()
    while True:
        fd = _libc.shm_open(SHM_NAME, O_RDWR, 0o666)
        if fd >= 0:
            break
        if time.time() - start > timeout:
            raise TimeoutError(
                f"Shared memory {SHM_NAME.decode()} not found after {timeout}s. "
                "Is GD running with the mod loaded?"
            )
        print("Waiting for GD to create shared memory...")
        time.sleep(1)

    buf = mmap.mmap(fd, SHM_SIZE)
    # fd can be closed after mmap
    import os
    os.close(fd)
    return buf


def read_state(buf):
    """Read full state from shared memory. Returns dict."""
    player = struct.unpack_from(PLAYER_FMT, buf, OFFSET_PLAYER)
    objects_raw = struct.unpack_from(OBJECTS_FMT, buf, OFFSET_OBJECTS)
    meta = struct.unpack_from(META_FMT, buf, OFFSET_META)

    return {
        'turn_flag': struct.unpack_from('<i', buf, OFFSET_TURN_FLAG)[0],
        'command': struct.unpack_from('<i', buf, OFFSET_COMMAND)[0],
        'action': struct.unpack_from('<i', buf, OFFSET_ACTION)[0],
        # Player
        'playerX': player[0],
        'playerY': player[1],
        'velocityY': player[2],
        'rotation': player[3],
        'gameMode': player[4],
        'isOnGround': player[5],
        'isDead': player[6],
        'speedMultiplier': player[7],
        'gravityFlipped': player[8],
        'levelPercent': player[9],
        # Objects: list of 25 tuples, each (relX, relY, cat, hazard, interact, size)
        'objects': [objects_raw[i*6:(i+1)*6] for i in range(25)],
        # Meta
        'numObjectsFound': meta[0],
        'reward': meta[1],
        'done': meta[2],
        'attemptNumber': meta[3],
        'prevPlayerX': meta[4],
    }


def write_action(buf, command=0, action=0):
    """Write command + action, then set turn_flag=0 (game's turn)."""
    struct.pack_into('<i', buf, OFFSET_COMMAND, command)
    struct.pack_into('<i', buf, OFFSET_ACTION, action)
    # Set turn_flag last — this is the synchronization signal
    struct.pack_into('<i', buf, OFFSET_TURN_FLAG, 0)


def wait_for_game(buf, timeout=5.0):
    """Spin-wait until turn_flag == 1 (agent's turn). Returns True on success."""
    start = time.time()
    while True:
        flag = struct.unpack_from('<i', buf, OFFSET_TURN_FLAG)[0]
        if flag == 1:
            return True
        if time.time() - start > timeout:
            raise TimeoutError("Game not responding (turn_flag stuck at 0)")
        time.sleep(0.0001)  # 100us


# --- Standalone mode: print state continuously ---
if __name__ == '__main__':
    print("Opening shared memory...")
    buf = open_shm()
    print(f"Connected to {SHM_NAME.decode()} ({SHM_SIZE} bytes)\n")

    GAME_MODES = {0: 'cube', 1: 'ship', 2: 'ball', 3: 'ufo'}

    try:
        while True:
            state = read_state(buf)
            mode = GAME_MODES.get(state['gameMode'], '?')
            print(
                f"X={state['playerX']:8.1f}  Y={state['playerY']:8.1f}  "
                f"vY={state['velocityY']:7.2f}  rot={state['rotation']:7.1f}  "
                f"mode={mode}  ground={state['isOnGround']}  "
                f"dead={state['isDead']}  speed={state['speedMultiplier']:.1f}  "
                f"grav={state['gravityFlipped']}  pct={state['levelPercent']:.1f}%  "
                f"objs={state['numObjectsFound']}  "
                f"reward={state['reward']:+.3f}  done={state['done']}  "
                f"attempt={state['attemptNumber']}",
                end='\r'
            )
            time.sleep(1/60)
    except KeyboardInterrupt:
        print("\nDone.")
