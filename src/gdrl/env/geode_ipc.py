from __future__ import annotations
from dataclasses import dataclass
import struct
from multiprocessing import shared_memory
import numpy as np

# v0 binary layout (little-endian)
# uint32 version
# uint32 tick
# float32[108] obs
# uint8 action_in
# uint8 reserved[3]
_HEADER_FMT = '<II'
_OBS_DIM = 108
_OBS_FMT = f'<{_OBS_DIM}f'
_ACTION_OFFSET = struct.calcsize(_HEADER_FMT) + struct.calcsize(_OBS_FMT)
_TOTAL_SIZE = _ACTION_OFFSET + 4


@dataclass
class GeodeIPCConfig:
    shm_name: str = 'gdrl_ipc'
    obs_dim: int = _OBS_DIM


class GeodeSharedMemoryAdapter:
    """Shared-memory adapter compatible with GDPrivilegedEnv IPCAdapter interface.

    Expects a Geode mod to maintain a shared memory segment with the v0 layout above.
    """

    def __init__(self, cfg: GeodeIPCConfig | None = None):
        self.cfg = cfg or GeodeIPCConfig()
        self.shm = shared_memory.SharedMemory(name=self.cfg.shm_name, create=False)
        self.buf = self.shm.buf

    def close(self):
        self.shm.close()

    def reset_level(self) -> None:
        # reserved for future control flags (e.g., restart request)
        pass

    def read_obs(self) -> np.ndarray:
        obs_bytes = self.buf[struct.calcsize(_HEADER_FMT):_ACTION_OFFSET]
        obs = struct.unpack(_OBS_FMT, obs_bytes)
        return np.asarray(obs, dtype=np.float32)

    def send_action(self, action: int) -> None:
        self.buf[_ACTION_OFFSET] = 1 if int(action) else 0
