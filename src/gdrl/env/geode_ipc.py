from __future__ import annotations
from dataclasses import dataclass
import struct
import time
from multiprocessing import shared_memory
import numpy as np

_HEADER_FMT = '<II'  # version, tick
_OBS_DIM = 108
_OBS_FMT = f'<{_OBS_DIM}f'
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_OBS_SIZE = struct.calcsize(_OBS_FMT)
_ACTION_OFFSET = _HEADER_SIZE + _OBS_SIZE
_LEVEL_COMPLETE_OFFSET = _ACTION_OFFSET + 1
_UPSIDE_DOWN_OFFSET = _ACTION_OFFSET + 2
_REVERSE_OFFSET = _ACTION_OFFSET + 3
_TOTAL_SIZE = _ACTION_OFFSET + 4
EXPECTED_VERSION = 1


@dataclass
class GeodeIPCConfig:
    shm_name: str = 'gdrl_ipc'
    obs_dim: int = _OBS_DIM
    expected_version: int = EXPECTED_VERSION


class GeodeSharedMemoryAdapter:
    """Shared-memory adapter compatible with GDPrivilegedEnv IPCAdapter interface."""

    def __init__(self, cfg: GeodeIPCConfig | None = None):
        self.cfg = cfg or GeodeIPCConfig()
        self.shm = shared_memory.SharedMemory(name=self.cfg.shm_name, create=False)
        self.buf = self.shm.buf
        if len(self.buf) < _TOTAL_SIZE:
            self.close()
            raise RuntimeError(
                f"Shared memory '{self.cfg.shm_name}' too small: {len(self.buf)} < {_TOTAL_SIZE}"
            )
        self._last_tick = None

    def close(self):
        self.shm.close()

    def _read_header(self) -> tuple[int, int]:
        return struct.unpack(_HEADER_FMT, self.buf[:_HEADER_SIZE])

    def verify_version(self) -> None:
        version, _ = self._read_header()
        if version != self.cfg.expected_version:
            raise RuntimeError(
                f"IPC version mismatch: got {version}, expected {self.cfg.expected_version}"
            )

    def read_tick(self) -> int:
        _, tick = self._read_header()
        return int(tick)

    def wait_next_tick(self, timeout_s: float = 0.2, poll_s: float = 0.001) -> bool:
        t0 = time.time()
        if self._last_tick is None:
            self._last_tick = self.read_tick()
        while time.time() - t0 < timeout_s:
            tick = self.read_tick()
            if tick != self._last_tick:
                self._last_tick = tick
                return True
            time.sleep(poll_s)
        return False

    def reset_level(self) -> None:
        # reserved for future control flags
        pass

    def read_obs(self) -> np.ndarray:
        obs_bytes = self.buf[_HEADER_SIZE:_ACTION_OFFSET]
        obs = struct.unpack(_OBS_FMT, obs_bytes)
        return np.asarray(obs, dtype=np.float32)

    def read_next_obs(self, timeout_s: float = 0.2) -> np.ndarray:
        self.verify_version()
        self.wait_next_tick(timeout_s=timeout_s)
        return self.read_obs()

    def send_action(self, action: int) -> None:
        self.buf[_ACTION_OFFSET] = 1 if int(action) else 0

    def read_level_complete_flag(self) -> bool:
        return bool(self.buf[_LEVEL_COMPLETE_OFFSET])

    def read_upside_down_flag(self) -> bool:
        return bool(self.buf[_UPSIDE_DOWN_OFFSET])

    def read_reverse_flag(self) -> bool:
        return bool(self.buf[_REVERSE_OFFSET])
