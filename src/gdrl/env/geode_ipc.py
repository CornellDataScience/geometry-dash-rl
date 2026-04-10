from __future__ import annotations
from dataclasses import dataclass
import struct
import time
from multiprocessing import shared_memory
import numpy as np

# V2 layout: version(4) + tick(4) + obs_dim(2) + obs[608](2432) + action_in(1) + ctrl_flags(1) + player_input(1) + level_done(1) + reserved(4)
_HEADER_FMT = '<IIH'  # version, tick, obs_dim
_OBS_DIM = 608
_OBS_FMT = f'<{_OBS_DIM}f'
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)  # 10
_OBS_SIZE = struct.calcsize(_OBS_FMT)        # 2432
_ACTION_OFFSET = _HEADER_SIZE + _OBS_SIZE     # 2442
_CTRL_FLAGS_OFFSET = _ACTION_OFFSET + 1       # 2443
_PLAYER_INPUT_OFFSET = _CTRL_FLAGS_OFFSET + 1 # 2444
_LEVEL_DONE_OFFSET = _PLAYER_INPUT_OFFSET + 1 # 2445
_TOTAL_SIZE = _LEVEL_DONE_OFFSET + 1 + 4      # 2450 (+ 4 reserved)
EXPECTED_VERSION = 2


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
        # Prevent Python's resource tracker from unlinking the segment on exit.
        # The mod owns the SHM lifetime, not Python.
        from multiprocessing import resource_tracker
        resource_tracker.unregister(f'/{self.cfg.shm_name}', 'shared_memory')
        self.buf = self.shm.buf
        if len(self.buf) < _TOTAL_SIZE:
            self.close()
            raise RuntimeError(
                f"Shared memory '{self.cfg.shm_name}' too small: {len(self.buf)} < {_TOTAL_SIZE}"
            )
        self._last_tick = None

    def close(self):
        self.shm.close()

    def _read_header(self) -> tuple[int, int, int]:
        return struct.unpack(_HEADER_FMT, self.buf[:_HEADER_SIZE])

    def verify_version(self) -> None:
        version, _, _ = self._read_header()
        if version != self.cfg.expected_version:
            raise RuntimeError(
                f"IPC version mismatch: got {version}, expected {self.cfg.expected_version}"
            )

    def read_tick(self) -> int:
        _, tick, _ = self._read_header()
        return int(tick)

    def read_obs_dim(self) -> int:
        _, _, obs_dim = self._read_header()
        return int(obs_dim)

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

    def send_reset(self) -> None:
        """Set reset_request bit in ctrl_flags."""
        self.buf[_CTRL_FLAGS_OFFSET] = self.buf[_CTRL_FLAGS_OFFSET] | 0x01

    def read_player_input(self) -> bool:
        """Read whether human pressed jump this frame."""
        return bool(self.buf[_PLAYER_INPUT_OFFSET])

    def read_level_complete_flag(self) -> bool:
        return bool(self.buf[_LEVEL_DONE_OFFSET])
