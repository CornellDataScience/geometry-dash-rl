"""V3 shared-memory adapter with ring buffer support.

Layout (must match mods/TrainingPipeline/src/main.cpp):

  Header (32 bytes):
    version       u32   =3
    tick          u32
    obs_dim       u16
    ring_capacity u16
    write_index   u32   (atomic, monotonic)
    frames_dropped u32  (reserved)
    episode_id    u32
    [pad to 32]   ... actually packed, so size is 4+4+2+2+4+4+4 = 24

  Latest-frame mirror:
    obs[608]      f32 * 608 = 2432
    action_in     u8
    ctrl_flags    u8
    player_input  u8
    level_done    u8
    reserved[8]   u8

  Ring buffer:
    frames[RING_CAPACITY] of FrameSlot
      tick          u32
      obs[608]      f32
      player_input  u8
      level_done    u8
      is_dead       u8
      pad0          u8
      episode_id    u32
      pad1[4]       u8
"""
from __future__ import annotations
from dataclasses import dataclass
import struct
import time
from multiprocessing import shared_memory
import numpy as np

OBS_DIM = 608
RING_CAPACITY = 512
EXPECTED_VERSION = 3

# header: <I I H H I I I  = 4+4+2+2+4+4+4 = 24
_HEADER_FMT = '<IIHHIII'
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)  # 24
assert _HEADER_SIZE == 24, _HEADER_SIZE

# latest-frame mirror: 608 floats + 4 u8 + 8 u8 reserved
_MIRROR_OBS_OFFSET = _HEADER_SIZE
_MIRROR_OBS_SIZE = OBS_DIM * 4
_MIRROR_ACTION_OFFSET = _MIRROR_OBS_OFFSET + _MIRROR_OBS_SIZE
_MIRROR_CTRL_OFFSET = _MIRROR_ACTION_OFFSET + 1
_MIRROR_PLAYER_INPUT_OFFSET = _MIRROR_CTRL_OFFSET + 1
_MIRROR_LEVEL_DONE_OFFSET = _MIRROR_PLAYER_INPUT_OFFSET + 1
_MIRROR_RESERVED_OFFSET = _MIRROR_LEVEL_DONE_OFFSET + 1
_MIRROR_END = _MIRROR_RESERVED_OFFSET + 8  # 24 + 2432 + 12 = 2468

# FrameSlot: u32 tick + 608f obs + 4 u8 + u32 ep + 4 u8 pad
# = 4 + 2432 + 4 + 4 + 4 = 2448
_FRAME_TICK_OFFSET = 0
_FRAME_OBS_OFFSET = 4
_FRAME_OBS_SIZE = OBS_DIM * 4
_FRAME_PLAYER_INPUT_OFFSET = _FRAME_OBS_OFFSET + _FRAME_OBS_SIZE  # 2436
_FRAME_LEVEL_DONE_OFFSET = _FRAME_PLAYER_INPUT_OFFSET + 1         # 2437
_FRAME_IS_DEAD_OFFSET = _FRAME_LEVEL_DONE_OFFSET + 1              # 2438
_FRAME_PAD0_OFFSET = _FRAME_IS_DEAD_OFFSET + 1                    # 2439
_FRAME_EPISODE_OFFSET = _FRAME_PAD0_OFFSET + 1                    # 2440
_FRAME_PAD1_OFFSET = _FRAME_EPISODE_OFFSET + 4                    # 2444
FRAME_SLOT_SIZE = _FRAME_PAD1_OFFSET + 4                          # 2448

_RING_OFFSET = _MIRROR_END
_TOTAL_SIZE = _RING_OFFSET + RING_CAPACITY * FRAME_SLOT_SIZE


@dataclass
class GeodeIPCV3Config:
    shm_name: str = 'gdrl_ipc_v3'
    expected_version: int = EXPECTED_VERSION


@dataclass
class Frame:
    """One captured frame from the ring buffer."""
    tick: int
    obs: np.ndarray            # shape (608,) float32
    player_input: int          # 0 or 1
    level_done: int
    is_dead: int
    episode_id: int


class GeodeV3Adapter:
    """Reads V3 shared memory with ring buffer.

    The mod writes one FrameSlot per game frame and increments write_index
    atomically. Python tracks last_consumed_index and drains all new frames
    on each call to drain_ring(), so it cannot miss frames as long as it
    catches up within RING_CAPACITY frames (~2s at 240fps).
    """

    def __init__(self, cfg: GeodeIPCV3Config | None = None):
        self.cfg = cfg or GeodeIPCV3Config()
        self.shm = shared_memory.SharedMemory(name=self.cfg.shm_name, create=False)
        # don't let Python's resource_tracker unlink the mod-owned segment
        from multiprocessing import resource_tracker
        try:
            resource_tracker.unregister(f'/{self.cfg.shm_name}', 'shared_memory')
        except Exception:
            pass
        self.buf = self.shm.buf
        if len(self.buf) < _TOTAL_SIZE:
            self.close()
            raise RuntimeError(
                f"V3 SHM '{self.cfg.shm_name}' too small: {len(self.buf)} < {_TOTAL_SIZE}"
            )
        self._last_consumed = None  # type: int | None
        self._last_tick = None      # type: int | None  (for wait_next_tick)

    def close(self):
        try:
            self.shm.close()
        except Exception:
            pass

    # --- header / mirror reads ---

    def _read_header(self) -> tuple[int, int, int, int, int, int, int]:
        return struct.unpack(_HEADER_FMT, self.buf[:_HEADER_SIZE])

    def verify_version(self) -> None:
        version, _, _, ring_cap, _, _, _ = self._read_header()
        if version != self.cfg.expected_version:
            raise RuntimeError(
                f"V3 IPC version mismatch: got {version}, expected {self.cfg.expected_version}"
            )
        if ring_cap != RING_CAPACITY:
            raise RuntimeError(
                f"V3 ring capacity mismatch: mod={ring_cap} python={RING_CAPACITY}"
            )

    def read_tick(self) -> int:
        _, tick, _, _, _, _, _ = self._read_header()
        return int(tick)

    def read_write_index(self) -> int:
        _, _, _, _, widx, _, _ = self._read_header()
        return int(widx)

    def read_episode_id(self) -> int:
        _, _, _, _, _, _, ep = self._read_header()
        return int(ep)

    def read_obs(self) -> np.ndarray:
        """Read the latest-frame mirror (for live monitor / inference)."""
        b = bytes(self.buf[_MIRROR_OBS_OFFSET:_MIRROR_OBS_OFFSET + _MIRROR_OBS_SIZE])
        return np.frombuffer(b, dtype=np.float32).copy()

    def send_action(self, action: int) -> None:
        self.buf[_MIRROR_ACTION_OFFSET] = 1 if int(action) else 0

    def send_reset(self) -> None:
        self.buf[_MIRROR_CTRL_OFFSET] = self.buf[_MIRROR_CTRL_OFFSET] | 0x01

    def read_player_input(self) -> bool:
        return bool(self.buf[_MIRROR_PLAYER_INPUT_OFFSET])

    def read_level_complete_flag(self) -> bool:
        return bool(self.buf[_MIRROR_LEVEL_DONE_OFFSET])

    def read_obs_dim(self) -> int:
        _, _, obs_dim, _, _, _, _ = self._read_header()
        return int(obs_dim)

    # --- V2-compat: frame-sync polling on the mirror ---

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

    def read_next_obs(self, timeout_s: float = 0.2) -> np.ndarray:
        self.wait_next_tick(timeout_s=timeout_s)
        return self.read_obs()

    # --- ring buffer reads ---

    def _read_slot(self, slot_index: int) -> Frame:
        base = _RING_OFFSET + (slot_index % RING_CAPACITY) * FRAME_SLOT_SIZE
        tick = struct.unpack_from('<I', self.buf, base + _FRAME_TICK_OFFSET)[0]
        obs_bytes = bytes(self.buf[base + _FRAME_OBS_OFFSET:base + _FRAME_OBS_OFFSET + _FRAME_OBS_SIZE])
        obs = np.frombuffer(obs_bytes, dtype=np.float32).copy()
        player_input = self.buf[base + _FRAME_PLAYER_INPUT_OFFSET]
        level_done = self.buf[base + _FRAME_LEVEL_DONE_OFFSET]
        is_dead = self.buf[base + _FRAME_IS_DEAD_OFFSET]
        episode_id = struct.unpack_from('<I', self.buf, base + _FRAME_EPISODE_OFFSET)[0]
        return Frame(
            tick=int(tick),
            obs=obs,
            player_input=int(player_input),
            level_done=int(level_done),
            is_dead=int(is_dead),
            episode_id=int(episode_id),
        )

    def drain_ring(self, max_frames: int | None = None) -> tuple[list[Frame], int]:
        """Consume all new frames since last call.

        Returns (frames, dropped). `dropped` is non-zero if Python fell
        behind by more than RING_CAPACITY (oldest frames overwritten).
        """
        write_idx = self.read_write_index()
        if self._last_consumed is None:
            # first call: start from current write_idx, don't backfill
            self._last_consumed = write_idx
            return [], 0

        available = write_idx - self._last_consumed
        if available <= 0:
            return [], 0

        dropped = 0
        if available > RING_CAPACITY:
            dropped = available - RING_CAPACITY
            self._last_consumed = write_idx - RING_CAPACITY

        if max_frames is not None and (write_idx - self._last_consumed) > max_frames:
            stop_at = self._last_consumed + max_frames
        else:
            stop_at = write_idx

        frames = []
        for idx in range(self._last_consumed, stop_at):
            frames.append(self._read_slot(idx))
        self._last_consumed = stop_at
        return frames, dropped

    def wait_for_frames(self, timeout_s: float = 0.5, poll_s: float = 0.0005) -> bool:
        """Block until at least one new frame is available."""
        t0 = time.time()
        if self._last_consumed is None:
            self._last_consumed = self.read_write_index()
        while time.time() - t0 < timeout_s:
            if self.read_write_index() > self._last_consumed:
                return True
            time.sleep(poll_s)
        return False


def total_shm_size() -> int:
    return _TOTAL_SIZE
