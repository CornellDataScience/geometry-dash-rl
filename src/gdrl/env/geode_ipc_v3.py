"""V4 shared-memory adapter (frame mirror + state ring buffer + level loading).

Layout (must match mods/TrainingPipeline/src/main.cpp):

  Header (40 bytes):
    version           u32  =4
    tick              u32
    obs_dim           u16
    ring_capacity     u16
    write_index       u32  (atomic, monotonic)
    frames_dropped    u32
    episode_id        u32
    frame_w           u16
    frame_h           u16
    frame_channels    u16
    reserved_hdr      u16
    level_length      f32
    current_level_id  u32

  Latest-frame mirror (state):
    obs[608]            f32 * 608  = 2432
    action_in           u8
    ctrl_flags          u8         (bit0=reset, bit1=load_level)
    player_input        u8
    level_done          u8
    requested_level_id  u32
    reset_target_pct    u8
    reserved_mirror[15] u8

  Frame mirror (pixels):
    mirror_frame_seq    u32        (lock-free seq)
    mirror_frame[FRAME_BYTES]  u8  (FRAME_W * FRAME_H grayscale, upright)

  Ring buffer (state-only):
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
EXPECTED_VERSION = 4
FRAME_W = 128
FRAME_H = 128
FRAME_CHANNELS = 1
FRAME_BYTES = FRAME_W * FRAME_H * FRAME_CHANNELS

# header: <I I H H I I I H H H H f I = 4+4+2+2+4+4+4+2+2+2+2+4+4 = 40
_HEADER_FMT = '<IIHHIIIHHHHfI'
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
assert _HEADER_SIZE == 40, _HEADER_SIZE

# mirror state
_MIRROR_OBS_OFFSET = _HEADER_SIZE
_MIRROR_OBS_SIZE = OBS_DIM * 4
_MIRROR_ACTION_OFFSET = _MIRROR_OBS_OFFSET + _MIRROR_OBS_SIZE        # 2472
_MIRROR_CTRL_OFFSET = _MIRROR_ACTION_OFFSET + 1                      # 2473
_MIRROR_PLAYER_INPUT_OFFSET = _MIRROR_CTRL_OFFSET + 1                # 2474
_MIRROR_LEVEL_DONE_OFFSET = _MIRROR_PLAYER_INPUT_OFFSET + 1          # 2475
_MIRROR_REQ_LEVEL_ID_OFFSET = _MIRROR_LEVEL_DONE_OFFSET + 1          # 2476
_MIRROR_RESET_PCT_OFFSET = _MIRROR_REQ_LEVEL_ID_OFFSET + 4           # 2480
_MIRROR_RESERVED_OFFSET = _MIRROR_RESET_PCT_OFFSET + 1               # 2481
# frame mirror
_MIRROR_FRAME_SEQ_OFFSET = _MIRROR_RESERVED_OFFSET + 15              # 2496
_MIRROR_FRAME_OFFSET = _MIRROR_FRAME_SEQ_OFFSET + 4                  # 2500
_MIRROR_END = _MIRROR_FRAME_OFFSET + FRAME_BYTES                     # 18884

# FrameSlot: u32 tick + 608f obs + 4 u8 + u32 ep + 4 u8 pad = 2448
_FRAME_TICK_OFFSET = 0
_FRAME_OBS_OFFSET = 4
_FRAME_OBS_SIZE = OBS_DIM * 4
_FRAME_PLAYER_INPUT_OFFSET = _FRAME_OBS_OFFSET + _FRAME_OBS_SIZE     # 2436
_FRAME_LEVEL_DONE_OFFSET = _FRAME_PLAYER_INPUT_OFFSET + 1            # 2437
_FRAME_IS_DEAD_OFFSET = _FRAME_LEVEL_DONE_OFFSET + 1                 # 2438
_FRAME_PAD0_OFFSET = _FRAME_IS_DEAD_OFFSET + 1                       # 2439
_FRAME_EPISODE_OFFSET = _FRAME_PAD0_OFFSET + 1                       # 2440
_FRAME_PAD1_OFFSET = _FRAME_EPISODE_OFFSET + 4                       # 2444
FRAME_SLOT_SIZE = _FRAME_PAD1_OFFSET + 4                             # 2448

_RING_OFFSET = _MIRROR_END
_TOTAL_SIZE = _RING_OFFSET + RING_CAPACITY * FRAME_SLOT_SIZE


@dataclass
class GeodeIPCV4Config:
    shm_name: str = 'gdrl_ipc_v4'
    expected_version: int = EXPECTED_VERSION


# back-compat alias for old import sites
GeodeIPCV3Config = GeodeIPCV4Config


@dataclass
class Frame:
    """One captured frame from the ring buffer (state, no pixels)."""
    tick: int
    obs: np.ndarray            # shape (608,) float32
    player_input: int
    level_done: int
    is_dead: int
    episode_id: int


class GeodeV4Adapter:
    """V4 shared-memory adapter.

    Adds frame mirror (latest 128x128 grayscale pixels via lock-free seq read),
    load-level + reset-to-percent control bits, header fields for level metadata.
    """

    def __init__(self, cfg: GeodeIPCV4Config | None = None):
        self.cfg = cfg or GeodeIPCV4Config()
        self.shm = shared_memory.SharedMemory(name=self.cfg.shm_name, create=False)
        from multiprocessing import resource_tracker
        try:
            resource_tracker.unregister(f'/{self.cfg.shm_name}', 'shared_memory')
        except Exception:
            pass
        self.buf = self.shm.buf
        if len(self.buf) < _TOTAL_SIZE:
            self.close()
            raise RuntimeError(
                f"V4 SHM '{self.cfg.shm_name}' too small: {len(self.buf)} < {_TOTAL_SIZE}"
            )
        self._last_consumed: int | None = None
        self._last_tick: int | None = None
        self._last_frame_seq: int | None = None

    def close(self):
        try:
            self.shm.close()
        except Exception:
            pass

    # --- header / mirror reads ---

    def _read_header(self):
        return struct.unpack(_HEADER_FMT, self.buf[:_HEADER_SIZE])

    def verify_version(self) -> None:
        h = self._read_header()
        version, _, _, ring_cap, _, _, _, fw, fh, fc, _, _, _ = h
        if version != self.cfg.expected_version:
            raise RuntimeError(
                f"V4 IPC version mismatch: got {version}, expected {self.cfg.expected_version}"
            )
        if ring_cap != RING_CAPACITY:
            raise RuntimeError(
                f"V4 ring capacity mismatch: mod={ring_cap} python={RING_CAPACITY}"
            )
        if (fw, fh, fc) != (FRAME_W, FRAME_H, FRAME_CHANNELS):
            raise RuntimeError(
                f"V4 frame layout mismatch: mod={fw}x{fh}x{fc} python={FRAME_W}x{FRAME_H}x{FRAME_CHANNELS}"
            )

    def read_tick(self) -> int:
        return int(self._read_header()[1])

    def read_write_index(self) -> int:
        return int(self._read_header()[4])

    def read_episode_id(self) -> int:
        return int(self._read_header()[6])

    def read_level_length(self) -> float:
        return float(self._read_header()[11])

    def read_current_level_id(self) -> int:
        return int(self._read_header()[12])

    def read_obs_dim(self) -> int:
        return int(self._read_header()[2])

    def read_obs(self) -> np.ndarray:
        b = bytes(self.buf[_MIRROR_OBS_OFFSET:_MIRROR_OBS_OFFSET + _MIRROR_OBS_SIZE])
        return np.frombuffer(b, dtype=np.float32).copy()

    def read_player_input(self) -> bool:
        return bool(self.buf[_MIRROR_PLAYER_INPUT_OFFSET])

    def read_level_complete_flag(self) -> bool:
        return bool(self.buf[_MIRROR_LEVEL_DONE_OFFSET])

    # --- control writes ---

    def send_action(self, action: int) -> None:
        self.buf[_MIRROR_ACTION_OFFSET] = 1 if int(action) else 0

    def send_reset(self, percent: int = 0) -> None:
        """Reset the current level. If percent>0, mod loads nearest checkpoint <= percent.

        Checkpoints are captured online at every 5%-bucket the player crosses
        in a healthy frame. First episodes per level only spawn at 0 until the
        bucket cache fills in.
        """
        pct = int(percent)
        if pct < 0 or pct > 99:
            raise ValueError(f"send_reset percent out of range: {pct}")
        self.buf[_MIRROR_RESET_PCT_OFFSET] = pct
        self.buf[_MIRROR_CTRL_OFFSET] = self.buf[_MIRROR_CTRL_OFFSET] | 0x01

    def send_load_level(self, level_id: int) -> None:
        """Switch to a different main level by GD level id (Stereo Madness=1, etc.)."""
        struct.pack_into('<I', self.buf, _MIRROR_REQ_LEVEL_ID_OFFSET, int(level_id))
        self.buf[_MIRROR_CTRL_OFFSET] = self.buf[_MIRROR_CTRL_OFFSET] | 0x02

    # --- frame mirror (pixels) ---

    def _read_frame_seq(self) -> int:
        return struct.unpack_from('<I', self.buf, _MIRROR_FRAME_SEQ_OFFSET)[0]

    def read_frame(self) -> np.ndarray:
        """Latest captured frame as (FRAME_H, FRAME_W) uint8.

        Lock-free read: snapshot seq, copy pixels, re-check seq. Retry up to
        8 times if a write happened mid-read. Raises RuntimeError if the
        mod hasn't published any frames yet (seq still 0).
        """
        for _ in range(8):
            seq0 = self._read_frame_seq()
            if seq0 == 0:
                raise RuntimeError("no frame published by mod yet (seq=0)")
            b = bytes(self.buf[_MIRROR_FRAME_OFFSET:_MIRROR_FRAME_OFFSET + FRAME_BYTES])
            seq1 = self._read_frame_seq()
            if seq0 == seq1:
                return np.frombuffer(b, dtype=np.uint8).reshape(FRAME_H, FRAME_W).copy()
        raise RuntimeError("frame seq unstable after 8 retries")

    def wait_next_frame_seq(self, timeout_s: float = 0.2, poll_s: float = 0.0005) -> bool:
        t0 = time.time()
        if self._last_frame_seq is None:
            self._last_frame_seq = self._read_frame_seq()
        while time.time() - t0 < timeout_s:
            seq = self._read_frame_seq()
            if seq != self._last_frame_seq:
                self._last_frame_seq = seq
                return True
            time.sleep(poll_s)
        return False

    # --- frame-sync polling on the state mirror ---

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
            tick=int(tick), obs=obs,
            player_input=int(player_input),
            level_done=int(level_done),
            is_dead=int(is_dead),
            episode_id=int(episode_id),
        )

    def drain_ring(self, max_frames: int | None = None) -> tuple[list[Frame], int]:
        write_idx = self.read_write_index()
        if self._last_consumed is None:
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
        t0 = time.time()
        if self._last_consumed is None:
            self._last_consumed = self.read_write_index()
        while time.time() - t0 < timeout_s:
            if self.read_write_index() > self._last_consumed:
                return True
            time.sleep(poll_s)
        return False


# back-compat alias for callsites importing the v3 class name
GeodeV3Adapter = GeodeV4Adapter


def total_shm_size() -> int:
    return _TOTAL_SIZE
