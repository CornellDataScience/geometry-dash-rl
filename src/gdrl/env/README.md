# env/

Shared memory bridges and Gym environment for live Geometry Dash interaction.

## Files

| File | Purpose |
| --- | --- |
| `geode_ipc.py` | V2 adapter — opens `/gdrl_ipc`, reads the latest frame from the **TelemetryObstacles** mod, writes actions back. Single-frame polling only. |
| `geode_ipc_v3.py` | V3 adapter — opens `/gdrl_ipc_v3`, reads the **TrainingPipeline** mod's ring buffer (512 slots) without dropping frames. Also exposes the latest-frame mirror for live monitor / inference. |
| `privileged_env.py` | Gym environment wrapping an IPC adapter — used by PPO training. |
| `live_monitor.py` | CLI that prints live game state (position, velocity, mode) while GD is running. |
| `geode_shm_cleanup.py` | Deletes a stale shared memory segment left over after a crash. |

## Manual smoke-test scripts

These are short interactive scripts under `gdrl.env` for verifying the mod end-to-end. Each requires GD to be running and usually in an active level.

| Script | What it tests | How to run |
| --- | --- | --- |
| `test_action.py` | Action injection: sends `jump` for 1000 frames then idle for 30, printing obs each frame. Verify in-game that the player actually jumps and `x` advances. | `python -m gdrl.env.test_action` |
| `test_reset.py` | Level reset: reads tick + x, calls `send_reset()`, sleeps briefly, verifies the level restarted (player back at start). | `python -m gdrl.env.test_reset` |
| `test_input_recording.py` | Human input capture: prints a line each time `player_input` is set. Press space in-game and confirm "JUMP detected" appears. | `python -m gdrl.env.test_input_recording` |
| `test_gym_wrapper.py` | Gym env end-to-end: `reset() → step() loop` alternating jump/idle every 30 frames, printing x/dx/reward. Verifies action propagation, reward shaping, and termination on death or level complete. | `python -m gdrl.env.test_gym_wrapper` |

## Unit tests (no game required)

Automated tests live under `tests/` at the repo root. They use fake shared-memory segments and synthetic shard files, so they do not need GD or the mod.

Run the full suite:

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

| File | What it covers |
| --- | --- |
| `tests/test_v3_ring_buffer.py` | V3 binary layout (header size, frame slot size, total SHM size), version verification, ring drain semantics (first-call no-backfill, incremental drain, overflow → `dropped > 0`), send_action / send_reset / read_obs on the mirror. Creates a fake SHM segment and writes frame slots from Python using the same struct offsets the adapter reads. |
| `tests/test_obs_dataset.py` | Shard format round-trip, `ShardIndex` flat indexing, frame-stack ordering (oldest→newest), replication at episode/session start, `find_shards` both legacy (shards directly in root) and multi-session (timestamped subdirs), session-namespaced episode ids (two sessions with colliding raw ids get disjoint effective ids), `train_val_split` disjointness, `ShardWriter` creates a session subdir without clobbering other sessions. |

## Live recording + monitoring

Monitor live telemetry while a level is running (V2, legacy):

```bash
python -m gdrl.env.live_monitor --print-every 25
```

Record human gameplay (V3, ring buffer):

```bash
python -m gdrl.data.record_human --out artifacts/recordings/ --shard-size 10000
```

Each recording run creates a new `artifacts/recordings/YYYYMMDD_HHMMSS/` subdir so sessions never overwrite each other.

## Cleanup

Delete a stale V2 or V3 segment after a crash:

```bash
python -m gdrl.env.geode_shm_cleanup --shm-name gdrl_ipc
python -m gdrl.env.geode_shm_cleanup --shm-name gdrl_ipc_v3
```
