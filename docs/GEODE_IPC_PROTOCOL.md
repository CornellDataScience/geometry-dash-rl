# Geode IPC Protocol (v0)

This document defines the binary shared-memory contract between the Geode mod and Python agent.

## Segment
- Name: `gdrl_ipc`
- Size: 444 bytes
- Endianness: little-endian

## Layout
- `uint32 version` (offset 0)
- `uint32 tick` (offset 4)
- `float32 obs[108]` (offset 8)
- `uint8 action_in` (offset 440)
- `uint8 reserved[3]` (offset 441..443)

## Observation map (current draft)
- `obs[0]`: player_x
- `obs[1]`: player_y
- `obs[2]`: velocity_y
- `obs[3]`: velocity_x
- `obs[4]`: on_ground (0/1)
- `obs[5]`: is_dead (0/1)
- `obs[6]`: speed_multiplier
- `obs[7]`: mode_id (0 cube, 1 ship, 2 ball, 3 ufo)
- `obs[8:108]`: reserved for nearby object features (`N x 5` flattened)

## Tick semantics
- Geode increments `tick` once per frame update.
- Python can poll tick and only read obs when tick changes.

## Action semantics
- Python writes `action_in` every decision step:
  - `0`: no-click / release
  - `1`: click / hold
- Geode reads action on update and applies input for that frame.

## Safety
- If version mismatch, Python should fail fast.
- If segment missing, Python should raise clear setup error.
