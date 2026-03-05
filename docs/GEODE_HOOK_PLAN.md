# Geode Hook Plan (next implementation steps)

## Goal
Populate `obs[0..7]` from live `PlayLayer` / `PlayerObject` each frame and apply `action_in`.

## Planned hooks
1. `PlayLayer::init` (or level start hook)
   - initialize/open shared memory bridge
   - write `version=1`, `tick=0`
2. `PlayLayer::update(float dt)`
   - read player ptr (`m_player1`)
   - fill core obs fields:
     - `obs[0]=x`
     - `obs[1]=y`
     - `obs[2]=m_yVelocity`
     - `obs[3]=m_xVelocity` (fallback derive from position delta)
     - `obs[4]=m_isOnGround?1:0`
     - `obs[5]=m_isDead?1:0`
     - `obs[6]=speed multiplier` (fallback 1.0)
     - `obs[7]=mode_id` mapped from player mode flags
   - increment `tick`
   - read `action_in` and inject press/release

## Mode mapping draft
- cube: 0
- ship: 1
- ball: 2
- ufo: 3
- unsupported modes can map to nearest safe fallback (0) until expanded.

## Input injection strategy
- If `action_in=1`: ensure jump/hold press is active.
- If `action_in=0`: ensure release.
- Keep edge-trigger helper to avoid repeated synthetic key spam.

## Validation checklist
- `python -m gdrl.env.geode_wait` returns ready when level starts.
- `python -m gdrl.env.run_env_smoke --mode geode` prints changing x/ticks.
- toggling actions in Python visibly affects player behavior.

## Caution
- Geode API symbols/fields may differ by GD version; resolve names against installed Geode headers before compiling.
