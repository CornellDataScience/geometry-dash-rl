# GD RL Geode Mod — Build Specification

## System Overview

Two processes on the same machine communicating through shared memory:

1. **C++ Geode mod** running inside Geometry Dash — extracts game state, freezes the game each frame, waits for an action from the agent, applies the action, unfreezes.
2. **Python RL trainer** running outside the game — reads state from shared memory, computes an action via a neural network, writes the action back.

IPC is **shared memory with atomic flag synchronization**. Not TCP, not WebSocket.

**Why not TCP:** Even on localhost, every send/receive goes through the full kernel networking stack — data copies from userspace to kernel socket buffers, gets wrapped in TCP segments, passes through the loopback driver, unwraps, and copies back to the receiving process. Most round-trips complete in 0.1–0.5ms but the kernel's scheduling and buffer management cause unpredictable spikes of 2–5ms. At 240fps the entire frame budget is 4.16ms — a single TCP jitter spike exceeds that. Nagle's algorithm batches small writes unless you set TCP_NODELAY. Partial reads mean your receive loop must handle fragmentation. Connection setup, drops, and timeouts add error-handling code unrelated to RL.

**Why not WebSocket:** WebSocket is TCP plus an HTTP upgrade handshake plus per-message framing overhead (2–14 bytes per message). It solves problems we don't have (browser compatibility, multi-client management) and adds latency on top of TCP's existing problems. Requires a WebSocket library dependency in the C++ mod or a hand-rolled protocol implementation.

**Why shared memory:** Both processes map the same physical RAM into their address spaces. Writing data means writing to a local pointer. Reading means reading from a local pointer. No kernel transitions, no copying, no serialization. Latency is nanoseconds (bounded by cache coherence). Synchronization is one atomic integer. No threads, no mutexes, no condition variables needed.

---

## Shared Memory Layout

A single fixed-size struct (~700 bytes) mapped into both processes. On Windows, create via `CreateFileMappingA` / `MapViewOfFile` with a named mapping string. On macOS/Linux, create via `shm_open` / `mmap` with a named POSIX shared memory path.

**Contents of the struct, in order:**

- `turn_flag` (int32, atomic): 0 = game's turn, 1 = agent's turn. The only synchronization primitive.
- `command` (int32): 0 = normal step, 1 = reset level, 2 = shutdown.
- `action` (int32): 0 = no click, 1 = click.
- Player features (float32 or int32):
  - `playerX`, `playerY`, `velocityY`, `rotation` (all float32)
  - `gameMode` (int32): 0=cube, 1=ship, 2=ball, 3=UFO
  - `isOnGround` (int32): 0 or 1
  - `isDead` (int32): 0 or 1
  - `speedMultiplier` (float32): discrete values 0.5/1.0/2.0/3.0/4.0 from speed portals
  - `gravityFlipped` (int32): 0 or 1
  - `levelPercent` (float32): 0–100
- Nearby objects: 25 slots × 6 float32 features each = 150 floats. Per object:
  - `relativeX`: object X minus player X
  - `relativeY`: object Y minus player Y
  - `typeCategory` (float-encoded int): 1=spike, 2=jump orb, 3=jump pad, 4=mode portal, 5=speed portal, 6=solid block, 0=unused slot
  - `isHazard`: 1.0 or 0.0
  - `isInteractable`: 1.0 or 0.0 (orbs, pads, portals)
  - `sizeClass`: max(width, height) of the object's scaled content size
- `numObjectsFound` (int32): how many of the 25 slots contain real data
- `reward` (float32)
- `done` (int32): 1 if episode ended, 0 otherwise
- `attemptNumber` (int32)
- `prevPlayerX` (float32): internal bookkeeping for reward delta

**Both C++ and Python must agree on exact byte offsets.** Define the layout once and derive offsets mechanically. Any mismatch causes silent data corruption.

---

## Frame Execution Sequence

Everything runs on the game's main thread inside the hooked `PlayLayer::update(float dt)`. No background threads.

1. **Run original physics.** Call the original `PlayLayer::update(dt)`. This advances the game one frame using the action applied at the end of the previous frame.
2. **Extract state.** Read player fields from `PlayerObject`. Scan `PlayLayer::m_objects` for nearby objects within ~800 pixels ahead. Categorize, filter out decorations, sort by distance, take closest 25, zero-pad. Compute reward as `(currentX - prevX) * 0.1`, override to `-10.0` on death, `+100.0` on level completion. Write everything to shared memory.
3. **Signal agent.** Atomic store `turn_flag = 1` (release semantics). Game is now frozen.
4. **Wait for agent.** Spin-read `turn_flag` until it becomes 0 (acquire semantics). Include 5-second timeout to avoid permanent hang if Python crashes — on timeout, disable RL mode and let game run normally.
5. **Handle command.** If `command == 1`: call `resetLevel()`, reset `prevPlayerX` to 0, increment `attemptNumber`, set `done = 0`, return from hook without applying action. If `command == 2`: disable RL mode, return.
6. **Apply action.** If `action == 1`: call `pushButton(PlayerButton::Jump)` on player. If `action == 0`: call `releaseButton(PlayerButton::Jump)`. This takes effect when `update(dt)` runs next frame.
7. **Return from hook.**

**Ordering matters:** Physics first (using last frame's action), then extract state, then agent decides, then apply new action. Reversing this causes a one-frame temporal shift that makes observations inconsistent with action effects.

---

## Object Type Categorization (C++)

Map raw `m_objectID` values to functional categories. Exact IDs depend on GD version — verify against Geode bindings and community docs.

- **Category 1 — Hazards:** All spike variants, saw blades, kill-on-contact objects.
- **Category 2 — Jump orbs:** Mid-air interactables activated by clicking (yellow, pink, blue, green orbs).
- **Category 3 — Jump pads:** Ground launchers activated on contact (yellow, pink, blue pads).
- **Category 4 — Mode portals:** Change game mode (cube, ship, ball, UFO, etc.).
- **Category 5 — Speed portals:** Change speed multiplier (0.5x through 4x).
- **Category 6 — Solid blocks:** Platforms and walls.
- **Category 0 — Decoration/unknown:** Filtered out, not included in observation.

Filtering out category 0 is critical — levels contain hundreds to thousands of decorative objects with no gameplay effect.

---

## Game Mode Detection (C++)

Check `PlayerObject` boolean flags: `m_isShip` → 1, `m_isBall` → 2, `m_isBird` → 3 (UFO), default → 0 (cube). Only these four modes are targeted.

---

## Reward Design (C++)

- **Per-frame:** `(currentPlayerX - prevPlayerX) * 0.1`
- **Death:** Override to `-10.0`, set `done = 1`
- **Level complete:** Override to `+100.0`, set `done = 1` (check `levelPercent >= 100`)
- Update `prevPlayerX` every frame, reset to 0 on level reset.

Computed in C++ to avoid extra round-trips and keep reward coupled with state extraction.

---

## Reset Handling (C++)

When `command == 1`: call `resetLevel()`, reset prevPlayerX, increment attemptNumber, set done=0, return from hook without running physics or applying action. Next `update(dt)` call is the first frame of the new attempt.

**If `resetLevel()` crashes when called from inside `update()`:** Use a `resetPending` flag instead. Set it when command=1 is received, return from hook, and at the very start of the next `update()` call check the flag and reset before doing anything else.

---

## Python Side

### Environment Wrapper

A Gymnasium-compatible class with `step(action)` and `reset()`.

**`step(action)`:** Write command=0 and action to shared memory. Set turn_flag=0. Spin-wait until turn_flag==1. Read obs, reward, done. Return `(obs, reward, done, info)`.

**`reset()`:** Write command=1 and action=0. Set turn_flag=0. Spin-wait until turn_flag==1. Read obs. Return `obs`.

Observation is a flat float32 vector: 11 player features + 150 object features = 161 floats.

Spin-wait should include 0.1ms sleep per iteration to reduce CPU burn, and a 5-second timeout raising an exception.

### PPO Training

Stable-Baselines3 PPO with MlpPolicy, device="cpu" (network too small to benefit from GPU). 3-layer MLP, 256 units per layer, ~200K parameters. Train on Stereo Madness first, then add levels with other modes. Initialize each new level from the previous checkpoint.

---

## Build Order

Each step is independently testable. Do not proceed to the next until the current step's success criterion is met.

### Step 1: Shared Memory Hello World

**C++:** Geode mod creates named shared memory on init. In `PlayLayer::update` hook, write an incrementing counter to the first 4 bytes every frame.

**Python:** Opens same named shared memory, reads and prints counter in a loop.

**Success:** Python prints numbers incrementing at ~60/sec during gameplay.

**Watch for:** Platform-specific naming conventions (Windows named mappings vs POSIX `/name` paths). Startup race condition if Python runs before the game creates the memory — Python should retry in a loop. Ensure both processes use the exact same name string.

### Step 2: Player State Extraction

**C++:** Expand mod to write all player fields (X, Y, velocityY, isDead, gameMode, isOnGround, speedMultiplier, gravityFlipped, levelPercent) to shared memory using the defined struct layout.

**Python:** Script reads and prints values while you play manually.

**Success:** Values match what you see on screen. PlayerX increases, isDead flips on spike contact, gameMode changes at portals.

**Watch for:** Field names on `PlayerObject` depend on Geode SDK / GD version. Expected names: `m_yVelocity`, `m_isShip`, `m_isBall`, `m_isBird`, `m_isOnGround`, `m_isDead`, `m_playerSpeed`, `m_isUpsideDown`. Check Geode headers if these don't compile. Struct padding between C++ and Python can cause offset mismatches — use `#pragma pack(1)` in C++ or account for padding explicitly in Python offsets. Test by writing known values and reading back.

### Step 3: Lockstep Synchronization

**C++:** Add atomic turn_flag and spin-wait to the update hook. After writing state, set flag=1 and spin until flag=0.

**Python:** Wait for flag==1, read state, sleep 10ms (simulate inference), write random action, set flag=0.

**Success:** Game visibly freezes during the 10ms and resumes when Python responds. Increasing delay to 1 second causes 1-second freeze per frame.

**Watch for:** If game hangs permanently, check atomic memory ordering (release on write, acquire on read). If game doesn't wait at all, the hook isn't firing (check `$modify` macro). Add timeout fallback so Python crashes don't permanently freeze GD.

### Step 4: Action Injection

**C++:** After receiving action, call `pushButton(PlayerButton::Jump)` for action=1, `releaseButton(PlayerButton::Jump)` for action=0.

**Python:** Hardcoded policy: if any object has relativeX between 0–100 and isHazard==1, jump. Otherwise don't.

**Success:** Bot clears at least a few spikes in Stereo Madness. Player visibly jumps at appropriate moments.

**Watch for:** `pushButton`/`releaseButton` may need a specific enum value — check Geode bindings for `PlayerButton::Jump` or equivalent. If player doesn't respond, try UILayer's `handleKeypress` as fallback. Verify action is applied AFTER the spin-wait and BEFORE returning from hook.

### Step 5: Nearby Object Extraction

**C++:** Iterate `PlayLayer::m_objects` (Cocos2d CCArray). Filter by X position (within window), categorize by `m_objectID`, discard category 0, sort by distance, take 25, zero-pad, write to shared memory.

**Python:** Print object list as a table (relX, relY, category, hazard, interactable, size) during manual play.

**Success:** Objects appear as you approach and disappear as you pass. Spikes show category 1 / hazard=1. Orbs show category 2 / interactable=1. No decorative objects appear.

**Watch for:** `m_objects` may contain thousands of objects — the X-position filter is critical for performance. Without it, the scan can take milliseconds and slow the game. The `m_objectID` to category mapping is fragile and community-reverse-engineered. Start with common IDs (spikes: 8, 39, 103; orbs: 36, 84, 141; pads: 35, 67, 140) and expand by logging unknown IDs during gameplay. IDs may differ between GD versions — verify empirically.

### Step 6: Reset Handling

**C++:** On command=1, call `resetLevel()`, reset internal state, return from hook.

**Python:** Loop: reset, step 100 times with random actions, reset. Repeat 50 times.

**Success:** Level restarts cleanly every time. No crashes over 50 resets. Post-reset observation shows playerX near 0, isDead=0.

**Watch for:** `resetLevel()` inside hooked `update()` may cause reentrancy issues. If it crashes, use the `resetPending` flag approach (set flag, return, reset at start of next update). Monitor memory usage over 1000+ resets for leaks.

### Step 7: PPO Training

**Python:** Gymnasium wrapper with observation_space=Box(161,) and action_space=Discrete(2). Connect to SB3 PPO. Train on Stereo Madness.

**Success:** Average episode length increases over 1–5M steps in TensorBoard.

**Watch for:** If agent plateaus, try adjusting reward (bigger death penalty, milestone bonuses at 25/50/75%). If agent never improves, normalize observations (divide playerX by level length, velocityY by max velocity, object distances by window size — PPO is sensitive to scale). Training speed bottleneck is 60fps game speed — 5M steps takes ~23 hours. No way around this without a separate simulator.

---

## Cross-Cutting Concerns

**Geode version compatibility:** Pin GD version in Steam (disable auto-update). Use alpha Geode build if stable doesn't support your version. Join Geode Discord.

**Struct alignment:** C++ compilers insert padding. Use `#pragma pack(1)` or manually match offsets in Python. Test with known values.

**Spin-wait CPU usage:** Add 0.1ms sleep per iteration on both sides. Without it, each spinning process maxes a CPU core, which can throttle laptops thermally.

**Game window focus:** GD may pause or reduce framerate when minimized. Keep window visible during training.

**Object scan performance:** O(N) per frame where N can be thousands. For extremely object-heavy community levels (50,000+ objects), add a spatial index or rescan every 4–5 frames instead of every frame.

**Player death timing:** Read `m_isDead` AFTER calling original `update(dt)`, not before — otherwise deaths during current-frame physics are missed.

**Dual mode levels:** The mod only reads `m_player1`. Avoid levels with dual sections initially. Expanding to `m_player2` is a future enhancement.