# geometry-dash-rl

Teacher-student RL prototype for Geometry Dash with a mock training pipeline and a Geode shared-memory bridge.

## Project status

Implemented:
- Gym-compatible privileged env (`GDPrivilegedEnv`) with mock IPC fallback
- Teacher PPO training entrypoint (SB3)
- Dataset collector writing compressed NPZ shards
- Student distillation trainer with weighted mode sampling
- Geode shared-memory adapter on the Python side
- Geode mod skeleton that writes core telemetry (`obs[0..7]`) + level-complete flag

Not finished yet:
- Real screen-frame capture in the collector
- Input injection from `action_in` into live gameplay
- Full production training/inference loop against real game frames

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/gdrl/env/privileged_env.py` | Gym env, reward logic, mock IPC adapter |
| `src/gdrl/teacher/train_ppo.py` | Teacher PPO training (saves SB3 model) |
| `src/gdrl/data/collector_cli.py` | Rollout collection + NPZ export |
| `src/gdrl/student/train_distill_dataset.py` | Student distillation from NPZ |
| `src/gdrl/student/train_distill.py` | Sanity-only random train step |
| `src/gdrl/env/geode_ipc.py` | Python shared-memory adapter (`gdrl_ipc`) |
| `src/gdrl/env/geode_ipc_test.py` | Fake writer/reader IPC smoke test |
| `src/gdrl/env/live_monitor.py` | Live telemetry monitor for Geode segment |
| `geode_mod/GDRLBridge/src/main.cpp` | Geode mod hook implementation |
| `docs/` | IPC protocol + implementation planning docs |

## Requirements

- Python `>=3.10`
- Virtualenv recommended
- Python deps from `requirements.txt` (`torch`, `stable-baselines3`, `gymnasium`, etc.)
- Optional Geode path requires Geometry Dash `2.2081` and Geode SDK (as required by `geode_mod/GDRLBridge/CMakeLists.txt`)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
mkdir -p artifacts
```

Or:

```bash
make setup
```

## Quick start (mock-only, no Geode needed)

1. Train teacher:

```bash
python -m gdrl.teacher.train_ppo
```

Expected output model:
- `artifacts/teacher_mock.zip`

2. Collect dataset shard from teacher rollouts:

```bash
python -m gdrl.data.collector_cli \
  --teacher-path artifacts/teacher_mock.zip \
  --episodes 3 \
  --out artifacts/datasets/mock_shard_000.npz
```

3. Train student from NPZ:

```bash
python -m gdrl.student.train_distill_dataset \
  --data artifacts/datasets/mock_shard_000.npz \
  --epochs 2 \
  --out artifacts/student_mock.pt
```

4. Optional quick sanity step (no dataset required):

```bash
python -m gdrl.student.train_distill
```

5. Optional env smoke run:

```bash
python -m gdrl.env.run_env_smoke --mode mock --steps 10
```

## Dataset format (NPZ shards)

`src/gdrl/data/collector.py` writes:
- `frames`: `uint8`, shape `[N, 84, 84]`
- `teacher_probs`: `float32`, shape `[N, 2]`
- `modes`: `int32`, shape `[N]`
- `level_ids`: `int32`, shape `[N]`
- `frame_idxs`: `int64`, shape `[N]`

Notes:
- Distillation stacks 4 consecutive frames.
- You need at least 5 frames in a shard or `train_distill_dataset` raises: `Dataset too small for stack_size=4`.

## Geode IPC protocol summary

Shared memory segment:
- name: `gdrl_ipc` (Python side)
- size: 444 bytes

| Field | Type | Notes |
| --- | --- | --- |
| `version` | `uint32` | Protocol version |
| `tick` | `uint32` | Incremented every update |
| `obs` | `float32[108]` | Privileged observation vector |
| `action_in` | `uint8` | Action byte written by Python |
| `reserved` | `uint8[3]` | `reserved[0]` is level-complete flag |

Action semantics:
- `0`: release/no-click
- `1`: press/hold

Reference: `docs/GEODE_IPC_PROTOCOL.md`.

## IPC smoke test without the game

Terminal A:

```bash
python -m gdrl.env.geode_ipc_test writer
```

Terminal B:

```bash
python -m gdrl.env.geode_ipc_test reader
```

Or run env smoke against shared memory:

```bash
python -m gdrl.env.run_env_smoke --mode geode --steps 10
```

## Geode CLI and SDK setup (macOS)

### 1. Install the Geode CLI

Download the latest release binary directly from GitHub:

```bash
curl -L https://github.com/geode-sdk/cli/releases/download/v3.7.4/geode-cli-v3.7.4-mac.zip -o geode-cli.zip
unzip geode-cli.zip
sudo mv geode /usr/local/bin/geode
geode --version
```

### 2. Configure the CLI with your Geometry Dash path

```bash
geode config setup
```

When prompted for the Geometry Dash path, use:

```
/Users/<your-username>/Library/Application Support/Steam/steamapps/common/Geometry Dash/Geometry Dash.app
```

### 3. Install the Geode SDK

```bash
geode sdk install
```

This sets the `GEODE_SDK` environment variable. If it is not set automatically, add it to your `~/.zshrc`:

```bash
export GEODE_SDK="$HOME/.geode/sdk"
source ~/.zshrc
```

### 4. Install Geode binaries into Geometry Dash

```bash
geode sdk install-binaries
```

This installs the Geode loader binaries into your Geometry Dash installation so the mod loader runs when you launch the game.

### 5. Set the mods output path

Add this to your `~/.zshrc` so built mods are automatically installed to the correct location:

```bash
echo 'export GEODE_MODS_PATH="/Users/<your-username>/Library/Application Support/Steam/steamapps/common/Geometry Dash/Geometry Dash.app/Contents/geode/mods"' >> ~/.zshrc
source ~/.zshrc
```

---

## Running with live Geode telemetry

1. Build the Geode mod (example CMake flow):

```bash
export GEODE_SDK=/path/to/geode-sdk
cmake -S geode_mod/GDRLBridge -B geode_mod/GDRLBridge/build
cmake --build geode_mod/GDRLBridge/build -j
```

2. Install/load the built mod in Geometry Dash via Geode.

3. Start a level, then verify the segment appears:

```bash
python -m gdrl.env.geode_wait
```

4. Monitor live values:

```bash
python -m gdrl.env.live_monitor --print-every 25
```

5. Drive the Gym env from shared memory:

```bash
python -m gdrl.env.run_env_smoke --mode geode --steps 20
```

6. If a stale segment remains after crashes:

```bash
python -m gdrl.env.geode_shm_cleanup --shm-name gdrl_ipc
```

## Geode mod behavior right now

Current implementation in `geode_mod/GDRLBridge/src/main.cpp`:
- Opens/creates POSIX shared memory `/gdrl_ipc`
- Writes `version`, increments `tick` every `PlayLayer::postUpdate`
- Fills core telemetry: `obs[0]=x`, `obs[1]=y`, `obs[2]=y_vel`, `obs[3]=x_delta`, `obs[4]=on_ground`, `obs[5]=is_dead`, `obs[6]=speed(placeholder)`, `obs[7]=mode`
- Sets `reserved[0]=1` on `levelComplete`

Pending:
- Applying `action_in` to actual jump/hold input (currently read only)

## Troubleshooting

- `FileNotFoundError` or timeout waiting for segment: make sure a level is running and the Geode mod is loaded.
- Version mismatch in Python adapter: compare `EXPECTED_VERSION` in `src/gdrl/env/geode_ipc.py` with `IPC_VERSION` in `geode_mod/GDRLBridge/src/main.cpp`.
- Dataset trainer says shard is too small: collect more frames by increasing `--episodes`.
- Stale shared-memory segment: run `python -m gdrl.env.geode_shm_cleanup --shm-name gdrl_ipc`.

## Additional docs

- `docs/IMPLEMENTATION_PLAN.md`
- `docs/GEODE_IPC_PROTOCOL.md`
- `docs/GEODE_HOOK_PLAN.md`
