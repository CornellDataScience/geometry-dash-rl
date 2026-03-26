# src/

Python RL pipeline for training and running an agent in Geometry Dash via the Geode shared memory bridge.

## Packages

| Package | Purpose |
| --- | --- |
| `gdrl/env/` | Gym environment, Geode IPC adapter, live monitor, shared memory utilities — see [gdrl/env/README.md](gdrl/env/README.md) |
| `gdrl/teacher/` | Teacher PPO training using privileged game observations |
| `gdrl/data/` | Rollout collector — records teacher episodes to NPZ shards |
| `gdrl/student/` | Student distillation trainer — learns to mimic teacher from frame data |
| `gdrl/infer/` | Inference stub for running a trained student agent |
| `gdrl/common/` | Shared config |

## Key files

| File | Purpose |
| --- | --- |
| `gdrl/env/geode_ipc.py` | Shared memory adapter that reads game state from `/gdrl_ipc` |
| `gdrl/env/privileged_env.py` | Gym env wrapping the IPC adapter with reward logic |
| `gdrl/env/live_monitor.py` | Prints live telemetry from a running GD session |
| `gdrl/env/geode_wait.py` | Waits for the shared memory segment to appear (connectivity check) |
| `gdrl/teacher/train_ppo.py` | Trains teacher PPO agent, saves to `artifacts/` |
| `gdrl/data/collector_cli.py` | Runs teacher rollouts and saves NPZ datasets |
| `gdrl/student/train_distill_dataset.py` | Trains student from NPZ via knowledge distillation |
