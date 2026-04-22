# src/

Python RL pipeline for training and running an agent in Geometry Dash via the Geode shared memory bridge.

## Packages

| Package | Purpose |
| --- | --- |
| `gdrl/env/` | Gym environment, Geode IPC adapter, live monitor, shared memory utilities — see [gdrl/env/README.md](gdrl/env/README.md) |
| `gdrl/teacher/` | Teacher training: BC warmup, DAgger iterations, PPO fine-tuning — see [gdrl/teacher/README.md](gdrl/teacher/README.md) |
| `gdrl/data/` | Data collection and dataset utilities — human recording, DAgger rollout, alignment, aggregation — see [gdrl/data/README.md](gdrl/data/README.md) |
| `gdrl/student/` | Student distillation trainer — learns to mimic teacher from pixel frames |
| `gdrl/infer/` | Inference stub for running a trained student agent |
| `gdrl/common/` | Shared config |

## Key files

| File | Purpose |
| --- | --- |
| `gdrl/env/geode_ipc.py` | V2 shared memory adapter (`/gdrl_ipc`) |
| `gdrl/env/geode_ipc_v3.py` | V3 shared memory adapter with ring buffer (`/gdrl_ipc_v3`) |
| `gdrl/env/privileged_env.py` | Gym env wrapping the IPC adapter with reward logic |
| `gdrl/env/live_monitor.py` | Prints live telemetry from a running GD session |
| `gdrl/teacher/model.py` | `TeacherPolicy` MLP — shared architecture for BC, DAgger, and PPO |
| `gdrl/teacher/bc_train.py` | BC warmup training on human recordings |
| `gdrl/teacher/dagger_loop.py` | Orchestrates the full DAgger loop end to end |
| `gdrl/teacher/train_ppo.py` | PPO fine-tuning after DAgger converges |
| `gdrl/data/record_human.py` | Records human gameplay to `.npz` shards via V3 ring buffer |
| `gdrl/data/dagger_rollout.py` | Rolls out a policy in the live game and records visited states |
| `gdrl/data/dagger_align.py` | Labels rollout states with human actions via X-position matching |
| `gdrl/data/dagger_dataset.py` | Aggregates human + DAgger labeled data into one growing dataset |
| `gdrl/student/train_distill_dataset.py` | Trains student from teacher rollouts via knowledge distillation |
