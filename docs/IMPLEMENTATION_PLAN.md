# Geometry Dash RL — Implementation Plan (v0)

## 0) Scope (Phase-1 deliverable)
Build an end-to-end baseline that proves the pipeline works:
1. Privileged-state teacher training loop (PPO) over a Gym-compatible env stub.
2. Data collection format for (frames, teacher_probs, metadata).
3. Student distillation training loop with auxiliary mode classification.
4. Real-time inference runner stub.

Out of scope in this phase:
- Full Geode production mod
- Demon-level performance claims
- TensorRT/CoreML optimization

## 1) Architecture
- `src/gdrl/env/privileged_env.py`: GDPrivilegedEnv + IPC adapter interface.
- `src/gdrl/teacher/train_ppo.py`: SB3 PPO training entrypoint.
- `src/gdrl/data/collector.py`: teacher rollout + dataset writer.
- `src/gdrl/student/model.py`: IMPALA-style student + mode head.
- `src/gdrl/student/train_distill.py`: KL distillation training loop.
- `src/gdrl/common/config.py`: config loading.

## 2) Milestones
### M1 (now): Repo bootstrap + executable stubs
- Python package layout
- Config files
- CLI scripts wired

### M2: Privileged env works with mock IPC
- Deterministic mock environment for local testing
- PPO can train and improve in mock env

### M3: Data pipeline
- Collect N episodes from teacher in mock env
- Write NPZ shards + metadata

### M4: Student distillation baseline
- Train on collected synthetic frames
- Validate KL + mode accuracy

### M5: Geode integration point
- Replace mock IPC with real shared-memory reader/writer
- Keep env API unchanged

## 3) Risks + mitigations
- **Geode instability** → strict interface boundary (`IPCAdapter`) + mock fallback.
- **Dataset explosion** → sharded NPZ with mmap loading.
- **Mode imbalance** → weighted sampler in distillation dataloader.
- **Latency unknowns** → defer optimization until baseline works.

## 4) Immediate TODO (next 48h)
1. Implement mock IPC + env returning plausible GD observations.
2. Run PPO script and verify training starts cleanly.
3. Implement collector output format and export test shard.
4. Implement student distill trainer on exported sample.
