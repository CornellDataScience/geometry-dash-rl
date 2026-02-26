# geometry-dash-rl

Initial scaffold for a teacher-student Geometry Dash RL pipeline.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

mkdir -p artifacts
python -m gdrl.teacher.train_ppo
python -m gdrl.student.train_distill
```

## Current status
- [x] Project structure + package setup
- [x] Mock privileged environment
- [x] PPO training entrypoint (mock)
- [x] Student architecture + distillation sanity step
- [x] Dataset shard schema + collector CLI (mock frames)
- [x] Distillation trainer over NPZ dataset + weighted mode sampler
- [x] Geode shared-memory adapter skeleton (`env/geode_ipc.py`)
- [x] Plan document in `docs/IMPLEMENTATION_PLAN.md`
- [ ] Real Geode memory map + mod implementation
- [ ] Real frame capture + input injection loop

## Demo path (today)
After dependencies are installed, you can run:
```bash
# 1) train mock teacher
python -m gdrl.teacher.train_ppo

# 2) collect mock dataset from teacher policy
python -m gdrl.data.collector_cli --teacher-path artifacts/teacher_mock.zip --episodes 3 --out artifacts/datasets/mock_shard_000.npz

# 3) train student from NPZ
python -m gdrl.student.train_distill_dataset --data artifacts/datasets/mock_shard_000.npz --epochs 2
```
This demonstrates the full teacher -> dataset -> student pipeline without Geode.
