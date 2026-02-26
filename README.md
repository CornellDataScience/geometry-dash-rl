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
- [x] Plan document in `docs/IMPLEMENTATION_PLAN.md`
- [ ] Geode IPC adapter
- [ ] Real data collection from game frames
