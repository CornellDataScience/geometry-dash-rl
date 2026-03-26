# env/

Shared memory bridge and Gym environment for live Geometry Dash interaction.

## Files

| File | Purpose |
| --- | --- |
| `geode_ipc.py` | Opens `/gdrl_ipc` shared memory, reads game state from the Telemetry mod, and writes actions back |
| `privileged_env.py` | Gym environment wrapping the IPC adapter — used by the teacher PPO agent during training |
| `live_monitor.py` | CLI tool that prints live game state (position, velocity, mode) while GD is running |
| `geode_shm_cleanup.py` | Deletes a stale `/gdrl_ipc` segment left over after a crash |

## Usage

Monitor live telemetry while a level is running:

```bash
python -m gdrl.env.live_monitor --print-every 25
```

Clean up a stale shared memory segment:

```bash
python -m gdrl.env.geode_shm_cleanup --shm-name gdrl_ipc
```
