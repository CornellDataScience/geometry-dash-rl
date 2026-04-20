"""Test action injection. Run while in a level."""
from gdrl.env.geode_ipc_v3 import GeodeV3Adapter
import time
import sys

ipc = GeodeV3Adapter()
ipc.verify_version()

# wait for ticks to start advancing (means you're in a level)
print("waiting for game to be in a level (ticks advancing)...")
t0 = time.time()
while time.time() - t0 < 30:
    if ipc.wait_next_tick(timeout_s=1.0):
        break
else:
    print("timed out waiting for level. enter a level first!")
    ipc.close()
    sys.exit(1)

print(f"game active at tick={ipc.read_tick()}, sending jumps...")

# send jump for 1000 frames
for i in range(1000):
    if not ipc.wait_next_tick(timeout_s=1.0):
        print("tick stalled, game paused?")
        break
    ipc.send_action(1)
    obs = ipc.read_obs()
    print(f"tick={ipc.read_tick()} JUMP  x={obs[0]:.1f} y={obs[1]:.1f} dead={obs[5]:.0f}")

# release for 30 frames
for i in range(30):
    if not ipc.wait_next_tick(timeout_s=1.0):
        print("tick stalled, game paused?")
        break
    ipc.send_action(0)
    obs = ipc.read_obs()
    print(f"tick={ipc.read_tick()} IDLE  x={obs[0]:.1f} y={obs[1]:.1f} dead={obs[5]:.0f}")

ipc.close()
