"""
Step 4 test: Action injection with heuristic spike-dodge policy.

Jumps when any hazard is detected within relX [0, 150] pixels ahead.
Should clear at least a few spikes in Stereo Madness.

Usage:
    python action_test.py
"""

import time
from shm_reader import open_shm, read_state, write_action, wait_for_game

print("Action test — heuristic spike dodge")
print("Start playing a level in GD. Press Ctrl+C to stop.\n")

buf = open_shm()
step = 0
jumps = 0

try:
    while True:
        wait_for_game(buf)
        state = read_state(buf)

        # Heuristic: jump if any hazard is within 0-150 px ahead
        should_jump = False
        for i in range(state['numObjectsFound']):
            obj = state['objects'][i]
            relX, relY, cat, isHazard, isInteract, size = obj
            if isHazard > 0.5 and 0 <= relX <= 150:
                should_jump = True
                break

        action = 1 if should_jump else 0
        if action == 1:
            jumps += 1

        write_action(buf, command=0, action=action)

        if step % 60 == 0:
            print(
                f"step={step:5d}  X={state['playerX']:8.1f}  "
                f"dead={state['isDead']}  pct={state['levelPercent']:.1f}%  "
                f"jumps={jumps}  objs={state['numObjectsFound']}"
            )

        step += 1

except KeyboardInterrupt:
    print(f"\nStopped after {step} steps, {jumps} jumps.")
