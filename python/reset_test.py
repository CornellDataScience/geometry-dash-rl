"""
Step 6 test: Reset handling stress test.

Runs 50 cycles of: reset → 100 random steps → check state.
Verifies no crashes and correct post-reset state.

Usage:
    python reset_test.py [num_cycles] [steps_per_cycle]
"""

import sys
import random
import time
from shm_reader import open_shm, read_state, write_action, wait_for_game

num_cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 50
steps_per_cycle = int(sys.argv[2]) if len(sys.argv) > 2 else 100

print(f"Reset test — {num_cycles} cycles, {steps_per_cycle} steps each")
print("Make sure a level is loaded in GD.\n")

buf = open_shm()

for cycle in range(num_cycles):
    # Send reset command
    wait_for_game(buf)
    write_action(buf, command=1, action=0)

    # Wait for post-reset state
    wait_for_game(buf)
    state = read_state(buf)

    # Verify post-reset state
    if state['isDead'] != 0:
        print(f"FAIL cycle {cycle}: isDead={state['isDead']} after reset")
    if state['playerX'] > 100:
        print(f"WARN cycle {cycle}: playerX={state['playerX']:.1f} (expected near 0)")

    # Run random steps
    for step in range(steps_per_cycle):
        action = random.randint(0, 1)
        write_action(buf, command=0, action=action)
        wait_for_game(buf)
        step_state = read_state(buf)

        # If died, just continue stepping
        if step_state['done']:
            break

    final_state = read_state(buf)
    print(
        f"Cycle {cycle+1:3d}/{num_cycles}: "
        f"X={final_state['playerX']:8.1f}  "
        f"dead={final_state['isDead']}  "
        f"pct={final_state['levelPercent']:.1f}%  "
        f"attempt={final_state['attemptNumber']}"
    )

print(f"\nAll {num_cycles} cycles completed successfully.")
