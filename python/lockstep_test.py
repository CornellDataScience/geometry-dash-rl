"""
Step 3 test: Lockstep synchronization.

Verifies that the game freezes while waiting for the agent and resumes
when the agent responds. The artificial sleep simulates inference time.

Usage:
    python lockstep_test.py [sleep_ms]

    sleep_ms: milliseconds to sleep per step (default 10).
              Try 1000 to verify the game freezes for 1 second per frame.
"""

import sys
import time
import random
from shm_reader import open_shm, read_state, write_action, wait_for_game

sleep_ms = int(sys.argv[1]) if len(sys.argv) > 1 else 10

print(f"Lockstep test — sleeping {sleep_ms}ms per step")
print("The game should visibly slow down. Press Ctrl+C to stop.\n")

buf = open_shm()
step = 0

try:
    while True:
        wait_for_game(buf)
        state = read_state(buf)

        if step % 60 == 0:
            print(
                f"step={step:5d}  X={state['playerX']:8.1f}  "
                f"dead={state['isDead']}  pct={state['levelPercent']:.1f}%"
            )

        # Simulate inference delay
        time.sleep(sleep_ms / 1000.0)

        # Random action
        action = random.randint(0, 1)
        write_action(buf, command=0, action=action)
        step += 1

except KeyboardInterrupt:
    print(f"\nStopped after {step} steps.")
