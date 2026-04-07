from gdrl.env.geode_ipc import GeodeSharedMemoryAdapter
import time
ipc = GeodeSharedMemoryAdapter(); ipc.verify_version()
ipc.wait_next_tick(timeout_s=2.0)
print(f"before reset: tick={ipc.read_tick()} x={ipc.read_obs()[0]:.1f}")
time.sleep(5)
ipc.send_reset()
print(f"after reset:  tick={ipc.read_tick()} x={ipc.read_obs()[0]:.1f}")
ipc.close()
