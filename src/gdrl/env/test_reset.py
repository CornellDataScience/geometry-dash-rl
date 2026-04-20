from gdrl.env.geode_ipc_v3 import GeodeV3Adapter
import time
ipc = GeodeV3Adapter(); ipc.verify_version()
ipc.wait_next_tick(timeout_s=2.0)
print(f"before reset: tick={ipc.read_tick()} x={ipc.read_obs()[0]:.1f}")
time.sleep(5)
ipc.send_reset()
print(f"after reset:  tick={ipc.read_tick()} x={ipc.read_obs()[0]:.1f}")
ipc.close()
