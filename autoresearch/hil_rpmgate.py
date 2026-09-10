import json, socket, subprocess, time, math

BIN = "/workspace/zig-out-hil/bin/dronestudio-headless"
FC = ("127.0.0.1", 5000)

p = subprocess.Popen([BIN], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
def call(d):
    p.stdin.write(json.dumps(d) + "\n"); p.stdin.flush()
    return json.loads(p.stdout.readline())
call({"cmd": "reset", "seed": 42, "scene": {"spawn": [0.0, 1.5, 0.0], "goal": [0.0, 0.0, 0.0], "obstacles": [], "extent": 10, "max_steps": 1000}})
call({"cmd": "set_dynamics", "path": "/workspace/DroneStudio/autoresearch/fixtures/chassis_v1.manifest.json"})
call({"cmd": "motor_v2", "on": True})
call({"cmd": "hil_listen", "port": 5100, "perm": [1, 0, 2, 3]})

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(3.0)
def send(m, expect=None):
    s.sendto(m.encode(), FC)
    if expect:
        r, _ = s.recvfrom(4096)
        assert expect in r.decode(), (expect, r.decode())
send("CONNECT", "ACK")
send(json.dumps({"dshot_protocol": 2, "motors": [{"pin": 17, "direction": 0}, {"pin": 27, "direction": 1}, {"pin": 22, "direction": 0}, {"pin": 23, "direction": 1}], "battery": {"cells": 3}}), "CONFIG_ACK")
send("Battery 16.4")
for i in range(4):
    send(f"Arm {i}"); time.sleep(1.4)
    s.sendto(b"HEARTBEAT", FC)
    try: s.recvfrom(4096)
    except Exception: pass
print("armed", flush=True)

MAXT = 11.0  # manifest max_thrust_n
print("pct | t_cmd(N) | omega_ss (rad/s) | kf_implied (N/(rad/s)^2)", flush=True)
for pct in (10, 20, 30, 40):
    for i in range(4):
        send(f"SetSpeed {i} {pct}")
    time.sleep(0.4)  # listener catches up
    call({"cmd": "hil_step", "ticks": 600})  # 1.2s spool to steady state
    st = call({"cmd": "hil_state"})
    om = sum(st["motor_omega"]) / 4.0
    t_cmd = pct / 100.0 * MAXT
    kf = t_cmd / (om * om) if om > 1 else 0.0
    print(f"{pct:3d} | {t_cmd:6.2f} | {om:9.1f} | {kf:.3e}", flush=True)
p.terminate()
print("RPMGATE_DONE", flush=True)
