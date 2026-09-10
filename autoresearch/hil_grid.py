import json, math, socket, subprocess, time

BIN = "/workspace/zig-out-hil/bin/dronestudio-headless"
FC = ("127.0.0.1", 5000)
C = (math.cos(-math.pi/4), math.sin(-math.pi/4), 0.0, 0.0)
C_INV = (C[0], -C[1], -C[2], -C[3])

def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return (w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2)

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(3.0)
def send(m):
    s.sendto(m.encode(), FC)
def hb():
    s.sendto(b"HEARTBEAT", FC)
    try: s.recvfrom(4096)
    except Exception: pass
import threading
def hb_daemon():
    hs = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); hs.settimeout(1.0)
    while True:
        try:
            hs.sendto(b"HEARTBEAT", FC); hs.recvfrom(4096)
        except Exception: pass
        time.sleep(0.4)
threading.Thread(target=hb_daemon, daemon=True).start()  # covers setup gaps (watchdog disarms at 2s)

send("CONNECT")
r, _ = s.recvfrom(4096); assert "ACK" in r.decode()
send(json.dumps({"dshot_protocol": 2, "motors": [{"pin": 17, "direction": 0}, {"pin": 27, "direction": 1}, {"pin": 22, "direction": 0}, {"pin": 23, "direction": 1}], "battery": {"cells": 3}}))
r, _ = s.recvfrom(4096); assert "CONFIG_ACK" in r.decode()
send("Battery 16.4")
for i in range(4):
    send(f"Arm {i}"); time.sleep(1.4); hb()
send("SetOrientation 1.0 0.0 0.0 0.0")
print("FC armed", flush=True)

def run_combo(kp, kd):
    send(f"UpdatePidParams Roll {kp} 0.05 {kd}")
    send(f"UpdatePidParams Pitch {kp} 0.05 {kd}")
    send(f"UpdatePidParams Yaw 0.1 0.0 0.0")  # weak: spin inversion unfixed (user-gated)
    p = subprocess.Popen([BIN], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    def call(d):
        p.stdin.write(json.dumps(d) + "\n"); p.stdin.flush()
        return json.loads(p.stdout.readline())
    call({"cmd": "reset", "seed": 42, "scene": {"spawn": [0.0, 1.5, 0.0], "goal": [0.0, 0.0, 0.0], "obstacles": [], "extent": 10, "max_steps": 1000}})
    call({"cmd": "set_dynamics", "path": "/workspace/DroneStudio/autoresearch/fixtures/chassis_v1.manifest.json"})
    call({"cmd": "motor_v2", "on": True})
    call({"cmd": "hil_listen", "port": 5100, "perm": [1, 0, 2, 3]})
    # straight to hover throttle with sim frozen; the closed loop absorbs the
    # spool transient (ramping while stepping caused ground strike at 1.5m spawn)
    send("UpdateBaseThrottle 11.1")
    hb()
    # 6s closed loop at 100Hz with 12ms extrapolation
    tilts = []; alts = []
    t0 = time.time(); n = 0
    while time.time() - t0 < 6.0:
        st = call({"cmd": "hil_state"})
        q = st["quat"]; om = st["omega"]
        q_sim = (q[3], q[0], q[1], q[2])
        DT = 0.012
        dq = (1.0, om[0]*DT/2, om[1]*DT/2, om[2]*DT/2)
        nn = math.sqrt(sum(v*v for v in dq)); dq = tuple(v/nn for v in dq)
        q_fc = qmul(C, qmul(qmul(q_sim, dq), C_INV))
        send(f"UpdateOrientation {q_fc[0]} {q_fc[1]} {q_fc[2]} {q_fc[3]}")
        call({"cmd": "hil_step", "ticks": 5})
        n += 1
        tilts.append(math.degrees(2 * math.acos(min(1.0, abs(q[3])))))
        alts.append(st["pos"][1])
        time.sleep(max(0, 0.01 - (time.time() - t0 - n * 0.01)))
    p.terminate()
    mx, end = max(tilts), tilts[-1]
    verdict = "PASS" if (mx < 10 and end < 5 and abs(alts[-1] - 1.5) < 1.0 and min(alts) > 0.3) else ("MARGINAL" if mx < 30 else "FAIL")
    return mx, end, verdict

print(f"{'Kp':>4} {'Kd':>4} {'maxTilt':>8} {'endTilt':>8}  verdict", flush=True)
for kp in (2, 3, 4):
    for kd in (2, 3, 4):
        try:
            mx, end, v = run_combo(kp, kd)
            print(f"{kp:4d} {kd:4d} {mx:8.1f} {end:8.1f}  {v}", flush=True)
        except Exception as e:
            print(f"{kp:4d} {kd:4d}  ERROR: {e}", flush=True)
print("GRID_DONE", flush=True)
