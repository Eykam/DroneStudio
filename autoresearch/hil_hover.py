import json, math, socket, subprocess, threading, time

BIN = "/workspace/zig-out-hil/bin/dronestudio-headless"
FC = ("127.0.0.1", 5000)

class Headless:
    def __init__(self):
        self.p = subprocess.Popen([BIN], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    def call(self, d):
        self.p.stdin.write(json.dumps(d) + "\n"); self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())

class FCClient:
    def __init__(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); self.s.settimeout(3.0)
        self.status = ""
    def send(self, m, expect=None):
        self.s.sendto(m.encode(), FC)
        if expect:
            r, _ = self.s.recvfrom(2048); self.status = r.decode()
            assert expect in self.status, (expect, self.status)
    def heartbeat(self):
        try:
            self.s.sendto(b"HEARTBEAT", FC)
            r, _ = self.s.recvfrom(2048); self.status = r.decode()
        except Exception: pass

def quat_level():
    return (1.0, 0.0, 0.0, 0.0)

h = Headless()
print("reset:", h.call({"cmd": "reset", "seed": 42, "scene": {"spawn": [0.0, 1.5, 0.0], "goal": [0.0, 0.0, 0.0], "obstacles": [], "extent": 10, "max_steps": 1000}}).get("ok", "?"), flush=True)
print("set_dynamics:", h.call({"cmd": "set_dynamics", "path": "/workspace/DroneStudio/autoresearch/fixtures/chassis_v1.manifest.json"}).get("ok", "?"), flush=True)
print("motor_v2:", h.call({"cmd": "motor_v2", "on": True}).get("ok", "?"), flush=True)
print("hil_listen:", h.call({"cmd": "hil_listen", "port": 5100}), flush=True)

fc = FCClient()
fc.send("CONNECT", "ACK")
fc.send(json.dumps({"dshot_protocol": 2, "motors": [{"pin": 17, "direction": 0}, {"pin": 27, "direction": 1}, {"pin": 22, "direction": 0}, {"pin": 23, "direction": 1}], "battery": {"cells": 3}}), "CONFIG_ACK")
fc.send("Battery 16.4")
for i in range(4):
    fc.send(f"Arm {i}"); time.sleep(1.4)
print("FC armed", flush=True)
fc.send("UpdateBaseThrottle 12.2")  # 0.4959kg chassis_v1, hover frac = m*g/(4*10N)
w0, x0, y0, z0 = quat_level()
fc.send(f"SetOrientation {w0} {x0} {y0} {z0}")
print("orientation control started; 6s free-run hover", flush=True)

log = []
t0 = time.time(); last_hb = 0.0
while time.time() - t0 < 6.0:
    st = h.call({"cmd": "hil_state"})
    q = st["quat"]
    fc.s.sendto(f"UpdateOrientation {q[3]} {q[0]} {q[1]} {q[2]}".encode(), FC)
    h.call({"cmd": "hil_step", "ticks": 10})
    log.append(st)
    if time.time() - last_hb > 0.5:
        fc.heartbeat(); last_hb = time.time()
    time.sleep(max(0, 0.02 - (time.time() - t0 - len(log) * 0.02)))

n = len(log)
y = [s["pos"][1] for s in log]
seqs = [s["hil_seq"] for s in log]
ages = [s["hil_age_ms"] for s in log]
print(f"steps: {n} ({n/6:.0f}/s)", flush=True)
print(f"alt y: start {y[0]:.3f} end {y[-1]:.3f} min {min(y):.3f} max {max(y):.3f}", flush=True)
print(f"hil_seq: {seqs[0]} -> {seqs[-1]} ({(seqs[-1]-seqs[0])/6:.0f} pkts/s), age p50 {sorted(ages)[len(ages)//2]:.2f}ms", flush=True)
print(f"last FC status: {fc.status.strip()[:200]}", flush=True)
print(f"final pos: {log[-1]['pos']} vel: {log[-1]['vel']}", flush=True)
