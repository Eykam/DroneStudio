import json, math, socket, subprocess, time

BIN = "/workspace/zig-out-hil/bin/dronestudio-headless"
FC = ("127.0.0.1", 5000)
# FC body (x fwd, y right, z down) vs sim body (x fwd, y up, z right):
# v_fc = M v_sim, M = R_x(-90 deg) -> c quat (w,x,y,z)
C = (math.cos(-math.pi/4), math.sin(-math.pi/4), 0.0, 0.0)
C_INV = (C[0], -C[1], -C[2], -C[3])

def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return (w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2)

class Headless:
    def __init__(self):
        self.p = subprocess.Popen([BIN], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    def call(self, d):
        self.p.stdin.write(json.dumps(d) + "\n"); self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())

h = Headless()
h.call({"cmd": "reset", "seed": 42, "scene": {"spawn": [0.0, 1.5, 0.0], "goal": [0.0, 0.0, 0.0], "obstacles": [], "extent": 10, "max_steps": 1000}})
h.call({"cmd": "set_dynamics", "path": "/workspace/DroneStudio/autoresearch/fixtures/chassis_v1.manifest.json"})
h.call({"cmd": "motor_v2", "on": True})
# corner-verified perm: m0->RR(sim1), m1->RL(sim0), m2->FR(sim2), m3->FL(sim3); yaw spin signs inverted (known rung-3 finding)
print("hil_listen:", h.call({"cmd": "hil_listen", "port": 5100, "perm": [1, 0, 2, 3]}), flush=True)

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(3.0)
status = ""
def send(m, expect=None):
    global status
    s.sendto(m.encode(), FC)
    if expect:
        r, _ = s.recvfrom(2048); status = r.decode()
        assert expect in status, (expect, status)
def hb():
    global status
    s.sendto(b"HEARTBEAT", FC)
    try:
        r, _ = s.recvfrom(2048); status = r.decode()
    except Exception: pass
send("CONNECT", "ACK")
send(json.dumps({"dshot_protocol": 2, "motors": [{"pin": 17, "direction": 0}, {"pin": 27, "direction": 1}, {"pin": 22, "direction": 0}, {"pin": 23, "direction": 1}], "battery": {"cells": 3}}), "CONFIG_ACK")
send("Battery 16.4")
for i in range(4):
    send(f"Arm {i}"); time.sleep(1.4); hb()
print("FC armed", flush=True)
send("UpdateBaseThrottle 11.1")  # hover frac = 0.4959*9.81/(4*11.0) = 0.1106
send("SetOrientation 1.0 0.0 0.0 0.0")  # level target (identity, FC frame)
# gains for radian-error authority: stock 0.5/rad has ~zero torque authority
for axis, kp, ki, kd in (("Roll", 6, 0.05, 1.0), ("Pitch", 6, 0.05, 1.0), ("Yaw", 2, 0.0, 0.5)):  # 100Hz feed: finer euler steps, ~10ms loop delay
    send(f"UpdatePidParams {axis} {kp} {ki} {kd}")
print("orientation control on; 6s hover run", flush=True)

log = []
t0 = time.time(); last_hb = 0.0
while time.time() - t0 < 6.0:
    st = h.call({"cmd": "hil_state"})
    log.append(st)
    q = st["quat"]  # sim data order [x,y,z,w]
    q_sim = (q[3], q[0], q[1], q[2])  # (w,x,y,z)
    q_fc = qmul(C, qmul(q_sim, C_INV))
    s.sendto(f"UpdateOrientation {q_fc[0]} {q_fc[1]} {q_fc[2]} {q_fc[3]}".encode(), FC)
    h.call({"cmd": "hil_step", "ticks": 5})
    if time.time() - last_hb > 0.5:
        hb(); last_hb = time.time()
        eul = status[status.find("CURR_EULER"):] if "CURR_EULER" in status else "?"
        parts = status.split()
        thr = [parts[3], parts[6], parts[9], parts[12]] if len(parts) > 12 else []
        sim_tilt = math.degrees(2 * math.acos(min(1.0, abs(st["quat"][3]))))
        print(f"t={time.time()-t0:.1f} sim_tilt={sim_tilt:.1f} {eul[:45]} thr={thr}", flush=True)
    time.sleep(max(0, 0.01 - (time.time() - t0 - len(log) * 0.01)))

n = len(log)
y = [v["pos"][1] for v in log]
qs = [v["quat"] for v in log]
tilt = [math.degrees(2 * math.acos(min(1.0, abs(q[3])))) for q in qs]
seqs = [v["hil_seq"] for v in log]
print(f"steps: {n} ({n/6:.0f}/s)", flush=True)
print(f"alt y: start {y[0]:.3f} end {y[-1]:.3f} min {min(y):.3f} max {max(y):.3f}", flush=True)
print(f"tilt deg: max {max(tilt):.2f} end {tilt[-1]:.2f}", flush=True)
print(f"hil_seq rate: {(seqs[-1]-seqs[0])/6:.0f} pkt/s", flush=True)
print(f"final pos: {log[-1]['pos']} vel: {log[-1]['vel']}", flush=True)
print(f"FC status: {status.strip()[:180]}", flush=True)
