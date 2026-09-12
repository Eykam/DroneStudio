import json, math, socket, subprocess, sys, time, threading
BIN = "/workspace/zig-out-hil/bin/dronestudio-headless"
FCBIN = "/workspace/zig-out-hil/bin/MotorController-hil"
FC = ("127.0.0.1", 5000)
C = (0.70710678, -0.70710678, 0.0, 0.0); C_INV = (0.70710678, 0.70710678, 0.0, 0.0)
def qmul(a, b):
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return (aw*bw - ax*bx - ay*by - az*bz, aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx, aw*bz + ax*by - ay*bx + az*bw)
KP = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0; KD = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0; DT_MS = 80.0
fc = subprocess.Popen([FCBIN], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1.5)
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(3.0)
def send(m): s.sendto(m.encode(), FC)
def hb_daemon():
    hs = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); hs.settimeout(1.0)
    while True:
        try: hs.sendto(b"HEARTBEAT", FC); hs.recvfrom(4096)
        except Exception: pass
        time.sleep(0.4)
threading.Thread(target=hb_daemon, daemon=True).start()
send("CONNECT"); r,_ = s.recvfrom(4096); assert "ACK" in r.decode()
send(json.dumps({"dshot_protocol": 300, "motors": [{"pin": 17, "direction": 0}, {"pin": 27, "direction": 1}, {"pin": 22, "direction": 0}, {"pin": 23, "direction": 1}], "battery": {"cells": 3}}))
r,_ = s.recvfrom(4096); assert "CONFIG_ACK" in r.decode()
send("Battery 16.4")
for i in range(4):
    send(f"Arm {i}"); time.sleep(1.4)
s.sendto(b"HEARTBEAT", FC); r,_ = s.recvfrom(4096)
st = r.decode()
assert " 0 1 " in st and " 1 1 " in st, f"not armed: {st}"
print("armed+battery ok:", st.strip()[:120], flush=True)
send(f"UpdatePidParams Roll {KP} 0.05 {KD}"); send(f"UpdatePidParams Pitch {KP} 0.05 {KD}")
send("UpdatePidParams Yaw 0.1 0.0 0.0")
p = subprocess.Popen([BIN], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
def call(d):
    p.stdin.write(json.dumps(d) + "\n"); p.stdin.flush()
    return json.loads(p.stdout.readline())
call({"cmd": "reset", "seed": 42, "scene": {"spawn": [0.0, 1.5, 0.0], "goal": [0.0, 0.0, 0.0], "obstacles": [], "extent": 10, "max_steps": 1000}})
call({"cmd": "set_dynamics", "path": "/workspace/DroneStudio/autoresearch/fixtures/chassis_v1.manifest.json"})
call({"cmd": "motor_v2", "on": True})
call({"cmd": "hil_listen", "port": 5100, "perm": [1, 0, 2, 3]})
send("UpdateBaseThrottle 11.1")

DT = DT_MS / 1000.0
def loop_window(dur, target_sim):
    """run closed loop for dur seconds with sim-frame target quat (w,x,y,z); returns [(t, ex, ey, ez)]"""
    q_fc_t = qmul(C, qmul(target_sim, C_INV))
    send(f"SetOrientation {q_fc_t[0]} {q_fc_t[1]} {q_fc_t[2]} {q_fc_t[3]}")
    out = []
    t0 = time.time(); n = 0
    while time.time() - t0 < dur:
        stj = call({"cmd": "hil_state"})
        q = stj["quat"]; om = stj["omega"]
        q_sim = (q[3], q[0], q[1], q[2])
        dq = (1.0, om[0]*DT/2, om[1]*DT/2, om[2]*DT/2)
        nn = math.sqrt(sum(v*v for v in dq)); dq = tuple(v/nn for v in dq)
        q_fc = qmul(C, qmul(qmul(q_sim, dq), C_INV))
        send(f"UpdateOrientation {q_fc[0]} {q_fc[1]} {q_fc[2]} {q_fc[3]}")
        send(f"UpdateGyro {om[0]:.5f} {om[2]:.5f} {-om[1]:.5f}")
        call({"cmd": "hil_step", "ticks": 5})
        n += 1
        w, x, y, z = q[3], q[0], q[1], q[2]
        # sim-frame euler (XYZ intrinsic)
        ex = math.degrees(math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)))
        ey = math.degrees(math.asin(max(-1, min(1, 2*(w*y - z*x)))))
        ez = math.degrees(math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
        out.append((time.time() - t0, ex, ey, ez))
        time.sleep(max(0, 0.01 - (time.time() - t0 - n * 0.01)))
    return out

def metrics(trace, axis, target_deg):
    vals = [(r[0], r[axis + 1]) for r in trace]
    t10 = t90 = None
    peak = -1e9; settle = None
    for t, v in vals:
        peak = max(peak, v)
    for t, v in vals:
        if t10 is None and v >= 0.1 * target_deg: t10 = t
        if t90 is None and v >= 0.9 * target_deg: t90 = t
    band = 0.05 * target_deg
    for i in range(len(vals) - 1, -1, -1):
        t, v = vals[i]
        if abs(v - target_deg) > band:
            settle = t
            break
    else:
        settle = 0.0
    rise = (t90 - t10) if (t10 is not None and t90 is not None) else None
    ovs = max(0.0, (peak - target_deg) / target_deg * 100.0)
    return rise, ovs, settle, peak

LEVEL = (1.0, 0.0, 0.0, 0.0)
c5, s5 = math.cos(math.radians(5)), math.sin(math.radians(5))
print("settle 1.5s level", flush=True)
loop_window(1.5, LEVEL)
print("STEP +10deg roll (sim-x)", flush=True)
tr = loop_window(2.5, (c5, s5, 0.0, 0.0))
r = metrics(tr, 0, 10.0)
print(f"ROLL: rise={r[0]}s overshoot={r[1]:.1f}% settle(5%)={r[2]}s peak={r[3]:.2f}deg", flush=True)
print("back to level 1.5s", flush=True)
loop_window(1.5, LEVEL)
print("STEP +10deg pitch (sim-z, world right axis)", flush=True)
tp = loop_window(2.5, (c5, 0.0, 0.0, s5))
r2 = metrics(tp, 2, 10.0)
print(f"PITCH: rise={r2[0]}s overshoot={r2[1]:.1f}% settle(5%)={r2[2]}s peak={r2[3]:.2f}deg", flush=True)
p.terminate(); fc.terminate()
print("STEPRESP_DONE", flush=True)
