import json, math, socket, subprocess, sys, time, threading
BIN = "/workspace/zig-out-hil/bin/dronestudio-headless"
FCBIN = "/workspace/zig-out-hil/bin/MotorController-hil"
FC = ("127.0.0.1", 5000)
C = (0.70710678, -0.70710678, 0.0, 0.0); C_INV = (0.70710678, 0.70710678, 0.0, 0.0)
def qmul(a, b):
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return (aw*bw - ax*bx - ay*by - az*bz, aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx, aw*bz + ax*by - ay*bx + az*bw)
kp = float(sys.argv[1]); kd = float(sys.argv[2]); DT_MS = float(sys.argv[3])
yawg = 0.1; dur = 20.0
fc = subprocess.Popen([FCBIN], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1.5)
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(3.0)
def send(m): s.sendto(m.encode(), FC)
def hb():
    try: s.sendto(b"HEARTBEAT", FC); s.recvfrom(4096)
    except Exception: pass
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
    send(f"Arm {i}"); time.sleep(1.4); hb()
s.sendto(b"HEARTBEAT", FC)
r, _ = s.recvfrom(4096)
print("STATUS after arm:", r.decode().strip()[:300], flush=True)
send("SetOrientation 1.0 0.0 0.0 0.0")
send(f"UpdatePidParams Roll {kp} 0.05 {kd}"); send(f"UpdatePidParams Pitch {kp} 0.05 {kd}")
send(f"UpdatePidParams Yaw {yawg} 0.0 0.0")
p = subprocess.Popen([BIN], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
def call(d):
    p.stdin.write(json.dumps(d) + "\n"); p.stdin.flush()
    return json.loads(p.stdout.readline())
call({"cmd": "reset", "seed": 42, "scene": {"spawn": [0.0, 1.5, 0.0], "goal": [0.0, 0.0, 0.0], "obstacles": [], "extent": 10, "max_steps": 1000}})
call({"cmd": "set_dynamics", "path": "/workspace/DroneStudio/autoresearch/fixtures/chassis_v1.manifest.json"})
call({"cmd": "motor_v2", "on": True})
call({"cmd": "hil_listen", "port": 5100, "perm": [1, 0, 2, 3]})
send("UpdateBaseThrottle 11.1"); hb()
t0 = time.time(); n = 0; tilts = []
print("  t   tilt  alt   omx    omy    omz", flush=True)
t30 = t90 = None
while time.time() - t0 < dur:
    st = call({"cmd": "hil_state"})
    q = st["quat"]; om = st["omega"]
    q_sim = (q[3], q[0], q[1], q[2])
    DT = DT_MS / 1000.0
    dq = (1.0, om[0]*DT/2, om[1]*DT/2, om[2]*DT/2)
    nn = math.sqrt(sum(v*v for v in dq)); dq = tuple(v/nn for v in dq)
    q_fc = qmul(C, qmul(qmul(q_sim, dq), C_INV))
    send(f"UpdateOrientation {q_fc[0]} {q_fc[1]} {q_fc[2]} {q_fc[3]}")
    call({"cmd": "hil_step", "ticks": 5})
    n += 1
    tilt = math.degrees(2 * math.acos(min(1.0, abs(q[3]))))
    tilts.append(tilt)
    if t30 is None and tilt > 30: t30 = time.time() - t0
    if t90 is None and tilt > 90: t90 = time.time() - t0

    time.sleep(max(0, 0.01 - (time.time() - t0 - n * 0.01)))
p.terminate(); fc.terminate()
print(f"kp={kp} kd={kd} DT={DT_MS}ms maxTilt={max(tilts):.1f} endTilt={tilts[-1]:.1f} t30={t30} t90={t90}", flush=True)
