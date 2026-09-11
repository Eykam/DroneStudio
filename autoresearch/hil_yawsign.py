import json, math, socket, subprocess, sys, time, threading
BIN = "/workspace/zig-out-hil/bin/dronestudio-headless"
FCBIN = "/workspace/zig-out-hil/bin/MotorController-hil"
FC = ("127.0.0.1", 5000)
C = (0.70710678, -0.70710678, 0.0, 0.0); C_INV = (0.70710678, 0.70710678, 0.0, 0.0)
def qmul(a, b):
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return (aw*bw - ax*bx - ay*by - az*bz, aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx, aw*bz + ax*by - ay*bx + az*bw)
YKP = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5
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
assert " 0 1 " in r.decode(), "not armed"
send("UpdatePidParams Roll 3 0.05 2"); send("UpdatePidParams Pitch 3 0.05 2")
send(f"UpdatePidParams Yaw {YKP} 0.0 0.5")
p = subprocess.Popen([BIN], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
def call(d):
    p.stdin.write(json.dumps(d) + "\n"); p.stdin.flush()
    return json.loads(p.stdout.readline())
call({"cmd": "reset", "seed": 42, "scene": {"spawn": [0.0, 1.5, 0.0], "goal": [0.0, 0.0, 0.0], "obstacles": [], "extent": 10, "max_steps": 1000}})
call({"cmd": "set_dynamics", "path": "/workspace/DroneStudio/autoresearch/fixtures/chassis_v1.manifest.json"})
call({"cmd": "motor_v2", "on": True})
call({"cmd": "hil_listen", "port": 5100, "perm": [1, 0, 2, 3]})
send("UpdateBaseThrottle 11.1")
DT = 0.080
def loop_window(dur, target_sim):
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
        call({"cmd": "hil_step", "ticks": 5})
        n += 1
        w, x, y, z = q[3], q[0], q[1], q[2]
        ey = math.degrees(math.asin(max(-1, min(1, 2*(w*y - z*x)))))
        out.append((time.time() - t0, ey))
        time.sleep(max(0, 0.01 - (time.time() - t0 - n * 0.01)))
    return out
LEVEL = (1.0, 0.0, 0.0, 0.0)
c75, s75 = math.cos(math.radians(7.5)), math.sin(math.radians(7.5))
loop_window(1.5, LEVEL)
tr = loop_window(4.0, (c75, 0.0, s75, 0.0))  # +15deg about sim world-y (up) = yaw step
yaws = [v for _, v in tr]
print(f"YAWTEST kp={YKP}: start={yaws[0]:.2f} end={yaws[-1]:.2f} min={min(yaws):.2f} max={max(yaws):.2f} target=+15", flush=True)
mid = len(yaws)//2
print(f"trace(0.4s): {[round(v,1) for v in yaws[::20]]}", flush=True)
p.terminate(); fc.terminate()
