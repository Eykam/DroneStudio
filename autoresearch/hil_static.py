import json, math, socket, time

FC = ("127.0.0.1", 5000)
C = (math.cos(-math.pi/4), math.sin(-math.pi/4), 0.0, 0.0)
C_INV = (C[0], -C[1], -C[2], -C[3])

def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return (w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2)

def aa(axis, deg):
    h = math.radians(deg) / 2
    return (math.cos(h), axis[0]*math.sin(h), axis[1]*math.sin(h), axis[2]*math.sin(h))

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(3.0)
def send(m, expect=None):
    s.sendto(m.encode(), FC)
    if expect:
        r, _ = s.recvfrom(4096)
        assert expect in r.decode(), (expect, r.decode())
def throttles():
    s.sendto(b"HEARTBEAT", FC)
    r, _ = s.recvfrom(4096)
    txt = r.decode()
    # STATUS <i> <armed> <thr> x4 -> parse first 8 ints after STATUS
    parts = txt.split()
    vals = [int(parts[3]), int(parts[6]), int(parts[9]), int(parts[12])]
    eul = txt[txt.find("CURR_EULER"):txt.find("CURR_EULER")+40] if "CURR_EULER" in txt else ""
    return vals, eul
send("CONNECT", "ACK")
send(json.dumps({"dshot_protocol": 2, "motors": [{"pin": 17, "direction": 0}, {"pin": 27, "direction": 1}, {"pin": 22, "direction": 0}, {"pin": 23, "direction": 1}], "battery": {"cells": 3}}), "CONFIG_ACK")
send("Battery 16.4")
for i in range(4):
    send(f"Arm {i}"); time.sleep(1.4)
send("UpdateBaseThrottle 11.1")
send("SetOrientation 1.0 0.0 0.0 0.0")
for axis, kp, ki, kd in (("Roll", 8, 0.2, 0.0), ("Pitch", 8, 0.2, 0.0), ("Yaw", 3, 0.0, 0.0)):
    send(f"UpdatePidParams {axis} {kp} {ki} {kd}")
time.sleep(0.3)
print("level:", throttles(), flush=True)
tests = [
    ("sim-x +10 (roll)", aa((1,0,0), 10)),
    ("sim-x -10 (roll)", aa((1,0,0), -10)),
    ("sim-z +10 (pitch)", aa((0,0,1), 10)),
    ("sim-z -10 (pitch)", aa((0,0,1), -10)),
]
for name, q_sim in tests:
    q_fc = qmul(C, qmul(q_sim, C_INV))
    for _ in range(5):
        s.sendto(f"UpdateOrientation {q_fc[0]} {q_fc[1]} {q_fc[2]} {q_fc[3]}".encode(), FC)
        time.sleep(0.05)
    time.sleep(0.3)
    v, e = throttles()
    print(f"{name}: thr={v} {e}", flush=True)
# back to level
q_fc = qmul(C, qmul((1.0,0,0,0), C_INV))
s.sendto(f"UpdateOrientation {q_fc[0]} {q_fc[1]} {q_fc[2]} {q_fc[3]}".encode(), FC)
print("STATIC_DONE", flush=True)
