import socket, struct, threading, time, math, json

FC = ("127.0.0.1", 5000)
collected = []
stop = False

def listener():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 5100))
    s.settimeout(0.5)
    while not stop:
        try:
            data, _ = s.recvfrom(64)
        except socket.timeout:
            continue
        if len(data) == 24 and data[:4] == b"HIL1":
            seq, = struct.unpack_from("<I", data, 4)
            thr = struct.unpack_from("<4H", data, 8)
            amask, = struct.unpack_from("<I", data, 16)
            collected.append((time.time(), seq, thr, amask))

t = threading.Thread(target=listener, daemon=True); t.start()
c = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); c.settimeout(3.0)
def send(m, expect=None):
    c.sendto(m.encode() if isinstance(m, str) else m, FC)
    if expect:
        r, _ = c.recvfrom(2048)
        assert expect in r.decode(), (expect, r)
        return r.decode()

def heartbeat():
    while not stop:
        try: c.sendto(b"HEARTBEAT", FC); c.recvfrom(2048)
        except Exception: pass
        time.sleep(0.5)

send("CONNECT", "ACK")
cfg = {"dshot_protocol": 2, "motors": [{"pin": 17, "direction": 0}, {"pin": 27, "direction": 1},
                                      {"pin": 22, "direction": 0}, {"pin": 23, "direction": 1}],
       "battery": {"cells": 3}}
send(json.dumps(cfg), "CONFIG_ACK")
send("Battery 16.4")   # clear the boot low-battery failsafe BEFORE anything else
print("configured+battery", flush=True)
hb = threading.Thread(target=heartbeat, daemon=True); hb.start()
time.sleep(1.0)
collected.clear(); time.sleep(0.5)
pre = len(collected)
print(f"packets pre-arm (0.5s): {pre}  (expect ~500, amask 0)", flush=True)
if collected:
    print("  sample:", collected[-1][1], collected[-1][2], "amask", collected[-1][3], flush=True)
for i in range(4):
    send(f"Arm {i}")
    time.sleep(1.4)
    print(f"armed {i}", flush=True)
send("UpdateBaseThrottle 45.0")

def quat(roll=0.0, pitch=0.0, yaw=0.0):
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    return (cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy, cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy)

w, x, y, z = quat(roll=math.radians(5.0))
send(f"SetOrientation {w} {x} {y} {z}")
print("target 5deg roll; streaming level current 50Hz x 3s", flush=True)
time.sleep(0.3); collected.clear()
w0, x0, y0, z0 = quat()
t0 = time.time()
while time.time() - t0 < 3.0:
    c.sendto(f"UpdateOrientation {w0} {x0} {y0} {z0}".encode(), FC)
    time.sleep(0.02)
time.sleep(0.3)
stop = True
n = len(collected)
print(f"HIL1 packets: {n} in ~3s ({n/3:.0f}/s)", flush=True)
if n > 10:
    seqs = [p[1] for p in collected]
    gaps = sum(1 for a, b in zip(seqs, seqs[1:]) if b - a != 1)
    amasks = set(p[3] for p in collected)
    thrs = [p[2] for p in collected[-200:]]
    means = [sum(t2[i] for t2 in thrs)/len(thrs) for i in range(4)]
    print(f"seq gaps: {gaps}, armed masks: {amasks}", flush=True)
    print(f"mean throttles (last 200): {[round(m,1) for m in means]}  spread {max(means)-min(means):.1f}", flush=True)
