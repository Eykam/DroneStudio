import json, socket, subprocess, time

BIN = "/workspace/zig-out-hil/bin/dronestudio-headless"
FC = ("127.0.0.1", 5000)

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(3.0)
def send(m, expect=None):
    s.sendto(m.encode(), FC)
    if expect:
        r, _ = s.recvfrom(2048)
        assert expect in r.decode(), (expect, r.decode())
def hb():
    s.sendto(b"HEARTBEAT", FC)
    try: s.recvfrom(2048)
    except Exception: pass

send("CONNECT", "ACK")
send(json.dumps({"dshot_protocol": 2, "motors": [{"pin": 17, "direction": 0}, {"pin": 27, "direction": 1}, {"pin": 22, "direction": 0}, {"pin": 23, "direction": 1}], "battery": {"cells": 3}}), "CONFIG_ACK")
send("Battery 16.4")
for i in range(4):
    send(f"Arm {i}"); time.sleep(1.4); hb()
print("armed", flush=True)

for i in range(4):
    hb()
    p = subprocess.Popen([BIN], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    def call(d):
        p.stdin.write(json.dumps(d) + "\n"); p.stdin.flush()
        return json.loads(p.stdout.readline())
    call({"cmd": "reset", "seed": 42, "scene": {"spawn": [0.0, 1.5, 0.0], "goal": [0.0, 0.0, 0.0], "obstacles": [], "extent": 10, "max_steps": 1000}})
    call({"cmd": "set_dynamics", "path": "/workspace/DroneStudio/autoresearch/fixtures/chassis_v1.manifest.json"})
    call({"cmd": "motor_v2", "on": True})
    call({"cmd": "hil_listen", "port": 5100})
    send(f"SetSpeed {i} 20")
    time.sleep(0.4)  # listener picks up the new throttle
    call({"cmd": "hil_step", "ticks": 40})
    st = call({"cmd": "hil_state"})
    send(f"SetSpeed {i} 0")
    time.sleep(0.2)
    w = [round(x, 3) for x in st["omega"]]
    mo = [round(x) for x in st["motor_omega"]]
    q = [round(x, 3) for x in st["quat"]]
    print(f"FC{i}: body_rate={w} sim_omega={mo} quat={q}", flush=True)
    p.terminate()
print("CORNERS3_DONE", flush=True)
