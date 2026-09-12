import json, math, socket, subprocess, sys, time, threading
BIN = '/workspace/zig-out-hil/bin/dronestudio-headless'
FCBIN = '/workspace/zig-out-hil/bin/MotorController-hil'
FC = ('127.0.0.1', 5000)
C = (0.70710678, -0.70710678, 0.0, 0.0); C_INV = (0.70710678, 0.70710678, 0.0, 0.0)
def qmul(a, b):
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return (aw*bw - ax*bx - ay*by - az*bz, aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx, aw*bz + ax*by - ay*bx + az*bw)
PKP = float(sys.argv[1]) if len(sys.argv) > 1 else 0.08
PKI = float(sys.argv[2]) if len(sys.argv) > 2 else 0.005
PKD = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15
SX = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0   # sim spawn x (fwd)
SZ = float(sys.argv[5]) if len(sys.argv) > 5 else -0.5  # sim spawn z (right)
fc = subprocess.Popen([FCBIN], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1.5)
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(3.0)
def send(m): s.sendto(m.encode(), FC)
def hb_daemon():
    hs = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); hs.settimeout(1.0)
    while True:
        try: hs.sendto(b'HEARTBEAT', FC); hs.recvfrom(4096)
        except Exception: pass
        time.sleep(0.4)
threading.Thread(target=hb_daemon, daemon=True).start()
send('CONNECT'); r,_ = s.recvfrom(4096); assert 'ACK' in r.decode()
send(json.dumps({'dshot_protocol': 300, 'motors': [{'pin': 17, 'direction': 0}, {'pin': 27, 'direction': 1}, {'pin': 22, 'direction': 0}, {'pin': 23, 'direction': 1}], 'battery': {'cells': 3}}))
r,_ = s.recvfrom(4096); assert 'CONFIG_ACK' in r.decode()
send('Battery 16.4')
for i in range(4):
    send(f'Arm {i}'); time.sleep(1.4)
s.sendto(b'HEARTBEAT', FC); r,_ = s.recvfrom(4096)
assert ' 0 1 ' in r.decode(), 'not armed'
send('UpdatePidParams Roll 9 2.0 0.6'); send('UpdatePidParams Pitch 9 2.0 0.6')
send('UpdatePidParams Yaw 1.5 0.0 0.5')
send('UpdatePidParams Altitude 15 4 8')
send(f'UpdatePidParams PosX {PKP} {PKI} {PKD}'); send(f'UpdatePidParams PosY {PKP} {PKI} {PKD}')
p = subprocess.Popen([BIN], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
def call(d):
    p.stdin.write(json.dumps(d) + '\n'); p.stdin.flush()
    return json.loads(p.stdout.readline())
call({'cmd': 'reset', 'seed': 42, 'scene': {'spawn': [SX, 1.5, SZ], 'goal': [0.0, 0.0, 0.0], 'obstacles': [], 'extent': 10, 'max_steps': 2500}})
call({'cmd': 'set_dynamics', 'path': '/workspace/DroneStudio/autoresearch/fixtures/chassis_v1.manifest.json'})
call({'cmd': 'motor_v2', 'on': True})
call({'cmd': 'hil_listen', 'port': 5100, 'perm': [1, 0, 2, 3]})
send('UpdateBaseThrottle 11.1')
DT = float(sys.argv[6]) if len(sys.argv) > 6 else 0.080
LEVEL = (1.0, 0.0, 0.0, 0.0)
q_fc_t = qmul(C, qmul(LEVEL, C_INV))
send(f'SetOrientation {q_fc_t[0]} {q_fc_t[1]} {q_fc_t[2]} {q_fc_t[3]}')
send('SetAltitude 1.5')
send('SetPosition 0.0 0.0')
out = []
t0 = time.time(); n = 0
DUR = 10.0
while time.time() - t0 < DUR:
    stj = call({'cmd': 'hil_state'})
    q = stj['quat']; om = stj['omega']; pos = stj['pos']; vel = stj['vel']
    q_sim = (q[3], q[0], q[1], q[2])
    dq = (1.0, om[0]*DT/2, om[1]*DT/2, om[2]*DT/2)
    nn = math.sqrt(sum(v*v for v in dq)); dq = tuple(v/nn for v in dq)
    q_fc = qmul(C, qmul(qmul(q_sim, dq), C_INV))
    send(f'UpdateOrientation {q_fc[0]} {q_fc[1]} {q_fc[2]} {q_fc[3]}')
    send(f'UpdateGyro {om[0]:.5f} {om[2]:.5f} {-om[1]:.5f}')
    send(f'UpdateAltitude {pos[1]:.4f} {vel[1]:.4f}')
    send(f'UpdatePosition {pos[0]:.4f} {pos[2]:.4f} {vel[0]:.4f} {vel[2]:.4f}')
    call({'cmd': 'hil_step', 'ticks': 5})
    n += 1
    out.append((time.time() - t0, pos[0], pos[2], pos[1]))
    time.sleep(max(0, 0.01 - (time.time() - t0 - n * 0.01)))
errs = [math.hypot(px, pz) for _, px, pz, _ in out]
alts = [a for _, _, _, a in out]
late = [e for t, e in zip([o[0] for o in out], errs) if t > 3.0]
print(f'POSHOLD kp={PKP} ki={PKI} kd={PKD} spawn=({SX},{SZ}): end_err={errs[-1]:.3f}m late_mean={sum(late)/max(1,len(late)):.3f} late_max={max(late):.3f} alt_end={alts[-1]:.2f}')
print('ERR(0.5s):', [round(e,2) for e in errs[::50]])
p.terminate(); fc.terminate()
