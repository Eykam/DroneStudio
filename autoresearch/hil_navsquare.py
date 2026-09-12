import json, math, socket, subprocess, sys, time, threading
BIN = '/workspace/zig-out-hil/bin/dronestudio-headless'
FCBIN = '/workspace/zig-out-hil/bin/MotorController-hil'
FC = ('127.0.0.1', 5000)
C = (0.70710678, -0.70710678, 0.0, 0.0); C_INV = (0.70710678, 0.70710678, 0.0, 0.0)
def qmul(a, b):
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return (aw*bw - ax*bx - ay*by - az*bz, aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx, aw*bz + ax*by - ay*bx + az*bw)
DT = float(sys.argv[1]) if len(sys.argv) > 1 else 0.080
# waypoints in FC world frame (x fwd, y right); 1m square back to origin
WPS = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
ARRIVE = 0.25; DWELL = 0.5; LEG_TIMEOUT = 8.0
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
send('UpdatePidParams PosX 0.12 0.02 0.3'); send('UpdatePidParams PosY 0.12 0.02 0.3')
p = subprocess.Popen([BIN], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
def call(d):
    p.stdin.write(json.dumps(d) + '\n'); p.stdin.flush()
    return json.loads(p.stdout.readline())
call({'cmd': 'reset', 'seed': 42, 'scene': {'spawn': [0.0, 1.5, 0.0], 'goal': [0.0, 0.0, 0.0], 'obstacles': [], 'extent': 10, 'max_steps': 4000}})
call({'cmd': 'set_dynamics', 'path': '/workspace/DroneStudio/autoresearch/fixtures/chassis_v1.manifest.json'})
call({'cmd': 'motor_v2', 'on': True})
call({'cmd': 'hil_listen', 'port': 5100, 'perm': [1, 0, 2, 3]})
send('UpdateBaseThrottle 11.1')
LEVEL = (1.0, 0.0, 0.0, 0.0)
q_fc_t = qmul(C, qmul(LEVEL, C_INV))
send(f'SetOrientation {q_fc_t[0]} {q_fc_t[1]} {q_fc_t[2]} {q_fc_t[3]}')
send('SetAltitude 1.5')
wi = 0
send(f'SetPosition {WPS[wi][0]} {WPS[wi][1]}')
leg_start = time.time(); dwell_start = None
t0 = time.time(); n = 0
trace = []
leg_times = []
MAXT = 50.0
while time.time() - t0 < MAXT and wi < len(WPS):
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
    err = math.hypot(pos[0] - WPS[wi][0], pos[2] - WPS[wi][1])
    trace.append((time.time() - t0, pos[0], pos[2], pos[1], err))
    if err < ARRIVE:
        if dwell_start is None: dwell_start = time.time()
        elif time.time() - dwell_start >= DWELL:
            leg_times.append(time.time() - leg_start)
            wi += 1
            if wi < len(WPS):
                send(f'SetPosition {WPS[wi][0]} {WPS[wi][1]}')
                leg_start = time.time(); dwell_start = None
    else:
        dwell_start = None
    if time.time() - leg_start > LEG_TIMEOUT:
        leg_times.append(-(time.time() - leg_start))  # negative = timed out
        wi += 1
        if wi < len(WPS):
            send(f'SetPosition {WPS[wi][0]} {WPS[wi][1]}')
            leg_start = time.time(); dwell_start = None
    time.sleep(max(0, 0.01 - (time.time() - t0 - n * 0.01)))
final_err = math.hypot(trace[-1][1], trace[-1][2])
alt_min = min(t[3] for t in trace); alt_max = max(t[3] for t in trace)
print(f'NAVSQUARE DT={DT}: legs_done={wi}/{len(WPS)} leg_times={[round(t,1) for t in leg_times]} final_err={final_err:.3f}m alt_range=[{alt_min:.2f},{alt_max:.2f}]')
print('TRACE(1s):', [(round(t[0]), round(t[1],2), round(t[2],2)) for t in trace[::100]])
p.terminate(); fc.terminate()
