#!/usr/bin/env python3
"""A/B: oracle-attitude HiL hover vs spec-noised IMU + complementary estimator.
Same seed, same gains, same duration both modes. Usage: hil_imu_ab.py [spec]"""
import json, math, socket, subprocess, sys, threading, time
sys.path.insert(0, '/workspace/DroneStudio/autoresearch')
from imu_model import ImuNoise, AttitudeEstimator, att_err_deg, qmul

FCBIN = '/workspace/zig-out-hil/bin/MotorController-hil'
BIN = '/workspace/zig-out-hil/bin/dronestudio-headless'
FC = ('127.0.0.1', 5000)
SPEC = sys.argv[1] if len(sys.argv) > 1 else '/workspace/DroneStudio/sensors/icm42688p.json'
C = (0.70710678, -0.70710678, 0.0, 0.0); C_INV = (0.70710678, 0.70710678, 0.0, 0.0)
AKP, AKI, AKD = 15.0, 4.0, 8.0
TARGET_ALT = SPAWN_ALT = 1.5
DUR = 12.0

def run_mode(mode):
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
    send('CONNECT'); r, _ = s.recvfrom(4096); assert 'ACK' in r.decode()
    send(json.dumps({'dshot_protocol': 300, 'motors': [{'pin': 17, 'direction': 0}, {'pin': 27, 'direction': 1}, {'pin': 22, 'direction': 0}, {'pin': 23, 'direction': 1}], 'battery': {'cells': 3}}))
    r, _ = s.recvfrom(4096); assert 'CONFIG_ACK' in r.decode()
    send('Battery 16.4')
    for i in range(4):
        send(f'Arm {i}'); time.sleep(1.4)
    s.sendto(b'HEARTBEAT', FC); r, _ = s.recvfrom(4096)
    assert ' 0 1 ' in r.decode(), 'not armed'
    send('UpdatePidParams Roll 9 2.0 0.6'); send('UpdatePidParams Pitch 9 2.0 0.6')
    send('UpdatePidParams Yaw 1.5 0.0 0.5')
    send(f'UpdatePidParams Altitude {AKP} {AKI} {AKD}')
    p = subprocess.Popen([BIN], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    def call(d):
        p.stdin.write(json.dumps(d) + '\n'); p.stdin.flush()
        return json.loads(p.stdout.readline())
    call({'cmd': 'reset', 'seed': 42, 'scene': {'spawn': [0.0, SPAWN_ALT, 0.0], 'goal': [0.0, 0.0, 0.0], 'obstacles': [], 'extent': 10, 'max_steps': 3000}})
    call({'cmd': 'set_dynamics', 'path': '/workspace/DroneStudio/autoresearch/fixtures/chassis_v1.manifest.json'})
    call({'cmd': 'motor_v2', 'on': True})
    call({'cmd': 'hil_listen', 'port': 5100, 'perm': [1, 0, 2, 3]})
    send('UpdateBaseThrottle 11.1')
    LEVEL = (1.0, 0.0, 0.0, 0.0)
    q_fc_t = qmul(C, qmul(LEVEL, C_INV))
    send(f'SetOrientation {q_fc_t[0]} {q_fc_t[1]} {q_fc_t[2]} {q_fc_t[3]}')
    send(f'SetAltitude {TARGET_ALT}')

    noise = ImuNoise(SPEC, seed=42)
    est = None
    out, t0, n, last_t = [], time.time(), 0, None
    while time.time() - t0 < DUR:
        now = time.time()
        dt_loop = (now - last_t) if last_t else 0.01
        last_t = now
        stj = call({'cmd': 'imu_truth'})
        om, ac = stj['omega'], stj['accel']
        q = call({'cmd': 'hil_state'})['quat']
        q_sim = (q[3], q[0], q[1], q[2])
        if est is None:
            est = AttitudeEstimator(q0=q_sim, k_acc=1.0)
        if mode == 'realistic':
            g_n, a_n = noise.sample(om, ac, dt_loop)
            q_att = est.update(g_n, a_n, dt_loop)
            g_send = g_n
        else:
            dq = (1.0, om[0]*dt_loop/2, om[1]*dt_loop/2, om[2]*dt_loop/2)
            nn = math.sqrt(sum(v*v for v in dq)); dq = tuple(v/nn for v in dq)
            q_att = qmul(q_sim, dq)
            g_send = om
        st2 = call({'cmd': 'hil_state'})
        pos, vel = st2['pos'], st2['vel']
        q_fc = qmul(C, qmul(q_att, C_INV))
        send(f'UpdateOrientation {q_fc[0]} {q_fc[1]} {q_fc[2]} {q_fc[3]}')
        send(f'UpdateGyro {g_send[0]:.5f} {g_send[2]:.5f} {-g_send[1]:.5f}')
        send(f'UpdateAltitude {pos[1]:.4f} {vel[1]:.4f}')
        call({'cmd': 'hil_step', 'ticks': 5})
        n += 1
        err = att_err_deg(est.q, (st2['quat'][3], st2['quat'][0], st2['quat'][1], st2['quat'][2])) if mode == 'realistic' else 0.0
        out.append((now - t0, pos[1], err))
        time.sleep(max(0, 0.01 - (time.time() - now)))
    alts = [a for _, a, _ in out]
    late = [(t, a, e) for t, a, e in out if t > 3.0]
    rmse = math.sqrt(sum((a - TARGET_ALT)**2 for _, a, _ in late) / max(1, len(late)))
    errs = [e for _, _, e in late]
    print(f'{mode}: end_alt={alts[-1]:.3f} late_rmse={rmse:.3f} alt_range=[{min(alts):.2f},{max(alts):.2f}] ' +
          (f'att_err_deg end={errs[-1]:.2f} mean={sum(errs)/len(errs):.2f} max={max(errs):.2f}' if errs else 'oracle attitude'))
    p.terminate(); fc.terminate()
    return rmse, errs

r_clean, _ = run_mode('clean')
r_noisy, errs = run_mode('realistic')
print(f'AB: alt_rmse clean={r_clean:.3f} realistic={r_noisy:.3f} delta={r_noisy-r_clean:+.3f}; att_err_deg end={errs[-1]:.2f} mean={sum(errs)/len(errs):.2f} max={max(errs):.2f}')
