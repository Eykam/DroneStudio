#!/usr/bin/env python3
"""Rate-controller step-response measurements on the real headless binary.

Implements the sim verification plan from RATE_CONTROLLER_ASSESSMENT.md.
Drives dronestudio-headless directly (rate_step command): constant body-rate
setpoint, 500 Hz fast loop with his actual RateController/PID + input
filters, sampled trajectory back over stdio.

Metrics per axis: rise time (10-90%), overshoot, settling time (+-5% band),
steady-state error, mean |torque| in the first 20ms (D-kick probe: the PID
differentiates the ERROR, so a setpoint step injects kd*d(sp)/dt).

Output: rate_tuning_report.json. These are SIM measurements of the firmware
gains against the Drone.zig inertia model - NOT a flight-readiness claim.
"""
import json, subprocess, os, sys
import numpy as np

BIN = os.environ.get("AUTORESEARCH_SIM_BIN", "/workspace/zig-out/bin/dronestudio-headless")
AXES = {"roll": 0, "pitch": 1, "yaw": 2}
SETPOINT = 3.0       # rad/s step
TICKS = 750          # 1.5 s at 500 Hz
SAMPLE_EVERY = 5     # 100 Hz sampling
HOVER = 1.5 * 9.81

def run_step(axis_idx, sp=SETPOINT, ticks=TICKS, gains=None):
    setpoint = [0.0, 0.0, 0.0]
    setpoint[axis_idx] = sp
    proc = subprocess.Popen([BIN], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    def cmd(obj):
        proc.stdin.write(json.dumps(obj) + "\n"); proc.stdin.flush()
        return json.loads(proc.stdout.readline())
    if gains:
        cmd({"cmd": "set_gains", **gains})
    cmd({"cmd": "reset", "seed": 1, "scene": {
        "spawn": [0, 1, 0], "goal": [0, 1, 12], "extent": 40,
        "max_steps": 250, "dynamics_noise": 0, "obstacles": []}})
    res = cmd({"cmd": "rate_step", "setpoint": setpoint, "ticks": ticks,
               "sample_every": SAMPLE_EVERY, "noise": 0})
    proc.stdin.write(json.dumps({"cmd": "close"}) + "\n"); proc.stdin.flush()
    proc.wait(timeout=10)
    return np.array(res["samples"])  # [t, wx, wy, wz, tqx, tqy, tqz]

def analyze(samples, axis_idx, sp):
    t = samples[:, 0]
    w = samples[:, 1 + axis_idx]
    tq = samples[:, 4 + axis_idx]
    lo, hi = 0.1 * sp, 0.9 * sp
    def first_cross(level):
        idx = np.nonzero(w >= level)[0]
        return float(t[idx[0]]) if len(idx) else None
    t10, t90 = first_cross(lo), first_cross(hi)
    rise = (t90 - t10) if (t10 is not None and t90 is not None) else None
    peak = float(w.max())
    overshoot = max(0.0, (peak - sp) / sp * 100.0)
    band = 0.05 * sp
    settle = None
    for i in range(len(t) - 1, -1, -1):
        if abs(w[i] - sp) > band:
            settle = float(t[i]) if i < len(t) - 1 else None
            break
    ss = w[int(len(w) * 0.9):]
    kick = float(np.mean(np.abs(tq[:4])))  # first ~20 ms
    return {
        "rise_time_s": None if rise is None else round(rise, 4),
        "t10_s": t10, "t90_s": t90,
        "overshoot_pct": round(overshoot, 2),
        "settling_time_s": settle,
        "steady_state_mean_rad_s": round(float(ss.mean()), 4),
        "steady_state_err_rad_s": round(float(sp - ss.mean()), 4),
        "peak_rate_rad_s": round(peak, 4),
        "mean_abs_torque_first_20ms_Nm": round(kick, 5),
        "mean_abs_torque_steady_Nm": round(float(np.mean(np.abs(tq[int(len(tq)*0.9):]))), 5),
        "reached_setpoint": bool(t90 is not None),
    }

def main():
    report = {
        "purpose": "step-response of his firmware RateController gains on the headless binary (Drone.zig inertia)",
        "binary": BIN, "setpoint_rad_s": SETPOINT, "ticks": TICKS, "sample_every": SAMPLE_EVERY,
        "note": "SIM measurement vs simulated inertia - tuning candidates need hardware validation; NOT flight-ready claims",
        "axes": {},
    }
    for name, idx in AXES.items():
        samples = run_step(idx)
        report["axes"][name] = analyze(samples, idx, SETPOINT)
        print(f"{name}: {json.dumps(report['axes'][name])}", flush=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rate_tuning_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print("WROTE", out)

if __name__ == "__main__":
    main()
