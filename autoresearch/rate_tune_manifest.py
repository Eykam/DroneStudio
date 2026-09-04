#!/usr/bin/env python3
"""In-sim rate-loop gain sweep against the MANIFEST airframe (chassis_v1,
real motor model + aero drag + 0.514 kg / 20x-smaller inertia).

Axis reality in headless: body +Y is vertical (thrust axis). PID channels
are labels on rates[0..2]: roll->axis0 (I=0.0022, arm-lever torque),
pitch->axis1 (VERTICAL, I=0.0039, ktau-only authority ~6.7x weaker),
yaw->axis2 (I=0.0021, arm-lever). The original abstract sweep treated
axis2 as the big-inertia vertical; that labeling was inconsistent with
the +Y-up body and is corrected here by tuning axes, not labels.

SIM-TUNED CANDIDATES ONLY - hardware re-validation required.
"""
import json, os, subprocess
import numpy as np
from rate_tuning import analyze, SETPOINT

BIN = "/workspace/zig-out/bin/dronestudio-headless"
MANIFEST = "/workspace/DroneStudio/autoresearch/fixtures/chassis_v1.manifest.json"
HOVER = 0.514324 * 9.81
TICKS = 750
SAMPLE_EVERY = 5

def run_step_manifest(axis_idx, gains, sp=SETPOINT):
    setpoint = [0.0, 0.0, 0.0]
    setpoint[axis_idx] = sp
    proc = subprocess.Popen([BIN], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    def cmd(obj):
        proc.stdin.write(json.dumps(obj) + "\n"); proc.stdin.flush()
        return json.loads(proc.stdout.readline())
    cmd({"cmd": "set_dynamics", "path": MANIFEST})
    cmd({"cmd": "set_gains", **gains})
    cmd({"cmd": "reset", "seed": 1, "scene": {
        "spawn": [0, 1, 0], "goal": [0, 1, 12], "extent": 40,
        "max_steps": 250, "dynamics_noise": 0, "obstacles": []}})
    res = cmd({"cmd": "rate_step", "setpoint": setpoint, "ticks": TICKS,
               "sample_every": SAMPLE_EVERY, "noise": 0, "thrust": HOVER})
    proc.stdin.write(json.dumps({"cmd": "close"}) + "\n"); proc.stdin.flush()
    proc.wait(timeout=10)
    return np.array(res["samples"])

GRID_HORIZ = [(kp, round(kp*0.1, 4), round(kp*0.02, 5)) for kp in (0.05, 0.1, 0.2, 0.4, 0.8)]
GRID_VERT  = [(kp, round(kp*0.05, 4), round(kp*0.01, 5)) for kp in (0.5, 1.0, 2.0, 4.0, 8.0)]

def sweep(channel, axis_idx, candidates):
    rows = []
    for kp, ki, kd in candidates:
        samples = run_step_manifest(axis_idx, {channel: [kp, ki, kd]})
        m = analyze(samples, axis_idx, SETPOINT)
        m["gains"] = {"kp": kp, "ki": ki, "kd": kd}
        rows.append(m)
        print(f"{channel}(axis{axis_idx}) kp={kp}: rise={m[chr(114)+chr(105)+chr(115)+chr(101)+chr(95)+chr(116)+chr(105)+chr(109)+chr(101)+chr(95)+chr(115)]}s ovr={m[chr(111)+chr(118)+chr(101)+chr(114)+chr(115)+chr(104)+chr(111)+chr(111)+chr(116)+chr(95)+chr(112)+chr(99)+chr(116)]}% ss={m[chr(115)+chr(116)+chr(101)+chr(97)+chr(100)+chr(121)+chr(95)+chr(115)+chr(116)+chr(97)+chr(116)+chr(101)+chr(95)+chr(101)+chr(114)+chr(114)+chr(95)+chr(114)+chr(97)+chr(100)+chr(95)+chr(115)]}", flush=True)
    return rows

def pick(rows):
    ok = [r for r in rows if r["reached_setpoint"] and r["overshoot_pct"] <= 20.0
          and abs(r["steady_state_err_rad_s"]) < 0.1]
    return min(ok, key=lambda r: (r["rise_time_s"] or 9e9)) if ok else None

def main():
    report = {
        "purpose": "rate-loop gain sweep vs manifest airframe (chassis_v1, motor model + aero) - SIM-TUNED ONLY",
        "dynamics": "chassis_v1 manifest, 0.5143 kg, I=[0.00220,0.00394,0.00213] sim-frame, EMAX params",
        "setpoint_rad_s": SETPOINT,
        "axis0_roll_channel": sweep("roll", 0, GRID_HORIZ),
        "axis1_vertical_pitch_channel": sweep("pitch", 1, GRID_VERT),
        "axis2_yaw_channel": sweep("yaw", 2, GRID_HORIZ),
    }
    picks = {k: (pick(v) or {}).get("gains") for k, v in report.items() if k.startswith("axis")}
    report["candidates"] = {
        "roll_axis0": picks["axis0_roll_channel"],
        "pitch_axis1_VERTICAL": picks["axis1_vertical_pitch_channel"],
        "yaw_axis2": picks["axis2_yaw_channel"],
        "note": "pitch channel drives the VERTICAL axis (weak ktau authority) - its gains are intentionally much higher",
        "warning": "SIM-TUNED against chassis_v1 manifest dynamics. NOT flight-ready. Validate on hardware.",
    }
    print("candidates:", json.dumps(report["candidates"], default=str)[:500])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rate_tune_manifest_report.json")
    json.dump(report, open(out, "w"), indent=2)
    print("WROTE", out)

if __name__ == "__main__":
    main()
