#!/usr/bin/env python3
"""In-sim PID gain sweep for the rate loop (RATE_CONTROLLER_ASSESSMENT.md,
verification step 5). Runs candidate gains through the real headless binary
(rate_step probe) and scores rise time / overshoot / settling / SS error.

OUTPUT IS SIM-TUNED CANDIDATES ONLY. The plant is the Drone.zig inertia
model with a sphere collider and no aero drag, motor lag, or mixer
quantization. Any candidate MUST be re-validated on hardware with
conservative first flights. Nothing here is a flight-ready gain.
"""
import json, os
from rate_tuning import run_step, analyze, SETPOINT, AXES

CANDIDATES_RP = [  # (kp, ki, kd) for roll (=pitch, same inertia)
    (0.1, 0.005, 0.001),   # firmware baseline
    (0.2, 0.010, 0.002),
    (0.4, 0.020, 0.004),
    (0.8, 0.030, 0.008),
    (1.2, 0.050, 0.012),
]
CANDIDATES_YAW = [
    (0.05, 0.003, 0.0005), # firmware baseline
    (0.15, 0.010, 0.002),
    (0.30, 0.020, 0.005),
]

def sweep(axis_name, axis_idx, candidates):
    rows = []
    for kp, ki, kd in candidates:
        gains = {axis_name: [kp, ki, kd]}
        samples = run_step(axis_idx, gains=gains)
        m = analyze(samples, axis_idx, SETPOINT)
        m["gains"] = {"kp": kp, "ki": ki, "kd": kd}
        rows.append(m)
        print(f"{axis_name} kp={kp} ki={ki} kd={kd}: rise={m['rise_time_s']}s "
              f"overshoot={m['overshoot_pct']}% settle={m['settling_time_s']}s "
              f"ss_err={m['steady_state_err_rad_s']}", flush=True)
    return rows

def pick(rows):
    ok = [r for r in rows if r["reached_setpoint"] and r["overshoot_pct"] <= 20.0
          and abs(r["steady_state_err_rad_s"]) < 0.1]
    if not ok:
        return None
    return min(ok, key=lambda r: (r["rise_time_s"] or 9e9))

def main():
    report = {
        "purpose": "in-sim rate-loop gain sweep - SIM-TUNED CANDIDATES ONLY, hardware validation required",
        "setpoint_rad_s": SETPOINT,
        "roll_pitch": sweep("roll", 0, CANDIDATES_RP),
        "yaw": sweep("yaw", 2, CANDIDATES_YAW),
    }
    rp, yw = pick(report["roll_pitch"]), pick(report["yaw"])
    report["candidates"] = {
        "roll_pitch": rp and rp["gains"], "yaw": yw and yw["gains"],
        "selection_rule": "fastest rise time with overshoot <= 20% and |SS err| < 0.1 rad/s",
        "warning": "SIM-TUNED against the Drone.zig inertia model (no aero drag, motor lag, or mixer). NOT flight-ready. Validate on hardware.",
    }
    print("candidates:", json.dumps(report["candidates"], default=str)[:400], flush=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rate_tune_sweep_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print("WROTE", out)

if __name__ == "__main__":
    main()
