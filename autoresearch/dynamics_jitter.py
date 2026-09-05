"""Per-episode dynamics tolerance jitter (2026-09-05, user-approved SIM track).

Writes a jittered copy of the base dynamics manifest for one episode seed:
mass / inertia / per-motor thrust / motor lag / drag ratio / CoM each get
independent draws, modeling build tolerance (battery placement, motor
variance, wiring). Deterministic per seed: the same episode always flies
the same airframe, and eval cells stay comparable.

Opt-in: factories only jitter when AUTORESEARCH_DYN_JITTER is set.
Ranges are anchors, not measurements - revisit after bench sysid.
"""
import json, os
import numpy as np

OUT_DIR = "/tmp/dyn_jitter"

MASS_TOL = 0.05        # +/-5% total mass
INERTIA_TOL = 0.08     # +/-8% diagonal inertia terms
MOTOR_THRUST_TOL = 0.04  # +/-4% per motor, independent (trim offsets)
MOTOR_LAG_TOL = 0.25   # +/-25% time constant
DRAG_RATIO_TOL = 0.15  # +/-15%
COM_TOL_M = 0.003      # +/-3mm per axis (battery strap placement)


def _stream(seed):
    # own stream: adding/removing jitter dimensions must not shift scenes
    return np.random.default_rng(np.uint64(seed) ^ np.uint64(0xD7A1))


def jitter_manifest(seed, base_path, out_dir=OUT_DIR):
    """Deterministic jittered copy of base manifest for this episode seed."""
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "dyn_%d.json" % (np.uint64(seed) % (2**63),))
    if os.path.exists(out):
        return out  # same seed -> same airframe, already written
    with open(base_path) as f:
        m = json.load(f)
    r = _stream(seed)

    def u(tol):
        return float(r.uniform(1.0 - tol, 1.0 + tol))

    dyn = m.get("dynamics", {})
    if "total_mass_kg" in dyn:
        dyn["total_mass_kg"] *= u(MASS_TOL)
    inertia = dyn.get("inertia_about_com_kgm2")
    if inertia:
        for k in ("ixx", "iyy", "izz"):
            if k in inertia:
                inertia[k] *= u(INERTIA_TOL)
    com = dyn.get("com_m")
    if com and len(com) == 3:
        dyn["com_m"] = [float(c + r.uniform(-COM_TOL_M, COM_TOL_M)) for c in com]
    for mo in m.get("motors", []):
        if "max_thrust_n" in mo:
            mo["max_thrust_n"] *= u(MOTOR_THRUST_TOL)
        if "time_constant_s" in mo:
            mo["time_constant_s"] *= u(MOTOR_LAG_TOL)
        if "drag_ratio" in mo:
            mo["drag_ratio"] *= u(DRAG_RATIO_TOL)
    tmp = out + ".tmp%d" % os.getpid()
    with open(tmp, "w") as f:
        json.dump(m, f)
    os.replace(tmp, out)  # atomic: parallel workers never see a torn file
    return out
