"""Evaluator for 3D-printed quad chassis candidates.

Checks, in order of cheapness:
  1. Geometry sanity: watertight, single body, bbox, prop clearance
  2. Printability (FDM, no supports): overhang faces, min wall thickness
  3. Mass properties: mass, CoM, inertia tensor (PETG baseline)
  4. Structural (optional, needs gmsh+ccx): linear static at max thrust + crash case
Each check returns (name, passed, detail, penalty). Score = 1 - sum(penalties).
"""
import sys, math, json
import numpy as np
import trimesh

RHO_PETG = 1240e-9   # kg/mm^3
OVERHANG_LIMIT_DEG = 45.0
MIN_WALL_MM = 1.2    # 3 perimeters at 0.4 mm nozzle
MAX_THRUST_PER_MOTOR_N = 9.7  # AKK RS2205 2300KV fitted full-throttle equilibrium (sim MOTOR_V2.md, kf=7.9e-7 @ 3560 rad/s cap); conservative availability basis for hover margin

def load(path):
    m = trimesh.load(path, force='mesh')
    return m

def check_sanity(m):
    out = []
    out.append(("watertight", m.is_watertight, f"watertight={m.is_watertight}", 0.0 if m.is_watertight else 0.3))
    bodies = m.split(only_watertight=False)
    single = len(bodies) == 1
    out.append(("single_body", single, f"{len(bodies)} bodies", 0.0 if single else 0.1))
    return out

def check_overhang(m):
    n = m.face_normals
    down = n[:, 2] < -math.sin(math.radians(OVERHANG_LIMIT_DEG))
    on_plate = m.triangles_center[:, 2] < m.bounds[0, 2] + 0.5  # build-plate face needs no support
    down = down & ~on_plate
    area = m.area_faces[down].sum()
    frac = area / m.area
    ok = frac < 0.02
    return ("overhang", ok, f"{frac*100:.2f}% of surface overhangs > {OVERHANG_LIMIT_DEG}deg ({area:.0f} mm^2)", 0.0 if ok else min(0.4, frac*4))

def check_wall_thickness(m, samples=4000):
    # cast rays inward from surface points; distance to next hit ~ local thickness
    pts, idx = m.sample(samples, return_index=True)
    normals = -m.face_normals[idx]
    # offset origin slightly outward to avoid self-hit at t=0
    origins = pts - normals * 0.05
    locs, hits, _ = m.ray.intersects_location(origins, normals, multiple_hits=False)
    thick = np.full(samples, np.inf)
    if len(hits):
        d = np.linalg.norm(locs - origins[hits], axis=1) - 0.05
        keep = d > 0.02  # discard self-hits
        thick[hits[keep]] = d[keep]
    finite = thick[np.isfinite(thick)]
    thin = (finite < MIN_WALL_MM).mean() if len(finite) else 0.0
    ok = thin < 0.01
    return ("wall_thickness", ok, f"{thin*100:.2f}% of samples below {MIN_WALL_MM} mm wall", 0.0 if ok else min(0.4, thin*10))

def mass_properties(m, motor_positions_mm, arm_length_mm):
    vol_mm3 = m.volume
    frame_mass_kg = vol_mm3 * RHO_PETG
    com = m.center_mass  # watertight mesh: volume-weighted
    I_frame_mm4 = m.moment_inertia  # geometric, mm^4
    I_frame = I_frame_mm4 * RHO_PETG * 1e-9  # kg*m^2 (mm^4 * kg/mm^3 = kg*mm; x1e-9? -> see below)
    # unit care: mm^4 * kg/mm^3 = kg*mm -> convert kg*mm^2? moment_inertia is integral r^2 dV -> mm^5? No:
    # trimesh moment_inertia: integral of r^2 dV over volume -> units mm^5. physical I = rho[kg/mm^3] * mm^5 -> kg*mm^2; x1e-6 -> kg*m^2
    I_frame = I_frame_mm4 * RHO_PETG * 1e-6
    MOTOR_KG, STACK_KG, BATTERY_KG, ELEC_KG = 0.032, 0.090, 0.180, 0.060
    a = arm_length_mm / math.sqrt(2)
    I_motors = np.zeros((3,3))
    for (mx, my) in motor_positions_mm:
        r2x, r2y, r2z = (my**2)*1e-6, (mx**2)*1e-6, (mx**2+my**2)*1e-6
        I_motors += MOTOR_KG*np.diag([r2x, r2y, r2z])
    I_pay = (STACK_KG+BATTERY_KG+ELEC_KG)*np.diag([(0.03**2),(0.03**2),(0.05**2)])
    total_mass = frame_mass_kg + 4*MOTOR_KG + STACK_KG + BATTERY_KG + ELEC_KG
    I_total = I_frame + I_motors + I_pay
    return {
        "frame_mass_g": round(frame_mass_kg*1000,1),
        "all_up_mass_kg": round(total_mass,3),
        "com_mm": [round(float(x),2) for x in com],
        "inertia_kgm2": [[round(float(v),8) for v in row] for row in I_total],
        "hover_thrust_frac": round(total_mass*9.81/(4*MAX_THRUST_PER_MOTOR_N),3),
    }, total_mass

def score(checks):
    pen = sum(c[3] for c in checks)
    return max(0.0, 1.0 - pen)

if __name__ == "__main__":
    from chassis import ChassisParams
    p = ChassisParams()
    m = load(sys.argv[1] if len(sys.argv) > 1 else "/home/sandbox/cad-researcher/chassis_v1.stl")
    checks = []
    checks += check_sanity(m)
    checks.append(check_overhang(m))
    checks.append(check_wall_thickness(m))
    ok_clear, adj, need = p.check_prop_clearance()
    checks.append(("prop_clearance", ok_clear, f"{adj:.0f} mm vs {need:.0f} mm needed", 0.0 if ok_clear else 0.5))
    props, total = mass_properties(m, p.motor_positions(), p.arm_length_mm)
    hover_ok = props["hover_thrust_frac"] < 0.65
    checks.append(("hover_margin", hover_ok, f"hover at {props['hover_thrust_frac']*100:.0f}% of max thrust", 0.0 if hover_ok else 0.4))
    for name, passed, detail, pen in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    print(json.dumps(props, indent=2))
    print(f"SCORE {score(checks):.3f}")
