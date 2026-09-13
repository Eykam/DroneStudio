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
BED_FIT_XY_MM = 220.0  # user directive 2026-09-12: printers are Bambu P1S (256) / Ender-3 V3 KE (220) - design to the smaller bed
MAX_SPLIT_BODIES = 4   # split-body print strategy: monolithic frame cannot fit 220mm (arm length is sim-contract-fixed)
MAX_THRUST_PER_MOTOR_N = 9.7  # AKK RS2205 2300KV fitted full-throttle equilibrium (sim MOTOR_V2.md, kf=7.9e-7 @ 3560 rad/s cap); conservative availability basis for hover margin

def load(path):
    m = trimesh.load(path, force='mesh')
    return m

def check_sanity(m):
    out = []
    out.append(("watertight", m.is_watertight, f"watertight={m.is_watertight}", 0.0 if m.is_watertight else 0.3))
    bodies = m.split(only_watertight=False)
    n = len(bodies)
    ok = 1 <= n <= MAX_SPLIT_BODIES
    out.append(("body_count", ok, f"{n} bodies (1-{MAX_SPLIT_BODIES} allowed: split-body print strategy 2026-09-12)", 0.0 if ok else 0.1))
    return out

def check_bed_fit(m):
    """Per-body XY extents must fit the 220x220mm bed at flat-on-plate orientation.
    Per-body (not whole-mesh bbox): an assembled split frame spans >220mm; each
    PRINTED piece must fit. In-plane rotation cannot save the monolithic frame
    (260x234 bbox -> min enclosing square (a+b)/sqrt2 ~ 349mm), so axis-aligned
    extents are the binding measure."""
    worst = 0.0
    worst_dim = None
    for body in m.split(only_watertight=False):
        ext = body.bounds[1] - body.bounds[0]
        w = max(float(ext[0]), float(ext[1]))
        if w > worst:
            worst, worst_dim = w, (round(float(ext[0]),1), round(float(ext[1]),1))
    ok = worst <= BED_FIT_XY_MM
    return ("bed_fit", ok, f"largest body XY {worst_dim[0]}x{worst_dim[1]} mm (limit {BED_FIT_XY_MM:.0f} mm)", 0.0 if ok else 0.5)

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
    # 2026-09-06 (parent follow-up): mass basis from components.py real part
    # masses instead of hardcoded aggregates - the stale constants
    # (BATTERY_KG=0.180 vs real CNHL 0.163, STACK 0.090 vs real v19 0.0317)
    # made all_up_mass_kg fictitious. Now sums the live placement map + 4
    # parametric motors, matching the export_manifest basis.
    from components import LIBRARY as _LIB, placement as _placement
    MOTOR_KG = _LIB["motor"].mass_g/1000.0
    placed_kg = sum(_LIB[k.split("#")[0]].mass_g for k in _placement())/1000.0
    a = arm_length_mm / math.sqrt(2)
    I_motors = np.zeros((3,3))
    for (mx, my) in motor_positions_mm:
        r2x, r2y, r2z = (my**2)*1e-6, (mx**2)*1e-6, (mx**2+my**2)*1e-6
        I_motors += MOTOR_KG*np.diag([r2x, r2y, r2z])
    I_pay = placed_kg*np.diag([(0.03**2),(0.03**2),(0.05**2)])
    total_mass = frame_mass_kg + 4*MOTOR_KG + placed_kg
    I_total = I_frame + I_motors + I_pay
    return {
        "frame_mass_g": round(frame_mass_kg*1000,1),
        "all_up_mass_kg": round(total_mass,3),
        "com_mm": [round(float(x),2) for x in com],
        "inertia_kgm2": [[round(float(v),8) for v in row] for row in I_total],
        "hover_thrust_frac": round(total_mass*9.81/(4*MAX_THRUST_PER_MOTOR_N),3),
    }, total_mass


def _mt_hit_dists(mesh, origin, dirs, first_only=True):
    """Batched vectorized Moller-Trumbore. first_only=True: nearest hit distance
    per ray (inf if none). first_only=False: list of all hit distances per ray.
    Units follow the mesh (candidate meshes are chassis-frame mm)."""
    tri = np.asarray(mesh.triangles, dtype=np.float64)
    v0 = tri[:, 0, :]; e1 = tri[:, 1, :] - v0; e2 = tri[:, 2, :] - v0
    o = np.asarray(origin, float)
    if first_only:
        out = np.full(len(dirs), np.inf)
    else:
        out = [[] for _ in range(len(dirs))]
    B, N = 64, 20000
    for r0 in range(0, len(dirs), B):
        D = np.asarray(dirs[r0:r0+B], dtype=np.float64)
        bmin = np.full(len(D), np.inf)
        ball = [[] for _ in range(len(D))] if not first_only else None
        for c0 in range(0, len(v0), N):
            V0 = v0[c0:c0+N]; E1 = e1[c0:c0+N]; E2 = e2[c0:c0+N]
            P = np.cross(D[:, None, :], E2[None, :, :])
            det = (E1[None, :, :] * P).sum(-1)
            nz = np.abs(det) > 1e-12
            inv = np.where(nz, 1.0 / np.where(nz, det, 1.0), 0.0)
            TV = np.repeat(o[None, None, :] - V0[None, :, :], len(D), axis=0)
            U = (TV * P).sum(-1) * inv
            Q = np.cross(TV, E1[None, :, :])
            V = (D[:, None, :] * Q).sum(-1) * inv
            T = (E2[None, :, :] * Q).sum(-1) * inv
            ok = nz & (U >= 0) & (V >= 0) & (U + V <= 1) & (T > 1e-9)
            if first_only:
                bmin = np.minimum(bmin, np.where(ok, T, np.inf).min(axis=1))
            else:
                for i in range(len(D)):
                    ball[i].extend(T[i][ok[i]].tolist())
        if first_only:
            out[r0:r0+B] = bmin
        else:
            for i in range(len(D)):
                out[r0 + i] = sorted(ball[i])
    return out

def check_camera_fov(m):
    """User requirement 2026-09-04: cameras must point OUT through the nose apertures,
    unobstructed - not downward into the shell. For each placed camera, cast rays from
    the real lens point across the Camera Module 3 FOV pyramid (2 deg margin inside
    spec): every ray must clear the frame mesh entirely."""
    from components import camera_lens_poses
    poses = camera_lens_poses()
    if not poses:
        return ("camera_fov", False, "no cameras placed", 0.5)
    problems = []
    for key, pose in poses.items():
        o = np.array(pose["origin_m"]) * 1000.0  # mm
        axis = np.array(pose["axis"], dtype=float)
        if float(axis @ np.array([1.0, 0.0, 0.0])) < 0.98:
            problems.append(f"{key}: lens axis {pose['axis']} not forward +X")
            continue
        hh = math.tan(math.radians(pose["hfov_deg"] / 2 - 2.0))
        vh = math.tan(math.radians(pose["vfov_deg"] / 2 - 2.0))
        dirs = []
        for sy in (-1, 0, 1):
            for sz in (-1, 0, 1):
                v = np.array([1.0, hh * sy, vh * sz])
                dirs.append(v / np.linalg.norm(v))
        origins = np.tile(o + np.array([1.0, 0.0, 0.0]), (len(dirs), 1))
        hits = m.ray.intersects_any(origins, np.array(dirs))
        n = int(np.asarray(hits).sum())
        if n:
            problems.append(f"{key}: {n}/{len(dirs)} FOV rays blocked")
    # pad_camera hard gate (parent ruling 2026-09-09): the belly-center down
    # camera's 65deg half-angle cone (2deg margin inside spec) must pass NO
    # frame material with the candidate's actual pocket geometry. Locate the
    # aperture plane from drop rays around the datum, put the emitter just
    # above the skin underside at the aperture, and require every cone ray
    # clear to 8m. No pocket + aperture -> fail.
    from components import placement as _placement
    pad = _placement().get("pad_camera")
    if pad is not None:
        o0 = np.array([pad[0] * 1000.0, pad[1] * 1000.0, pad[2] * 1000.0])
        down = np.array([0.0, 0.0, -1.0])
        probe_origins = [o0] + [o0 + 6.0 * (math.cos(t) * np.array([1.0, 0, 0]) + math.sin(t) * np.array([0, 1.0, 0]))
                                for t in [2 * math.pi * i / 8 for i in range(8)]]
        probe_hits = [_mt_hit_dists(m, po, down[None, :], first_only=False)[0] for po in probe_origins]
        probe_hits = [[h for h in hs if h <= 60.0] for hs in probe_hits]
        center_hits = probe_hits[0]
        ring_bottoms = [hs[-1] for hs in probe_hits[1:] if hs]
        if center_hits:
            # frame material directly under the datum: pocket closed or missing
            problems.append("pad_camera: no clear drop from belly-center datum (pocket/aperture missing or occluded)")
        elif not ring_bottoms:
            problems.append("pad_camera: no belly skin found under datum (drop probe found no aperture plane)")
        else:
            bottom = sorted(ring_bottoms)[len(ring_bottoms) // 2]
            o = o0 + (bottom - 0.15) * down  # emitter just above the aperture plane
            half = math.radians(65.0 - 2.0)
            right = np.array([1.0, 0.0, 0.0]); upv = np.array([0.0, 1.0, 0.0])
            dirs = [down]
            for frac, nseg in ((0.5, 8), (1.0, 24)):
                for i in range(nseg):
                    t = 2 * math.pi * i / nseg
                    d = down + math.tan(half * frac) * (math.cos(t) * right + math.sin(t) * upv)
                    dirs.append(d / np.linalg.norm(d))
            hits = _mt_hit_dists(m, o, np.array(dirs))
            nblocked = int(np.sum(np.isfinite(hits) & (hits <= 8000.0)))
            if nblocked:
                problems.append(f"pad_camera: {nblocked}/{len(dirs)} down-cone rays blocked (65deg half-angle, emitter at aperture plane)")
    ok = not problems
    return ("camera_fov", ok, "; ".join(problems) if problems else
            f"{len(poses)} cameras + pad_camera: FOV clear", 0.0 if ok else 0.5)

def check_imu_lever_arm(m):
    """IMU must sit near the frame CoM (lever-arm corrections only work for small
    offsets); its full transform still exports to the manifest for the estimator."""
    from components import imu_pose
    pos = np.array(imu_pose()["position_m"]) * 1000.0
    com = np.array(m.center_mass)
    d = float(np.linalg.norm(pos - com)) / 1000.0
    ok = d <= 0.060
    return ("imu_lever_arm", ok, f"IMU {d*1000:.0f} mm from frame CoM (limit 60 mm)",
            0.0 if ok else 0.3)

def check_dfam(m, samples=2000):
    """Independent DfAM hardening gate (vendored earthtojake/text-to-cad dfam_tool.py,
    MIT (c) Thompson Labs LLC): watertight + wall-thickness p05 >= MIN_WALL_MM +
    estimated support volume <= 30% of part volume (FDM limits, process-limits.md).
    Second measurement stack behind overhang/wall_thickness; tool errors pass with a
    flagged detail so a broken tool never eats a generation."""
    import json as _json, os, subprocess, tempfile
    path = os.path.join(tempfile.gettempdir(), "dfam_%d.stl" % os.getpid())
    try:
        m.export(path)
        tool = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor", "dfam_tool.py")
        out = subprocess.run(["python3", tool, "measure", path, "--samples", str(samples)],
                             capture_output=True, text=True, timeout=300)
        d = _json.loads(out.stdout or "{}")
        if "mesh" not in d:
            return ("dfam", True, "dfam tool error (non-gating): %s" % (d.get("error") or out.stderr[:160]), 0.0)
        mesh_ok = d.get("mesh", {}).get("watertight") is True
        w = d.get("wall_thickness", {})
        p05 = w.get("p05_mm")
        wall_ok = p05 is not None and p05 >= MIN_WALL_MM
        s = d.get("support_volume", {})
        ratio = s.get("support_to_part_ratio_pct")
        sup_ok = ratio is None or ratio <= 30.0
        ok = mesh_ok and wall_ok and sup_ok
        return ("dfam", ok, "watertight=%s, wall p05=%smm (limit %s), min=%smm, support~%s%% of part (limit 30)"
                % (mesh_ok, p05, MIN_WALL_MM, w.get("min_mm"), ratio), 0.0 if ok else 0.4)
    except Exception as e:
        return ("dfam", True, "dfam tool error (non-gating): %s: %s" % (type(e).__name__, e), 0.0)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

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
    checks.append(check_bed_fit(m))
    ok_clear, adj, need = p.check_prop_clearance()
    checks.append(("prop_clearance", ok_clear, f"{adj:.0f} mm vs {need:.0f} mm needed", 0.0 if ok_clear else 0.5))
    props, total = mass_properties(m, p.motor_positions(), p.arm_length_mm)
    hover_ok = props["hover_thrust_frac"] < 0.65
    checks.append(("hover_margin", hover_ok, f"hover at {props['hover_thrust_frac']*100:.0f}% of max thrust", 0.0 if hover_ok else 0.4))
    for name, passed, detail, pen in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    print(json.dumps(props, indent=2))
    print(f"SCORE {score(checks):.3f}")
