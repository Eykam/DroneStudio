"""FoV tab JSON per the dashboard contract (2026-09-08): one JSON per accepted
variant, ALL coordinates in the published GLB's frame (Y-up), per-sensor
frustum/cone ray grids (<=2000 rays), per-ray first-hit distances in mm
(null = clear to max range), datasheet-true FoV, versioned against the mesh.

Datasheet anchors:
- Pi Camera Module 3 (standard, IMX708): 66h x 41v deg (RP-008151-DS-1).
- VL53L9CX: 55h x 42v deg, range 0.05-8.8m (ST DS14879 Rev 7 Table 3, via sim sibling).
- Proposed pad-marker down camera: cone 65deg half-angle, 8m (sim sibling marker spec).
"""
import os, sys, json, math, hashlib
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
CAM_FOV = (66.0, 41.0); CAM_REF = "RP-008151-DS-1 Camera Module 3 product brief (standard IMX708)"
TOF_FOV = (55.0, 42.0); TOF_REF = "ST DS14879 Rev 7 Table 3 (VL53L9CX)"
TOF_LENS_Z_M = 0.0034  # lens = placement z + 3.4mm (components.py carrier note)
N_AZ, N_EL = 48, 36    # 1728 rays <= 2000 budget


def ch2glb(p):
    """chassis frame (Z-up, m) -> published GLB frame (Y-up)."""
    x, y, z = p
    return np.array([x, z, -y], float)


def quat_look(axis_glb):
    """Minimal-rotation quat (x,y,z,w) mapping canonical -Z look dir onto the optical axis."""
    a = np.array([0, 0, -1.0]); b = np.asarray(axis_glb, float); b /= np.linalg.norm(b)
    d = float(np.dot(a, b))
    if d < -0.999999:
        return [1.0, 0.0, 0.0, 0.0]
    v = np.cross(a, b); s = math.sqrt((1 + d) * 2)
    return [float(v[0]/s), float(v[1]/s), float(v[2]/s), float(s/2)]


def basis(axis):
    up_ref = np.array([0, 1, 0.0])
    if abs(float(np.dot(axis, up_ref))) > 0.999:
        up_ref = np.array([0, 0, 1.0])
    right = np.cross(axis, up_ref); right /= np.linalg.norm(right)
    upv = np.cross(right, axis); upv /= np.linalg.norm(upv)
    return right, upv


def frustum_dirs(axis, hfov, vfov):
    right, upv = basis(axis)
    dirs = []
    for el in np.linspace(-vfov/2, vfov/2, N_EL):
        for az in np.linspace(-hfov/2, hfov/2, N_AZ):
            d = axis + math.tan(math.radians(az))*right + math.tan(math.radians(el))*upv
            dirs.append(d/np.linalg.norm(d))
    return np.array(dirs)


def cone_dirs(axis, half_angle_deg):
    right, upv = basis(axis)
    dirs = []
    for r in np.linspace(0, half_angle_deg, N_EL):
        for t in np.linspace(0, 360, N_AZ, endpoint=False):
            rr, tt = math.radians(r), math.radians(t)
            d = axis*math.cos(rr) + (math.cos(tt)*right + math.sin(tt)*upv)*math.sin(rr)
            dirs.append(d/np.linalg.norm(d))
    return np.array(dirs)


def raycast(mesh, origin_glb, dirs, range_max_m):
    """Batch-vectorized Moller-Trumbore, first-hit per ray (B rays x N tris blocks).
    trimesh's rtree caster hangs on certain diagonal rays; this is deterministic
    and memory-flat."""
    tri = np.asarray(mesh.triangles, dtype=np.float64)
    v0 = tri[:, 0, :]; e1 = tri[:, 1, :] - v0; e2 = tri[:, 2, :] - v0
    o = np.asarray(origin_glb, float)
    nR = len(dirs)
    best = np.full(nR, np.inf)
    B, N = 128, 20000
    for r0 in range(0, nR, B):
        D = dirs[r0:r0+B]
        bmin = np.full(len(D), np.inf)
        for c0 in range(0, len(v0), N):
            V0 = v0[c0:c0+N]; E1 = e1[c0:c0+N]; E2 = e2[c0:c0+N]
            P = np.cross(D[:, None, :], E2[None, :, :])          # (b,n,3)
            det = (E1[None, :, :] * P).sum(-1)                    # (b,n)
            nz = np.abs(det) > 1e-12
            inv = np.where(nz, 1.0 / np.where(nz, det, 1.0), 0.0)
            TV = o[None, None, :] - V0[None, :, :]                # (1,n,3)
            TV = np.repeat(TV, len(D), axis=0)                    # (b,n,3)
            U = (TV * P).sum(-1) * inv
            Q = np.cross(TV, E1[None, :, :])
            V = (D[:, None, :] * Q).sum(-1) * inv
            T = (E2[None, :, :] * Q).sum(-1) * inv
            ok = nz & (U >= 0) & (V >= 0) & (U + V <= 1) & (T > 1e-9)
            bmin = np.minimum(bmin, np.where(ok, T, np.inf).min(axis=1))
        best[r0:r0+B] = bmin
    return [None if (not np.isfinite(d)) or d > range_max_m else round(d*1000, 1) for d in best]


def raycast_one(variant, sensor_id):
    """Single-sensor raycast in a fresh process (cgroup OOM workaround)."""
    pose = json.load(open(os.path.join(HERE, "coverage", variant + ".poses.json")))
    s = next(x for x in pose["sensors"] if x["id"] == sensor_id)
    s["origin"] = np.array(s["origin"], float); s["axis"] = np.array(s["axis"], float)
    ray_mesh = os.environ.get("FOV_RAY_MESH") or os.path.join(HERE, "snapshots", variant, "chassis.sim.glb")
    mesh = trimesh.load(ray_mesh, force="mesh", process=False)
    if s["fov"]["type"] == "frustum":
        dirs = frustum_dirs(s["axis"], s["fov"]["hfov_deg"], s["fov"]["vfov_deg"])
    else:
        dirs = cone_dirs(s["axis"], s["fov"]["half_angle_deg"])
    hits = raycast(mesh, s["origin"], dirs, s["range_m"]["max"])
    clear = sum(1 for h in hits if h is None)
    entry = {"id": s["id"], "label": s["label"],
             "mount": {"pos_mm": [round(v*1000, 2) for v in s["origin"]], "quat_xyzw": quat_look(s["axis"])},
             "fov": s["fov"], "range_m": s["range_m"],
             "coverage_fraction": round(clear/len(dirs), 4),
             "raycast": {"dirs": [[round(float(c), 6) for c in d] for d in dirs], "hit_mm": hits}}
    if s["proposed"]:
        entry["proposed"] = True
    part_path = os.path.join(HERE, "coverage", f"{variant}.{sensor_id}.json")
    json.dump(entry, open(part_path, "w"))
    print(f"{s['id']:12s} coverage {clear/len(dirs)*100:5.1f}%  rays {len(dirs)} -> {part_path}", flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


SENSOR_IDS = ["cam_left", "cam_right", "tof_e", "tof_n", "tof_ne", "tof_nw", "tof_s", "tof_se", "tof_sw", "tof_w", "cam_down"]


def merge(variant, outpath):
    pose = json.load(open(os.path.join(HERE, "coverage", variant + ".poses.json")))
    pl_hash = pose["placement_sha256"]
    out_sensors = [json.load(open(os.path.join(HERE, "coverage", f"{variant}.{sid}.json")))
                   for sid in SENSOR_IDS]
    doc = {"variant": variant, "units": "mm", "frame": "glb",
           "mesh": {"url": f"api/cad/designs/cad-chassis-{variant}",
                    "version": 0,
                    "published_glb_sha256": sha256_file(os.path.join(HERE, "snapshots", variant, "chassis.glb")),
                    "raycast_mesh": f"snapshots/{variant}/chassis.sim.glb",
                    "raycast_mesh_sha256": sha256_file(os.path.join(HERE, "snapshots", variant, "chassis.sim.glb")),
                    "note": "raycast ran against the published display mesh itself (same geometry the dashboard renders; gltfpack -noq decode of chassis.glb, 724787 faces) because the 14M-face unsimplified sim twin OOM-killed casters on this box. Published mesh includes component carrier solids; ToF origins are per-station empirical emitter points marched just past each carrier's own solid along the optical axis (cardinal ~+1.97mm radial, diagonals further - the rotated carrier sweeps past the nominal module-face note), camera origins at the real lens apex. url/version are dashboard-side values the tab owner should confirm"},
           "placement_sha256": pl_hash,
           "sensors": out_sensors}
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    json.dump(doc, open(outpath, "w"))
    print("->", outpath, f"({os.path.getsize(outpath)//1024} KB)", flush=True)


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "v97-g96a"
    mode = sys.argv[2] if len(sys.argv) > 2 else "--all"
    if mode == "--poses":
        sys.path.insert(0, HERE)
        pose = build_sensors()
        pose_path = os.path.join(HERE, "coverage", variant + ".poses.json")
        os.makedirs(os.path.dirname(pose_path), exist_ok=True)
        json.dump(pose, open(pose_path, "w"))
        print("poses ->", pose_path, flush=True)
    elif mode == "--one":
        raycast_one(variant, sys.argv[3])
    elif mode == "--merge":
        outpath = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "coverage", variant + ".fov.json")
        merge(variant, outpath)
    else:
        print("use --poses | --one <id> | --merge [outpath]", file=sys.stderr); sys.exit(2)
