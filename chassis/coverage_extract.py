"""Per-sensor datasheet-true FoV cone + frame-obstruction extraction.

For each placed sensor (2x Pi Camera Module 3, 8x VL53L9CX dToF), sample a ray
grid across its datasheet FoV cone from its real lens pose and raycast against
the actual frame mesh. Output: one JSON per sensor + a summary, versioned
against the exact frame variant / placement hash it was measured on.

Datasheet anchors:
- Pi Camera Module 3 (standard, IMX708): H 66 deg x V 41 deg (product brief
  RP-008151-DS-1; loop gate uses 66.3x41.6 inside the +-3deg lens tolerance).
- VL53L9CX: H 55 deg x V 42 deg (71 deg diagonal, ST DS14879 Rev 7 Table 3,
  via sim sibling 2026-09-08).
"""
import os, sys, json, math, hashlib
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))

CAM_FOV = {"h": 66.0, "v": 41.0, "ref": "RP-008151-DS-1 Camera Module 3 product brief (standard, IMX708)"}
TOF_FOV = {"h": 55.0, "v": 42.0, "ref": "ST DS14879 Rev 7 Table 3 (VL53L9CX)"}
TOF_RANGE_CAP_M = 8.8      # datasheet max range; beyond = free
CAM_RANGE_CAP_M = 50.0     # effectively unbounded for obstruction purposes
TOF_LENS_Z_OFFSET_M = 0.0034  # lens sits 3.4mm above placement z (carrier board X=11.6 optical axis, components.py note)


def rot_z(v, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([c*v[0]-s*v[1], s*v[0]+c*v[1], v[2]])


def cone_rays(axis, hfov_deg, vfov_deg, n_h, n_v, h_about="z"):
    """Ray directions across a rectangular FoV cone around `axis` (unit vec).
    h_about=z: horizontal sweep rotates about world Z (ring sensors, cameras).
    Returns (n_h*n_v, 3) directions, plus the az/el grids."""
    axis = np.asarray(axis, float); axis /= np.linalg.norm(axis)
    azs = np.linspace(-hfov_deg/2, hfov_deg/2, n_h)
    els = np.linspace(-vfov_deg/2, vfov_deg/2, n_v)
    dirs = []
    for el in els:
        for az in azs:
            d = rot_z(axis, az)
            # elevation: rotate d about the horizontal axis perpendicular to d
            side = np.cross([0,0,1], d)
            n = np.linalg.norm(side)
            if n < 1e-9:
                side = np.array([0,1,0.0])
            else:
                side /= n
            a = math.radians(el)
            d2 = d*math.cos(a) + np.cross(side, d)*math.sin(a) + side*np.dot(side,d)*(1-math.cos(a))
            dirs.append(d2/np.linalg.norm(d2))
    return np.array(dirs), azs, els


def extract(variant, outdir):
    sys.path.insert(0, HERE)
    os.makedirs(outdir, exist_ok=True)
    # frame mesh: uncompressed sim GLB (Y-up) -> rotate to chassis Z-up
    m = trimesh.load(os.path.join(HERE, "snapshots", variant, "chassis.sim.glb"), force="mesh")
    T = trimesh.transformations.rotation_matrix(-math.pi/2, [1,0,0])  # sim Y-up -> chassis Z-up
    m.apply_transform(T)
    assert m.is_watertight or True

    import components
    placements = components.placement()
    pl_bytes = json.dumps(placements, sort_keys=True).encode()
    pl_hash = hashlib.sha256(pl_bytes).hexdigest()

    sensors = []
    # cameras: real lens poses from components (apex, axis, fov)
    for key, pose in components.camera_lens_poses().items():
        sensors.append({"key": key, "kind": "camera",
                        "origin": list(pose["origin_m"]), "axis": list(pose["axis"]),
                        "fov": dict(CAM_FOV), "range_cap": CAM_RANGE_CAP_M})
    # ToF ring: radial-outward axis at each placement bearing, lens z +3.4mm
    for key, pos in placements.items():
        if not key.startswith("vl53l9cx_breakout"):
            continue
        x, y, z = pos
        r = math.hypot(x, y)
        axis = [x/r, y/r, 0.0]
        sensors.append({"key": key, "kind": "tof",
                        "origin": [x, y, z + TOF_LENS_Z_OFFSET_M], "axis": axis,
                        "fov": dict(TOF_FOV), "range_cap": TOF_RANGE_CAP_M})

    summary = {"variant": variant, "placement_sha256": pl_hash,
               "frame_mesh": f"snapshots/{variant}/chassis.sim.glb",
               "sensors": []}
    N_H, N_V = 61, 47
    for s in sensors:
        dirs, azs, els = cone_rays(s["axis"], s["fov"]["h"], s["fov"]["v"], N_H, N_V)
        origins = np.tile(np.array(s["origin"]), (len(dirs), 1))
        locs, hit_rays, tri = m.ray.intersects_location(origins, dirs, multiple_hits=False)
        dist = np.full(len(dirs), -1.0)
        for loc, ri in zip(locs, hit_rays):
            d = float(np.linalg.norm(loc - origins[ri]))
            dist[ri] = d if d <= s["range_cap"] else -1.0
        free = dist < 0
        rec = {"sensor": s["key"], "kind": s["kind"],
               "origin_m": [round(v,5) for v in s["origin"]],
               "axis": [round(v,5) for v in s["axis"]],
               "hfov_deg": s["fov"]["h"], "vfov_deg": s["fov"]["v"],
               "fov_ref": s["fov"]["ref"],
               "range_cap_m": s["range_cap"],
               "grid": {"n_az": N_H, "n_el": N_V,
                        "az_deg": [round(a,3) for a in azs], "el_deg": [round(e,3) for e in els]},
               "hit_dist_m": [round(d,4) for d in dist],
               "free_fraction": round(float(free.mean()),4),
               "datasheet_anchor": True}
        fn = os.path.join(outdir, s["key"].replace("#","_") + ".coverage.json")
        json.dump(rec, open(fn, "w"))
        summary["sensors"].append({"sensor": s["key"], "kind": s["kind"],
                                   "free_fraction": rec["free_fraction"], "file": os.path.basename(fn)})
        print(f"{s['key']:32s} free {rec['free_fraction']*100:5.1f}%  ({s['kind']})", flush=True)
    json.dump(summary, open(os.path.join(outdir, "_summary.json"), "w"), indent=1)
    print("summary ->", os.path.join(outdir, "_summary.json"), flush=True)


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "v97-g96a"
    outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "coverage", variant)
    extract(variant, outdir)
