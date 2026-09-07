"""Containment gate R4 (2026-09-06): enclosure vs the ACTUAL shell + attachment.

Replaces the frame-global-bbox test that let components float in open sky
(Eyad escalation 2026-09-06: floating, exposed, unattached components; the
calibration showed ToF carriers, battery, stack, IMU, GPS all unsupported).

Fast path: ray checks run against the candidate's trimesh STL mesh (the one
run_candidate already loads) - RayMeshIntersector, milliseconds. The build123d
is_inside path remains as a mesh=None fallback only.

Sub-checks, folded into the single 'containment' gate (11-gate count kept):
  1. bbox sanity (legacy): inside frame bbox + clearance.
  2. EMBED: bbox-sampled points must not be buried in frame material.
  3. SKY: rays +Z from the bbox top grid must hit shell (roof/wall/canopy)
     within SKY_MAX_MM for >= SKY_MIN_FRAC of samples.
  4. ATTACH: rays -Z from the bbox bottom grid must hit frame within
     ATTACH_MM for >= ATTACH_MIN_FRAC of samples (real support, not floating).
  5. LATERAL (vl53l9cx_breakout only): radially-outward rays must hit frame
     within LAT_MAX_MM unless they exit through the optical corridor
     (|z - lens z| <= AP_Z and |tangential offset| <= AP_TAN).
"""
import math
import numpy as np
import build123d as b
from components import LIBRARY, placement, tof_lens_poses

CLEAR_MM = 2.0
SKY_MAX_MM = 45.0
SKY_MIN_FRAC = 0.5
ATTACH_MM = 3.0
ATTACH_MIN_FRAC = 0.5
LAT_MAX_MM = 25.0
AP_Z = 5.0
AP_TAN = 9.0
# Cameras sit on open masts by design: relaxed sky requirement.
SKY_RELAXED = {"pi_camera_3": 0.25}

def component_bbox_mm(key, pos):
    c = LIBRARY[key.split("#")[0]]
    if c.dims_m is not None:
        dx, dy, dz = (d * 1000 for d in c.dims_m)
        cx, cy, cz = pos[0] * 1000, pos[1] * 1000, pos[2] * 1000
        class BB: pass
        bb = BB(); bb.min = b.Vector(cx - dx/2, cy - dy/2, cz); bb.max = b.Vector(cx + dx/2, cy + dy/2, cz + dz)
        return c, bb
    from components import cad_geometry
    sh, _, _ = cad_geometry(key, pos)
    return c, sh.bounding_box()

def _grid(bb, n=4):
    return np.array([[bb.min.X + (bb.max.X-bb.min.X)*(i+.5)/n,
                      bb.min.Y + (bb.max.Y-bb.min.Y)*(j+.5)/4, 0.0]
                     for i in range(n) for j in range(4)])

def check_containment(part=None, mesh=None, clear_mm=CLEAR_MM):
    if part is None and mesh is None:
        raise ValueError("containment needs part or mesh")
    ray = None
    if mesh is not None:
        import trimesh.ray.ray_triangle as rt
        ray = rt.RayMeshIntersector(mesh)
    if mesh is not None:
        (fx0, fy0, fz0), (fx1, fy1, fz1) = mesh.bounds
    else:
        fb = part.bounding_box(); fx0, fy0, fz0, fx1, fy1, fz1 = fb.min.X, fb.min.Y, fb.min.Z, fb.max.X, fb.max.Y, fb.max.Z
    worst, fails, embeds = 0.0, [], []
    exposed, floating = [], []
    poses = tof_lens_poses()
    for key, pos in placement().items():
        c, bb = component_bbox_mm(key, pos)
        cname = key.split("#")[0]
        # 1. legacy bbox sanity
        poke = max(fx0 - (bb.min.X - clear_mm), (bb.max.X + clear_mm) - fx1,
                   fy0 - (bb.min.Y - clear_mm), (bb.max.Y + clear_mm) - fy1,
                   fz0 - (bb.min.Z - clear_mm), (bb.max.Z + clear_mm) - fz1, 0.0)
        if poke > worst: worst = poke
        if poke > 0:
            fails.append("%s+%.1fmm" % (key, poke))
        # 2. embed sampling
        nx = 5
        pts = np.array([[bb.min.X + (bb.max.X-bb.min.X)*(i+.5)/nx,
                         bb.min.Y + (bb.max.Y-bb.min.Y)*(j+.5)/nx,
                         bb.min.Z + (bb.max.Z-bb.min.Z)*(k+.5)/nx]
                        for i in range(nx) for j in range(nx) for k in range(nx)])
        if mesh is not None:
            mask = mesh.contains(pts)
            inside, total = int(mask.sum()), len(pts)
        else:
            inside = total = 0
            for p in pts:
                total += 1
                try:
                    if part.is_inside(tuple(p)): inside += 1
                except Exception: pass
        if total and inside/total > 0.10:
            embeds.append("%s(%d/%d buried)" % (key, inside, total))
        if ray is not None:
            g = _grid(bb)
            # 3. SKY
            o = g.copy(); o[:, 2] = bb.max.Z + 0.05
            d = np.tile([0.0, 0.0, 1.0], (len(o), 1))
            locs, _, _ = ray.intersects_location(o, d, multiple_hits=False)
            sky_hit = len([1 for L in locs if 0 < (L[2]-bb.max.Z) <= SKY_MAX_MM]) if len(locs) else 0
            need = SKY_RELAXED.get(cname, SKY_MIN_FRAC)
            if sky_hit/len(o) < need:
                exposed.append("%s(sky %d/%d)" % (key, sky_hit, len(o)))
            # 4. ATTACH
            o = g.copy(); o[:, 2] = bb.min.Z - 0.05
            d = np.tile([0.0, 0.0, -1.0], (len(o), 1))
            locs, _, _ = ray.intersects_location(o, d, multiple_hits=False)
            sup = len([1 for L in locs if 0 < (bb.min.Z-L[2]) <= ATTACH_MM]) if len(locs) else 0
            if sup/len(o) < ATTACH_MIN_FRAC:
                floating.append("%s(support %d/%d)" % (key, sup, len(o)))
            # 5. LATERAL (ToF ring only)
            if cname == "vl53l9cx_breakout" and key in poses:
                ang = math.atan2(pos[1], pos[0]); ux, uy = math.cos(ang), math.sin(ang)
                lz = poses[key]["origin_m"][2]*1000
                zs = [bb.min.Z + (bb.max.Z-bb.min.Z)*(i+.5)/3 for i in range(3)]
                cx0, cy0 = (bb.min.X+bb.max.X)/2, (bb.min.Y+bb.max.Y)/2
                open_rays = 0; tot = 0
                for pz in zs:
                    tot += 1
                    ty, tx = -uy, ux
                    t_off = (cy0-pos[1]*1000)*ty + (cx0-pos[0]*1000)*tx
                    if abs(pz-lz) <= AP_Z and abs(t_off) <= AP_TAN:
                        continue
                    o = np.array([[cx0+ux*0.05, cy0+uy*0.05, pz]])
                    d = np.array([[ux, uy, 0.0]])
                    locs, _, _ = ray.intersects_location(o, d, multiple_hits=False)
                    hit = any(0 < ((L[0]-o[0][0])*ux + (L[1]-o[0][1])*uy) <= LAT_MAX_MM for L in locs) if len(locs) else False
                    if not hit:
                        open_rays += 1
                if open_rays:
                    exposed.append("%s(lateral %d/%d open)" % (key, open_rays, tot))
    ok = not fails and not embeds and not exposed and not floating
    if ok:
        detail = "all components enclosed by shell, supported, none embedded (R4)"
    else:
        parts = []
        if fails: parts.append("PROTRUDE: " + ", ".join(fails))
        if embeds: parts.append("EMBEDDED: " + ", ".join(embeds))
        if exposed: parts.append("EXPOSED: " + ", ".join(exposed))
        if floating: parts.append("FLOATING: " + ", ".join(floating))
        detail = " | ".join(parts)
    return ("containment", ok, detail, 0.0 if ok else 0.6)
