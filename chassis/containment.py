"""Containment gate: every placed component must sit fully inside the frame
envelope with service clearance (user directive 2026-09-04: 'PCBs sticking
out of the sides is not good enough'). Protrusion = hard fail.
Motors/props are exempt by design (arm-mounted); cameras must pass too
(lens sits behind the nose shell - recess it, don't poke it out).
"""
import build123d as b
from components import LIBRARY, placement, cad_geometry

CLEAR_MM = 2.0  # service clearance around every component

def component_bbox_mm(key, pos):
    c = LIBRARY[key.split("#")[0]]
    sh, _, _ = cad_geometry(key, pos)
    if sh is not None:
        return c, sh.bounding_box()
    dx, dy, dz = (d * 1000 for d in c.dims_m)
    cx, cy, cz = pos[0] * 1000, pos[1] * 1000, pos[2] * 1000
    class BB: pass
    bb = BB()
    bb.min = b.Vector(cx - dx/2, cy - dy/2, cz)
    bb.max = b.Vector(cx + dx/2, cy + dy/2, cz + dz)
    return c, bb

def check_containment(part, clear_mm=CLEAR_MM):
    """(name, passed, detail, penalty) in the evaluate.py check contract."""
    fbb = part.bounding_box()
    worst, fails = 0.0, []
    embeds = []
    for key, pos in placement().items():
        c, bb = component_bbox_mm(key, pos)
        poke = max(fbb.min.X - (bb.min.X - clear_mm),
                   (bb.max.X + clear_mm) - fbb.max.X,
                   fbb.min.Y - (bb.min.Y - clear_mm),
                   (bb.max.Y + clear_mm) - fbb.max.Y,
                   fbb.min.Z - (bb.min.Z - clear_mm),
                   (bb.max.Z + clear_mm) - fbb.max.Z, 0.0)
        if poke > worst:
            worst = poke
        if poke > 0:
            fails.append("%s(%s)+%.1fmm" % (key, c.name.split("(")[0].strip()[:20], poke))
        # interpenetration: component must not be buried in frame material
        # (closing the z=0.002-into-the-floor exploit found 2026-09-04).
        # Sampled is_inside grid: a boolean intersect() against the full frame
        # solid costs minutes; point classification costs ms.
        c2, bb2 = component_bbox_mm(key, pos)
        nx = 5
        inside = total = 0
        for i in range(nx):
            for j in range(nx):
                for k in range(nx):
                    px = bb2.min.X + (bb2.max.X - bb2.min.X) * (i + 0.5) / nx
                    py = bb2.min.Y + (bb2.max.Y - bb2.min.Y) * (j + 0.5) / nx
                    pz = bb2.min.Z + (bb2.max.Z - bb2.min.Z) * (k + 0.5) / nx
                    total += 1
                    try:
                        if part.is_inside((px, py, pz)):
                            inside += 1
                    except Exception:
                        pass
        if total and inside / total > 0.10:  # >10% of sampled volume in frame solid
            embeds.append("%s(%d/%d pts buried)" % (key, inside, total))
    ok = not fails and not embeds
    if ok:
        detail = "all components inside frame envelope + %.0fmm clearance, none embedded" % clear_mm
    else:
        parts = (["PROTRUDE: " + ", ".join(fails)] if fails else []) + \
                (["EMBEDDED: " + ", ".join(embeds)] if embeds else [])
        detail = " | ".join(parts)
    return ("containment", ok, detail, 0.0 if ok else 0.6)
