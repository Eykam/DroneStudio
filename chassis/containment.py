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
    ok = not fails
    detail = "all components inside frame envelope + %.0fmm clearance" % clear_mm if ok \
             else "PROTRUDE: " + ", ".join(fails)
    return ("containment", ok, detail, 0.0 if ok else 0.6)
