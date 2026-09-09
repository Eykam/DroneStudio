"""Rewrite the 8 tof_* origins in v97-g96a.poses.json: march each lens apex
along its optical axis past the carrier's own solid (last inside point + 0.1mm),
so raycasts start at the true emitter plane of the built assembly."""
import json, math, sys
sys.path.insert(0, "/work/DroneStudio/chassis")
import numpy as np
import components, build123d as b

POSES = "/work/DroneStudio/chassis/coverage/v97-g96a.poses.json"
TOF_LENS_Z_M = 0.0034
pose = json.load(open(POSES))
pl = components.placement()

def ch2glb(p):
    x, y, z = p
    return [x, z, -y]

for s in pose["sensors"]:
    if not s["id"].startswith("tof_"):
        continue
    bearing = s["id"][4:]
    key = "vl53l9cx_breakout#" + bearing
    x, y, z = pl[key]
    r = math.hypot(x, y)
    axc = np.array([x/r, y/r, 0.0])
    apex = np.array([x*1000, y*1000, (z + TOF_LENS_Z_M)*1000])
    shape = components.component_shape(key, pl[key])
    last_inside = None
    t = 0.0
    while t <= 20.0:
        if shape.is_inside(b.Vertex(*(apex + t*axc))):
            last_inside = t
        t += 0.1
    old = s["origin"]
    fo = (last_inside + 0.1)/1000.0 if last_inside is not None else 0.0
    s["origin"] = ch2glb((x + fo*x/r, y + fo*y/r, z + TOF_LENS_Z_M))
    print(f"{s['id']}: last-inside {last_inside}mm -> radial offset {fo*1000:.2f}mm; origin {old} -> {[round(v,6) for v in s['origin']]}")
json.dump(pose, open(POSES, "w"))
print("poses.json updated")
