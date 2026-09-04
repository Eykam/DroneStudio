"""Export chassis to GLB (meters, sim-ready) + physics manifest JSON.

Manifest dronestudio.chassis/1.1 - the contract DroneStudio's chassis-branch
loader reads instead of hardcoded values in prefabs/Drone.zig.
v1.1 adds: dynamics (composed rigid body incl. payload, inertia about CoM)
and aero (projected areas for drag), so the sim needs no derivation of its own.
"""
import json, math
import numpy as np
import build123d as b
from chassis import ChassisParams, build_chassis

RHO_PETG = 1240e-9  # kg/mm^3
# Real motor params: AKK RS2205 2300KV on 4S (user named the motor 2026-09-04).
# Max thrust 11.0N bench peak on 5045-class props (oscarliang/EMAX thrust stand); fitted 9.7N full-throttle equilibrium.
# drag_ratio 0.015m + motor mass 28.8g from sim-side motor sysid 2026-09-04 (AKK RS2205 2300KV).
MOTOR = {"max_thrust_n": 11.0, "time_constant_s": 0.04, "drag_ratio": 0.015,
         "kv": 2300, "cells": 4, "source": "AKK RS2205 2300KV (user motor, https://a.co/d/066TbARJ); EMAX RS2205 clone - bench 989-1155g on 5045-class, 65mOhm, 4S"}
MOTOR_DIRS = ["cw", "ccw", "cw", "ccw"]  # sim quad-X order M1..M4
from components import LIBRARY, placed_items, placed_cad_items
MOTOR_MASS_KG = LIBRARY["motor"].mass_g / 1000.0  # 28.8 g AKK RS2205 2300KV (sysid-measured)
PROP_DIA_M = 0.127  # 5 inch

def _solid_inertia_box(m, sx, sy, sz):
    return np.diag([m*(sy*sy+sz*sz)/12, m*(sx*sx+sz*sz)/12, m*(sx*sx+sy*sy)/12])

def compose_dynamics(part, p: ChassisParams):
    rho = RHO_PETG
    m_f = part.volume * rho
    c = part.center(); com_f = np.array([c.X, c.Y, c.Z]) * 0.001
    I_f_org = np.array([list(row) for row in part.matrix_of_inertia], dtype=float) * rho * 1e-6  # kg*m^2 about origin
    r = com_f
    I_f_com = I_f_org - m_f * ((r @ r) * np.eye(3) - np.outer(r, r))  # about frame CoM
    items = [{"name": "frame", "mass_kg": m_f, "com": com_f, "I_self": I_f_com}]
    for (mx, my) in p.motor_positions():
        pos = np.array([mx*0.001, my*0.001, p.body_thickness_mm*0.001])
        items.append({"name": "motor", "mass_kg": MOTOR_MASS_KG, "com": pos,
                      "I_self": _solid_inertia_box(MOTOR_MASS_KG, 0.028, 0.028, 0.030)})
    prop = LIBRARY["prop"]
    r_p, h_p = prop.dims_m[0] / 2, prop.dims_m[2]
    m_p = prop.mass_g / 1000.0
    I_prop = np.diag([m_p*(3*r_p*r_p+h_p*h_p)/12]*2 + [m_p*r_p*r_p/2])
    z_prop = (p.body_thickness_mm + 34.0) * 0.001
    for (mx, my) in p.motor_positions():
        items.append({"name": "prop", "mass_kg": m_p, "com": np.array([mx*0.001, my*0.001, z_prop]),
                      "I_self": I_prop})
    cad_items = placed_cad_items()
    for it in placed_items():
        if it["shape"] == "cylinder-z":
            r_, h_ = it["size_m"][0] / 2, it["size_m"][2]
            I_self = np.diag([it["mass_kg"]*(3*r_*r_+h_*h_)/12]*2 + [it["mass_kg"]*r_*r_/2])
        else:
            I_self = _solid_inertia_box(it["mass_kg"], *it["size_m"])
        items.append({"name": it["component"], "mass_kg": it["mass_kg"], "com": np.array(it["position_m"]),
                      "I_self": I_self, "source": it.get("source"), "mount": it.get("mount")})
    name_to_cad = {}
    for k, (sh, I_self, com_m) in cad_items.items():
        name_to_cad.setdefault(LIBRARY[k.split("#")[0]].name, []).append((I_self, com_m))
    for i in items:
        lst = name_to_cad.get(i.get("name"))
        if lst:
            # match the placed instance nearest to the item's com
            best = min(lst, key=lambda t: sum((t[1][j]-i["com"][j])**2 for j in range(3)))
            i["I_self"] = best[0]
    M = sum(i["mass_kg"] for i in items)
    com = sum(i["mass_kg"]*i["com"] for i in items) / M
    I = np.zeros((3, 3))
    for i in items:
        d = i["com"] - com
        I += i["I_self"] + i["mass_kg"] * ((d @ d) * np.eye(3) - np.outer(d, d))
    return M, com, I, items

def aero_areas(out_stl):
    """Projected areas (m^2) facing +/- each axis, from the exported mesh."""
    import trimesh
    m = trimesh.load(out_stl, force='mesh')
    v = m.vertices * 0.001
    tri = m.triangles * 0.001
    n = m.face_normals
    a = m.area_faces * 1e-6
    areas = {}
    for i, ax in enumerate("xyz"):
        proj = np.abs(n[:, i]) * a
        areas[ax] = round(float(proj.sum()) / 2, 6)  # /2: front+back faces both counted
    return areas

def export(p: ChassisParams, out_base: str):
    part = build_chassis(p)
    b.export_stl(part, out_base + ".stl")
    rho = RHO_PETG
    vol_mm3 = part.volume
    frame_mass = vol_mm3 * rho
    com_mm = part.center()
    I_mm5 = part.matrix_of_inertia
    I = [[I_mm5[i][j] * rho * 1e-6 for j in range(3)] for i in range(3)]
    # STEP before GLB: the gltf writer on a big Compound leaves OCC unable to
    # write STEP afterwards (export_step "Failed to write STEP file").
    b.export_step(part, out_base + ".step")
    from components import assembly_shapes, prop_shapes, PROP_PLANE_ABOVE_MOTOR_MOUNT_MM
    cad_shapes = assembly_shapes()  # every placed component, CAD or primitive
    # motors: 4x CAD at parametric motor positions (not in placement map)
    if LIBRARY["motor"].step_path:
        from components import cad_geometry
        for i, (mx, my) in enumerate(p.motor_positions()):
            g, _, _ = cad_geometry("motor#%d" % i, [mx * 0.001, my * 0.001, p.body_thickness_mm * 0.001])
            if g is not None:
                cad_shapes.append(g)
    # rotor discs + hubs at the prop plane
    cad_shapes += prop_shapes(p.motor_positions(), p.body_thickness_mm)
    if cad_shapes:
        b.export_gltf(b.Compound(children=[part] + cad_shapes), out_base + ".glb", binary=True, linear_deflection=0.1, angular_deflection=0.5)
    else:
        b.export_gltf(part, out_base + ".glb", binary=True)  # mm -> m on write
    _slim_glb(out_base + ".glb")


def _slim_glb(glb_path):
    """Post-process the assembly GLB for the /cad viewer: build123d writes one
    primitive PER FACE (50k draw calls -> orbit lag). gltfpack welds, merges
    primitives, simplifies (error-bounded), quantizes (KHR_mesh_quantization,
    natively supported by three.js/babylon - no decoder needed). No meshopt
    compression (-cc): dashboard drei useGLTF wires MeshoptDecoder by default (verified 2026-09-04)."""
    import shutil as _sh, subprocess as _sp, os as _os
    gp = _sh.which("gltfpack")
    if not gp:
        return
    tmp = glb_path + ".slim"
    try:
        r = _sp.run([gp, "-i", glb_path, "-o", tmp, "-si", "0.7", "-kn", "-cc"],
                    capture_output=True, timeout=300)
        if r.returncode == 0 and _os.path.exists(tmp) and _os.path.getsize(tmp) > 1000:
            _os.replace(tmp, glb_path)
        elif _os.path.exists(tmp):
            _os.remove(tmp)
    except Exception:
        pass  # raw GLB stays if slimming fails
    M, com, I_tot, items = compose_dynamics(part, p)
    z_top_m = p.body_thickness_mm * 0.001
    motors = []
    for i, (mx, my) in enumerate(p.motor_positions()):
        motors.append({
            "id": i + 1,
            "position_m": [round(mx*0.001, 6), round(my*0.001, 6), round(z_top_m, 6)],
            "axis": [0, 0, 1],
            "direction": MOTOR_DIRS[i],
            "mass_kg": MOTOR_MASS_KG,
            "prop_diameter_m": PROP_DIA_M,
            **MOTOR,
        })
    manifest = {
        "schema": "dronestudio.chassis/1.1",
        "name": out_base.split("/")[-1],
        "geometry": {"file": out_base.split("/")[-1] + ".glb", "units": "meters", "forward": "+X", "up": "+Z"},
        "material": {"name": "PETG", "density_kg_m3": 1240, "e_mpa": 2100, "yield_mpa": 50},
        "dynamics": {
            "total_mass_kg": round(M, 6),
            "com_m": [round(float(x), 6) for x in com],
            "inertia_about_com_kgm2": {"ixx": float(I_tot[0][0]), "iyy": float(I_tot[1][1]), "izz": float(I_tot[2][2]),
                                        "ixy": float(I_tot[0][1]), "ixz": float(I_tot[0][2]), "iyz": float(I_tot[1][2])},
            "composition": [{"name": i["name"], "mass_kg": round(i["mass_kg"], 6),
                             "com_m": [round(float(x), 6) for x in i["com"]],
                             **({"source": i["source"], "mount": i["mount"]} if i.get("source") else {})} for i in items],
            "note": "fully composed rigid body: frame + 4 motors + payload (real components, see chassis/components.py). Sim should use these directly.",
        },
        "inertial": {  # frame-only, kept for reference/debug
            "frame_mass_kg": round(frame_mass, 6),
            "frame_com_m": [round(c*0.001, 6) for c in com_mm],
            "frame_inertia_kgm2_about_origin": {"ixx": I[0][0], "iyy": I[1][1], "izz": I[2][2],
                                                 "ixy": I[0][1], "ixz": I[0][2], "iyz": I[1][2]},
        },
        "aero": {
            "projected_area_m2": aero_areas(out_base + ".stl"),
            "cd_flat_plate_estimate": 1.1,
            "note": "linear drag ~ 0.5*rho_air*cd*area*v^2 per axis; rho_air=1.225",
        },
        "collision": {"type": "convex_hull", "fallback": "vhacd", "max_hulls": 8},
        "motors": motors,
        "imu": {"position_m": [0, 0, round(z_top_m, 6)], "note": "IMU sits on the FC stack"},
        "stack": {"pattern_mm": p.stack_spacing_mm, "hole_dia_mm": p.stack_hole_dia_mm, "z_bottom_m": round(z_top_m, 6)},
    }
    with open(out_base + ".manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest

if __name__ == "__main__":
    p = ChassisParams()
    m = export(p, "/home/sandbox/cad-researcher/chassis_v1")
    d = m["dynamics"]
    print("total_mass_kg", d["total_mass_kg"], "com", d["com_m"])
    print("inertia_about_com", d["inertia_about_com_kgm2"])
    print("aero", m["aero"]["projected_area_m2"])
