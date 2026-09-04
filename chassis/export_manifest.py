"""Export chassis to GLB (meters, sim-ready) + physics manifest JSON.

Manifest is the contract DroneStudio's chassis-branch loader reads instead of
the hardcoded values in Studio/src/core/ecs/prefabs/Drone.zig. Field names
follow MJCF/URDF prior art (inertial, collision, joint/motor sites).
"""
import json, math, sys
import build123d as b
from chassis import ChassisParams, build_chassis

RHO_PETG = 1240e-9  # kg/mm^3
MOTOR = {"max_thrust_n": 10.0, "time_constant_s": 0.04, "drag_ratio": 0.15}  # sim FlightController defaults
MOTOR_DIRS = ["cw", "ccw", "cw", "ccw"]  # sim quad-X order M1..M4
MOTOR_MASS_KG = 0.032

def export(p: ChassisParams, out_base: str):
    part = build_chassis(p)
    rho = RHO_PETG
    vol_mm3 = part.volume
    frame_mass = vol_mm3 * rho
    com_mm = part.center()
    I_mm5 = part.matrix_of_inertia  # geometric, mm^5
    I = [[I_mm5[i][j] * rho * 1e-6 for j in range(3)] for i in range(3)]  # kg*m^2 (frame only)
    # build123d converts the declared unit (MM) to glTF-native meters on write
    b.export_gltf(part, out_base + ".glb", binary=True)
    b.export_step(part, out_base + ".step")  # mm, for printing/CAD interchange
    z_top_m = p.body_thickness_mm * 0.001
    motors = []
    for i, (mx, my) in enumerate(p.motor_positions()):
        motors.append({
            "id": i + 1,
            "position_m": [round(mx*0.001, 6), round(my*0.001, 6), round(z_top_m, 6)],
            "axis": [0, 0, 1],
            "direction": MOTOR_DIRS[i],
            "mass_kg": MOTOR_MASS_KG,
            **MOTOR,
        })
    # rigid-body totals the sim should use: frame + point-mass motors at mounts
    a2 = (p.arm_length_mm/math.sqrt(2)*0.001)**2  # squared |x|=|y| of each motor, m^2
    Ixx_mot = 4 * MOTOR_MASS_KG * a2       # sum m*y^2 over 4 motors
    Izz_mot = 4 * MOTOR_MASS_KG * 2 * a2   # sum m*(x^2+y^2)
    manifest = {
        "schema": "dronestudio.chassis/1",
        "name": out_base.split("/")[-1],
        "geometry": {"file": out_base.split("/")[-1] + ".glb", "units": "meters", "forward": "+X", "up": "+Z"},
        "material": {"name": "PETG", "density_kg_m3": 1240},
        "inertial": {
            "frame_mass_kg": round(frame_mass, 6),
            "frame_com_m": [round(c*0.001, 6) for c in com_mm],
            "frame_inertia_kgm2": {"ixx": I[0][0], "iyy": I[1][1], "izz": I[2][2],
                                    "ixy": I[0][1], "ixz": I[0][2], "iyz": I[1][2]},
            "motor_inertia_add_kgm2": {"ixx": Ixx_mot, "iyy": Ixx_mot, "izz": Izz_mot},
            "note": "sim should compose frame_inertia + motor_inertia_add + payload (battery/stack/cameras) point masses",
        },
        "collision": {"type": "convex_hull", "fallback": "vhacd", "max_hulls": 8},
        "motors": motors,
        "imu": {"position_m": [0, 0, round(z_top_m, 6)]},
        "stack": {"pattern_mm": p.stack_spacing_mm, "hole_dia_mm": p.stack_hole_dia_mm, "z_bottom_m": round(z_top_m, 6)},
    }
    with open(out_base + ".manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest

if __name__ == "__main__":
    p = ChassisParams()
    m = export(p, "/home/sandbox/cad-researcher/chassis_v1")
    print(json.dumps({k: m[k] for k in ("inertial", "motors")}, indent=2)[:1200])
    import os
    print("glb bytes:", os.path.getsize("/home/sandbox/cad-researcher/chassis_v1.glb"))
