"""Component library: REAL parts with sourced masses/dims.

Each entry: datasheet/mfr-grounded mass + bounding dims, source URL.
Where a manufacturer STEP is unavailable (most FPV parts), the model is a
spec-exact primitive (cylinder/box) with REAL mass - inertia is box/cylinder-
approx but mass and envelope are true. Import manufacturer STEP into
components/step/ and set step_path to upgrade a part to exact B-rep inertia.

Placement is an optimization surface: placement.json (cwd) overrides the
default x/y/z of any placeable component; the mutation loop may edit it
without touching chassis.py.
"""
from dataclasses import dataclass
import os, json

_STEP_CACHE = {}

def cad_geometry(key, pos_m):
    """Real B-rep for a component with step_path, placed at pos_m (meters).
    Returns (shape_in_mm, I_self_kgm2 about part CoM, com_m); (None,None,None) if no CAD."""
    c = LIBRARY[key.split("#")[0]]
    if not c.step_path:
        return None, None, None
    import build123d as b
    import numpy as np
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), c.step_path)
    if path not in _STEP_CACHE:
        brep = path + ".brep"
        if os.path.exists(brep):
            sh = b.import_brep(brep)
        else:
            sh = b.import_step(path)
            try:
                b.export_brep(sh, brep)
            except Exception:
                pass
        _STEP_CACHE[path] = sh
    src = _STEP_CACHE[path]
    bb = src.bounding_box()
    cx, cy = (bb.min.X + bb.max.X) / 2, (bb.min.Y + bb.max.Y) / 2
    sh = src.moved(b.Location((pos_m[0] * 1000 - cx, pos_m[1] * 1000 - cy,
                               pos_m[2] * 1000 - bb.min.Z)))
    m_kg = c.mass_g / 1000.0
    rho_eff = m_kg / sh.volume          # kg/mm3 (datasheet mass over real volume)
    I_mm5 = sh.matrix_of_inertia        # mm^5, about origin
    I_org = np.array([[I_mm5[i][j] for j in range(3)] for i in range(3)]) * rho_eff  # kg*mm^2
    ctr = sh.center()
    r = np.array([ctr.X, ctr.Y, ctr.Z])  # mm
    I_com = I_org - m_kg * ((r @ r) * np.eye(3) - np.outer(r, r))
    return sh, I_com * 1e-6, r * 0.001

def placed_cad_items():
    """{placement_key: (shape_mm, I_self_kgm2, com_m)} for CAD-backed parts."""
    out = {}
    for key, pos in placement().items():
        g = cad_geometry(key, pos)
        if g[0] is not None:
            out[key] = g
    return out

@dataclass
class Component:
    name: str
    mass_g: float
    dims_m: tuple          # bounding box (x, y, z), meters
    shape: str             # "box" | "cylinder-z" (for inertia approx)
    mount: str             # "motor_pad" | "stack" | "deck" | "nose"
    source: str            # datasheet/product URL or "estimate"
    step_path: str = None  # optional manufacturer STEP for exact B-rep

LIBRARY = {
    "motor": Component(
        "EMAX Eco II 2207", 33.4, (0.0275, 0.0275, 0.0332), "cylinder-z", "motor_pad",
        "https://www.mantisfpv.com.au/emax-eco-ii-series-2207-3-6s-1700-1900-2400kv/",
        step_path="parts/motor_2207.step"),  # GrabCAD generic 2207 (bbox 27.0x27.0x32.3mm, within 1.2mm of ECO II); no exact EMAX STEP exists - https://grabcad.com/library/2207-brushless-motor-1
    "prop": Component(
        "Gemfan Hurricane 51466 V2", 4.2, (0.1318, 0.1318, 0.0068), "cylinder-z", "motor_pad",
        "https://www.gemfanhobby.com/hurricane-51466-v2-pc-3-blade.html"),
    "fc_esc_stack": Component(
        "30.5mm FC+ESC stack (BLHELI_S, his hardware per repo assets/Drone/drone_esc_mount_30mm.stl)",
        90.0, (0.036, 0.036, 0.020), "box", "stack",
        "estimate - replace with his actual stack datasheet"),
    "battery": Component(
        "4S 1300mAh LiPo (Tattu-class)", 180.0, (0.075, 0.035, 0.035), "box", "deck",
        "user-confirmed 4S build 2026-09-04; capacity/mass still Tattu-class estimate until he names the battery model"),
    "pi_zero_2w": Component(
        "Raspberry Pi Zero 2W", 11.0, (0.065, 0.030, 0.005), "box", "nose",
        "https://www.raspberrypi.com/products/raspberry-pi-zero-2-w/",
        step_path="parts/pizero2w.step"),  # GrabCAD community w/ 40-pin header, bbox 65.8x30.6x11.5mm - https://grabcad.com/library/raspberry-pi-zero-2-w-with-40-pin-male-connector-1
    "pi_camera_3": Component(
        "Raspberry Pi Camera Module 3", 4.0, (0.025, 0.024, 0.012), "box", "nose",
        "https://pip-assets.raspberrypi.com/categories/1207-design-files/documents/RP-008154-DS-1-camera-module-3-step.zip",
        step_path="parts/Camera_module_3_std_model_simple.stp"),
    "gps": Component(
        "Beitian BN-220 GPS", 10.8, (0.022, 0.020, 0.007), "box", "deck",
        "https://grabcad.com/library/gps-beitian-bn-220-1 (bbox 22.0x20.0x6.9mm vs 22x20x6mm spec); mass from vendor listings, verify when he names his GPS",
        step_path="parts/gps_bn220.step"),
    "mpu9250": Component(
        "MPU-9250 breakout (GY-9250)", 3.0, (0.025, 0.015, 0.003), "box", "stack",
        "estimate - generic GY-9250 module"),
}

# default placements relative to frame origin (m); z=0 is arm-plate bottom.
# His physical layout: stereo pair 60mm apart at the nose (repo README),
# stack center, battery on deck above the stack.
DEFAULT_PLACEMENT = {
    "fc_esc_stack": [0.0, 0.0, 0.016],
    "battery": [0.0, 0.0, 0.045],
    "pi_zero_2w": [0.030, -0.030, 0.010],
    "pi_camera_3#left": [0.035, -0.030, 0.012],
    "pi_camera_3#right": [0.035, 0.030, 0.012],
    "mpu9250": [0.0, 0.0, 0.022],
    "gps": [-0.045, 0.0, 0.045],  # rear deck, typical FPV GPS perch
}

def placement():
    p = dict(DEFAULT_PLACEMENT)
    if os.path.exists("placement.json"):
        p.update(json.load(open("placement.json")))
    return p

def placed_items():
    """Manifest payload entries: real component per placement, props on motor pads."""
    pl = placement()
    out = []
    for key, pos in pl.items():
        cname = key.split("#")[0]
        c = LIBRARY[cname]
        out.append({"component": c.name, "mass_kg": round(c.mass_g / 1000, 6),
                    "position_m": [round(float(x), 6) for x in pos],
                    "size_m": list(c.dims_m), "shape": c.shape, "source": c.source,
                    "mount": c.mount})
    return out
