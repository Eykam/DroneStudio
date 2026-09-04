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
        "https://www.mantisfpv.com.au/emax-eco-ii-series-2207-3-6s-1700-1900-2400kv/"),
    "prop": Component(
        "Gemfan Hurricane 51466 V2", 4.2, (0.1318, 0.1318, 0.0068), "cylinder-z", "motor_pad",
        "https://www.gemfanhobby.com/hurricane-51466-v2-pc-3-blade.html"),
    "fc_esc_stack": Component(
        "30.5mm FC+ESC stack (BLHELI_S, his hardware per repo assets/Drone/drone_esc_mount_30mm.stl)",
        90.0, (0.036, 0.036, 0.020), "box", "stack",
        "estimate - replace with his actual stack datasheet"),
    "battery": Component(
        "4S 1300mAh LiPo (Tattu-class)", 180.0, (0.075, 0.035, 0.035), "box", "deck",
        "estimate - replace with his actual battery datasheet"),
    "pi_zero_2w": Component(
        "Raspberry Pi Zero 2W", 11.0, (0.065, 0.030, 0.005), "box", "nose",
        "https://www.raspberrypi.com/products/raspberry-pi-zero-2-w/"),
    "pi_camera_3": Component(
        "Raspberry Pi Camera Module 3", 4.0, (0.025, 0.024, 0.012), "box", "nose",
        "https://www.raspberrypi.com/products/camera-module-3/"),
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
