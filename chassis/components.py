import math
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
    src = _apply_orientation(key.split("#")[0], _STEP_CACHE[path])
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


def primitive_geometry(key, pos_m):
    """Primitive solid for a component WITHOUT manufacturer CAD (battery, stack,
    mpu9250) so it appears in the assembly GLB and containment checks."""
    c = LIBRARY[key.split("#")[0]]
    if c.step_path:
        return None
    import build123d as b
    dx, dy, dz = (d * 1000 for d in c.dims_m)
    cx, cy, z0 = pos_m[0] * 1000, pos_m[1] * 1000, pos_m[2] * 1000
    if c.shape == "cylinder-z":
        sh = b.Cylinder(radius=dx / 2, height=dz)
    else:
        sh = b.Box(dx, dy, dz)
    # Box/Cylinder are centered at origin; placement z = component bottom
    sh = _apply_orientation(key.split("#")[0], sh)
    return sh.moved(b.Location((cx, cy, z0 + dz / 2)))


def component_shape(key, pos):
    """CAD shape when available, else primitive."""
    sh, _, _ = cad_geometry(key, pos)
    return sh if sh is not None else primitive_geometry(key, pos)


def assembly_shapes():
    """One shape per placed component (CAD or primitive) for the assembly GLB."""
    out = []
    for key, pos in placement().items():
        sh = component_shape(key, pos)
        if sh is not None:
            out.append(sh)
    return out


PROP_PLANE_ABOVE_MOTOR_MOUNT_MM = 34.0  # motor bell top + shaft clearance

def prop_shapes(motor_positions_mm, motor_mount_z_mm):
    """Rotor disc + hub per motor pad, at the prop plane."""
    import build123d as b
    r = LIBRARY["prop"].dims_m[0] * 1000 / 2
    z = motor_mount_z_mm + PROP_PLANE_ABOVE_MOTOR_MOUNT_MM
    out = []
    for mx, my in motor_positions_mm:
        out.append(b.Cylinder(radius=4, height=6).moved(b.Location((mx, my, z - 3))))
        out.append(b.Cylinder(radius=r, height=1.5).moved(b.Location((mx, my, z + 4.5))))
    return out


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
        "AKK RS2205 2300KV", 28.8, (0.0275, 0.0275, 0.0332), "cylinder-z", "motor_pad",
        "https://www.mantisfpv.com.au/emax-eco-ii-series-2207-3-6s-1700-1900-2400kv/",
        step_path="parts/motor_2207.step"),  # GrabCAD generic 2207 (bbox 27.0x27.0x32.3mm, within 1.2mm of ECO II); no exact EMAX STEP exists - https://grabcad.com/library/2207-brushless-motor-1
    "prop": Component(
        "5x4.5 3-blade prop (5045-class, AKK RS2205 recommended)", 4.0, (0.127, 0.127, 0.006), "cylinder-z", "motor_pad",
        "5045-class per AKK RS2205 2300KV specs (user motor 2026-09-04); specific prop model TBD"),
    "fc_esc_stack": Component(
        "ee-flight single-board PCBA (CM4 + STM32F405 + 4x ESC + power tree)",
        31.73, (0.108, 0.052, 0.022), "box", "stack",
        "REAL board via pcba/1.0 contract: ee-flight v19 cand-097 (72 comps, SB1+SB2 populated, mass is spec-exact estimate; outline 108x52x1.635mm interim/unoptimized - SB6 targets stack-compatible shrink). GLB: parts/ee_flight_v19_pcba.glb. Pin ee-flight v19, re-pull on version bumps: https://dronestudio-dashboard-production.up.railway.app"),
    "battery": Component(
        "CNHL Black Series 1300mAh 14.8V 4S 100C (stock 1301004BK)", 163.0, (0.074, 0.034, 0.0335), "box", "deck",
        "user-named battery 2026-09-06. Manufacturer specs: 33.5x34x74mm, 163g incl. wire+connector, XT60 + JST-XH balance, 12AWG (https://chinahobbyline.com/products/cnhl-black-series-1300mah-14-8v-4s-100c-lipo-battery-with-xt60-plug; vendor tolerance +/-1-5mm, +/-5g). Replaces 180g Tattu-class estimate."),
    "pi_camera_3": Component(
        "Raspberry Pi Camera Module 3", 4.0, (0.025, 0.024, 0.012), "box", "nose",
        "https://pip-assets.raspberrypi.com/categories/1207-design-files/documents/RP-008154-DS-1-camera-module-3-step.zip",
        step_path="parts/Camera_module_3_std_model_simple.stp"),
    "gps": Component(
        "Beitian BN-220 GPS", 10.8, (0.022, 0.020, 0.007), "box", "deck",
        "https://grabcad.com/library/gps-beitian-bn-220-1 (bbox 22.0x20.0x6.9mm vs 22x20x6mm spec); mass from vendor listings, verify when he names his GPS",
        step_path="parts/gps_bn220.step"),
    "mpu9250": Component(
        "MPU-9250 breakout (GY-9250)", 0.0, (0.025, 0.015, 0.003), "box", "stack",
        "mass zeroed 2026-09-05: IMU is U6 on the ee-flight PCBA (0.03g, carried in fc_esc_stack mass); entry kept for pose/lever-arm gate"),
    "vl53l9cx_breakout": Component(
        "VL53L9CX dToF breakout", 2.0, (0.020, 0.016, 0.005), "box", "perimeter",
        "ST VL53L9CX module OPTICAL LGA 12.1x5.1x4.5mm (ST part table, EE-verified 2026-09-05) on assumed 20x16x5mm custom carrier (2x M2); STEVAL-VL53L9 carrier exists but 30-week lead - custom breakout path, STEVAL data brief as reference schematic; EE owns final carrier dims"),
}

ORIENTATIONS = {
    # Pi Camera Module 3 STEP is modeled board-flat with the LENS POINTING -Z
    # (lens-barrel solids sit at native z < -2mm, below the board face). To mount
    # lens-forward (+X, out through the nose aperture): rotate -90 deg about Y.
    # (+90 stands the board up but points the lens BACKWARD, -X; shipped that way
    # in d69c9de - user reported cameras still wrong 2026-09-04. Verified against
    # the actual STEP solid geometry: board at native z=-0.7..0, lens at z=-7.6..-3.)
    "pi_camera_3": ("Y", -90.0),
}

CAMERA_SPEC = {
    # Standard (non-wide) lens. Sources: raspberrypi.com/products/camera-module-3
    "pi_camera_3": {"hfov_deg": 66.3, "vfov_deg": 41.6},
}

def _apply_orientation(cname, sh):
    spec = ORIENTATIONS.get(cname)
    if not spec:
        return sh
    import build123d as b
    ax = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[spec[0]]
    return sh.rotate(b.Axis((0, 0, 0), ax), spec[1])  # returns a copy

# default placements relative to frame origin (m); z=0 is arm-plate bottom.
# His physical layout: stereo pair 60mm apart at the nose (repo README),
# stack center, battery on deck above the stack.
DEFAULT_PLACEMENT = {
    "fc_esc_stack": [0.0, 0.0, 0.016],
    "battery": [0.0, 0.0, 0.045],
    "pi_camera_3#left": [0.083, -0.028, 0.002],  # user directive 2026-09-05: +4mm to nose apertures (was 0.079)
    "pi_camera_3#right": [0.083, 0.028, 0.002],
    "mpu9250": [0.0, 0.0, 0.022],
    "gps": [-0.045, 0.0, 0.045],  # rear deck, typical FPV GPS perch
    # VL53L9CX 360-degree ring (user directive 2026-09-05): 8 breakouts at 45 deg
    # bearing spacing, recessed at the shell, sensor facing radially outward
    # (55x42 deg FoV -> ~10 deg overlaps). z = board bottom.
    "vl53l9cx_breakout#n":   [ 0.088,  0.000, 0.014],
    "vl53l9cx_breakout#ne":  [ 0.045,  0.045, 0.014],
    "vl53l9cx_breakout#e":   [ 0.000,  0.048, 0.014],
    "vl53l9cx_breakout#se":  [-0.045,  0.045, 0.014],
    "vl53l9cx_breakout#s":   [-0.130,  0.000, 0.014],
    "vl53l9cx_breakout#sw":  [-0.045, -0.045, 0.014],
    "vl53l9cx_breakout#w":   [ 0.000, -0.048, 0.014],
    "vl53l9cx_breakout#nw":  [ 0.045, -0.045, 0.014],
}

# User-directive fixed placements (2026-09-05): the exact 45-degree ToF ring
# bearings and the nose-hole camera positions (+4mm) are REQUIREMENTS, not
# optimization variables. placement.json (the codex mutation surface) may move
# anything else but may NOT override these keys - gen-46/47 drift (cameras back
# to 0.079, asymmetric carrier ring) showed the prompt alone does not hold them.
FIXED_PLACEMENT_KEYS = frozenset({
    "vl53l9cx_breakout#n", "vl53l9cx_breakout#ne", "vl53l9cx_breakout#e",
    "vl53l9cx_breakout#se", "vl53l9cx_breakout#s", "vl53l9cx_breakout#sw",
    "vl53l9cx_breakout#w", "vl53l9cx_breakout#nw",
    "pi_camera_3#left", "pi_camera_3#right",
})

def placement():
    p = dict(DEFAULT_PLACEMENT)
    if os.path.exists("placement.json"):
        overrides = json.load(open("placement.json"))
        for k in FIXED_PLACEMENT_KEYS:
            overrides.pop(k, None)
        p.update(overrides)
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



TOF_SPEC = {"vl53l9cx_breakout": {"hfov_deg": 55.0, "vfov_deg": 42.0}}

def tof_lens_poses():
    """{placement_key: {origin_m, axis, hfov_deg, vfov_deg}} for the VL53L9CX ring.
    Axis = radially outward from frame origin in XY at the placement bearing."""
    out = {}
    for key, pos in placement().items():
        cname = key.split("#")[0]
        if cname not in TOF_SPEC:
            continue
        ang = math.atan2(pos[1], pos[0])
        out[key] = {"origin_m": [pos[0], pos[1], pos[2] + 0.003],
                    "axis": [round(math.cos(ang), 4), round(math.sin(ang), 4), 0.0],
                    **TOF_SPEC[cname]}
    return out

def camera_lens_poses():
    """{placement_key: {origin_m, axis, hfov_deg, vfov_deg}} - REAL lens apex of
    each placed camera, measured from the rotated STEP solids (never inferred
    from bbox dims: the lens barrel is offset from the board bbox center).
    Lens-barrel solids = native-z center < -2mm (they protrude from the board
    face). Placement convention (cad_geometry): rotated shape centered in x/y,
    bottom-aligned in z on the placement point. origin = lens front-face center;
    axis = +X, the lens-forward mount direction enforced by ORIENTATIONS."""
    import build123d as b
    out = {}
    for key, pos in placement().items():
        cname = key.split("#")[0]
        if cname not in CAMERA_SPEC:
            continue
        c = LIBRARY[cname]
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), c.step_path)
        brep = path + ".brep"
        if path not in _STEP_CACHE:
            _STEP_CACHE[path] = b.import_brep(brep) if os.path.exists(brep) else b.import_step(path)
        nat = _STEP_CACHE[path]
        rot = _apply_orientation(cname, nat)
        bb = rot.bounding_box()
        cx, cy = (bb.min.X + bb.max.X) / 2, (bb.min.Y + bb.max.Y) / 2
        lens = [s for s in nat.solids()
                if (s.bounding_box().min.Z + s.bounding_box().max.Z) / 2 < -2.0]
        # keep the lens assembly only (barrel + housing): the two largest
        # sub-board protrusions; tiny standoff/screw solids near board corners
        # otherwise skew the lens center.
        lens = sorted(lens, key=lambda s: s.volume, reverse=True)[:2]
        spec = ORIENTATIONS.get(cname)
        ax = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[spec[0]]
        rl = [s.rotate(b.Axis((0, 0, 0), ax), spec[1]) for s in lens]
        fx = max(s.bounding_box().max.X for s in rl)
        lcy = (min(s.bounding_box().min.Y for s in rl) + max(s.bounding_box().max.Y for s in rl)) / 2
        lcz = (min(s.bounding_box().min.Z for s in rl) + max(s.bounding_box().max.Z for s in rl)) / 2
        fov = CAMERA_SPEC[cname]
        out[key] = {"origin_m": [(pos[0] * 1000 + fx - cx) / 1000,
                                 (pos[1] * 1000 + lcy - cy) / 1000,
                                 (pos[2] * 1000 + lcz - bb.min.Z) / 1000],
                    "axis": [1.0, 0.0, 0.0],
                    "hfov_deg": fov["hfov_deg"], "vfov_deg": fov["vfov_deg"]}
    return out


def imu_pose():
    """IMU site: real placement + orientation (identity unless ORIENTATIONS says otherwise).
    The sim/estimator corrects readings with this transform relative to CoM."""
    pos = placement()["mpu9250"]
    rot = ORIENTATIONS.get("mpu9250")
    quat = [0.0, 0.0, 0.0, 1.0]
    if rot:
        import math as _m
        ax = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[rot[0]]
        h = _m.radians(rot[1]) / 2
        quat = [ax[0] * _m.sin(h), ax[1] * _m.sin(h), ax[2] * _m.sin(h), _m.cos(h)]
    return {"position_m": [float(x) for x in pos], "rotation_quat_xyzw": quat}
