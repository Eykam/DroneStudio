"""Parametric 5-inch quad chassis (quad-X), build123d.

Constraints from DroneStudio sim (Studio/src/core/ecs/components/FlightController.zig):
  - quad-X, motor_arm_length 0.15 m (center -> motor axis)
  - motor order: M1 FR (CW), M2 FL (CCW), M3 RL (CW), M4 RR (CCW)
  - max thrust 10 N per motor, AUW target ~1.5 kg
Print target: FDM, PETG baseline (rho = 1240 kg/m^3), 0.4 mm nozzle, no supports.
"""
from dataclasses import dataclass, field, asdict
import math
import build123d as b

@dataclass
class ChassisParams:
    arm_length_mm: float = 150.0        # sim motor_arm_length (center to motor axis)
    arm_width_mm: float = 14.0
    arm_thickness_mm: float = 23.0      # maximum T-beam depth at the loaded root
    arm_root_width_mm: float = 22.0
    center_plate_len_mm: float = 90.0
    center_plate_wid_mm: float = 46.0
    top_plate_thickness_mm: float = 2.5
    body_thickness_mm: float = 1.4      # margin above the 1.2 mm printable minimum
    arm_rib_thickness_mm: float = 1.4
    arm_rib_offset_mm: float = 3.3      # twin ribs clear holes on arm centerline
    arm_rib_root_mm: float = 12.0
    motor_pad_thickness_mm: float = 5.0
    motor_pad_dia_mm: float = 26.0
    motor_hole_spacing_mm: float = 16.0 # 16x16 M3 pattern (22xx/23xx motors)
    motor_hole_dia_mm: float = 3.2
    motor_center_hole_dia_mm: float = 9.0
    stack_spacing_mm: float = 30.5      # standard FC/ESC stack
    stack_hole_dia_mm: float = 3.2
    stack_standoff_dia_mm: float = 6.0
    fillet_radius_mm: float = 2.0
    prop_dia_mm: float = 127.0          # 5 inch
    prop_clearance_mm: float = 10.0     # min tip-to-tip margin between adjacent props

    def motor_positions(self):
        """Quad-X motor XY positions (mm), sim order M1 FR, M2 FL, M3 RL, M4 RR."""
        a = self.arm_length_mm / math.sqrt(2.0)
        return [(a, a), (-a, a), (-a, -a), (a, -a)]

    def check_prop_clearance(self):
        adjacent = self.arm_length_mm * math.sqrt(2.0)
        need = self.prop_dia_mm + self.prop_clearance_mm
        return adjacent >= need, adjacent, need

def build_chassis(p: ChassisParams) -> b.Part:
    arms = []
    for (mx, my) in p.motor_positions():
        ang = math.degrees(math.atan2(my, mx))
        # A thin, wide lower flange carries torsion.  Two tall longitudinal webs
        # put most material far from the neutral axis for vertical stiffness.
        # Splitting the web also clears the stack and motor holes, both of which
        # lie on an arm's centreline in this quad-X layout.
        L = p.arm_length_mm
        prof = b.Polyline((0, -p.arm_root_width_mm/2), (L, -p.arm_width_mm/2),
                          (L, p.arm_width_mm/2), (0, p.arm_root_width_mm/2), close=True)
        arm = b.extrude(b.make_face(prof), p.body_thickness_mm)
        pad = b.extrude(b.Circle(p.motor_pad_dia_mm/2).face(), p.motor_pad_thickness_mm)
        pad = pad.locate(b.Pos(L, 0, 0))

        # A small overlap with the pad avoids transferring thrust through only
        # the thin lower flange at the arm-to-motor junction.
        rib_end = L - p.motor_pad_dia_mm / 2 + 2.0
        piece = arm + pad
        for off in (-p.arm_rib_offset_mm, p.arm_rib_offset_mm):
            # Bending moment falls toward the motor, so taper web depth roughly
            # with sqrt(moment), retaining depth where it earns the most
            # stiffness.  The final point meets the motor pad flush.
            x0 = p.arm_rib_root_mm
            span = rib_end - x0
            y0 = off - p.arm_rib_thickness_mm / 2
            side = [(x0, y0, p.body_thickness_mm),
                    (rib_end, y0, p.body_thickness_mm),
                    (rib_end, y0, p.motor_pad_thickness_mm)]
            for frac in (0.90, 0.75, 0.50, 0.25):
                x = x0 + frac * span
                remaining = (rib_end - x) / span
                z = p.motor_pad_thickness_mm + (
                    p.arm_thickness_mm - p.motor_pad_thickness_mm
                ) * math.sqrt(remaining)
                side.append((x, y0, z))
            side.append((x0, y0, p.arm_thickness_mm))
            rib = b.extrude(
                b.make_face(b.Polyline(*side, close=True)),
                p.arm_rib_thickness_mm,
                dir=(0, 1, 0),
            )
            piece = piece + rib
        piece = piece.rotate(b.Axis.Z, ang)
        arms.append(piece)
    body = arms[0]
    for a in arms[1:]:
        body = body + a
    # center hub
    hub = b.extrude(b.make_face(b.Circle(p.arm_root_width_mm * 1.15)), p.body_thickness_mm)
    body = body + hub
    # motor bolt holes (16x16 M3) + center bore, through each pad
    cut_height = max(p.arm_thickness_mm, p.motor_pad_thickness_mm) + 2
    holes = []
    for (mx, my) in p.motor_positions():
        hs = p.motor_hole_spacing_mm / 2
        for dx, dy in ((hs, hs), (-hs, hs), (-hs, -hs), (hs, -hs)):
            holes.append(b.Pos(mx+dx, my+dy, -1) * b.Cylinder(p.motor_hole_dia_mm/2, cut_height))
        holes.append(b.Pos(mx, my, -1) * b.Cylinder(p.motor_center_hole_dia_mm/2, cut_height))
    # stack holes (30.5 mm square) in hub
    sh = p.stack_spacing_mm / 2
    for dx, dy in ((sh, sh), (-sh, sh), (-sh, -sh), (sh, -sh)):
        holes.append(b.Pos(dx, dy, -1) * b.Cylinder(p.stack_hole_dia_mm/2, cut_height))
    for h in holes:
        body = body - h
    return b.Part(body.wrapped)

if __name__ == "__main__":
    p = ChassisParams()
    ok, adjacent, need = p.check_prop_clearance()
    print(f"prop clearance: adjacent motor spacing {adjacent:.1f} mm, need {need:.1f} mm -> {'OK' if ok else 'FAIL'}")
    part = build_chassis(p)
    print(f"volume {part.volume:.0f} mm^3")
    rho = 1240e-9  # kg/mm^3
    print(f"mass {part.volume*rho*1000:.1f} g (PETG)")
    b.export_stl(part, "/home/sandbox/cad-researcher/chassis_v1.stl")
    b.export_step(part, "/home/sandbox/cad-researcher/chassis_v1.step")
    print("exported chassis_v1.stl / .step")
