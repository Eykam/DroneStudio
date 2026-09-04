"""Parametric 5-inch quad chassis (quad-X), build123d.

Constraints from DroneStudio sim (Studio/src/core/ecs/components/FlightController.zig):
  - quad-X, motor_arm_length 0.15 m (center -> motor axis)
  - motor order: M1 FR (CW), M2 FL (CCW), M3 RL (CW), M4 RR (CCW)
  - EMAX ECO II 2207 2400KV / 4S, max thrust 17.27 N per motor
Print target: FDM, PETG baseline (rho = 1240 kg/m^3), 0.4 mm nozzle, no supports.
"""
from dataclasses import dataclass, field, asdict
import math
import build123d as b

@dataclass
class ChassisParams:
    arm_length_mm: float = 150.0        # sim motor_arm_length (center to motor axis)
    arm_width_mm: float = 8.2
    arm_thickness_mm: float = 27.0      # maximum closed-section depth at the loaded root
    arm_root_width_mm: float = 14.0
    arm_shell_root_width_mm: float = 11.0
    arm_sweep_mm: float = 3.0           # chiral plan-view bow; motor axes stay fixed
    arm_roof_slope: float = 1.15        # >1 gives a support-free (>45 deg) inner roof
    arm_height_falloff: float = 0.58    # deep root, lean span; moment-shaped shell
    center_plate_len_mm: float = 90.0
    center_plate_wid_mm: float = 46.0
    top_plate_thickness_mm: float = 2.5
    body_thickness_mm: float = 1.22     # slight margin above the 1.2 mm minimum
    arm_rib_thickness_mm: float = 1.22  # closed arm shell wall
    arm_rib_offset_mm: float = 3.3      # retained for parameter-file compatibility
    arm_rib_root_mm: float = 12.0
    body_fairing_height_mm: float = 2.5
    body_fairing_draft_mm: float = 0.8
    body_corner_radius_mm: float = 12.0
    motor_pad_thickness_mm: float = 4.6
    motor_pad_dia_mm: float = 28.7      # boss-to-boss envelope, not a solid disk
    motor_boss_wall_mm: float = 1.2
    motor_spoke_width_mm: float = 3.0
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
    def sweep_center(x):
        # Both endpoints remain on the fixed motor radial.  Bowing the middle of
        # every arm produces a subtle pinwheel sweep without moving a motor axis.
        return -p.arm_sweep_mm * math.sin(math.pi * x / p.arm_length_mm)

    def section_wire(x, center, width, height, inner=False):
        """Support-free pentagonal arm section in the local YZ plane."""
        wall = p.arm_rib_thickness_mm
        slope = p.arm_roof_slope
        roof_rise = slope * width / 2
        if not inner:
            shoulder = height - roof_rise
            points = [
                (x, center - width/2, 0),
                (x, center + width/2, 0),
                (x, center + width/2, shoulder),
                (x, center, height),
                (x, center - width/2, shoulder),
            ]
        else:
            # Offset each wall inward by its true normal thickness.  The cavity
            # remains open at both ends for wiring, inspection, and powder-free
            # printing; along its span the arm is a torsionally closed section.
            roof_normal = math.hypot(slope, 1.0)
            half_inner = width/2 - wall
            shoulder = height - slope * half_inner - wall * roof_normal
            apex = height - wall * roof_normal
            points = [
                (x, center - half_inner, wall),
                (x, center + half_inner, wall),
                (x, center + half_inner, shoulder),
                (x, center, apex),
                (x, center - half_inner, shoulder),
            ]
        return b.Wire.make_polygon(points, close=True)

    arms = []
    for (mx, my) in p.motor_positions():
        ang = math.degrees(math.atan2(my, mx))
        # A swept, root-flared lower chine gives a broad printable first layer.
        # The smooth plan outline keeps the arm-to-body junction from reading as
        # a flat plate even though it is printed directly on the build surface.
        L = p.arm_length_mm
        x0 = p.arm_rib_root_mm
        bolt_radius = p.motor_hole_spacing_mm / math.sqrt(2.0)
        rib_end = L - bolt_radius
        plan_x = [0, x0, x0 + 0.25*(rib_end-x0), x0 + 0.50*(rib_end-x0),
                  x0 + 0.75*(rib_end-x0), rib_end, L]
        lower = []
        upper = []
        for x in plan_x:
            width = p.arm_root_width_mm + (
                p.arm_width_mm - p.arm_root_width_mm
            ) * x / L
            center = sweep_center(x)
            lower.append((x, center - width/2))
            upper.append((x, center + width/2))
        outline = b.Polyline(*(lower + list(reversed(upper))), close=True)
        outline = b.fillet(outline.vertices(), p.fillet_radius_mm)
        arm = b.extrude(b.make_face(outline), p.body_thickness_mm)

        # Loft a hollow structural skin in six bending-moment stations.  The
        # tall root and rapidly falling depth retain the v5-g4 bending economy,
        # while the roof closes the section for torsion.  Its two pitched faces
        # (rather than a flat ceiling) make the internal cavity self-supporting.
        span = rib_end - x0
        tube_sections = []
        for frac in (0.0, 0.25, 0.50, 0.75, 0.90, 1.0):
            x = x0 + frac * span
            width = p.arm_shell_root_width_mm + (
                p.arm_width_mm - p.arm_shell_root_width_mm
            ) * frac
            roof_normal = math.hypot(p.arm_roof_slope, 1.0)
            min_tip_height = (
                p.arm_roof_slope * width/2
                + p.arm_rib_thickness_mm * (1 + roof_normal - p.arm_roof_slope)
                + 0.15
            )
            tip_height = max(p.motor_pad_thickness_mm, min_tip_height)
            height = tip_height + (
                p.arm_thickness_mm - tip_height
            ) * max(0.0, 1.0 - frac) ** p.arm_height_falloff
            tube_sections.append((x, sweep_center(x), width, height))
        outer = b.Solid.make_loft(
            [section_wire(*section) for section in tube_sections], ruled=True
        )
        cavity = b.Solid.make_loft(
            [section_wire(*section, inner=True) for section in tube_sections],
            ruled=True,
        )
        arm = arm + (outer - cavity)

        # Close the high-shear motor end of the monocoque with one perimeter-
        # thickness diaphragm.  The span remains hollow and root-accessible for
        # wiring, while eliminating the open-shell notch that concentrated crash
        # stress where the arm flows into the motor nacelle.
        cap_x = rib_end - p.arm_rib_thickness_mm
        cap_frac = (cap_x - x0) / span
        cap_width = p.arm_shell_root_width_mm + (
            p.arm_width_mm - p.arm_shell_root_width_mm
        ) * cap_frac
        cap_min_height = (
            p.arm_roof_slope * cap_width/2
            + p.arm_rib_thickness_mm * (
                1 + math.hypot(p.arm_roof_slope, 1.0) - p.arm_roof_slope
            )
            + 0.15
        )
        cap_tip_height = max(p.motor_pad_thickness_mm, cap_min_height)
        cap_height = cap_tip_height + (
            p.arm_thickness_mm - cap_tip_height
        ) * max(0.0, 1.0 - cap_frac) ** p.arm_height_falloff
        end_diaphragm = b.Solid.make_loft([
            section_wire(cap_x, sweep_center(cap_x), cap_width, cap_height),
            section_wire(rib_end, sweep_center(rib_end),
                         p.arm_width_mm, tip_height),
        ], ruled=True)
        arm = arm + end_diaphragm

        # A cruciform motor mount follows the four bolt load paths instead of
        # carrying a mostly unstressed solid disk.  Circular bosses retain a
        # full printable wall around both the shaft bore and every M3 hole.
        bolt_boss_radius = p.motor_hole_dia_mm / 2 + p.motor_boss_wall_mm
        center_boss_radius = p.motor_center_hole_dia_mm / 2 + p.motor_boss_wall_mm
        spoke_length = 2 * (bolt_radius + bolt_boss_radius)
        pad = b.extrude(
            b.Rectangle(spoke_length, p.motor_spoke_width_mm).face(),
            p.motor_pad_thickness_mm,
        )
        pad = pad + b.extrude(
            b.Rectangle(p.motor_spoke_width_mm, spoke_length).face(),
            p.motor_pad_thickness_mm,
        )
        pad = pad + b.extrude(
            b.Circle(center_boss_radius).face(), p.motor_pad_thickness_mm
        )
        for bx, by in ((bolt_radius, 0), (-bolt_radius, 0),
                       (0, bolt_radius), (0, -bolt_radius)):
            pad = pad + b.Pos(bx, by, 0) * b.extrude(
                b.Circle(bolt_boss_radius).face(), p.motor_pad_thickness_mm
            )
        pad = pad.locate(b.Pos(L, 0, 0))

        # A plan-view throat and a tiny pyramidal crown spread the tube into the
        # longitudinal motor spoke.  This removes the old shell-edge hotspot
        # without turning the low-stress motor mount back into a solid disk.
        neck_x = L - center_boss_radius
        end_center = sweep_center(rib_end)
        throat = b.Polyline(
            (rib_end, end_center - p.arm_width_mm/2),
            (neck_x, -p.motor_spoke_width_mm/2),
            (neck_x, p.motor_spoke_width_mm/2),
            (rib_end, end_center + p.arm_width_mm/2),
            close=True,
        )
        pad = pad + b.extrude(b.make_face(throat), p.motor_pad_thickness_mm)
        crown_root = b.Wire.make_polygon([
            (rib_end, end_center - p.arm_width_mm/2, p.motor_pad_thickness_mm),
            (rib_end, end_center + p.arm_width_mm/2, p.motor_pad_thickness_mm),
            (rib_end, end_center + p.arm_width_mm/2, tip_height),
            (rib_end, end_center - p.arm_width_mm/2, tip_height),
        ], close=True)
        crown = b.Solid.make_loft(
            [crown_root, b.Vertex(neck_x, 0, p.motor_pad_thickness_mm)],
            ruled=True,
        )
        pad = pad + crown

        # The tube overlaps the inboard edge of this load-path mount, fairing
        # the former vertical pedestal step into the arm.
        piece = arm + pad
        piece = piece.rotate(b.Axis.Z, ang)
        arms.append(piece)
    body = arms[0]
    for a in arms[1:]:
        body = body + a
    # A drafted rounded coaming makes the center structure read as one fuselage
    # and braces all four swept arm roots.  It encloses an open-top/open-bottom
    # recessed service bay: battery straps and stack wiring stay accessible,
    # while the crossed lower arm skins form local capture ledges.  Both inner
    # and outer walls lean by far less than 45 degrees and need no support.
    fair_h = p.body_fairing_height_mm
    draft = p.body_fairing_draft_mm
    wall = p.body_thickness_mm

    def rounded_wire(length, width, radius, z):
        radius = min(radius, length/2 - 0.1, width/2 - 0.1)
        return b.RectangleRounded(length, width, radius).face().outer_wire().locate(
            b.Pos(0, 0, z)
        )

    outer_fairing = b.Solid.make_loft([
        rounded_wire(p.center_plate_len_mm, p.center_plate_wid_mm,
                     p.body_corner_radius_mm, 0),
        rounded_wire(p.center_plate_len_mm - 2*draft,
                     p.center_plate_wid_mm - 2*draft,
                     p.body_corner_radius_mm - draft, fair_h),
    ], ruled=True)
    inner_fairing = b.Solid.make_loft([
        rounded_wire(p.center_plate_len_mm - 2*wall,
                     p.center_plate_wid_mm - 2*wall,
                     p.body_corner_radius_mm - wall, -0.1),
        rounded_wire(p.center_plate_len_mm - 2*(draft + wall),
                     p.center_plate_wid_mm - 2*(draft + wall),
                     p.body_corner_radius_mm - draft - wall, fair_h + 0.1),
    ], ruled=True)
    body = body + (outer_fairing - inner_fairing)

    # Leave the center of the bay open underneath instead of filling it with a
    # legacy circular plate; the four overlapping arm chines remain the floor
    # and the perimeter coaming supplies the cross-arm tie.
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
