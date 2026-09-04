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
    arm_width_mm: float = 9.2
    arm_thickness_mm: float = 32.0      # maximum closed-section depth at the loaded root
    arm_root_width_mm: float = 14.0     # broad root chine flows into the cabin shell
    arm_shell_root_width_mm: float = 13.6
    arm_sweep_mm: float = 3.0           # chiral plan-view bow; motor axes stay fixed
    arm_roof_slope: float = 1.15        # >1 gives a support-free (>45 deg) inner roof
    arm_crown_width_mm: float = 3.0     # narrow high-fiber facet on the closed arm shell
    arm_height_falloff: float = 0.55    # deep root, lean span; moment-shaped shell
    arm_tip_height_mm: float = 10.8     # faired nacelle depth at the motor load transfer
    center_plate_len_mm: float = 120.0
    center_plate_wid_mm: float = 86.0
    top_plate_thickness_mm: float = 2.5
    body_thickness_mm: float = 1.60     # four printable perimeters for the fuselage skin
    arm_rib_thickness_mm: float = 1.40  # closed arm shell wall
    arm_rib_offset_mm: float = 3.3      # retained for parameter-file compatibility
    arm_rib_root_mm: float = 12.0
    body_fairing_height_mm: float = 43.0
    body_fairing_draft_mm: float = 0.8
    body_roof_slope: float = 1.10       # support-free inner canopy faces (>45 deg)
    body_roof_top_len_mm: float = 86.0
    body_roof_top_wid_mm: float = 46.0
    body_hatch_len_mm: float = 80.0     # dorsal battery/avionics service opening
    body_hatch_wid_mm: float = 40.0
    payload_rail_width_mm: float = 2.4
    body_corner_radius_mm: float = 12.0
    motor_pad_thickness_mm: float = 4.6
    motor_pad_dia_mm: float = 28.7      # boss-to-boss envelope, not a solid disk
    motor_boss_wall_mm: float = 1.2
    motor_spoke_width_mm: float = 3.4
    motor_hole_spacing_mm: float = 16.0 # 16x16 M3 pattern (22xx/23xx motors)
    motor_hole_dia_mm: float = 3.2
    motor_center_hole_dia_mm: float = 9.0
    stack_spacing_mm: float = 30.5      # standard FC/ESC stack
    stack_hole_dia_mm: float = 3.2
    stack_standoff_dia_mm: float = 9.0
    stack_standoff_height_mm: float = 10.0
    camera_aperture_dia_mm: float = 10.0
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
        """Support-free crowned arm section in the local YZ plane."""
        wall = p.arm_rib_thickness_mm
        slope = p.arm_roof_slope
        roof_normal = math.hypot(slope, 1.0)
        crown = min(p.arm_crown_width_mm, width - 2.5 * wall)
        roof_rise = slope * (width - crown) / 2
        if not inner:
            shoulder = height - roof_rise
            points = [
                (x, center - width/2, 0),
                (x, center + width/2, 0),
                (x, center + width/2, shoulder),
                (x, center + crown/2, height),
                (x, center - crown/2, height),
                (x, center - width/2, shoulder),
            ]
        else:
            # The narrow crown moves skin onto the upper bending fiber while
            # shortening the two pitched roof faces.  Their inner surfaces stay
            # steeper than 45 degrees, so the cavity prints without support.
            half_inner = width/2 - wall
            shoulder = (
                height + slope*crown/2 - wall*roof_normal
                - slope*half_inner
            )
            inner_crown_half = (
                crown/2 - wall*(roof_normal - 1.0)/slope
            )
            crown_z = height - wall
            points = [
                (x, center - half_inner, wall),
                (x, center + half_inner, wall),
                (x, center + half_inner, shoulder),
                (x, center + inner_crown_half, crown_z),
                (x, center - inner_crown_half, crown_z),
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
        profile_end = L - bolt_radius
        rib_end = profile_end
        plan_x = [0, x0, x0 + 0.25*(profile_end-x0),
                  x0 + 0.50*(profile_end-x0),
                  x0 + 0.75*(profile_end-x0), profile_end, L]
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
        # tall root and rapidly falling depth retain the bending economy, while
        # a narrow dorsal facet turns the old peaked tube into a more efficient
        # faired box.  Pitched cavity ceilings remain self-supporting.
        span = profile_end - x0
        tube_sections = []
        for frac in (0.0, 0.25, 0.50, 0.75, 0.90, 1.0):
            x = x0 + frac * span
            width = p.arm_shell_root_width_mm + (
                p.arm_width_mm - p.arm_shell_root_width_mm
            ) * frac
            roof_normal = math.hypot(p.arm_roof_slope, 1.0)
            min_tip_height = (
                p.arm_roof_slope * (width - p.arm_crown_width_mm)/2
                + p.arm_rib_thickness_mm * (1 + roof_normal - p.arm_roof_slope)
                + 0.15
            )
            profile_tip_height = max(p.motor_pad_thickness_mm,
                                     min_tip_height)
            height = profile_tip_height + (
                p.arm_thickness_mm - profile_tip_height
            ) * max(0.0, 1.0 - frac) ** p.arm_height_falloff
            if frac == 1.0:
                height = max(height, p.arm_tip_height_mm)
            tube_sections.append((x, sweep_center(x), width, height))
        tip_height = tube_sections[-1][3]
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
        end_diaphragm = b.Solid.make_loft([
            section_wire(cap_x, sweep_center(cap_x),
                         p.arm_width_mm, tip_height),
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

        # A short solid nacelle insert begins inside the hollow tip, closes its
        # cavity before the inboard bolt, and fans into the shaft boss.  It
        # removes the shell-end stress concentration with far less material
        # than deepening the whole motor plate.
        bridge_start = rib_end - 2*p.arm_rib_thickness_mm
        bridge_center = sweep_center(bridge_start)
        bridge = b.Polyline(
            (bridge_start, bridge_center - p.arm_width_mm/2),
            (L, -center_boss_radius),
            (L, center_boss_radius),
            (bridge_start, bridge_center + p.arm_width_mm/2),
            close=True,
        )
        pad = pad + b.extrude(b.make_face(bridge), tip_height)

        piece = arm + pad
        piece = piece.rotate(b.Axis.Z, ang)
        arms.append(piece)
    body = arms[0]
    for a in arms[1:]:
        body = body + a
    # A tall structural-skin fuselage replaces the vestigial deck coaming.  The
    # battery, computer and stack are packed side-by-side below its shoulder;
    # the compound-pitched roof closes around a dorsal service hatch.  Thus the
    # payload is inside the airframe rather than strapped to a top plate, while
    # every downward-facing cavity facet remains steeper than 45 degrees.
    fair_h = p.body_fairing_height_mm
    draft = p.body_fairing_draft_mm
    wall = p.body_thickness_mm

    def faired_wire(length, width, radius, z):
        """A filleted, fore-aft faired fuselage perimeter."""
        hl, hw = length/2, width/2
        points = [
            (hl, 0, z),
            (0.96*hl, 0.55*hw, z),
            (0.82*hl, 0.90*hw, z),
            (0.62*hl, hw, z),
            (-0.62*hl, hw, z),
            (-0.82*hl, 0.90*hw, z),
            (-0.96*hl, 0.55*hw, z),
            (-hl, 0, z),
            (-0.96*hl, -0.55*hw, z),
            (-0.82*hl, -0.90*hw, z),
            (-0.62*hl, -hw, z),
            (0.62*hl, -hw, z),
            (0.82*hl, -0.90*hw, z),
            (0.96*hl, -0.55*hw, z),
        ]
        wire = b.Polyline(*points, close=True)
        corner = min(0.4*radius, 0.11*width)
        return b.fillet(wire.vertices(), corner)

    shoulder_len = p.center_plate_len_mm - 2*draft
    shoulder_wid = p.center_plate_wid_mm - 2*draft
    roof_run = max(
        (shoulder_len - p.body_roof_top_len_mm) / 2,
        (shoulder_wid - p.body_roof_top_wid_mm) / 2,
    )
    roof_z = fair_h + p.body_roof_slope * roof_run
    outer_fairing = b.Solid.make_loft([
        faired_wire(p.center_plate_len_mm, p.center_plate_wid_mm,
                    p.body_corner_radius_mm, 0),
        faired_wire(shoulder_len, shoulder_wid,
                    p.body_corner_radius_mm - draft, fair_h),
        faired_wire(p.body_roof_top_len_mm, p.body_roof_top_wid_mm,
                    p.body_corner_radius_mm - 2*draft, roof_z),
    ], ruled=True)
    # The inner loft pierces both the underside and crown.  This makes a true
    # hollow monocoque with a large dorsal hatch rather than a mass-heavy floor
    # or an unprintable horizontal cavity ceiling.
    inner_shoulder_z = fair_h - wall/2
    inner_fairing = b.Solid.make_loft([
        faired_wire(p.center_plate_len_mm - 2*wall,
                    p.center_plate_wid_mm - 2*wall,
                    p.body_corner_radius_mm - wall, -0.2),
        faired_wire(p.center_plate_len_mm - 2*(draft + wall),
                    p.center_plate_wid_mm - 2*(draft + wall),
                    p.body_corner_radius_mm - draft - wall,
                    inner_shoulder_z),
        faired_wire(p.body_hatch_len_mm, p.body_hatch_wid_mm,
                    p.body_corner_radius_mm - 2*wall, roof_z + 1.0),
    ], ruled=True)
    body = body + (outer_fairing - inner_fairing)

    # Sparse first-layer rails support and capture the two side-by-side payload
    # lanes.  Each rail crosses a pair of arm chines, distributing battery-jolt
    # load without paying the mass of a full 120 x 86 mm belly plate.
    rail_len = 86.0
    rail_x = -10.0
    for rail_y in (-31.0, -19.0, -7.0, 8.0, 20.0, 32.0):
        rail = b.Pos(rail_x, rail_y, 0) * b.extrude(
            b.Rectangle(rail_len, p.payload_rail_width_mm).face(), wall
        )
        body = body + rail
    nose_rail = b.Pos(40.0, 0, 0) * b.extrude(
        b.Rectangle(p.payload_rail_width_mm, 82.0).face(), wall
    )
    body = body + nose_rail

    # Annular stack hard-points fill the hollow arm locally around the FEA
    # fixture/mount holes.  Their depth spreads peak root stress into the shell
    # instead of thickening the entire center deck.
    sh = p.stack_spacing_mm / 2
    for dx, dy in ((sh, sh), (-sh, sh), (-sh, -sh), (sh, -sh)):
        boss = b.Pos(dx, dy, p.stack_standoff_height_mm/2) * b.Cylinder(
            p.stack_standoff_dia_mm/2, p.stack_standoff_height_mm
        )
        body = body + boss

    # Twin forward apertures leave only the lenses exposed; the camera PCBs sit
    # aft of the nose skin.  Horizontal holes are small enough to bridge and do
    # not compromise the continuous roof hoop around the service hatch.
    for camera_y in (-24.0, 24.0):
        aperture = b.Pos(54.0, camera_y, 9.0) * b.Cylinder(
            p.camera_aperture_dia_mm/2, 20.0
        ).rotate(b.Axis.Y, 90)
        body = body - aperture

    # motor bolt holes (16x16 M3) + center bore, through each pad
    cut_height = max(roof_z, p.arm_thickness_mm,
                     p.motor_pad_thickness_mm) + 2
    holes = []
    for (mx, my) in p.motor_positions():
        hs = p.motor_hole_spacing_mm / 2
        for dx, dy in ((hs, hs), (-hs, hs), (-hs, -hs), (hs, -hs)):
            holes.append(b.Pos(mx+dx, my+dy, -1) * b.Cylinder(p.motor_hole_dia_mm/2, cut_height))
        holes.append(b.Pos(mx, my, -1) * b.Cylinder(p.motor_center_hole_dia_mm/2, cut_height))
    # stack holes (30.5 mm square) in hub
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
