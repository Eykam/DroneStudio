"""Candidate A: low coaming monocoque with a shared stereo nose and slender belly.

Parametric 5-inch quad chassis (quad-X), build123d.

Constraints from DroneStudio sim (Studio/src/core/ecs/components/FlightController.zig):
  - quad-X, motor_arm_length 0.15 m (center -> motor axis)
  - motor order: M1 FR (CW), M2 FL (CCW), M3 RL (CW), M4 RR (CCW)
  - AKK RS2205 2300KV / 4S, max thrust 11.0 N per motor
Print target: FDM, PETG baseline (rho = 1240 kg/m^3), 0.4 mm nozzle, no supports.
"""
from dataclasses import dataclass, field, asdict
import math
import build123d as b

@dataclass
class ChassisParams:
    arm_length_mm: float = 150.0        # sim motor_arm_length (center to motor axis)
    arm_width_mm: float = 9.2
    arm_thickness_mm: float = 24.5      # closed-section root stays 2 mm below the recessed stack
    arm_root_width_mm: float = 14.0     # broad root chine flows into the cabin shell
    arm_shell_root_width_mm: float = 16.2
    arm_sweep_mm: float = 3.0           # chiral plan-view bow; motor axes stay fixed
    arm_roof_slope: float = 1.15        # >1 gives a support-free (>45 deg) inner roof
    arm_crown_width_mm: float = 3.6     # broader bending flange; <2.5 mm inner bridge
    arm_height_falloff: float = 0.68    # retain root depth, shed low-moment span skin
    arm_tip_height_mm: float = 10.8     # faired nacelle depth at the motor load transfer
    center_plate_len_mm: float = 242.0  # compact twin-cheek nose to rear avionics fin
    center_plate_wid_mm: float = 68.0
    top_plate_thickness_mm: float = 2.5
    body_thickness_mm: float = 1.25     # structural skin; all faces use normal offsets
    arm_rib_thickness_mm: float = 1.25  # >=1.20 mm normal to the default sloping crown
    arm_rib_offset_mm: float = 3.3      # retained for parameter-file compatibility
    arm_rib_root_mm: float = 16.0
    body_fairing_height_mm: float = 46.5 # stack canopy follows the recessed mounting ring
    body_fairing_draft_mm: float = 4.7   # inward side inset at the stack shoulder
    body_roof_slope: float = 1.10       # support-free inner canopy faces (>45 deg)
    body_roof_top_len_mm: float = 86.0
    body_roof_top_wid_mm: float = 46.0
    body_hatch_len_mm: float = 80.0     # dorsal battery/avionics service opening
    body_hatch_wid_mm: float = 40.0
    payload_rail_width_mm: float = 1.4
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
    stack_standoff_dia_mm: float = 6.0   # 1.4 mm annular wall around each M3 bore
    stack_standoff_height_mm: float = 24.5
    camera_aperture_dia_mm: float = 10.0
    fillet_radius_mm: float = 4.0
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
        # Reinforce only the root, fading the inner wall back to span gauge.
        # The outer chine remains continuous through the fuselage junction.
        root_blend = max(0.0, min(1.0, (75.0 - x) / 30.0))
        wall = p.arm_rib_thickness_mm + 0.20 * root_blend
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

        # Hollow six-sided spars put depth just beyond the stack ring, where
        # the cantilever begins. A slimmer outer span replaces the old broad
        # shallow tube; its roof descends continuously into the motor fairing.
        # The terminal depth never dips below the nacelle, removing a notch.
        span = profile_end-x0
        spar_stations = [
            (0.00, 15.5, 22.0),
            (0.08, 15.5, 24.0),
            (0.20, 15.0, 24.2),
            (0.40, 12.8, 20.2),
            (0.60, 10.5, 15.8),
            (0.80, 9.2, 11.8),
            (0.92, 9.2, 10.8),
            (1.00, 9.2, 10.8),
        ]
        tube_sections = []
        for frac, nominal_width, nominal_height in spar_stations:
            x = x0+frac*span
            width = nominal_width*p.arm_shell_root_width_mm/16.2
            width = max(p.arm_width_mm, width)
            height = nominal_height*p.arm_thickness_mm/24.5
            if frac == 1.0:
                width, height = p.arm_width_mm, p.arm_tip_height_mm
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

        # A descending nacelle fairing closes the tube at full depth through
        # the inboard bolt, then sheds height toward the shaft boss.  Its roof
        # follows the falling bending moment instead of carrying a solid,
        # full-height block all the way across the motor center.  The belly
        # stays on the plate, and the four bolt locations remain unchanged.
        bridge_start = rib_end - 2*p.arm_rib_thickness_mm
        bridge_center = sweep_center(bridge_start)
        nacelle_sections = []
        for x, height in ((bridge_start, tip_height),
                          (rib_end, tip_height),
                          (L, p.motor_pad_thickness_mm)):
            frac = (x - bridge_start)/(L - bridge_start)
            center = bridge_center*(1-frac)
            half_width = p.arm_width_mm/2*(1-frac) + center_boss_radius*frac
            nacelle_sections.append(b.Wire.make_polygon([
                (x, center-half_width, 0), (x, center+half_width, 0),
                (x, center+half_width, height),
                (x, center-half_width, height),
            ], close=True))
        pad = pad + b.Solid.make_loft(nacelle_sections, ruled=True)

        piece = arm + pad
        piece = piece.rotate(b.Axis.Z, ang)
        arms.append(piece)
    body = arms[0]
    for a in arms[1:]:
        body = body + a
    # A longitudinal monocoque gives each payload a real, disjoint cavity.
    # The low battery well sits BETWEEN the rear arms, the stack clears their
    # crowned junction, and the upright Pi occupies the narrow forward bay.
    # Keeping the deep arm roots intact is cheaper than weakening them with
    # electronics cutouts and recovering stiffness with a thick belly plate.
    wall = p.body_thickness_mm
    slope = p.body_roof_slope
    draft = p.body_fairing_draft_mm / p.body_fairing_height_mm
    sx = p.center_plate_len_mm / 242.0
    sy = p.center_plate_wid_mm / 68.0
    cockpit_h = p.body_fairing_height_mm

    # x, belly breadth, shoulder height, dorsal crown breadth (mm). The
    # sidewalls lean inward with height: broad first-layer chines still collect
    # the arm loads, while the upper body wraps closely around the payloads.
    # The low GPS/IMU tail follows the avionics envelope before a short swept
    # shoulder rises into the battery well. This removes the old tall wedge
    # above the rear electronics and gives the battery a defined aft coaming.
    # The lowered stack canopy and recessed battery remain accessible from
    # above; all transitions preserve the payloads' 2 mm service envelopes.
    stations = [
        (-145.0, 26.0, 15.0, 12.0),
        (-141.0, 31.0, 18.5, 18.0),
        (-120.0, 31.0, 18.5, 18.0),
        (-114.0, 52.0, 41.5, 43.5),
        (-105.0, 50.2, 41.5, 41.7),
        (-56.0, 50.2, 41.5, 41.7),
        (-45.0, 52.0, 41.5, 43.5),
        (-33.0, 56.0, 42.0, 40.0),
        (-24.0, 57.0, cockpit_h, 43.0),
        (24.0, 57.0, cockpit_h, 43.0),
        (38.0, 26.0, 35.6, 16.0),
        (92.5, 26.0, 35.6, 16.0),
        (97.0, 18.0, 30.0, 8.0),
    ]
    # Hold the payload shoulders while pulling the belly chines inward.
    # A wider dorsal opening lowers unused canopy skin above the upright Pi;
    # the stack, battery, and rear fin retain their complete service envelopes.
    old_draft = draft
    draft = 0.055 * p.body_fairing_draft_mm / 4.7
    stations = [(x*sx, (w-2*(old_draft-draft)*h)*sy, h, c*sy)
                for x, w, h, c in stations]
    shell_envelopes = []

    def cabin_shell(stations):
        """Mitered, normal-gauge shell with bottom and dorsal service access."""
        # Each side plane is y + draft*z = breadth/2; each roof plane is
        # z + slope*y = shoulder*(1-slope*draft) + slope*breadth/2.
        # Offset their full 3D normals, including the longitudinal sweep. This
        # preserves the printable gauge through both tapered sides and roof folds.
        side_offsets, roof_offsets = [], []
        for a, z in zip(stations, stations[1:]):
            dx = z[0] - a[0]
            side_gradient = (z[1]-a[1])/(2*dx)
            roof_gradient = (1-slope*draft)*(z[2]-a[2])/dx + slope*side_gradient
            side_offsets.append(wall*math.sqrt(1+draft*draft+side_gradient**2))
            roof_offsets.append(wall*math.sqrt(1+slope*slope+roof_gradient**2))

        def offset_profile(values, offsets):
            """Miter adjacent offset planes, preserving gauge through each chine."""
            lines = []
            for i, offset in enumerate(offsets):
                x0, x1 = stations[i][0], stations[i+1][0]
                gradient = (values[i+1]-values[i])/(x1-x0)
                lines.append((gradient, values[i]-gradient*x0-offset))
            joints = [stations[0][0] + wall]
            for i, (a, z) in enumerate(zip(lines, lines[1:]), 1):
                x = ((z[1]-a[1])/(a[0]-z[0])
                     if abs(a[0]-z[0]) > 1e-9 else stations[i][0])
                joints.append(x)
            joints.append(stations[-1][0] - wall)
            return [(x, lines[min(i, len(lines)-1)][0]*x
                     + lines[min(i, len(lines)-1)][1]) for i, x in enumerate(joints)]

        def interpolate(profile, x):
            for a, z in zip(profile, profile[1:]):
                if x <= z[0]:
                    return a[1] + (z[1]-a[1])*(x-a[0])/(z[0]-a[0])
            return profile[-1][1]

        inner_sides = offset_profile([w/2 for _, w, _, _ in stations], side_offsets)
        inner_roofs = offset_profile(
            [(1-slope*draft)*h+slope*w/2 for _, w, h, _ in stations], roof_offsets)
        outer_roofs = [(x, h+slope*((w-c)/2-draft*h)) for x, w, h, c in stations]

        def cabin_wire(station, inner=False):
            x, width, shoulder, crown = station
            roof = shoulder + slope*((width-crown)/2-draft*shoulder)
            if inner:
                side_constant = interpolate(inner_sides, x)
                roof_constant = interpolate(inner_roofs, x)
                # Offset the pitched surface along its normal, then extend it
                # through the crown: an open service hatch, with no broad bridge.
                shoulder = (roof_constant - slope*side_constant)/(1-slope*draft)
                roof = interpolate(outer_roofs, x) + 0.5
                crown_half = (roof_constant-roof)/slope
                bottom = -0.2
            else:
                side_constant, crown_half, bottom = width/2, crown/2, 0.0
            belly_half = side_constant - draft*bottom
            shoulder_half = side_constant - draft*shoulder
            return b.Wire.make_polygon([
                (x, -belly_half, bottom), (x, belly_half, bottom),
                (x, shoulder_half, shoulder), (x, crown_half, roof),
                (x, -crown_half, roof), (x, -shoulder_half, shoulder),
            ], close=True)

        outer_fairing = b.Solid.make_loft(
            [cabin_wire(s) for s in stations], ruled=True)
        # End bulkheads join both skins without closing the service openings.
        inner_x = sorted({x for profile in (inner_sides, inner_roofs, outer_roofs)
                          for x, _ in profile
                          if inner_sides[0][0] <= x <= inner_sides[-1][0]})
        inner_fairing = b.Solid.make_loft(
            [cabin_wire((x, 0, 0, 0), inner=True) for x in inner_x], ruled=True)
        shell_envelopes.append((outer_fairing, inner_fairing))
        return outer_fairing - inner_fairing, max(z for _, z in outer_roofs)

    fairing, roof_z = cabin_shell(stations)
    # The shared outer hull is assembled below before any cavity is cut.

    # Low camera cheeks flank the tall Pi spine. Each recessed pocket has its
    # own pitched coaming, so the wide stereo nose no longer needs a tall full-
    # width canopy or a long prow ahead of the avionics. The inward cheek skin
    # overlaps the spine, creating a continuous monocoque junction on the bed.
    # All camera service boxes clear the inner faces by 2 mm; the lens ports
    # below are the only forward openings.
    cheek_stations = [
        (62.5, 26.0, 14.0, 16.0),
        (65.0, 35.0, 16.8, 27.0),
        (93.5, 35.0, 16.8, 27.0),
        (97.0, 26.0, 14.0, 16.0),
    ]
    cheek_stations = [(x*sx, (w-2*(old_draft-draft)*h)*sy, h, c*sy)
                      for x, w, h, c in cheek_stations]
    cheek, cheek_roof = cabin_shell(cheek_stations)
    roof_z = max(roof_z, cheek_roof)
    # Hollow the joined nose once: shared voids remove doubled internal
    # partitions while the outer cheeks become one continuous faired shell.
    outer_hull, inner_hull = shell_envelopes[0]
    for camera_y in (-28.0*sy, 28.0*sy):
        outer_hull = outer_hull + b.Pos(0, camera_y, 0)*shell_envelopes[1][0]
        inner_hull = inner_hull + b.Pos(0, camera_y, 0)*shell_envelopes[1][1]
    body = body + (outer_hull-inner_hull)
    for camera_y in (-28.0*sy, 28.0*sy):
        # Short first-layer seat rails tie each cheek's end bulkheads together.
        seat = b.Pos(79.75*sx, camera_y, 0) * b.extrude(
            b.Rectangle(33.25*sx, p.payload_rail_width_mm).face(), wall)
        body = body + seat

    # Two first-layer longerons support the recessed payloads, tie the end
    # bulkheads to all four arm chines, and leave underside service access.
    # They also carry battery jolt loads into the central mounting ring.
    rail_len = p.center_plate_len_mm - wall
    rail_x = (stations[0][0] + stations[-1][0])/2
    for rail_y in (-10.0*sy, 10.0*sy):
        rail = b.Pos(rail_x, rail_y, 0) * b.extrude(
            b.Rectangle(rail_len, p.payload_rail_width_mm).face(), wall)
        body = body + rail

    # First-layer diagonal ties brace nose and tail against lateral sway.
    # Their triangular load paths replace shell thickening and sit below the
    # 2 mm service clearance of the recessed battery, Pi, GPS and cameras.
    for root_x, end_x in ((-27.0, -144.0), (27.0, 96.0)):
        for side in (-1, 1):
            ax, ay = root_x*sx, side*27.0*sy
            bx, by = end_x*sx, -side*11.0*sy
            length = math.hypot(bx-ax, by-ay)
            ox = -(by-ay)/length * p.payload_rail_width_mm/2
            oy = (bx-ax)/length * p.payload_rail_width_mm/2
            tie = b.Wire.make_polygon([
                (ax+ox, ay+oy, 0), (bx+ox, by+oy, 0),
                (bx-ox, by-oy, 0), (ax-ox, ay-oy, 0),
            ], close=True)
            body = body + b.Solid.extrude(b.Face(tie), (0, 0, wall))

    # Full-depth annular stack pylons transfer load into the existing crowned
    # arms. The PCB sits 2 mm above them, entirely clear of structural skin.
    sh = p.stack_spacing_mm / 2
    for dx, dy in ((sh, sh), (-sh, sh), (-sh, -sh), (sh, -sh)):
        boss = b.Pos(dx, dy, p.stack_standoff_height_mm/2) * b.Cylinder(
            p.stack_standoff_dia_mm/2, p.stack_standoff_height_mm)
        body = body + boss

    # Teardrop lens ports have a 45-degree roof, so the nose needs no support.
    # Only the sight lines pierce the shell; both camera boards remain inside.
    aperture_r = p.camera_aperture_dia_mm/2
    for camera_y in (-28.0*sy, 28.0*sy):
        port = b.Wire.make_polygon([
            (93.0*sx, camera_y-aperture_r, 9.0),
            (93.0*sx, camera_y-aperture_r, 9.0-aperture_r),
            (93.0*sx, camera_y+aperture_r, 9.0-aperture_r),
            (93.0*sx, camera_y+aperture_r, 9.0),
            (93.0*sx, camera_y, 9.0+aperture_r),
        ], close=True)
        body = body - b.Solid.extrude(b.Face(port), (20.0*sx, 0, 0))

    # motor bolt holes (16x16 M3) + center bore, through each pad
    cut_height = max(roof_z, p.arm_thickness_mm,
                     p.motor_pad_thickness_mm) + 2
    holes = []
    for (mx, my) in p.motor_positions():
        hs = p.motor_hole_spacing_mm / 2
        for dx, dy in ((hs, hs), (-hs, hs), (-hs, -hs), (hs, -hs)):
            holes.append(b.Pos(mx+dx, my+dy, -1) * b.Cylinder(
                p.motor_hole_dia_mm/2, cut_height,
                align=(b.Align.CENTER, b.Align.CENTER, b.Align.MIN)))
        holes.append(b.Pos(mx, my, -1) * b.Cylinder(
            p.motor_center_hole_dia_mm/2, cut_height,
            align=(b.Align.CENTER, b.Align.CENTER, b.Align.MIN)))
    # stack holes (30.5 mm square) in hub
    for dx, dy in ((sh, sh), (-sh, sh), (-sh, -sh), (sh, -sh)):
        holes.append(b.Pos(dx, dy, -1) * b.Cylinder(
            p.stack_hole_dia_mm/2, cut_height,
            align=(b.Align.CENTER, b.Align.CENTER, b.Align.MIN)))
    for h in holes:
        body = body - h

    # Wiring/drain throats open every arm cavity through its first-layer skin.
    # The structural shell remains continuous above each small vertical port;
    # enclosed air pockets no longer create separate internal STL surfaces.
    for mx, my in p.motor_positions():
        x = 45.0
        throat = b.Pos(x, sweep_center(x), -0.2) * b.Cylinder(
            1.6, p.arm_rib_thickness_mm + 0.6,
            align=(b.Align.CENTER, b.Align.CENTER, b.Align.MIN))
        body = body - throat.rotate(b.Axis.Z, math.degrees(math.atan2(my, mx)))
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
