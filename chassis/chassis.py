"""Forward stereo pockets, blind spar ToF seats and ee-flight v19 service cabin.

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
    arm_rib_thickness_mm: float = 1.35  # normal spar skin, including local recess load paths
    arm_rib_offset_mm: float = 3.3      # retained for parameter-file compatibility
    arm_rib_root_mm: float = 19.75     # 2 mm Pi service gap at the open spar root
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
    # Read placement once: long CAD Booleans must use one coherent layout
    # even if a separate optimization process updates placement.json.
    from components import LIBRARY, placement as _placement, tof_lens_poses, camera_lens_poses
    placements = _placement()
    tof_poses = tof_lens_poses()
    camera_poses = camera_lens_poses()
    def sweep_center(x):
        # Both endpoints remain on the fixed motor radial.  Bowing the middle of
        # every arm produces a subtle pinwheel sweep without moving a motor axis.
        return -p.arm_sweep_mm * math.sin(math.pi * x / p.arm_length_mm)

    def section_wire(x, center, width, height, inner=False):
        """Five-facet closed spar with an unbridged ridge and broad landing keel."""
        root_blend = max(0.0, min(1.0, (75.0-x)/30.0))
        wall = p.arm_rib_thickness_mm + 0.10*root_blend
        half = width/2
        keel = max(p.arm_crown_width_mm/2, 0.52223*half)
        # The broad shoulder moves upward into the lateral load path; two
        # continuous pitched webs close at a ridge instead of a flat crown.
        # That roof needs no internal bridge. Spanwise depth and plan taper
        # recover vertical stiffness, while the broad keel prints on the bed.
        # Lower the shoulder at the shallow motor end so even that roof
        # retains the explicit >45-degree support-free slope constraint.
        shoulder_z = min(0.58484*height,
                         height-p.arm_roof_slope*half)
        points = [(-keel,0),(keel,0),(half,shoulder_z),
                  (0,height),(-half,shoulder_z)]
        if inner:
            # Offset every face in its local normal; the extra 3.5% preserves
            # the minimum gauge through the longitudinal taper and sweep.
            gauge = wall*1.035
            lines = []
            for (y0,z0),(y1,z1) in zip(points,points[1:]+points[:1]):
                dy,dz = y1-y0,z1-z0
                length = math.hypot(dy,dz)
                ny,nz = -dz/length,dy/length
                lines.append((ny,nz,ny*y0+nz*z0+gauge))
            inset = []
            for (ay,az,ac),(by,bz,bc) in zip(lines[-1:]+lines[:-1],lines):
                det = ay*bz-by*az
                inset.append(((ac*bz-bc*az)/det,(ay*bc-by*ac)/det))
            points = inset
        return b.Wire.make_polygon([(x,center+y,z) for y,z in points],close=True)

    arms = []
    arm_cavities = []
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
        # Only a short hub saddle is needed: the closed spar already has a
        # continuous lower skin. Removing the old full-span flat apron leaves
        # an integrated chine instead of a plate edge beside every arm.
        plan_x = [0.0, x0, x0+5.0]
        lower, upper = [], []
        for x in plan_x:
            width = p.arm_root_width_mm*(1-0.10*x/(x0+5.0))
            center = sweep_center(x)
            lower.append((x,center-width/2))
            upper.append((x,center+width/2))
        outline = b.Polyline(*(lower+list(reversed(upper))),close=True)
        outline = b.fillet(outline.vertices(),p.fillet_radius_mm)
        arm = b.extrude(b.make_face(outline),p.body_thickness_mm)

        # A ridge-vault wing narrows sooner outside its broad root shoulder.
        # The deep root stays below the FC; the outboard crown grows only where
        # needed to retain vertical bending stiffness after removing the flat
        # roof flange. All motor axes and terminal load-transfer depths persist.
        span = profile_end-x0
        spar_stations = [
            (0.00, 15.050, 22.500),
            (0.10, 15.805, 24.241),
            (0.24, 17.801, 23.919),
            (0.42, 16.027, 20.921),
            (0.62, 13.472, 18.147),
            (0.81, 10.693, 14.362),
            (0.93, 9.200, 10.800),
            (1.00, 9.200, 10.800),
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
        def tube_center_at(x):
            # The loft uses straight segments between its sweep stations.
            # Every overlapping motor-end face must use that same centerline;
            # evaluating the sine again leaves micron-scale Boolean slivers.
            for left, right in zip(tube_sections, tube_sections[1:]):
                if x <= right[0]:
                    f = (x-left[0])/(right[0]-left[0])
                    return left[1]+f*(right[1]-left[1])
            return tube_sections[-1][1]
        outer = b.Solid.make_loft(
            [section_wire(*section) for section in tube_sections], ruled=True
        )
        cavity = b.Solid.make_loft(
            [section_wire(*section, inner=True) for section in tube_sections],
            ruled=True,
        )
        arm_cavities.append(cavity.rotate(b.Axis.Z, ang))
        arm = arm + (outer - cavity)

        # Close the high-shear motor end of the monocoque with one perimeter-
        # thickness diaphragm.  The span remains hollow and root-accessible for
        # wiring, while eliminating the open-shell notch that concentrated crash
        # stress where the arm flows into the motor nacelle.
        cap_x = rib_end - p.arm_rib_thickness_mm
        end_diaphragm = b.Solid.make_loft([
            section_wire(cap_x, tube_center_at(cap_x),
                         p.arm_width_mm, tip_height),
            section_wire(rib_end, tube_center_at(rib_end),
                         p.arm_width_mm, tip_height),
        ], ruled=True)
        arm = arm + end_diaphragm

        # A cruciform motor mount follows the four bolt load paths instead of
        # carrying a mostly unstressed solid disk.  Circular bosses retain a
        # full printable wall around both the shaft bore and every M3 hole.
        bolt_boss_radius = p.motor_hole_dia_mm / 2 + p.motor_boss_wall_mm
        center_boss_radius = p.motor_center_hole_dia_mm / 2 + p.motor_boss_wall_mm
        spoke_length = 2 * (bolt_radius + bolt_boss_radius)
        # Recess the connecting webs below the annular mounting seats. The
        # four full-height bolt collars and shaft ring locate the motor; these
        # short webs transmit load at the first-layer keel. This takes mass off
        # the arm tips without cutting a lateral slot or an enclosed overhang.
        web_height = max(2*p.arm_rib_thickness_mm,
                         0.54*p.motor_pad_thickness_mm)
        pad = b.extrude(
            b.Rectangle(spoke_length, p.motor_spoke_width_mm).face(),
            web_height,
        )
        pad = pad + b.extrude(
            b.Rectangle(p.motor_spoke_width_mm, spoke_length).face(),
            web_height,
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

        # Continue the spar's pointed crown into a tapering motor nacelle.
        # A pitched five-face exterior replaces the old solid rectangular
        # shoulder: the root ridge becomes a flat annular landing only at the
        # shaft boss. Material stays along the two shear webs and lower keel,
        # reducing motor-end mass without thinning any boss or arm skin.
        bridge_start = rib_end - 2*p.arm_rib_thickness_mm
        bridge_center = tube_center_at(bridge_start)
        def nacelle_center_at(x):
            if x <= rib_end:
                return tube_center_at(x)
            return tube_center_at(rib_end)*(L-x)/(L-rib_end)
        nacelle_sections = []
        for x, height in ((bridge_start, tip_height),
                          (rib_end, tip_height),
                          (L, p.motor_pad_thickness_mm)):
            frac = (x-bridge_start)/(L-bridge_start)
            center = nacelle_center_at(x)
            half_width = p.arm_width_mm/2*(1-frac)+center_boss_radius*frac
            # The first two sections retain a fully pitched crown. At the
            # terminal shaft ring the roof spreads onto the motor seating plane.
            roof_rise = p.arm_roof_slope*half_width*(1.0 if x<=rib_end else 0.0)
            shoulder = height-roof_rise
            nacelle_sections.append(b.Wire.make_polygon([
                (x,center-half_width,0),(x,center+half_width,0),
                (x,center+half_width,shoulder),(x,center,height),
                (x,center-half_width,shoulder),
            ],close=True))
        pad = pad + b.Solid.make_loft(nacelle_sections,ruled=True)

        # A pointed wiring gallery cores the falling motor fairing. Excluding
        # the shaft and bolt collars leaves their full 1.2 mm radial walls.
        gallery = []
        for x, height in ((bridge_start+0.1, tip_height),
                          (rib_end, tip_height),
                          (L-center_boss_radius-0.2, 7.2)):
            frac = (x-bridge_start)/(L-bridge_start)
            center = nacelle_center_at(x)
            gallery.append(b.Wire.make_polygon([
                (x,center-2.7,p.arm_rib_thickness_mm*1.04),
                (x,center+2.7,p.arm_rib_thickness_mm*1.04),
                (x,center,height-p.arm_rib_thickness_mm*1.6),
            ],close=True))
        pocket = b.Solid.make_loft(gallery,ruled=True)
        for bx,radius in ((L-bolt_radius,bolt_boss_radius),(L,center_boss_radius)):
            pocket = pocket - b.Pos(bx,0,-1)*b.Cylinder(radius,tip_height+2,
                align=(b.Align.CENTER,b.Align.CENTER,b.Align.MIN))
        piece = arm + pad
        for void in pocket.solids():
            if void.volume>1.0:
                piece = piece - void
        # A bed-facing throat connects the gallery to the outside for wiring
        # and drainage, avoiding a sealed secondary internal mesh surface.
        drain_x = L-7.6
        drain_y = nacelle_center_at(drain_x)
        piece = piece - b.Pos(drain_x,drain_y,-0.2)*b.Cylinder(
            0.8,p.arm_rib_thickness_mm+0.7,
            align=(b.Align.CENTER,b.Align.CENTER,b.Align.MIN))
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
        (-143.0, 28.6, 13.1, 23.0),
        (-141.0, 29.0, 13.1, 24.0),
        (-122.0, 29.0, 13.1, 24.0),
        (-114.0, 52.0, 41.5, 43.5),
        (-105.0, 50.2, 41.5, 41.7),
        (-56.0, 50.2, 41.5, 41.7),
        (-45.0, 52.0, 41.5, 43.5),
        (-33.0, 56.0, 42.0, 40.0),
        (-24.0, 57.0, cockpit_h, 43.0),
        (24.0, 57.0, cockpit_h, 43.0),
        (38.0, 26.0, 35.6, 16.0),
        (91.8, 26.0, 35.6, 16.0),
        (93.2, 22.0, 34.0, 12.0),
    ]
    # Hold the payload shoulders while pulling the belly chines inward.
    # A wider dorsal opening lowers unused canopy skin above the upright Pi;
    # the stack, battery, and rear fin retain their complete service envelopes.
    old_draft = draft
    draft = 0.035 * p.body_fairing_draft_mm / 4.7
    stations = [(x*sx, (w-2*(old_draft-draft)*h)*sy, h, c*sy)
                for x, w, h, c in stations]
    # Size the cabin from the live PCBA, whose placement z is its bottom.
    # The v19 board at z=41.5 needs x +/-56, y +/-28, z=39.5..65.5
    # service space. The shoulder extends 1.5 mm above that complete box;
    # a normal-gauge pitched coaming leaves an open, printable dorsal hatch.
    board = LIBRARY['fc_esc_stack']
    board_x, board_y, board_z = (v*1000 for v in placements['fc_esc_stack'])
    board_dx, board_dy, board_dz = (v*1000 for v in board.dims_m)
    board_top = board_z+board_dz+2.0
    board_shoulder = board_top+1.5
    board_half_length = board_dx/2+4.0
    board_half_width = abs(board_y)+board_dy/2+2.0
    cabin_width = max(64.2*sy, 2*(board_half_width+draft*board_top
                                  +wall*math.sqrt(1+draft*draft)+0.5))
    cabin_crown = 2*(board_half_width+1.1)
    board_aft, board_front = board_x-board_half_length, board_x+board_half_length
    stations = [s for s in stations if s[0] < board_aft-7 or s[0] > board_front+15]
    stations += [
        (board_aft-7, 48.0*sy, 41.5, 40.0*sy),
        (board_aft, cabin_width, board_shoulder, cabin_crown),
        (board_front, cabin_width, board_shoulder, cabin_crown),
        (board_front+15, 25.0*sy, 35.6, 16.0*sy),
    ]
    stations.sort()
    shell_envelopes = []

    def cabin_shell(stations):
        """Mitered, normal-gauge shell with bottom and dorsal service access."""
        # Each side plane is y + draft*z = breadth/2; each roof plane is
        # z + slope*y = shoulder*(1-slope*draft) + slope*breadth/2.
        # Offset their full 3D normals, including the longitudinal sweep. This
        # preserves the printable gauge through both tapered sides and roof folds.
        side_offsets, roof_offsets, chine_offsets = [], [], []
        # The lower side is a 45-degree keel chine rather than a square skirt.
        # Offset its complete 3D plane just like the upper side and canopy.
        chine_height = 1.65
        chine_inset = 1.65
        chine_slope = chine_inset/chine_height-draft
        for a, z in zip(stations, stations[1:]):
            dx = z[0] - a[0]
            side_gradient = (z[1]-a[1])/(2*dx)
            roof_gradient = (1-slope*draft)*(z[2]-a[2])/dx + slope*side_gradient
            side_offsets.append(wall*math.sqrt(1+draft*draft+side_gradient**2))
            roof_offsets.append(wall*math.sqrt(1+slope*slope+roof_gradient**2))
            chine_offsets.append(wall*math.sqrt(1+chine_slope**2+side_gradient**2))

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
        inner_chines = offset_profile(
            [w/2-chine_inset for _, w, _, _ in stations], chine_offsets)
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
            chine_constant = (interpolate(inner_chines, x) if inner
                              else width/2-chine_inset)
            chine_z = (side_constant-chine_constant)/(draft+chine_slope)
            belly_half = chine_constant+chine_slope*bottom
            chine_half = side_constant-draft*chine_z
            shoulder_half = side_constant-draft*shoulder
            return b.Wire.make_polygon([
                (x, -belly_half, bottom), (x, belly_half, bottom),
                (x, chine_half, chine_z), (x, shoulder_half, shoulder),
                (x, crown_half, roof), (x, -crown_half, roof),
                (x, -shoulder_half, shoulder), (x, -chine_half, chine_z),
            ], close=True)

        outer_fairing = b.Solid.make_loft(
            [cabin_wire(s) for s in stations], ruled=True)
        # End bulkheads join both skins without closing the service openings.
        inner_x = sorted({x for profile in (inner_sides, inner_chines, inner_roofs, outer_roofs)
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
    # Swept camera shoulders grow continuously from the narrow avionics neck.
    # Their diagonal skins replace the blunt transverse cheek bulkhead and
    # carry stereo-nose side loads into the spine as a folded monocoque.
    # The full breadth is reached before either camera's 2 mm service box.
    # The cheek deck rises only as it approaches the camera boards. Its
    # low swept saddle removes the unused full-height wedge behind the
    # stereo pair; the independent central spine still encloses the Pi.
    # Reach the original full section before the cameras' aft service plane.
    # Every side begins on the bed and the roof is pitched, with no bridge.
    # Pull the unoccupied aft wedges into distinct swept cheek pods. The
    # intermediate shoulder wraps the cameras before their service envelope;
    # the central avionics spine still carries the nose longitudinally.
    cheek_stations = [
        (59.0, 20.0, 7.5, 12.0),
        (64.5, 46.0, 12.0, 34.0),
        (68.0, 76.0, 22.0, 64.0),
        (71.3, 94.2, 28.8, 84.0),
        (86.8, 94.2, 28.8, 84.0),
        (88.5, 89.0, 28.5, 78.0),
    ]
    # Pull the unoccupied aft wedges into distinct swept cheek pods. The
    # intermediate shoulder wraps the cameras before their service envelope;
    # the central avionics spine still carries the nose longitudinally.
    # The camera placements advanced 4 mm; translate their complete seat vault.
    cheek_stations = [(x*sx+4.0, (w-2*(old_draft-draft)*h)*sy, h, c*sy)
                      for x, w, h, c in cheek_stations]
    cheek, cheek_roof = cabin_shell(cheek_stations)
    roof_z = max(roof_z, cheek_roof)
    # Hollow the joined nose once: shared voids remove doubled internal
    # partitions while the outer cheeks become one continuous faired shell.
    outer_hull, inner_hull = shell_envelopes[0]
    outer_hull = outer_hull + shell_envelopes[1][0]
    inner_hull = inner_hull + shell_envelopes[1][1]
    # A longer longitudinal vault eliminates doubled spine partitions inside
    # the swept shoulders. Keep its first-layer sill, aft/front piers and a
    # 1.6+ mm crown ligament: the two haunches close at >45 degrees, so the
    # dorsal coaming remains supported while the shared bay opens for service.
    cross_passage = b.Wire.make_polygon([
        (57.5*sx+4.0,-13.5*sy,1.5), (86.0*sx+4.0,-13.5*sy,1.5),
        (86.0*sx+4.0,-13.5*sy,19.0), (71.75*sx+4.0,-13.5*sy,35.5),
        (57.5*sx+4.0,-13.5*sy,19.0),
    ],close=True)
    spine = fairing-b.Solid.extrude(b.Face(cross_passage),(0,27.0*sy,0))
    shell = (outer_hull-inner_hull)+spine

    # Taper the battery's stiffening blisters into the parent side skin.
    # The folded cheek now has pointed fore/aft runouts instead of full-depth
    # transverse end walls. Its hollow upper and lower facets carry pack-bay
    # shear while all of the original 2 mm service space remains available.
    battery_width = next(w for x,w,h,c in stations if abs(x+105.0*sx)<1e-6)
    # Put the battery shear belt lower on the fuselage and deepen its fold.
    # This preserves a continuous tray-to-stack load path while making room
    # for three large pitched service gills and their wider diagonal pillars.
    # The 6 mm rise / 3.6 mm projection is self-supporting on the build plate.
    zlo, zmid, zhi = 4.0, 10.0, 16.0
    def side_y(z):
        return battery_width/2-draft*z
    blister_stations = [(-109.0,0.15),(-98.0,3.6),(-63.0,3.6),(-48.5,0.15)]
    def blister_wire(x, depth, side, inner=False):
        lower_slope = (side_y(zmid)+depth-(side_y(zlo)-0.1))/(zmid-zlo)
        upper_slope = ((side_y(zhi)-0.1)-(side_y(zmid)+depth))/(zhi-zmid)
        lower_c = side_y(zlo)-0.1-lower_slope*zlo
        upper_c = side_y(zhi)-0.1-upper_slope*zhi
        if inner:
            # The 11 mm runout adds an X component to each surface normal.
            # A conservative normal offset keeps the lofted skin >=1.25 mm.
            runout_gradient = 3.45*sy/(11.0*sx)
            lower_ci = lower_c-wall*math.sqrt(1+lower_slope**2+runout_gradient**2)
            upper_ci = upper_c-wall*math.sqrt(1+upper_slope**2+runout_gradient**2)
            peak_z = (upper_ci-lower_ci)/(lower_slope-upper_slope)
            peak_y = lower_slope*peak_z+lower_ci
            low,high = zlo+wall,zhi-wall
            profile = [(side_y(low)-3.0,low),
                       (lower_slope*low+lower_ci,low),(peak_y,peak_z),
                       (upper_slope*high+upper_ci,high),(side_y(high)-3.0,high)]
        else:
            profile = [(side_y(zlo)-0.1,zlo),
                       (side_y(zmid)+depth,zmid),(side_y(zhi)-0.1,zhi)]
        return b.Wire.make_polygon([(x*sx,side*y,z) for y,z in profile],close=True)
    for side in (-1,1):
        outer_blister = b.Solid.make_loft([
            blister_wire(x,d*sy,side) for x,d in blister_stations],ruled=True)
        inner_stations = list(blister_stations)
        # At each pointed end the cavity fades inside the existing sidewall;
        # stop short of the external tip to keep a continuous skin ligament.
        inner_stations[0] = (-109.0+wall/sx,0.15+3.45*wall/(11.0*sx))
        inner_stations[-1] = (-48.5-wall/sx,0.15+3.45*wall/(14.5*sx))
        inner_blister = b.Solid.make_loft([
            blister_wire(x,d*sy,side,True) for x,d in inner_stations],ruled=True)
        shell = (shell+outer_blister)-inner_blister

        # Three elongated gills leave a 2.75 mm minimum dorsal ligament,
        # 2.5 mm end-to-end pillars, and a continuous deep lower chine.
        # The 10.75 mm pitched rise exceeds the longest 10 mm roof run;
        # broader openings replace unstressed skin without thinning a wall.
        for cx,cz,hw,hh,y0 in [
            (-99.0,28.75,9.25,10.75,18.5),
            (-78.0,28.75,9.25,10.75,18.5),
            (-57.0,28.75,9.25,10.75,18.5),
            (-10.0,36.5,6.0,7.5,20.0),
            (10.0,36.5,6.0,7.5,20.0),
            (49.5,20.0,6.0,9.0,6.5),
            (65.5,20.0,6.0,9.0,6.5),
        ]:
            # Swept gills align their diagonal webs with the battery cheek.
            # The widest battery-gill roof run is 10 mm with a 10.75 mm rise.
            skew = 0.75 if cx < -40.0 else 0.0
            opening = b.Wire.make_polygon([
                ((cx-hw)*sx,side*y0*sy,cz),
                ((cx+skew)*sx,side*y0*sy,cz-hh),
                ((cx+hw)*sx,side*y0*sy,cz),
                ((cx-skew)*sx,side*y0*sy,cz+hh),
            ],close=True)
            shell = shell - b.Solid.extrude(b.Face(opening),(0,side*14.0*sy,0))
    # Recessed pointed cheek gills sit behind the camera boards, outside
    # both sight-line pyramids. A continuous sill and dorsal brow frame each
    # opening; the 1.4:1 pitched heads need no bridge or support material.
    # Cutting across the swept wall exposes the common service vault without
    # leaving a doubled partition inside the cheek.
    for side in (-1, 1):
        gill = b.Wire.make_polygon([
            (63.0*sx+4.0,side*14.0*sy,12.5),
            (66.5*sx+4.0,side*14.0*sy,7.5),
            (70.0*sx+4.0,side*14.0*sy,12.5),
            (66.5*sx+4.0,side*14.0*sy,17.5),
        ],close=True)
        shell = shell-b.Solid.extrude(b.Face(gill),(0,side*38.0*sy,0))
    # Recessed ToF carriers follow the actual, axis-aligned board envelopes;
    # only the optical port rotates to the placement's radial bearing.
    # Diagonal seats include a closed, ribbed spar saddle. Their blind service
    # cuts apply to the joined spar and shell and retain a >=1.6 mm floor.
    spar_services = []
    spar_vaults = []
    coaming_trims = []

    def convex_outline(points):
        points = sorted(set(points))
        def cross(a, c, d):
            return (c[0]-a[0])*(d[1]-a[1])-(c[1]-a[1])*(d[0]-a[0])
        halves = []
        for sequence in (points, list(reversed(points))):
            half = []
            for point in sequence:
                while len(half) >= 2 and cross(half[-2], half[-1], point) <= 0:
                    half.pop()
                half.append(point)
            halves.append(half[:-1])
        return halves[0]+halves[1]

    def outset_outline(points, gauge):
        lines = []
        for a, c in zip(points, points[1:]+points[:1]):
            dx, dy = c[0]-a[0], c[1]-a[1]
            length = math.hypot(dx, dy)
            nx, ny = dy/length, -dx/length
            lines.append((nx, ny, nx*a[0]+ny*a[1]+gauge))
        result = []
        for a, c in zip(lines[-1:]+lines[:-1], lines):
            det = a[0]*c[1]-c[0]*a[1]
            result.append(((a[2]*c[1]-c[2]*a[1])/det,
                           (a[0]*c[2]-c[0]*a[2])/det))
        return result

    for key, pos in placements.items():
        if key not in tof_poses:
            continue
        cx, cy, z0 = (v*1000 for v in pos)
        diagonal = abs(cx) > 1 and abs(cy) > 1
        # Construct opposing saddles from the same half-plane to avoid OCCT
        # tolerance slivers at mirrored, coincident spar/corbel intersections.
        half_turn = diagonal and cx < 0
        if half_turn:
            cx,cy = -cx,-cy
        dx, dy, dz = (v*1000 for v in LIBRARY[key.split('#')[0]].dims_m)
        hx, hy = dx/2+2.0, dy/2+2.0
        angle = math.atan2(cy, cx)
        ux, uy = math.cos(angle), math.sin(angle)
        tx, ty = -uy, ux
        front = abs(ux)*hx+abs(uy)*hy
        # A short flat facet faces the sensor even at a diagonal bearing.
        # The complete service rectangle remains inside this convex pocket.
        outline_points = [
            (-hx,-hy), (hx,-hy), (hx,hy), (-hx,hy),
            (front*ux-4.5*tx, front*uy-4.5*ty),
            (front*ux+4.5*tx, front*uy+4.5*ty),
        ]
        shared_payload = ('pi_zero_2w' if key.endswith('#n') else
                          'gps' if key.endswith('#s') else None)
        shared_bb = None
        if shared_payload is not None:
            # Only the oriented envelope is needed; avoid calculating the CAD
            # model's inertia every time a seat is built.
            from components import _apply_orientation
            from pathlib import Path
            import components
            spec = LIBRARY[shared_payload]
            path = Path(components.__file__).parent / spec.step_path
            brep = Path(str(path)+'.brep')
            shape = b.import_brep(str(brep)) if brep.exists() else b.import_step(str(path))
            shape = _apply_orientation(shared_payload, shape)
            bb = shape.bounding_box()
            sp = placements[shared_payload]
            offset = b.Vector(sp[0]*1000-(bb.min.X+bb.max.X)/2,
                              sp[1]*1000-(bb.min.Y+bb.max.Y)/2,
                              sp[2]*1000-bb.min.Z)
            shared_bb = shape.moved(b.Location(offset)).bounding_box()
        if shared_payload == 'gps':
            # The GPS sits below this board. Wrap both service envelopes
            # from the bed instead of bridging a flat cut above the GPS.
            outline_points.extend([
                (x-cx,y-cy)
                for x in (shared_bb.min.X-2,shared_bb.max.X+2)
                for y in (shared_bb.min.Y-2,shared_bb.max.Y+2)
            ])
        footprint = convex_outline(outline_points)
        front = max(x*ux+y*uy for x,y in footprint)
        seat_wall = max(wall, 2.2) if diagonal else wall
        outer_footprint = outset_outline(footprint, seat_wall)
        def pocket_wire(points, z):
            return b.Wire.make_polygon([(cx+x,cy+y,z) for x,y in points], close=True)
        top = z0+dz+3.5
        outer_seat = b.Solid.extrude(b.Face(pocket_wire(outer_footprint,0)), (0,0,top))
        inner_seat = b.Solid.extrude(b.Face(pocket_wire(footprint,-0.2)), (0,0,top+0.4))
        mount = outer_seat-inner_seat
        if diagonal:
            # Two pitched corbels close beneath a 1.6 mm blind floor.
            # The tiny final ridge spans <0.5 mm; each underside rises more
            # than its horizontal run. The vertical perimeter is also the
            # reinforcing rib that transfers seat loads into the spar webs.
            floor_z = z0-2.0
            floor_gauge = 1.6
            apex_z = floor_z-floor_gauge
            # Intersect a constant-pitch vault with the irregular footprint.
            # This keeps a 1.1:1 underside everywhere without filling the
            # rectangular seat to the depth required only at its front ears.
            vault_half = (apex_z+0.2)/1.1
            vault_x = max(abs(x*ux+y*uy) for x,y in outer_footprint)+2
            vx,vy = cx-vault_x*ux,cy-vault_x*uy
            vault = b.Wire.make_polygon([
                (vx-tx*vault_half,vy-ty*vault_half,-0.2),
                (vx+tx*vault_half,vy+ty*vault_half,-0.2),
                (vx,vy,apex_z),
            ],close=True)
            # Align the ridge with the spar so wires can enter through both
            # end ribs while the transverse roof stays steeper than 45 deg.
            vault_tool = b.Solid.extrude(b.Face(vault),(2*vault_x*ux,2*vault_x*uy,0))
            underside = inner_seat & vault_tool
            saddle = b.Solid.extrude(b.Face(pocket_wire(outer_footprint,0)),
                                     (0,0,floor_z))-underside
            # The existing gallery is re-opened through this pitched vault
            # after the shell and spars join; no flat internal ceiling is cut.
            zone = b.Solid.extrude(b.Face(pocket_wire(outer_footprint,-.2)),(0,0,100))
            if half_turn:
                zone = zone.rotate(b.Axis.Z,180)
                vault_tool = vault_tool.rotate(b.Axis.Z,180)
            spar_vaults.append((zone,vault_tool))
            mount = mount+saddle
        # Two ledges finish 2 mm below the board. Their 2.2:2 undersides
        # grow from the vertical walls without a suspended floor or bridge.
        for side in (-1,1):
            if shared_payload == 'gps' or diagonal:
                # The stacked service volumes leave no space for a floor;
                # this shared pocket retains bottom access like the cameras.
                continue
            ledge = b.Wire.make_polygon([
                (cx+side*hx,cy-hy,z0-4.2),
                (cx+side*(hx+0.15),cy-hy,z0-2.0),
                (cx+side*(hx-2.0),cy-hy,z0-2.0),
            ],close=True)
            mount = mount+b.Solid.extrude(b.Face(ledge),(0,2*hy,0))
        # Pointed skirt windows save material below the board recess. Both
        # the skirt-window heads and sensor-port head rise faster than 1:1.
        for a,c in zip(footprint,footprint[1:]+footprint[:1]):
            ex,ey = c[0]-a[0],c[1]-a[1]
            length = math.hypot(ex,ey)
            if length < 9.0:
                continue
            ex,ey = ex/length,ey/length
            nx,ny = ey,-ex
            hw = min(4.5,(length-3.0)/2)
            if diagonal:
                hw = min(hw,(z0-3.6-5.5)/1.1)
            x,y = cx+(a[0]+c[0])/2,cy+(a[1]+c[1])/2
            window = b.Wire.make_polygon([
                (x-ex*hw-nx*.2,y-ey*hw-ny*.2,5.5),
                (x-nx*.2,y-ny*.2,5.5-hw*1.1),
                (x+ex*hw-nx*.2,y+ey*hw-ny*.2,5.5),
                (x-nx*.2,y-ny*.2,5.5+hw*1.1),
            ],close=True)
            mount = mount-b.Solid.extrude(b.Face(window),(nx*(wall+1),ny*(wall+1),0))
        # Preserve the original wiring galleries through the added coaming;
        # a transverse mount wall must not seal an existing hollow spar.
        for arm_cavity in arm_cavities:
            if not diagonal:
                mount = mount-arm_cavity
        # The nose and tail coamings share their bays with the Pi and GPS.
        # Keep those existing payloads' complete 2 mm service envelopes free.
        if shared_payload is not None:
            bb = shared_bb
            size = bb.max-bb.min+b.Vector(4,4,4)
            service_box = b.Pos(bb.min.X-2,bb.min.Y-2,bb.min.Z-2)*b.Box(
                size.X,size.Y,size.Z,align=(b.Align.MIN,b.Align.MIN,b.Align.MIN))
            mount = mount-service_box
        if not mount.is_valid:
            raise ValueError(f'Invalid ToF seat: {key}')
        if half_turn:
            mount = mount.rotate(b.Axis.Z,180)
        shell = shell+mount
        # Open upward for installation; this cut clears existing bulkheads
        # as well as the new coaming, with no enclosed horizontal ceiling.
        service = b.Solid.extrude(b.Face(pocket_wire(footprint,z0-2.0)),
                                  (0,0,max(100.0,roof_z+10)))
        if half_turn:
            service = service.rotate(b.Axis.Z,180)
        shell = shell-service
        if diagonal:
            spar_services.append(service)
            # Above the seat rim, remove the thin remnant of the cabin skin
            # behind the service cut. The full 1.6 mm seat wall and floor
            # remain below this opening; no doubled foil runs up the canopy.
            trim = b.Pos(cx,cy,top)*b.Box(2*hx,2*hy+4,100,
                align=(b.Align.CENTER,b.Align.CENTER,b.Align.MIN))
            if half_turn:
                trim = trim.rotate(b.Axis.Z,180)
            coaming_trims.append(trim)
        oz = tof_poses[key]['origin_m'][2]*1000
        px,py = cx+ux*(front-.2),cy+uy*(front-.2)
        hw = 3.0
        port = b.Wire.make_polygon([
            (px-tx*hw,py-ty*hw,oz), (px,py,oz-hw*1.1),
            (px+tx*hw,py+ty*hw,oz), (px,py,oz+hw*1.1),
        ],close=True)
        optical_cut = b.Solid.extrude(b.Face(port),(ux*(seat_wall+1),uy*(seat_wall+1),0))
        if half_turn:
            optical_cut = optical_cut.rotate(b.Axis.Z,180)
        shell = shell-optical_cut
        if abs(cx) < 1e-6 and abs(cy) > 1:
            # Bed-founded ties attach the two lateral seats to the cabin.
            side = 1 if cy > 0 else -1
            for sign in (-1,1):
                ax,ay = sign*8.0*sy,side*24.0*sy
                bx,by = cx+sign*(hx+wall/2),cy-side*(hy+wall/2)
                length = math.hypot(bx-ax,by-ay)
                ox,oy = -(by-ay)/length*p.payload_rail_width_mm/2,(bx-ax)/length*p.payload_rail_width_mm/2
                tie = b.Wire.make_polygon([
                    (ax+ox,ay+oy,0),(bx+ox,by+oy,0),
                    (bx-ox,by-oy,0),(ax-ox,ay-oy,0),
                ],close=True)
                shell = shell+b.Solid.extrude(b.Face(tie),(0,0,max(wall,z0-4.2)))

    # Boolean the apertures on the shell alone to retain the complete
    # closed arm sections where they join the cabin's lower shoulders.
    body = body + shell
    # The enlarged cabin crosses the old wiring galleries. Remove those
    # internal partitions from the joined hull so the existing drains still
    # reach the complete span. Restrict this to the cabin region, preserving
    # the original high-shear motor-end diaphragms and nacelle galleries.
    gallery_limit = b.Cylinder(95,50,
        align=(b.Align.CENTER,b.Align.CENTER,b.Align.MIN))
    for cavity in arm_cavities:
        passage = cavity & gallery_limit
        for zone,vault in spar_vaults:
            # Within a recess, the wiring roof follows the printable vault
            # beneath its 1.6 mm floor. Elsewhere retain the full spar cavity.
            local = passage & zone
            if local is not None and local.volume > 1e-7:
                passage = (passage-zone)+(local & vault)
        body = body-passage
    for service in spar_services:
        body = body-service
    # A single upward-open service cut clears any overlapping sensor coaming
    # or older internal partition from the real board's complete 2 mm box.
    board_service = b.Pos(board_x,board_y,board_z-2.0)*b.Box(
        board_dx+4,board_dy+4,max(100.0,roof_z-board_z+4),
        align=(b.Align.CENTER,b.Align.CENTER,b.Align.MIN))
    body = body-board_service
    for trim in coaming_trims:
        body = body-trim
    # Keep the complete camera service rectangles free of the expanded
    # cabin/cheek junction. Both pockets remain open for vertical insertion.
    from components import _apply_orientation
    for key,pos in placements.items():
        if not key.startswith('pi_camera_3#'):
            continue
        from pathlib import Path
        import components
        path = Path(components.__file__).parent/LIBRARY['pi_camera_3'].step_path
        brep = Path(str(path)+'.brep')
        camera = b.import_brep(str(brep)) if brep.exists() else b.import_step(str(path))
        camera = _apply_orientation('pi_camera_3',camera)
        bb = camera.bounding_box()
        camera_service = b.Pos(pos[0]*1000,pos[1]*1000,pos[2]*1000-2)*b.Box(
            bb.size.X+4,bb.size.Y+4,100,
            align=(b.Align.CENTER,b.Align.CENTER,b.Align.MIN))
        body = body-camera_service
    # Two first-layer longerons support the recessed payloads, tie the end
    # bulkheads to all four arm chines, and leave underside service access.
    # They also carry battery jolt loads into the central mounting ring.
    rail_len = stations[-1][0] - stations[0][0] - wall
    rail_x = (stations[0][0] + stations[-1][0])/2
    for rail_y in (-10.0*sy, 10.0*sy):
        rail = b.Pos(rail_x, rail_y, 0) * b.extrude(
            b.Rectangle(rail_len, p.payload_rail_width_mm).face(), wall)
        body = body + rail

    # First-layer diagonal ties brace nose and tail against lateral sway.
    # Their triangular load paths replace shell thickening and sit below the
    # 2 mm service clearance of the recessed battery, Pi, GPS and cameras.
    for root_x, end_x in ((-27.0, -142.0), (27.0, 92.0)):
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

    # Camera sight lines: one FOV-pyramid void per camera, apex at the real lens
    # point (components.camera_lens_poses, boards mounted lens-forward +X).
    # Half-angles = Camera Module 3 spec (66.3h x 41.6v deg) + 1 deg margin, so no
    # frame material sits inside the field of view - evaluate.py:check_camera_fov
    # gates exactly this. Pyramid ceiling slopes down at ~22 deg: printable, no supports.
    for _key, _pose in camera_poses.items():
        _ox, _oy, _oz = (v * 1000 for v in _pose["origin_m"])
        _hh = math.tan(math.radians(_pose["hfov_deg"] / 2 + 1.0))
        _vh = math.tan(math.radians(_pose["vfov_deg"] / 2 + 1.0))
        def _fov_rect(_x, _ox=_ox, _oy=_oy, _oz=_oz, _hh=_hh, _vh=_vh):
            _w = _hh * (_x - _ox) + 0.5
            _h = _vh * (_x - _ox) + 0.5
            return b.Wire.make_polygon([
                (_x, _oy - _w, _oz - _h), (_x, _oy + _w, _oz - _h),
                (_x, _oy + _w, _oz + _h), (_x, _oy - _w, _oz + _h),
            ], close=True)
        body = body - b.Solid.make_loft([_fov_rect(_ox + 0.5), _fov_rect(125.0)])

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
        ang = math.degrees(math.atan2(my,mx))
        # Bed-founded annular ribs round out the high-shear drain junction;
        # the bore stays open through their full height into the gallery.
        drain_rib = b.Pos(x,sweep_center(x),0)*b.Cylinder(3.2,3.2,
            align=(b.Align.CENTER,b.Align.CENTER,b.Align.MIN))
        body = body+drain_rib.rotate(b.Axis.Z,ang)
        throat = b.Pos(x, sweep_center(x), -0.2) * b.Cylinder(
            1.6, max(p.arm_rib_thickness_mm + 0.6,3.6),
            align=(b.Align.CENTER, b.Align.CENTER, b.Align.MIN))
        body = body - throat.rotate(b.Axis.Z,ang)
    # Mitered ribs can trap tiny closed air wedges where three skins meet.
    # Fill only those sub-150 mm3 wedges; the open wiring galleries remain
    # hollow. This also removes fragile internal slivers from the print.
    for boundary in list(body.shells()):
        enclosed = b.Solid(boundary)
        if 0 < enclosed.volume < 150:
            body = body+enclosed
    if not body.is_valid or len(body.solids()) != 1:
        raise ValueError('The chassis must be one valid solid')
    body = b.Part(children=body.solids())
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
