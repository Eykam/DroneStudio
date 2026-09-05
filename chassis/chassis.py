"""Candidate A: swept stereo shoulders with a vaulted, shared avionics nose.

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
    arm_rib_root_mm: float = 19.0
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
        """Faceted lenticular shell: deep belly chines and a broad shoulder."""
        root_blend = max(0.0, min(1.0, (75.0-x)/30.0))
        wall = p.arm_rib_thickness_mm + 0.10*root_blend
        half = width/2
        crown = min(p.arm_crown_width_mm/2, half-1.25*wall)
        keel = max(crown, half-0.22*height)
        # The middle shoulder keeps material far from the lateral bending
        # axis; deep belly and crown folds shorten the perimeter and carry
        # torsional shear as a closed cell. Both slopes print without support.
        points = [(-keel,0),(keel,0),(half,0.24*height),
                  (half,0.66*height),(crown,height),(-crown,height),
                  (-half,0.66*height),(-half,0.24*height)]
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

        # Hollow octagonal spars put depth just beyond the stack ring, where
        # the cantilever begins. A slimmer outer span replaces the old broad
        # shallow tube; its roof descends continuously into the motor fairing.
        # The terminal depth never dips below the nacelle, removing a notch.
        span = profile_end-x0
        # Section-area and biaxial beam-compliance sizing redistributes
        # depth into the outer half-span, where the original taper was soft.
        # Roots flare into the fuselage while axes and terminal heights stay fixed.
        spar_stations = [
            (0.00, 15.05, 22.50),
            (0.10, 15.65, 24.20),
            (0.24, 15.40, 25.85),
            (0.42, 13.50, 22.05),
            (0.62, 11.50, 17.65),
            (0.81, 10.15, 13.25),
            (0.93, 9.20, 10.80),
            (1.00, 9.20, 10.80),
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

        # A pointed wiring gallery cores the falling motor fairing. Excluding
        # the shaft and bolt collars leaves their full 1.2 mm radial walls.
        gallery = []
        for x, height in ((bridge_start+0.1, tip_height),
                          (rib_end, tip_height),
                          (L-center_boss_radius-0.2, 7.2)):
            frac = (x-bridge_start)/(L-bridge_start)
            center = bridge_center*(1-frac)
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
        drain_y = bridge_center*(1-(drain_x-bridge_start)/(L-bridge_start))
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
        (-143.0, 29.4, 13.5, 22.0),
        (-141.0, 30.4, 13.5, 23.0),
        (-122.0, 30.4, 13.5, 23.0),
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
    cheek_stations = [
        (56.0, 20.0, 28.5, 12.0),
        (71.3, 94.2, 28.8, 84.0),
        (86.8, 94.2, 28.8, 84.0),
        (88.5, 89.0, 28.5, 78.0),
    ]
    cheek_stations = [(x*sx, (w-2*(old_draft-draft)*h)*sy, h, c*sy)
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
        (57.5*sx,-13.5*sy,1.5), (86.0*sx,-13.5*sy,1.5),
        (86.0*sx,-13.5*sy,19.0), (71.75*sx,-13.5*sy,35.5),
        (57.5*sx,-13.5*sy,19.0),
    ],close=True)
    spine = fairing-b.Solid.extrude(b.Face(cross_passage),(0,27.0*sy,0))
    shell = (outer_hull-inner_hull)+spine

    # Fold the middle of each battery sidewall into an outward hollow chine.
    # These shallow longitudinal corrugations brace the tall pack well
    # without thickening its skin or encroaching on the service envelope.
    # The steep lower and upper facets grow from the existing wall, so the
    # belt prints in place; its cavity opens directly into the battery bay.
    battery_width = next(w for x,w,h,c in stations if abs(x+105.0*sx)<1e-6)
    zlo, zmid, zhi = 8.0, 15.0, 22.0
    depth = 3.0*sy
    def side_y(z):
        return battery_width/2-draft*z
    lower_slope = (side_y(zmid)+depth-(side_y(zlo)-0.1))/(zmid-zlo)
    upper_slope = ((side_y(zhi)-0.1)-(side_y(zmid)+depth))/(zhi-zmid)
    lower_c = side_y(zlo)-0.1-lower_slope*zlo
    upper_c = side_y(zhi)-0.1-upper_slope*zhi
    lower_ci = lower_c-wall*math.sqrt(1+lower_slope**2)
    upper_ci = upper_c-wall*math.sqrt(1+upper_slope**2)
    inner_peak_z = (upper_ci-lower_ci)/(lower_slope-upper_slope)
    inner_peak_y = lower_slope*inner_peak_z+lower_ci
    for side in (-1,1):
        outer_wire = b.Wire.make_polygon([
            (-104.0*sx, side*(side_y(zlo)-0.1), zlo),
            (-104.0*sx, side*(side_y(zmid)+depth), zmid),
            (-104.0*sx, side*(side_y(zhi)-0.1), zhi),
        ],close=True)
        shell = shell + b.Solid.extrude(b.Face(outer_wire),(47.0*sx,0,0))
        low, high = zlo+wall, zhi-wall
        inner_wire = b.Wire.make_polygon([
            (-104.0*sx+wall,side*(side_y(low)-3.0),low),
            (-104.0*sx+wall,side*(lower_slope*low+lower_ci),low),
            (-104.0*sx+wall,side*inner_peak_y,inner_peak_z),
            (-104.0*sx+wall,side*(upper_slope*high+upper_ci),high),
            (-104.0*sx+wall,side*(side_y(high)-3.0),high),
        ],close=True)
        shell = shell - b.Solid.extrude(b.Face(inner_wire),(47.0*sx-2*wall,0,0))

        # Small pointed vents leave continuous upper coamings, the new
        # folded belt, and generous pillars between every opening. Their
        # pitched heads close above 45 degrees without a horizontal bridge.
        for cx,cz,hw,hh,y0 in [
            (-96.0,31.5,5.5,6.5,18.5),
            (-80.0,31.5,5.5,6.5,18.5),
            (-64.0,31.5,5.5,6.5,18.5),
            (-10.0,36.5,6.0,7.5,20.0),
            (10.0,36.5,6.0,7.5,20.0),
            (49.5,20.0,6.0,9.0,6.5),
            (65.5,20.0,6.0,9.0,6.5),
        ]:
            opening = b.Wire.make_polygon([
                ((cx-hw)*sx,side*y0*sy,cz),
                (cx*sx,side*y0*sy,cz-hh),
                ((cx+hw)*sx,side*y0*sy,cz),
                (cx*sx,side*y0*sy,cz+hh),
            ],close=True)
            shell = shell - b.Solid.extrude(b.Face(opening),(0,side*14.0*sy,0))
    # Boolean the apertures on the shell alone to retain the complete
    # closed arm sections where they join the cabin's lower shoulders.
    body = body + shell
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
    from components import camera_lens_poses as _clp
    for _key, _pose in _clp().items():
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
        throat = b.Pos(x, sweep_center(x), -0.2) * b.Cylinder(
            1.6, p.arm_rib_thickness_mm + 0.6,
            align=(b.Align.CENTER, b.Align.CENTER, b.Align.MIN))
        body = body - throat.rotate(b.Axis.Z, math.degrees(math.atan2(my, mx)))
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
