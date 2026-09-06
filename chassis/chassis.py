"""Chevron-pier cabin with close-pitched battery gills and a continuous sill.

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
    body_thickness_mm: float = 1.22     # structural skin; all faces use normal offsets
    arm_rib_thickness_mm: float = 1.35  # normal spar skin, including local recess load paths
    arm_rib_offset_mm: float = 3.3      # retained for parameter-file compatibility
    arm_rib_root_mm: float = 19.75     # wiring access at the open spar root
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

def _tof_seat_rear_pads(placements, arm_envelopes):
    """Internal radius fillets on the NE/SW seats' rear service corners."""
    from components import LIBRARY
    pads = None
    radius, overlap = 1.8, 0.2
    for key, pos in placements.items():
        if not key.startswith('vl53l9cx_breakout#'):
            continue
        cx, cy, z0 = (v*1000 for v in pos)
        if cx*cy <= 1:
            continue
        dx, dy, dz = (v*1000 for v in LIBRARY[key.split('#')[0]].dims_m)
        sx, sy = math.copysign(1,cx), math.copysign(1,cy)
        rx, ry = cx-sx*(dx/2+2), cy-sy*(dy/2+2)
        bottom = z0-2.1  # overlap the existing 1.6 mm blind seat floor
        def point(u,v):
            return (rx+sx*u,ry+sy*v,bottom)
        a,c,d = point(-overlap,-overlap),point(radius,-overlap),point(radius,0)
        f,g = point(0,radius),point(-overlap,radius)
        mid = radius*(1-1/math.sqrt(2))
        wire = b.Wire([
            b.Edge.make_line(a,c),b.Edge.make_line(c,d),
            b.Edge.make_three_point_arc(d,point(mid,mid),f),
            b.Edge.make_line(f,g),b.Edge.make_line(g,a),
        ])
        # The 1.8 mm tangent radius rounds only the empty service corner.
        # Minimum separation from the real carrier corner exceeds 2.0 mm.
        # The seat floor supports the fillet's entire underside.
        zone = b.Solid.extrude(b.Face(wire),(0,0,30-bottom))
        # Small bed-founded pads close the thin rear wall/floor intersection
        # where the same seats join the spar. Keep both the radius fillet
        # and this root junction reinforcement within the old spar envelope.
        root_pad = b.Pos(cx-sx*0.6,ry-sy*2.2,0)*b.Box(3.2,2.6,3.2,
            align=(b.Align.CENTER,b.Align.CENTER,b.Align.MIN))
        zone = zone+root_pad
        for envelope in arm_envelopes:
            pad = zone & envelope
            if pad is None or pad.volume < 1e-7:
                continue
            pads = pad if pads is None else pads+pad
    return pads

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
        # Bend the outboard span around the diagonal optical window. The
        # root and motor axes retain the baseline centerline; an uncut closed
        # spar now passes beside the beam instead of across its lower edge.
        bypass = (15.0*math.sin(math.pi*(x-48.0)/84.0)**2
                  if 48.0 < x < 132.0 else 0.0)
        # The corrected 12.83 x 6.10 mm module window reaches slightly
        # farther down and sideways than the earlier optical package. A
        # local 1.9 mm outward bow carries the closed spar past its lower
        # corner; no arm is slit and the main-shell aperture stays minimal.
        window_bypass = (1.9*math.sin(math.pi*(x-101.0)/30.0)**2
                         if 101.0 < x < 131.0 else 0.0)
        return (-p.arm_sweep_mm*math.sin(math.pi*x/p.arm_length_mm)
                -bypass-window_bypass)

    def spar_profile(x, width, height):
        """Eight-facet closed section with a broad keel and pitched crown."""
        transition = max(0.0,min(1.0,(x-75.0)/23.0))
        half = width/2
        # A short lower chine moves the side webs outward sooner, putting
        # material on useful bending flanges rather than along a long taper.
        # The narrower plan retains lateral stiffness through its broad keel.
        keel = max(p.arm_crown_width_mm/2, 0.80*half)
        crown = 1.6-0.4*transition
        chine = 0.15*height
        shoulder = min(0.72*height, height-p.arm_roof_slope*(half-crown))
        return [(-keel,0),(keel,0),(half,chine),(half,shoulder),
                (crown,height),(-crown,height),(-half,shoulder),(-half,chine)]

    def section_wire(x, center, width, height, inner=False):
        """Offset swept spar faces in 3D, including their spanwise gradients."""
        points = spar_profile(x,width,height)
        if inner:
            root_blend = max(0.0,min(1.0,(75.0-x)/30.0))
            wall = max(1.22,p.arm_rib_thickness_mm-0.13)+0.10*root_blend
            # The section's YZ normal alone underestimates wall thickness on
            # the optical bypass. Include both adjacent loft spans and both
            # endpoints of each face, then miter those true normal offsets.
            # Only swept/tapered faces receive this extra material; the keel
            # and narrow crown keep their minimum printable normal gauge.
            neighbors = []
            for left,right in zip(tube_sections,tube_sections[1:]):
                if left[0]-1e-7 <= x <= right[0]+1e-7:
                    neighbors.extend(n for n in (left,right) if abs(n[0]-x)>1e-7)
            lines = []
            for index,((y0,z0),(y1,z1)) in enumerate(zip(points,points[1:]+points[:1])):
                dy,dz = y1-y0,z1-z0
                length = math.hypot(dy,dz)
                ny,nz = -dz/length,dy/length
                gradient = 0.0
                for nx,ncenter,nwidth,nheight in neighbors:
                    other = spar_profile(nx,nwidth,nheight)
                    for j in (index,(index+1)%len(points)):
                        delta_y = ncenter+other[j][0]-center-points[j][0]
                        delta_z = other[j][1]-points[j][1]
                        gradient = max(gradient,abs((ny*delta_y+nz*delta_z)/(nx-x)))
                gauge = wall*max(1.035,1.01*math.sqrt(1+gradient*gradient))
                lines.append((ny,nz,ny*y0+nz*z0+gauge))
            inset = []
            for (ay,az,ac),(by,bz,bc) in zip(lines[-1:]+lines[:-1],lines):
                det = ay*bz-by*az
                inset.append(((ac*bz-bc*az)/det,(ay*bc-by*ac)/det))
            points = inset
        return b.Wire.make_polygon([(x,center+y,z) for y,z in points],close=True)

    arms = []
    arm_cavities = []
    arm_envelopes = []
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
            (0.50, 15.005, 19.811),
            (0.62, 13.472, 18.147),
            (0.71, 12.156, 16.354),
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
            # Recover bending stiffness through section depth at the root
            # and the optical bypass, tapering back to the original shallow
            # outer span before it enters the bottom of the optical window.
            height *= 1.0+0.16*max(0.0,min(1.0,(0.70-frac)/0.20))
            # Trade a little apex height for a broader load-bearing crown;
            # raised shoulders recover stiffness while reducing surface area.
            height *= 0.95
            free_span = max(0.0,min(1.0,(x-75.0)/23.0))
            width *= 1.0-0.04*free_span
            # Redistribute width into depth through the loaded span, then
            # return smoothly to the existing motor nacelle. The optical
            # bypass and all four motor positions keep their original axes.
            taper = max(0.0,min(1.0,(132.0-x)/22.0))
            width *= 1.0-0.18*taper
            height *= 1.0+0.04*max(0.0,min(1.0,(0.94-frac)/0.20))
            crest = max(0.0,min(1.0,(x-53.0)/14.0,(110.0-x)/22.0))
            height += 2.6*crest
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
        arm_envelopes.append(outer.rotate(b.Axis.Z, ang))
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
    # crowned junction, and the stereo pair shares the low forward bay.
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
        (38.0, 26.0, 22.5, 16.0),
        (91.8, 26.0, 22.5, 16.0),
        (93.2, 22.0, 22.5, 12.0),
    ]
    # Hold the payload shoulders while pulling the belly chines inward.
    # A wider dorsal opening lowers unused canopy skin above the forward electronics;
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
        (board_front+15, 25.0*sy, 22.5, 16.0*sy),
    ]
    # End the upper cabin at the ee-flight front service wall. The former
    # longitudinal Pi bay is completely absent; the low common shell carries
    # the cameras and forward ToF seat.
    stations = [station for station in stations if station[0] <= board_front]
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

    # One low perimeter encloses all eight carriers, the GPS, battery and
    # stereo nose. The side facets continue between sensor bearings; no
    # sensor has an external housing or a carrier-sized outside recess.
    # The tall cabin above this shared lower vault is only the ee-flight bay.
    cheek_stations = [
        (63.0, 20.0, 7.5, 12.0),
        (68.5, 46.0, 12.0, 34.0),
        (72.0, 76.0, 22.0, 64.0),
        (75.3, 94.2, 28.8, 84.0),
        (90.8, 94.2, 28.8, 84.0),
        (92.5, 89.0, 28.5, 78.0),
    ]
    cheek_stations = [(x*sx, (w-2*(old_draft-draft)*h)*sy, h, c*sy)
                      for x,w,h,c in cheek_stations]
    _, cheek_roof = cabin_shell(cheek_stations)
    roof_z = max(roof_z, cheek_roof)
    spar_services, spar_vaults, optical_cuts = [], [], []

    def convex_outline(points):
        points = sorted(set(points))
        def cross(a, c, d):
            return (c[0]-a[0])*(d[1]-a[1])-(c[1]-a[1])*(d[0]-a[0])
        halves = []
        for sequence in (points, list(reversed(points))):
            half = []
            for point in sequence:
                while len(half) >= 2 and cross(half[-2], half[-1], point) <= 1e-8:
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

    # The diagonal internal guide ribs retain the original spar load path.
    # They terminate at the common shell rim, wholly inside its outline.
    # A continuous folded plan connects the cardinal and diagonal optical
    # faces. 1.22 mm normal wall offsets preserve the gauge at every miter.
    # The diagonals have x +/- y = 114 mm and a flat 18.4 mm optical face.
    # Front and rear terminate beyond the complete +2 mm carrier envelopes.
    perimeter = [
        (-143.5,-13.5), (-130.0,-13.5), (-108.0,-23.0),
        (-64.5,-49.5), (-51.5,-62.5), (-14.0,-59.25),
        (14.0,-59.25), (51.5,-62.5), (64.5,-49.5),
        (75.0,-43.5), (91.0,-43.5), (101.5,-12.0),
        (101.5,12.0), (91.0,43.5), (75.0,43.5),
        (64.5,49.5), (51.5,62.5), (14.0,59.25),
        (-14.0,59.25), (-51.5,62.5), (-64.5,49.5),
        (-108.0,23.0), (-130.0,13.5), (-143.5,13.5),
    ]
    perimeter = [(x*sx,y*sy) for x,y in perimeter]
    def plan_prism(points, bottom, height):
        wire = b.Wire.make_polygon([(x,y,bottom) for x,y in points],close=True)
        return b.Solid.extrude(b.Face(wire),(0,0,height))
    lower_outer = plan_prism(perimeter,0,22.5)
    lower_inner = plan_prism(outset_outline(perimeter,-wall),-.2,23.2)
    outer_hull, inner_hull = shell_envelopes[0]
    for outer,inner in shell_envelopes[1:]:
        outer_hull,inner_hull = outer_hull+outer,inner_hull+inner
    # Hollow the union once so overlapping shells leave no internal fairing.
    shell = (outer_hull+lower_outer)-(inner_hull+lower_inner)
    # Bed-founded sections of the inner cabin wall carry the upper board
    # and battery coamings into the arm roots and lower longerons. Keep only
    # these narrow piers instead of a full-height doubled inner fuselage.
    for pier_x in (-100.0,-78.0,-25.0,25.0):
        pier_width = 4.0 if pier_x < -50 else 6.0
        zone = b.Pos(pier_x,0,0)*b.Box(pier_width,100,24.5,
            align=(b.Align.CENTER,b.Align.CENTER,b.Align.MIN))
        shell = shell+(fairing & zone)
    # Reinforce the inside of the steep battery-to-board shoulder. Its
    # external surface stays on the original hull; the local thicker miter
    # avoids a fragile acute seam where the two side planes meet the roof.
    band = b.Pos(board_aft-3.5,0,43.0)*b.Box(9.0,90,26.0,
        align=(b.Align.CENTER,b.Align.CENTER,b.Align.MIN))
    shoulder_void = (inner_hull.moved(b.Pos(0,.6,0)) &
                     inner_hull.moved(b.Pos(0,-.6,0)))
    shell = shell+((outer_hull-shoulder_void) & band)
    # A four-millimetre lap backs the acute aft battery roof fold, where
    # crash bending concentrates stress. Clip it to the existing hull and
    # keep it above the complete battery service box; no exterior grows.
    aft_lap_zone = b.Pos(-105.0*sx,0,39.6)*b.Box(4.0,60.0,6.0,
        align=(b.Align.CENTER,b.Align.CENTER,b.Align.MIN))
    aft_lap_void = (inner_hull.moved(b.Pos(0,0.5,0)) &
                    inner_hull.moved(b.Pos(0,-0.5,0)))
    shell = shell+((outer_hull-aft_lap_void) & aft_lap_zone)
    # Three close-pitched gills follow the recessed battery. Their short
    # roofs remove unused upper panel area while keeping the continuous
    # lower sill, aft fold lap, and battery-to-cabin shoulder intact.
    for side in (-1,1):
        for cx in (-101.5,-82.5,-63.5):
            gill = b.Wire.make_polygon([
                (cx-8.0,side*18.5,25.0), (cx+8.0,side*18.5,25.0),
                (cx+9.0,side*18.5,30.1), (cx+1.0,side*18.5,39.7),
                (cx-7.0,side*18.5,30.1),
            ],close=True)
            shell = shell-b.Solid.extrude(b.Face(gill),(0,side*55,0))

    # Six raked arches put the coaming material into opposed diagonal piers.
    # Their chevron pattern braces the dorsal belt longitudinally; the short
    # pitched roofs close without supports. All cuts stay above the common
    # sensor shell, preserving each internal seat and its small optical port.
    # Paired piers keep at least 2.8 mm in-plane width and 1.22 mm normal skin.
    for side in (-1,1):
        for vent_x in (-45.5,-27.3,-9.1,9.1,27.3,45.5):
            rake = -1.4 if vent_x < 0 else 1.4
            opening = b.Wire.make_polygon([
                (vent_x-7.7,side*20,24.8), (vent_x+7.7,side*20,24.8),
                (vent_x+7.7+rake,side*20,55.5),
                (vent_x+rake,side*20,64.8),
                (vent_x-7.7+rake,side*20,55.5),
            ],close=True)
            shell = shell-b.Solid.extrude(b.Face(opening),(0,side*20,0))
    # Four matching peaked windows complete the light, continuous coaming
    # around the forward bulkhead; narrow piers link its sill and roof belt.
    for vent_y in (-21.0,-7.0,7.0,21.0):
        opening = b.Wire.make_polygon([
            (board_front-2,vent_y-5.6,28.0),
            (board_front-2,vent_y+5.6,28.0),
            (board_front-2,vent_y+5.6,57.0),
            (board_front-2,vent_y,64.0),
            (board_front-2,vent_y-5.6,57.0),
        ],close=True)
        shell = shell-b.Solid.extrude(b.Face(opening),(4,0,0))

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
        shared_payload = 'gps' if key.endswith('#s') else None
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
        top = z0+dz+3.5 if diagonal else z0-2.0
        outer_seat = b.Solid.extrude(b.Face(pocket_wire(outer_footprint,0)), (0,0,top))
        inner_seat = b.Solid.extrude(b.Face(pocket_wire(footprint,-0.2)), (0,0,top+0.4))
        mount = outer_seat-inner_seat
        if diagonal:
            # The carrier needs a thin internal guide lip, not a 2.2 mm
            # wall around every unloaded edge. Retain the full-gauge ribs
            # wherever the seat intersects a spar, and keep the original
            # blind floor and printable vault below the carrier unchanged.
            guide_footprint = outset_outline(footprint, wall)
            guide = b.Solid.extrude(b.Face(pocket_wire(guide_footprint,0)),
                                   (0,0,top))-inner_seat
            for envelope in arm_envelopes:
                rib = mount & envelope
                if rib is not None and rib.volume > 1e-7:
                    guide = guide+rib
            mount = guide
        if not diagonal and shared_payload is None and not key.endswith('#n'):
            # The lateral carriers need only the two PCB-edge rails and their
            # pitched ledges. The forward seat retains its side braces to
            # anchor the nose longerons and prevent a low-frequency free tip.
            rail_zones = None
            for sign in (-1,1):
                zone = b.Pos(cx+sign*(hx+wall/2),cy,0)*b.Box(
                    wall+0.2,2*(hy+wall+1),top+1,
                    align=(b.Align.CENTER,b.Align.CENTER,b.Align.MIN))
                rail_zones = zone if rail_zones is None else rail_zones+zone
            mount = mount & rail_zones
        if diagonal:
            # Three parallel underside vaults hollow the wide sensor-seat
            # haunches. Two 1.3 mm webs rise from the print bed and support
            # every valley, leaving a continuous 1.6 mm blind carrier floor.
            # The full peripheral spar ribs and upper carrier guides persist.
            floor_z = z0-2.0
            floor_gauge = 1.6
            apex_z = floor_z-floor_gauge
            vault_half = (apex_z+0.2)/1.1
            vault_x = max(abs(x*ux+y*uy) for x,y in outer_footprint)+2
            vx,vy = cx-vault_x*ux,cy-vault_x*uy
            vault_tool = None
            for center_t,t0,t1 in ((-8.0,-40.0,-4.65),
                                    (0.0,-3.35,3.35),
                                    (8.0,4.65,40.0)):
                vault = b.Wire.make_polygon([
                    (vx+tx*(center_t-vault_half),vy+ty*(center_t-vault_half),-.2),
                    (vx+tx*(center_t+vault_half),vy+ty*(center_t+vault_half),-.2),
                    (vx+tx*center_t,vy+ty*center_t,apex_z),
                ],close=True)
                chamber = b.Solid.extrude(b.Face(vault),(2*vault_x*ux,2*vault_x*uy,0))
                divider = b.Wire.make_polygon([
                    (vx+tx*t0,vy+ty*t0,-.3),(vx+tx*t1,vy+ty*t1,-.3),
                    (vx+tx*t1,vy+ty*t1,30),(vx+tx*t0,vy+ty*t0,30),
                ],close=True)
                chamber = chamber & b.Solid.extrude(b.Face(divider),(2*vault_x*ux,2*vault_x*uy,0))
                vault_tool = chamber if vault_tool is None else vault_tool+chamber
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
        # Preserve the original wiring galleries through the added coaming;
        # a transverse mount wall must not seal an existing hollow spar.
        for arm_cavity in arm_cavities:
            if not diagonal:
                mount = mount-arm_cavity
        # The tail seat shares its bay with the GPS.
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
        mount = mount & lower_outer
        shell = shell+mount
        # The carrier service void is internal. Diagonal seats retain their
        # vaulted spar floor while freeing the entire carrier +2 mm envelope.
        service = b.Solid.extrude(b.Face(pocket_wire(footprint,z0-2.0)),
                                  (0,0,100.0))
        if half_turn:
            service = service.rotate(b.Axis.Z,180)
        spar_services.append(service)
        # Only 13.9 x 7.2 mm reaches the outer surface: the VL53L9CX body
        # 12.83 x 6.10 mm (ST DS14879 Rev 7 Fig 23, +tol) plus 1 mm total
        # clearance. Supersedes stale 12.1 x 5.1 package metadata. The 2.45 mm top ligament
        # and 1.22 mm parent wall remain continuous around the optical rim.
        oz = tof_poses[key]['origin_m'][2]*1000
        hw,hh = 13.9/2,7.2/2
        port = b.Wire.make_polygon([
            (cx-tx*hw,cy-ty*hw,oz-hh), (cx+tx*hw,cy+ty*hw,oz-hh),
            (cx+tx*hw,cy+ty*hw,oz+hh), (cx-tx*hw,cy-ty*hw,oz+hh),
        ],close=True)
        # Stop at this main-shell facet. A long radial tool would also cut
        # the structural spar outside the enclosure, beyond the optical rim.
        ray_lengths = []
        for a,c in zip(perimeter,perimeter[1:]+perimeter[:1]):
            ex,ey = c[0]-a[0],c[1]-a[1]
            den = ux*ey-uy*ex
            if abs(den) < 1e-9:
                continue
            ax,ay = a[0]-cx,a[1]-cy
            reach = (ax*ey-ay*ex)/den
            along = (ax*uy-ay*ux)/den
            if reach>0 and -1e-8 <= along <= 1+1e-8:
                ray_lengths.append(reach)
        reach = min(ray_lengths)+0.05
        optical_cut = b.Solid.extrude(b.Face(port),(ux*reach,uy*reach,0))
        if half_turn:
            optical_cut = optical_cut.rotate(b.Axis.Z,180)
        optical_cuts.append(optical_cut)

    # Boolean the apertures on the shell alone to retain the complete
    # closed arm sections where they join the cabin's lower shoulders.
    for optical_cut in optical_cuts:
        shell = shell-optical_cut
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
    # Relieve the rear corner on the two highly loaded diagonal seats.
    # The radius fillets are clipped to the original spar exterior,
    # retain >2 mm carrier clearance, and grow from the existing seat floors.
    body = body + _tof_seat_rear_pads(placements, arm_envelopes)
    # A single upward-open service cut clears any overlapping sensor coaming
    # or older internal partition from the real board's complete 2 mm box.
    board_service = b.Pos(board_x,board_y,board_z-2.0)*b.Box(
        board_dx+4,board_dy+4,max(100.0,roof_z-board_z+4),
        align=(b.Align.CENTER,b.Align.CENTER,b.Align.MIN))
    body = body-board_service
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
    # 2 mm service clearance of the recessed battery, GPS and cameras.
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
        # The optical bypass divides the root and outboard wiring vaults.
        # A small reinforced bed-facing drain opens the outer gallery too,
        # preserving one connected watertight surface without filling it.
        outer_x = 115.0
        outer_y = sweep_center(outer_x)
        collar = b.Pos(outer_x,outer_y,0)*b.Cylinder(2.3,2.5,
            align=(b.Align.CENTER,b.Align.CENTER,b.Align.MIN))
        drain = b.Pos(outer_x,outer_y,-.2)*b.Cylinder(1.0,3.0,
            align=(b.Align.CENTER,b.Align.CENTER,b.Align.MIN))
        body = body+collar.rotate(b.Axis.Z,ang)
        body = body-drain.rotate(b.Axis.Z,ang)
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
