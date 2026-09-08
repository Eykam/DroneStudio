"""v83-g82a: lower cockpit shoulders with a forward-rising twin-ridge aft hip.

A compound inward fold rises from the existing aft hip into two lower
structural roof ridges. Slightly shallower support-free outer pitches
reduce canopy area while clearing the complete CM4 service envelope.
Original closed spars, optical seats and payload mounting datums persist.

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

    ring_roof_cut_z_mm: float = 36.5  # continuous lower roof strip between carrier covers
    ring_carrier_cover_mm: float = 28.0  # full cover over the fixed carrier envelope
    bezel_surround_mm: float = 3.2  # preserve optical aperture, recess and 0.8 mm land
    cradle_foot_rail_mm: float = 2.0  # three bed-facing radial ties replace the broad apron
    deck_arch_half_span_mm: float = 8.0  # pitched openings in the tall deck side piers
    cradle_post_depth_mm: float = 4.0  # forward boss seat stays radial -3.77 mm
    structural_gauge_mm: float = 1.24  # nominal 1.2 mm construction with print margin
    cradle_ring_web_mm: float = 1.24
    tray_rib_pitch_mm: float = 12.0
    arm_box_depth_scale: float = 0.97
    arm_box_width_scale: float = 0.98

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
    arm_sweep_sign=1.0
    def sweep_center(x):
        # Both endpoints remain on the fixed motor radial. Mirrored sweeps
        # carry the arms around the optical ring and away from the stereo bay.
        # Bend the outboard span around the diagonal optical window. The
        # root and motor axes retain the baseline centerline; an uncut closed
        # spar now passes beside the beam instead of across its lower edge.
        bypass = (23.5*math.sin(math.pi*(x-48.0)/84.0)**2
                  if 48.0 < x < 132.0 else 0.0)
        # The corrected 12.83 x 6.10 mm module window reaches slightly
        # farther down and sideways than the earlier optical package. A
        # local 1.9 mm outward bow carries the closed spar past its lower
        # corner; no arm is slit and the main-shell aperture stays minimal.
        window_bypass = (1.9*math.sin(math.pi*(x-101.0)/30.0)**2
                         if 101.0 < x < 131.0 else 0.0)
        # Carry the inboard shoulder outside the lower optical corner.
        # The short extra sweep removes a notch at the diagonal carrier
        # while leaving the lens corridor, saddle and motor axes fixed.
        root_bypass=(1.4*math.sin(math.pi*(x-48.0)/44.0)**2
                     if 48.0 < x < 92.0 else 0.0)
        return arm_sweep_sign*(-p.arm_sweep_mm*math.sin(math.pi*x/p.arm_length_mm)
                -bypass-window_bypass-root_bypass)

    def spar_profile(x, width, height):
        """Deep lenticular wing with a narrow keel and broad upper shoulders."""
        # Broader crowns and raised shoulders put skin near the bending
        # flanges, permitting slimmer roots without thinning the walls.
        # The cant blends out ahead of the motor-end diaphragm; normal
        # offsets below account for both the section and its swept loft.
        transition=max(0.0,min(1.0,(x-75.0)/23.0))
        blend=max(0.0,min(1.0,(x-75.0)/23.0,(138.0-x)/12.0))
        root=max(0.0,min(1.0,(70.0-x)/22.0))
        half=width/2*(1.0-0.06*blend-0.04*root)
        height-=0.2*blend+0.3*root
        # Shorten the internal bridge without thinning the normal skin.
        # The pointed roof keeps its depth and >45-degree side pitches.
        crown=1.45-0.4*transition+0.7*blend+0.2*root
        keel0=max(p.arm_crown_width_mm/2,0.80*half)
        keel=keel0+(0.98*half-keel0)*blend
        chine=max(1.5*p.arm_rib_thickness_mm,(0.15-0.13*blend)*height)
        shoulder_half=half*(1.0-0.04*blend)
        shoulder0=min(0.72*height,height-p.arm_roof_slope*(half-crown))
        shoulder=shoulder0+(height-p.arm_roof_slope*(shoulder_half-crown)-shoulder0)*max(blend,root)
        # Keep the complete original profile through the internal sensor
        # ring; the lenticular chine begins beyond its carrier clearances.
        # A narrower belly and deeper free-span section retain bending
        # stiffness. Fade the resection out
        # ahead of the fixed motor diaphragm; the broad upper shoulders
        # remain the loaded compression flange. The lower facets rise
        # well above 45 degrees from a continuous printable keel.
        lens=max(0.0,min(1.0,(x-82.0)/15.0,(130.0-x)/32.0))
        depth=1.0+0.045*lens
        height*=depth
        shoulder*=depth
        chine*=depth
        keel+=(0.60*half-keel)*lens
        chine+=(0.26*height-chine)*lens
        crown-=0.20*lens
        shoulder=min(shoulder,height-p.arm_roof_slope*(shoulder_half-crown))
        return [(-keel,0),(keel,0),(half,chine),(shoulder_half,shoulder),
                (crown,height),(-crown,height),(-shoulder_half,shoulder),(-half,chine)]

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
        arm_sweep_sign=-1.0 if mx*my>0 else 1.0
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
        # Divide the shallow hub saddle into two continuous tapered rails
        # around a bed-facing wiring port. Solid end tongues preserve the
        # central junction and open spar mouth; the bolt ring is outboard.
        # Each side chord is over 2.4 mm wide at its narrowest section.
        port=[(4.0,0.0),(7.0,-4.0),(16.0,-4.0),(18.0,0.0),
              (16.0,4.0),(7.0,4.0)]
        wire=b.Wire.make_polygon([(x,sweep_center(x)+y,-.2) for x,y in port],close=True)
        arm=arm-b.Solid.extrude(b.Face(wire),(0,0,p.body_thickness_mm+.4))

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
            box_blend=max(0.0,min(1.0,(x-75.0)/23.0,(.93-frac)/.12))
            height *= 1.0+(p.arm_box_depth_scale-1.0)*box_blend
            width *= 1.0+(p.arm_box_width_scale-1.0)*box_blend
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
        # Two short internal roof ribs brace the long shallow root roof
        # and interrupt its nearly tangent junction with the carrier bed.
        # The lower wiring gallery stays continuous; the undersides inherit
        # the pitched roof, and the rib breadth exceeds the 1.2 mm floor.
        # Subtract them from the gallery itself so the later root re-opening
        # preserves these load paths instead of erasing them.
        roof_band=cavity-cavity.moved(b.Pos(0,0,-1.8))
        for rib_x in (51.8,58.8):
            slab=b.Pos(rib_x,0,0)*b.Box(1.5,100,40,
                align=(b.Align.CENTER,b.Align.CENTER,b.Align.MIN))
            cavity=cavity-(roof_band & slab)
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
        # A 0.02 mm collar allowance keeps the tessellated curved walls
        # above the 1.2 mm floor; the fixed bores and seating heights stay.
        boss_wall = max(1.22, p.motor_boss_wall_mm)
        bolt_boss_radius = p.motor_hole_dia_mm / 2 + boss_wall
        center_boss_radius = p.motor_center_hole_dia_mm / 2 + boss_wall
        spoke_length = 2 * (bolt_radius + bolt_boss_radius)
        # Taller blade ribs put material into depth instead of a wide
        # shallow pad. Their 1.30 mm minimum width exceeds the DFAM
        # floor, and their tops remain below the fixed annular seats.
        # Each rib prints directly from the bed. Relative to the old
        # tapered webs, vertical section inertia increases while rib
        # volume falls; all four screw collars and shaft bores persist.
        web_height=max(2*p.arm_rib_thickness_mm,
                       0.80*p.motor_pad_thickness_mm)
        root_half=max(0.65,0.56*p.motor_spoke_width_mm/2)
        end_half=max(0.65,0.38*p.motor_spoke_width_mm/2)
        rib_outline=b.Wire.make_polygon([
            (0,-root_half,0),(bolt_radius,-end_half,0),
            (bolt_radius,end_half,0),(0,root_half,0)],close=True)
        spoke=b.Solid.extrude(b.Face(rib_outline),(0,0,web_height))
        pad=spoke
        for rib_angle in (90,180,270):
            pad=pad+spoke.rotate(b.Axis.Z,rib_angle)
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
        nacelle_polygons = []
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
            points=[(center-half_width,0),(center+half_width,0),
                    (center+half_width,shoulder),(center,height),
                    (center-half_width,shoulder)]
            nacelle_polygons.append((x,points))
            nacelle_sections.append(b.Wire.make_polygon(
                [(x,y,z) for y,z in points],close=True))
        pad = pad + b.Solid.make_loft(nacelle_sections,ruled=True)

        # Core the motor saddle with a conformal five-face gallery.
        # The former narrow triangular bore left solid wedges beside its
        # lower corners. True 3D normal offsets retain >=1.24 mm walls,
        # including the falling roof and the lateral centerline gradient.
        # The fixed collars are excluded from the complete tool below.
        gallery=[]
        for x in (bridge_start+0.1,rib_end,L-center_boss_radius-0.2):
            for left,right in zip(nacelle_polygons,nacelle_polygons[1:]):
                if x<=right[0]+1e-8:
                    t=(x-left[0])/(right[0]-left[0])
                    points=[(a[0]+t*(c[0]-a[0]),a[1]+t*(c[1]-a[1]))
                            for a,c in zip(left[1],right[1])]
                    break
            neighbors=[]
            for left,right in zip(nacelle_polygons,nacelle_polygons[1:]):
                if left[0]-1e-8<=x<=right[0]+1e-8:
                    neighbors.extend(n for n in (left,right) if abs(n[0]-x)>1e-8)
            lines=[]
            for i,((y0,z0),(y1,z1)) in enumerate(zip(points,points[1:]+points[:1])):
                dy,dz=y1-y0,z1-z0
                length=math.hypot(dy,dz);ny,nz=-dz/length,dy/length
                gradient=0.0
                for nx,other in neighbors:
                    for j in (i,(i+1)%len(points)):
                        dy0=other[j][0]-points[j][0]
                        dz0=other[j][1]-points[j][1]
                        gradient=max(gradient,abs((ny*dy0+nz*dz0)/(nx-x)))
                gauge=max(1.24,p.arm_rib_thickness_mm-0.11)
                gauge*=max(1.035,1.01*math.sqrt(1+gradient*gradient))
                lines.append((ny,nz,ny*y0+nz*z0+gauge))
            inner=[]
            for (ay,az,ac),(by,bz,bc) in zip(lines[-1:]+lines[:-1],lines):
                det=ay*bz-by*az
                inner.append(((ac*bz-bc*az)/det,(ay*bc-by*ac)/det))
            # The external roof flattens into the motor seat. Keep the
            # internal eaves lower so both gallery roof faces stay >45 deg.
            for j in (2,4):
                y,z=inner[j]
                inner[j]=(y,min(z,inner[3][1]-1.12*abs(y-inner[3][0])))
            gallery.append(b.Wire.make_polygon([(x,y,z) for y,z in inner],close=True))
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
    # The optical facets retain their XY datums. Their 28 mm eave leaves
    # the full lens bezel and its upper land intact, while shortening the
    # broad vertical skirt. The raised inboard hip still covers each carrier.
    # A separate longitudinal rake below follows the battery-to-CM4 envelope.
    wall = p.body_thickness_mm
    # Raise the inboard roof through its pitch, rather than restoring
    # the heavy skirt. The aft hip starts 4 mm earlier so its inner face
    # clears the CM4 service corners without a flat clipped underside.
    slope = p.body_roof_slope+0.04
    sx = p.center_plate_len_mm / 242.0
    sy = p.center_plate_wid_mm / 68.0

    def box(x, y, z, dx, dy, dz):
        return b.Pos(x, y, z)*b.Box(dx, dy, dz,
            align=(b.Align.CENTER, b.Align.CENTER, b.Align.MIN))

    def prism(points, z, height):
        return b.Solid.extrude(b.Face(b.Wire.make_polygon(
            [(x,y,z) for x,y in points], close=True)), (0,0,height))

    # x, half breadth. The diagonal optical facets are 30.55 mm wide,
    # leaving the entire recessed sheet and land inside the shell line.
    # All eight module-face standoffs are consequently about 9.2 mm.
    stations = [(-144.0,14.0),(-130.0,14.0),(-114.0,24.8),
                (-108.0,25.5),(-65.4,43.8),(-43.8,65.4),
                (-14.0,61.6),(14.0,61.6),(43.8,65.4),
                (65.4,43.8),(75.0,43.5),(91.0,43.5),(101.6,12.0)]
    stations = [(x*sx,h*sy) for x,h in stations]
    perimeter = [(x,-h) for x,h in stations] + [(x,h) for x,h in reversed(stations)]
    def clip_x(points,edge,sign):
        result=[]
        for a,d in zip(points,points[1:]+points[:1]):
            aa=(a[0]-edge)*sign>=-1e-8; dd=(d[0]-edge)*sign>=-1e-8
            if aa: result.append(a)
            if aa != dd:
                f=(edge-a[0])/(d[0]-a[0]);result.append((edge,a[1]+f*(d[1]-a[1])))
        return result

    def convex(points):
        pts=sorted(set(points)); halves=[]
        for seq in (pts,list(reversed(pts))):
            h=[]
            for p in seq:
                while len(h)>=2 and (h[-1][0]-h[-2][0])*(p[1]-h[-2][1])-(h[-1][1]-h[-2][1])*(p[0]-h[-2][0])<=1e-8: h.pop()
                h.append(p)
            halves.extend(h[:-1])
        return halves

    def offset(points,gauge):
        lines=[]
        for a,d in zip(points,points[1:]+points[:1]):
            dx,dy=d[0]-a[0],d[1]-a[1]; l=math.hypot(dx,dy);nx,ny=dy/l,-dx/l
            lines.append((nx,ny,nx*a[0]+ny*a[1]+gauge))
        out=[]
        for a,d in zip(lines[-1:]+lines[:-1],lines):
            det=a[0]*d[1]-d[0]*a[1]
            if abs(det)<1e-8:
                # Collinear station edges carry the same inward offset.
                # The perimeter supplied here has no repeated collinear vertices.
                raise ValueError('collinear offset')
            out.append(((a[2]*d[1]-d[2]*a[1])/det,(a[0]*d[2]-d[0]*a[2])/det))
        return out
    outer_plan=prism(perimeter,0,150)
    inner_plan=prism(offset(perimeter,-wall),-.2,151)
    outs=[];ins=[]
    for x0,x1 in [(-145,-40),(-90,84),(40,103)]:
        pts=convex(clip_x(clip_x(perimeter,x0,1),x1,-1))
        outer=prism(pts,0,150); inner=prism(pts,-.2,151)
        for a,d in zip(pts,pts[1:]+pts[:1]):
            ex,ey=d[0]-a[0],d[1]-a[1]; el=math.hypot(ex,ey)
            nx,ny=ey/el,-ex/el;c=nx*a[0]+ny*a[1]
            pl=b.Plane(origin=(nx*c,ny*c,28.0),z_dir=(slope*nx,slope*ny,1))
            half=pl*b.Box(600,600,500,align=(b.Align.CENTER,b.Align.CENTER,b.Align.MAX))
            outer=outer & half
            inner=inner & half.moved(b.Pos(0,0,-wall*1.025*math.sqrt(1+slope*slope)))
        outs.append(outer);ins.append(inner)
    outer=outs[0];inner=ins[0]
    for o in outs[1:]: outer=outer+o
    for i in ins[1:]: inner=inner+i
    # Follow the two payload heights with a low battery turtledeck and a
    # short, steep aft cockpit hip. The shallow battery ridge clears the
    # 35.5 mm pack top at both outer corners; the delayed 2.30:1 hip
    # leaves its inner face above the aft CM4 service corner at X=-56 mm. The tail returns into the GPS
    # fin using the original aft pitch. Each transverse inner roof remains
    # steeper than 45 degrees; all offsets include longitudinal rake.
    roof_outers=[]; roof_inners=[]
    roof_stations=[(-104.0,57.8,0.08),(-72.5,60.32,2.30),
                   (-104.0,56.0,-0.50)]
    for ridge_x,ridge_z,rake in roof_stations:
        roof_outer=box(0,0,-.2,600,600,200)
        roof_inner=box(0,0,-.2,600,600,200)
        for side in (-1,1):
            pl=b.Plane(origin=(ridge_x*sx,0,ridge_z),
                       z_dir=(-rake/sx,side*slope,1))
            half=pl*b.Box(800,800,600,
                align=(b.Align.CENTER,b.Align.CENTER,b.Align.MAX))
            roof_outer=roof_outer & half
            drop=wall*1.025*math.sqrt(1+slope*slope+(rake/sx)**2)
            roof_inner=roof_inner & half.moved(b.Pos(0,0,-drop))
        roof_outers.append(roof_outer);roof_inners.append(roof_inner)
    roof_outer=roof_outers[0];roof_inner=roof_inners[0]
    for ro,ri in zip(roof_outers[1:],roof_inners[1:]):
        roof_outer=roof_outer+ro;roof_inner=roof_inner+ri
    outer=outer & roof_outer
    inner=inner & roof_inner
    # Rake both cockpit shoulders toward the forward service portal.
    # Their intersection with the original aft hip forms a wedge instead
    # of a constant-height dorsal extrusion. Each shoulder still grows
    # inward at 1.12:1 from the perimeter; the longitudinal rake does not
    # create an unsupported inner eave or a horizontal bridging panel.
    # The full forward CM4 service corner (X=56,Y=28), including the
    # thin portal jamb, retains over 0.8 mm clearance above the 61.2 mm
    # service volume. The bay cut therefore cannot leave a flat underside
    # where it meets these shoulders.
    roof_rake=0.065/sx
    for roof_side in (-1,1):
        pl=b.Plane(origin=(0,0,99.0),
                   z_dir=(roof_rake,roof_side*1.12,1))
        half=pl*b.Box(800,800,600,
            align=(b.Align.CENTER,b.Align.CENTER,b.Align.MAX))
        outer=outer & half
        drop=wall*1.025*math.sqrt(1+1.12**2+roof_rake**2)
        inner=inner & half.moved(b.Pos(0,0,-drop))
    # A second, opposing longitudinal pitch shortens the high aft
    # cockpit shoulders. The compound hip uses shallower 1.06:1 inner
    # roof slopes and >=1.22 mm normal skin; its ridge intersects the
    # forward rake rather than adding a suspended transverse bulkhead.
    # At the full rear service corner (X=-56,Y=28), the inner roof
    # remains above Z=61.6 mm, clearing the 61.2 mm CM4 service box.
    # Retain the original outboard shoulder plane as an envelope limit.
    # The shallower cockpit pitch below intersects it at |Y|=33.3 mm:
    # only the high central skin moves inward, while all optical-ring
    # eaves and carrier-cover edges keep their original height and gauge.
    for roof_side in (-1,1):
        plane=b.Plane(origin=(0,0,96.0),
                      z_dir=(-0.015/sx,roof_side*1.12,1))
        half=plane*b.Box(800,800,600,
            align=(b.Align.CENTER,b.Align.CENTER,b.Align.MAX))
        outer=outer & half
        drop=wall*1.025*math.sqrt(1+1.12**2+(-0.015/sx)**2)
        inner=inner & half.moved(b.Pos(0,0,-drop))
    aft_rake=-0.015/sx
    for roof_side in (-1,1):
        plane=b.Plane(origin=(0,0,94.0),
                      z_dir=(aft_rake,roof_side*1.06,1))
        half=plane*b.Box(800,800,600,
            align=(b.Align.CENTER,b.Align.CENTER,b.Align.MAX))
        outer=outer & half
        drop=wall*1.025*math.sqrt(1+1.06**2+aft_rake**2)
        inner=inner & half.moved(b.Pos(0,0,-drop))
    # An inward fold splits the back of the cockpit into two raked
    # ridges, cutting the high aft shoulders down toward the CM4 envelope.
    # Unlike a constant-height trough, this face rises forward at 2.00:1:
    # its first layer meets the bed-founded aft hip, then each higher layer
    # advances less than its height, including the flared hatch edge:
    # (2.00-1.12*(3.8/6))/sqrt(1+(3.8/6)**2) = 1.09 > 1.0.
    # The inner lip therefore grows from supported material throughout.
    # Outboard, the original roof remains lower, retaining the carrier
    # covers and ring. Full normal offsets include both roof gradients.
    folds_outer=[]; folds_inner=[]
    for fold_side in (-1,1):
        plane=b.Plane(origin=(-65.0*sx,0,59.4),
                      z_dir=(-2.00/sx,-fold_side*1.12,1))
        half=plane*b.Box(800,800,600,
            align=(b.Align.CENTER,b.Align.CENTER,b.Align.MAX))
        folds_outer.append(half)
        drop=wall*1.025*math.sqrt(1+(2.00/sx)**2+1.12**2)
        folds_inner.append(half.moved(b.Pos(0,0,-drop)))
    # Restrict the forward rake to the cockpit. At X=-70 its full
    # skin is already above the old aft hip outside the open hatch,
    # so the boundary makes no step; the battery and tail are untouched.
    aft_keep=box(-370.0*sx,0,-.2,600.0*sx,600,200)
    outer=outer & (folds_outer[0]+folds_outer[1]+aft_keep)
    inner=inner & (folds_inner[0]+folds_inner[1]+aft_keep)

    # A three-fold nose follows the two camera boards and central radial
    # carrier. Intersect it with the existing shell: every fold is inward,
    # and the fixed optical facets, seats and clearance tools are retained.
    # Ridge valleys have 1.12:1 inner pitches and a true normal skin offset.
    # A steep aft hip blends the folds into the forward cockpit jamb;
    # the union of the four roofs leaves no unsupported horizontal ledge.
    nose_outer=[]; nose_inner=[]
    for ridge_y in (-28.0,0.0,28.0):
        no=box(0,0,-.2,600,600,200)
        ni=box(0,0,-.2,600,600,200)
        rake=.06/sx
        # The camera brows sit lower than the central radial carrier
        # hood. Their inner corners clear the real 25.862 mm board top;
        # the existing steep hip joins them to the cockpit service jamb.
        brow_z=45.3 if ridge_y == 0.0 else 43.0
        for side in (-1,1):
            plane=b.Plane(origin=(83.0*sx,ridge_y*sy,brow_z),
                          z_dir=(rake,side*1.12/sy,1))
            half=plane*b.Box(800,800,600,
                align=(b.Align.CENTER,b.Align.CENTER,b.Align.MAX))
            no=no & half
            drop=wall*1.025*math.sqrt(1+(1.12/sy)**2+rake*rake)
            ni=ni & half.moved(b.Pos(0,0,-drop))
        nose_outer.append(no);nose_inner.append(ni)
    hip=b.Plane(origin=(71.0*sx,0,45.3),z_dir=(1.20/sx,0,1))
    half=hip*b.Box(800,800,600,
        align=(b.Align.CENTER,b.Align.CENTER,b.Align.MAX))
    nose_o=half;nose_i=half.moved(b.Pos(0,0,-wall*1.025*math.sqrt(1+(1.20/sx)**2)))
    for no,ni in zip(nose_outer,nose_inner):
        nose_o=nose_o+no;nose_i=nose_i+ni
    outer=outer & nose_o
    inner=inner & nose_i
    outer_hull=outer&outer_plan; inner_hull=inner&inner_plan
    shell=outer_hull-inner_hull
    # Flared access shoulders follow the battery bay instead of carrying
    # a uniform narrow slot through surplus dorsal skin. The 12 mm ends
    # keep the nose/tail cross ties; the broad middle admits fingers and
    # battery leads while leaving both continuous canopy edge chords.
    # Close the flare before the flight-board envelope: its aft canopy
    # shoulders keep their complete original coverage over the CM4 bay.
    hatch=[(-107.0,-6.0),(-91.0,-9.8),(-63.0,-9.8),(-57.0,-6.0),
           (-29.0,-6.0),(-29.0,6.0),(-57.0,6.0),(-63.0,9.8),
           (-91.0,9.8),(-107.0,6.0)]
    shell=shell-prism(hatch,37.2,100)
    # Continue the narrow channel to the existing forward service portal.
    # Both complete shoulder ridges and their hip-to-ring ties persist.
    shell=shell-box(-14.0,0,37.2,30.0,12.0,100)
    shell=shell-box(28.0,0,34.0,56.0,54.6,110)

    # A continuous lower roof strip follows the unchanged shell line.
    # Full carrier covers and swept shear bands retain enclosure and
    # connect the ring to the battery/avionics canopy across scalloped bays.
    protected=box(-28,0,0,56,56,150)+box(-68,0,0,78,38,150)
    # Swept roof bands connect the battery/FC spine to the optical-ring
    # shoulders. Three paired diagonals replace four full transverse bands:
    # the roof itself is the shear member, with no added decorative skin.
    # Their 2.6 mm plan-normal width stays above the printable floor even
    # at the sloping hip intersections. Covers below remain independent.
    for side in (-1,1):
        for spine_x, ring_x in ((-82.0,-68.0),(-54.0,-31.0),(-27.0,-2.0)):
            y0,y1=18.0,66.0
            # Fan the load path into the optical-ring shoulder: a 2.0 mm
            # neck at the canopy grows to 3.2 mm at the continuous rim.
            # These are pieces of structural roof, not added fairing.
            factor=math.sqrt(1+((ring_x-spine_x)/(y1-y0))**2)
            neck,foot=1.0*factor,1.6*factor
            band=[(spine_x-neck,side*y0),(ring_x-foot,side*y1),
                  (ring_x+foot,side*y1),(spine_x+neck,side*y0)]
            protected=protected+prism(band,0,150)
    shell=shell-(box(0,0,53.5,350,250,110)-protected)
    for key,pos in placements.items():
        if key in tof_poses:
            cx,cy,_=(v*1000 for v in pos)
            angle=math.degrees(math.atan2(cy,cx))
            # A radial hood covers the full 12.14 x 21.2 mm mechanical
            # service envelope with clearance. Its outer end fans into the
            # continuous lower roof strip; the narrowed inboard end removes
            # the unused square corners above the carrier's rear wiring bay.
            # Every hood is an area of the existing normally-offset skin,
            # so there are no added pods, thin edge laps or new overhangs.
            scale=p.ring_carrier_cover_mm/28.0
            # Chamfer the unloaded outer shoulder corners while keeping
            # the entire PCB/connector hood and its shell-root connection.
            # These tapered panels are still the single structural shell.
            # Start each clipped corner exactly on the original taper.
            # Keeping this polygon a subset of the old cover avoids a
            # narrow projecting lip where the hood meets the hip rim.
            corner_t=12.6+(14.0+8.4)*(1.4/26.4)
            hood=[(-8.4,-12.6),(14.0,-corner_t),(18.0,-11.4),
                  (18.0,11.4),(14.0,corner_t),(-8.4,12.6)]
            # Broaden the east/west hood-to-rim fan at its acute waist
            # junction. This retains a little existing structural skin
            # across the crash-stress notch, within the original hull.
            if key in ('vl53l9cx_breakout#e','vl53l9cx_breakout#w'):
                hood=[(-8.4,-13.4),(14.0,-14.8),(18.0,-11.4),
                      (18.0,11.4),(14.0,14.8),(-8.4,13.4)]
            hood=[(r*scale,t*scale) for r,t in hood]
            cap=prism(hood,0,150).rotate(b.Axis.Z,angle).moved(b.Pos(cx,cy,0))
            protected=protected+cap
    for key in ('gps','pi_camera_3#left','pi_camera_3#right'):
        cx,cy,_=(v*1000 for v in placements[key])
        dx,dy,_=(v*1000 for v in LIBRARY[key.split('#')[0]].dims_m)
        protected=protected+box(cx,cy,0,dx+2*wall,dy+2*wall,150)
    # A continuous 2.0 mm plan-width hip rim closes the shoulder load
    # path. The former broad strip above it becomes scalloped service
    # access between the carrier covers, with full hood footprints intact.
    # Preserve the lower sidewall, every optical facet and all bezel lands.
    rim=outer_plan-prism(offset(perimeter,-2.0),0,150)
    # Lower the rim cutoff with the eaves, preserving its 2 mm plan width.
    # Terminate the rim at the pitched shoulder: remove the narrow
    # cantilevered returns at the two waist corners. The full carrier
    # hoods and swept shear bands remain independent of this lower rim.
    rim=rim & box(0,0,0,350,250,p.ring_roof_cut_z_mm-6.0)
    protected=protected+rim
    upper_tool=box(0,0,p.ring_roof_cut_z_mm-8.5,350,250,110)-protected
    shell=shell-upper_tool
    # Carry the existing scallops down with the lower battery roof.
    # Otherwise a lower pitch restores broad panels below the old fixed
    # Z cutoff and consumes the area saved by the compact turtledeck.
    # The full 2 mm perimeter rim, diagonal roof ties, battery spine and
    # every carrier hood remain protected. Only the empty aft shoulders
    # between those continuous load paths are opened for service access.
    aft_shoulder_tool=box(-88.0*sx,0,21.5,42.0*sx,250,110)-protected
    shell=shell-aft_shoulder_tool

    # Pointed ventilation bays turn the high cockpit shoulders into
    # a shear lattice within the existing skin. Both longitudinal edge
    # chords remain continuous, and broad webs separate the openings.
    # Cut perpendicular to the aft roof plane to retain the complete
    # normal wall gauge at the rims instead of leaving feather edges.
    # The tips close over 6 mm of plan run with only 3.0 mm lateral
    # advance: the 1.06:1 roof gives a >45-degree print trajectory.
    # Only the upper shoulders are relieved; all carrier covers and
    # lower ring load paths lie below these tools. The wider openings
    # retain 6.0 mm webs between their 12 mm stations and continuous
    # longitudinal chords along both canopy edges.
    for side in (-1,1):
        for vent_x in (-44.0,-32.0,-20.0,-8.0):
            vent_y=17.0*side
            surface_z=94.0-aft_rake*vent_x-1.06*abs(vent_y)
            plane=b.Plane(origin=(vent_x,vent_y,surface_z),
                          x_dir=(1,0,-aft_rake),
                          z_dir=(aft_rake,side*1.06,1))
            along=math.sqrt(1+1.06**2)
            outline=[(0,-9.0*along),(3.0,-3.0*along),
                     (3.0,3.0*along),(0,9.0*along),
                     (-3.0,3.0*along),(-3.0,-3.0*along)]
            wire=b.Wire.make_polygon([(u,v,-2*wall) for u,v in outline],close=True)
            shell=shell-plane*b.Solid.extrude(b.Face(wire),(0,0,4*wall))

    # R2: stepped, inward-only IR-sheet bezels. The printed chassis contains
    # the bonding land; 0.75 mm dark IR-pass sheets are separate consumables.
    # No optical crop: use the full Table-37 union at each actual depth.
    # Board face is -2.77 mm radial and module height 4.64 mm, hence the
    # emitting module face is +1.87 mm relative to the placement axis.
    bezel_cuts=[]
    bezel_rings=[]
    for key,pos in placements.items():
        if key not in tof_poses:
            continue
        cx,cy,z0=(v*1000 for v in pos)
        angle=math.degrees(math.atan2(cy,cx))
        ux,uy=math.cos(math.radians(angle)),math.sin(math.radians(angle))
        tx,ty=-uy,ux
        def local(shape):
            return shape.rotate(b.Axis.Z,angle).moved(b.Pos(cx,cy,0))
        reaches=[]
        for a,d in zip(perimeter,perimeter[1:]+perimeter[:1]):
            ex,ey=d[0]-a[0],d[1]-a[1]
            den=ux*ey-uy*ex
            if abs(den)<1e-8: continue
            ax,ay=a[0]-cx,a[1]-cy
            r=(ax*ey-ay*ex)/den
            f=(ax*uy-ay*ux)/den
            if r>0 and -1e-8<=f<=1+1e-8: reaches.append(r)
        outer_r=min(reaches)
        plane_r=outer_r-2.5
        standoff=plane_r-1.87
        window_w=21.1+1.24*(standoff-9.2)
        window_h=12.3+0.90*(standoff-9.2)
        oz=tof_poses[key]['origin_m'][2]*1000
        # Backing ring is wholly inboard of the unmodified shell line.
        ring=local(box(outer_r-1.86,-1.25,oz-window_h/2-p.bezel_surround_mm,
                       3.72,window_w+2*p.bezel_surround_mm,
                       window_h+2*p.bezel_surround_mm)) & outer_hull
        # A rear sheet land and four gauge-thickness returns share the
        # shell wall, replacing the filled outer half of the backing block.
        # Pitch the cavity ceiling so the return prints continuously
        # from the rear land; a flat blind slot needs support at this height.
        half_t=window_w/2+p.bezel_surround_mm-p.structural_gauge_mm
        low_z=oz-window_h/2-p.bezel_surround_mm+p.structural_gauge_mm
        high_z=oz+window_h/2+p.bezel_surround_mm-p.structural_gauge_mm*math.sqrt(1+1.12**2)
        w=b.Wire.make_polygon([(plane_r,-1.25-half_t,low_z),
            (plane_r+2.8,-1.25-half_t,low_z),
            (plane_r+2.8,-1.25-half_t,high_z-1.12*2.8),
            (plane_r,-1.25-half_t,high_z)],close=True)
        relief=local(b.Solid.extrude(b.Face(w),(0,2*half_t,0)))
        ring=ring-relief
        bezel_rings.append(ring)
        opening=local(box(plane_r-1.3,-1.25,oz-window_h/2,
                          5.2,window_w,window_h))
        # A 0.8 mm perimeter land accepts a sheet recessed 2.5 mm. The
        # outward reveal flares with the same exclusion-zone slopes.
        def window_wire(r,w,h):
            return b.Wire.make_polygon([(r,-1.25-w/2,oz-h/2),
                (r,-1.25+w/2,oz-h/2),(r,-1.25+w/2,oz+h/2),
                (r,-1.25-w/2,oz+h/2)],close=True)
        reveal=local(b.Solid.make_loft([
            window_wire(plane_r,window_w+1.6,window_h+1.6),
            window_wire(outer_r+3,window_w+1.6+1.24*5.5,
                        window_h+1.6+2.24*5.5)],ruled=True))
        # The interior optical corridor clears the complete growing zone.
        # The inboard half-shelf and M2 post remain behind the module face.
        corridor=local(b.Solid.make_loft([
            window_wire(1.87,window_w-1.24*standoff,window_h-.90*standoff),
            window_wire(plane_r,window_w,window_h)],ruled=True))
        bezel_cuts.extend([opening,reveal,corridor])

    # The rear battery flanks become a swept, ventilated shear panel.
    # Cut only the shell skin: the recessed tray, longerons and internal
    # sensor seats remain continuous. Each opening stops 2.6 mm above the
    # bed and 3 mm below the eave; inclined piers connect these two
    # chords. The aperture roofs rise >1.12:1 and grow from both jambs,
    # so the flanks print without suspended horizontal lintels.
    # These bays lie between the aft cardinal and diagonal ToF stations,
    # outside every optical facet, carrier hood and PCB service envelope.
    for gx in (-106.0,-92.0,-78.0):
        outline=[(gx-5.5,2.6),(gx+3.5,2.6),(gx+6.2,14.0),
                 (gx+0.5,25.0),(gx-4.0,14.0)]
        wire=b.Wire.make_polygon([(x*sx,-100.0,z) for x,z in outline],close=True)
        shell=shell-b.Solid.extrude(b.Face(wire),(0,200.0,0))

    # Swept cheek vaults replace the broad forward skirt with a deep
    # shear panel: continuous 4 mm belly and >5 mm upper chords surround
    # two inclined piers. The side cuts stay in the near-vertical camera
    # cheeks, ahead of the diagonal ToF carrier and behind the nose facet.
    # Pointed roofs rise at least 1.2:1 and print inward from both jambs;
    # all camera pads, retaining ears and optical cuts are added below.
    for gx in (71.5,85.0):
        outline=[(gx-5.0,4.0),(gx+3.0,4.0),(gx+5.0,13.0),
                 (gx,21.5),(gx-3.5,13.0)]
        wire=b.Wire.make_polygon([(x*sx,-100.0,z) for x,z in outline],close=True)
        shell=shell-b.Solid.extrude(b.Face(wire),(0,200.0,0))

    # The waist between each cardinal and diagonal optical facet has
    # no payload behind its lower skirt. Convert those broad panels into
    # swept shear piers, retaining 2.6 mm belly and 3.7 mm eave chords.
    # Only the shell is cut: the arm tubes, internal sensor saddles and
    # every optical bezel are independent and retain their original forms.
    # Cut normal to each slanted facet to avoid feathering its 1.22 mm
    # skin. Both pointed roof edges rise >1.25:1 from their side jambs.
    for waist_x in (-26.5,26.5):
        for side in (-1,1):
            flank_grade=side*math.copysign(3.8/29.8,waist_x)*sy/sx
            length=math.sqrt(1+flank_grade*flank_grade)
            tx,ty=1/length,flank_grade/length
            nx,ny=-ty,tx
            cy=side*(61.6+(abs(waist_x)-14.0)*3.8/29.8)*sy
            lean=math.copysign(1.0,waist_x)
            outline=[(-6.3,2.6),(6.3,2.6),(6.3+lean,12.8),
                     (lean,24.3),(-6.3+lean,12.8)]
            wire=b.Wire.make_polygon([
                (waist_x*sx+tx*u-4*nx,cy+ty*u-4*ny,z)
                for u,z in outline],close=True)
            shell=shell-b.Solid.extrude(b.Face(wire),(8*nx,8*ny,0))

    for cut in bezel_cuts: shell=shell-cut
    body=body+shell

    # Re-open the arm roots through the enclosure walls. The original
    # motor diaphragms and motor bolt load paths retain their fixed datums.
    gallery_limit=b.Cylinder(95,31,align=(b.Align.CENTER,b.Align.CENTER,b.Align.MIN))
    for cavity in arm_cavities:
        body=body-(cavity & gallery_limit)

    # The component exporter recenters each rotated STEP by its actual
    # world AABB. The asymmetric header gives the diagonal carriers a
    # different datum from a rotated nominal box. Derive the mechanical
    # mount origin with the identical transform; pinned poses never move.
    from pathlib import Path
    import components as component_models
    carrier_path=Path(component_models.__file__).parent/LIBRARY['vl53l9cx_breakout'].step_path
    carrier_brep=Path(str(carrier_path)+'.brep')
    carrier_cad=(b.import_brep(str(carrier_brep)) if carrier_brep.exists()
                 else b.import_step(str(carrier_path))).rotate(b.Axis.Y,90)
    carrier_bb=carrier_cad.bounding_box()
    datum_r=(carrier_bb.min.X+carrier_bb.max.X)/2
    datum_t=(carrier_bb.min.Y+carrier_bb.max.Y)/2
    def carrier_mount_frame(pos):
        cx,cy,z0=(v*1000 for v in pos)
        ang=math.degrees(math.atan2(cy,cx)); theta=math.radians(ang)
        bb=carrier_cad.rotate(b.Axis.Z,ang).bounding_box()
        ex=datum_r*math.cos(theta)-datum_t*math.sin(theta)
        ey=datum_r*math.sin(theta)+datum_t*math.cos(theta)
        return (cx+ex-(bb.min.X+bb.max.X)/2,
                cy+ey-(bb.min.Y+bb.max.Y)/2,z0,ang)

    # R1: eight folded, bed-founded cradles with shared ring braces.
    # The real PCB remains radial -3.77 to -2.77, tangential +/-10,
    # with its lower edge at the unchanged placement Z.
    for key,pos in placements.items():
        if key not in tof_poses: continue
        cx,cy,z0,angle=carrier_mount_frame(pos)
        def local(shape):
            return shape.rotate(b.Axis.Z,angle).moved(b.Pos(cx,cy,0))
        # Work in world XY for a compact square shelf containing the rotated
        # carrier as well as its mounting ear. The pitched vaults remain open
        # to the print bed; none is an inaccessible sealed cavity.
        # A folded skin replaces the filled top of the vaulted block.
        # Its ridge lines meet the fixed PCB bottom plane; the unchanged
        # board-edge ledge supplies continuous contact along the PCB.
        # Vertical webs below the valleys carry the folds to the bed.
        pitch=5.4
        grade=1.11
        normal=p.structural_gauge_mm
        shelf=box(cx,cy,0,24.8,24.8,z0)
        for offset in (-1.5*pitch,-.5*pitch,.5*pitch,1.5*pitch):
            half=(pitch-normal)/2
            apex=z0-normal*math.sqrt(1+grade*grade)
            eave=apex-grade*half
            v=b.Wire.make_polygon([(cx-12.6,cy+offset-half,-.2),
                (cx-12.6,cy+offset+half,-.2),(cx-12.6,cy+offset+half,eave),
                (cx-12.6,cy+offset,apex),(cx-12.6,cy+offset-half,eave)],close=True)
            shelf=shelf-b.Solid.extrude(b.Face(v),(25.2,0,0))
        for offset in (-2*pitch,-pitch,0.0,pitch,2*pitch):
            v=b.Wire.make_polygon([(cx-12.6,cy+offset-pitch/2,z0),
                (cx-12.6,cy+offset,z0-grade*pitch/2),
                (cx-12.6,cy+offset+pitch/2,z0),
                (cx-12.6,cy+offset+pitch/2,z0+1),
                (cx-12.6,cy+offset-pitch/2,z0+1)],close=True)
            shelf=shelf-b.Solid.extrude(b.Face(v),(25.2,0,0))
        # Cross-vault the cardinal seats perpendicular to their folds.
        # Diagonal seats also carry the swept arm roots: keep their full
        # folded shear webs to avoid grazing saddle/spar intersections.
        # Each opening rises 1.12:1 from the build plate and stops below
        # the corrugated valleys, retaining a deep two-direction saddle.
        # The 4 mm central pier and 2.4 mm end piers receive the original
        # contact skin; no carrier support or mounting feature is removed.
        cardinal=abs(math.sin(math.radians(2*angle)))<0.5
        for radial in ((-6.0,6.0) if cardinal else ()):
            half=4.0
            apex=z0-5.4*1.11/2-p.structural_gauge_mm
            eave=apex-1.12*half
            w=b.Wire.make_polygon([(cx+radial-half,cy-12.6,-.2),
                (cx+radial+half,cy-12.6,-.2),
                (cx+radial+half,cy-12.6,eave),
                (cx+radial,cy-12.6,apex),
                (cx+radial-half,cy-12.6,eave)],close=True)
            shelf=shelf-b.Solid.extrude(b.Face(w),(0,25.2,0))
        # The cradle service volume frees the original spar crossing while
        # retaining its full section on either side of this tied-in saddle.
        carrier_clear=local(box(0,0,z0,12.14,21.2,16.8))
        body=body-carrier_clear
        # A narrow board-edge shelf reaches the physical edge, not a remote
        # global bounding box. Side rails have 0.3 mm insertion clearance.
        # A low radial foot joins the shelf to the common perimeter sill.
        # It stays well below the RX/TX optical corridor.
        # Three first-layer rails tie the existing vaulted shelf into the
        # perimeter sill. Open bays between them remove the redundant apron;
        # the pitched shelf vaults, PCB support ledge and bosses stay intact.
        foot=local(box(5.5,0,0,23.0,p.cradle_foot_rail_mm,wall))
        for t in (-9.4,9.4):
            foot=foot+local(box(5.5,t,0,23.0,p.cradle_foot_rail_mm,wall))
        # At the nose the stereo optical cuts interrupt narrow floor ties.
        # Keep this one full apron to connect its cradle to the common shell.
        if key == 'vl53l9cx_breakout#n':
            foot=local(box(5.5,0,0,23.0,22.8,wall))
        foot=foot & outer_hull
        # Trim the unused inboard skirt of the folded bed. The full PCB
        # edge, rear screw post and both shell-sharing triangular braces
        # retain their contact. Diagonal boards need a broader saddle for
        # their rotated carrier envelope; use a continuous perimeter cut.
        diagonal=abs(math.sin(math.radians(2*angle)))>0.5
        # Stop the folded skirt before the revised spar's grazing
        # lower-wall intersection. Its rear datum is still 3.2 mm
        # behind the mounting post, leaving the complete PCB seat.
        back=-11.0 if diagonal else -10.0
        front=4.4
        # The diagonally rotated square bed left tangential corner
        # skirts far beyond the PCB, rails and gussets. End those skirts
        # at +/-12.0 mm: the 20 mm board and both mounting ears retain full
        # support, while the empty corner no longer grazes the spar floor.
        # This also removes the nearly coplanar cradle/spar wedge that
        # produced degenerate tetrahedra in otherwise valid solid exports.
        skirt_width=24.0 if diagonal else 50.0
        shelf=shelf & local(box((back+front)/2,0,-.1,
                                front-back,skirt_width,z0+1))
        # A 1.4 mm ledge bears directly on the PCB bottom edge. The broad
        # shock-support shelf sits 1.7 mm below the carrier envelope.
        ridge=local(box(-3.27,0,z0-1.7,1.4,20.0,1.7))
        cradle=shelf+foot+ridge
        for side in (-1,1):
            rail=local(box(-3.27,side*(10.3+wall/2),z0-3.0,
                           3.44,wall,18.6))
            cradle=cradle+rail
        # Both mounting holes are on the same tangential mounting ear.
        # A continuous rear post connects both bosses to the floor/load path.
        post=local(box(-3.77-p.cradle_post_depth_mm/2,8.4,z0-3.0,
                       p.cradle_post_depth_mm,4.6,18.6))
        cradle=cradle+post
        for hole_z in (z0+2.0,z0+13.0):
            # 1.6 mm pilot, >=1.2 mm surrounding PETG for M2 self-tappers.
            pilot=b.Cylinder(.8,6.4,rotation=(0,90,0),
                align=(b.Align.CENTER,b.Align.CENTER,b.Align.CENTER))
            pilot=local(pilot.moved(b.Pos(-6.3,8.4,hole_z)))
            cradle=cradle-pilot
        # Keep J1's entire 8.54 mm front niche and the inboard J2 FFC
        # passage unobstructed. Neither crosses the mounting ear at t=8.4.
        j1=local(box(1.50,-2.15,z0+9.7,8.54+0.6,10.7,5.7))
        j2=local(box(-6.4,-2.0,z0+2.0,5.8,12.0,9.0))
        cradle=cradle-j1-j2
        # The released carrier has a rear component between H1 and H2
        # (native X 6.1..8.9, Y 16.85..19.75). Split the two boss lobes
        # around it, leaving a 2.8 mm rear spine and a printable pitched roof.
        notch=b.Wire.make_polygon([(-5.62,6.0,19.3),(-3.40,6.0,19.3),
            (-3.40,6.0,24.792),(-5.62,6.0,22.35)],close=True)
        cradle=cradle-local(b.Solid.extrude(b.Face(notch),(0,4.8,0)))
        # Open triangular diaphragms share the cradle and shell sill.
        # The inboard triangle transfers the M2 shelf load to the bed rail;
        # the outboard triangle carries that rail into the lower bezel wall.
        # Both inclined chords have >=1.24 mm normal gauge and rise >45 deg.
        for tangent in (-9.4,9.4):
            g=p.structural_gauge_mm
            y=tangent-p.cradle_ring_web_mm/2
            grade=1.12
            def triangle_web(x0,x1,height,rising):
                def poly(points):
                    w=b.Wire.make_polygon([(x,y,z) for x,z in points],close=True)
                    return b.Solid.extrude(b.Face(w),(0,p.cradle_ring_web_mm,0))
                if rising:
                    outer=poly([(x0,0),(x1,0),(x1,height)])
                    c=-grade*x0-g*math.sqrt(1+grade*grade)
                    inner=poly([((g-c)/grade,g),(x1-g,g),
                                (x1-g,grade*(x1-g)+c)])
                else:
                    outer=poly([(x0,0),(x1,0),(x0,height)])
                    c=height+grade*x0-g*math.sqrt(1+grade*grade)
                    inner=poly([(x0+g,g),((c-g)/grade,g),
                                (x0+g,c-grade*(x0+g))])
                return outer-inner
            cradle=cradle+local(triangle_web(-7.7,-7.7+(z0-1.7)/grade,z0-1.7,False))
            cradle=cradle+local(triangle_web(6.8,16.5,grade*(16.5-6.8),True))
        # All cradle features stay within the common hull, never outside it.
        body=body+(cradle & outer_hull)

    # Restore the bezel lands after the internal mounting clearances; their
    # 0.8 mm sheet overlap is independent of the conservative carrier box.
    for ring in bezel_rings: body=body+ring
    for cut in bezel_cuts: body=body-(cut & outer_hull)
    # Round the service-corner junctions inside the original spar
    # envelope. These radii occupy empty carrier corners, outside the real PCB, header, rear-component and FFC envelopes.
    for key,pos in placements.items():
        if key not in tof_poses: continue
        cx,cy,z0,angle=carrier_mount_frame(pos)
        for radial_sign,sign in ((-1,-1),(-1,1),(1,-1),(1,1)):
            def point(u,v): return (radial_sign*(6.07-u),sign*(10.6-v),0.0)
            # Spread the outer carrier/spar corner into the pitched web.
            # The larger outer radius removes a tiny acute return at the
            # crash-load junction; the inboard connector-side radii retain
            # their original clearance. Clip every pad to the closed spar.
            # A broader connector-side inboard return spreads the roof
            # load without a near-tangent lip on the spar's inner pitch.
            # Stop 0.2 mm short of the FFC passage; the screw-ear-side
            # return retains its original radius and pilot clearance.
            radius=2.4 if radial_sign > 0 or sign < 0 else 1.8
            # Extend the return into the existing spar by 0.28 mm.
            # This removes a near-coplanar cradle/roof sliver while
            # keeping the same inner corner radius and carrier clearance.
            overlap=.28
            a,c,d=point(-overlap,-overlap),point(radius,-overlap),point(radius,0)
            f,g=point(0,radius),point(-overlap,radius)
            mid=radius*(1-1/math.sqrt(2))
            wire=b.Wire([b.Edge.make_line(a,c),b.Edge.make_line(c,d),
                b.Edge.make_three_point_arc(d,point(mid,mid),f),
                b.Edge.make_line(f,g),b.Edge.make_line(g,a)])
            # The service cut starts at the carrier's seating plane.
            # Round only that cut: extending the pad to the bed deposited
            # redundant islands on the sloping internal spar floor and
            # produced near-zero-volume tetrahedra at the overlap return.
            # Lap 1.5 mm into the supporting cradle instead of meeting
            # its seating-plane edge tangentially (a non-manifold T join).
            pad_z=max(3.0,z0-1.5)
            pad=b.Solid.extrude(b.Face(wire),(0,0,31.0-pad_z))
            pad=pad.rotate(b.Axis.Z,angle).moved(b.Pos(cx,cy,pad_z))
            for envelope in arm_envelopes:
                rib=pad & envelope
                if rib is not None and rib.volume>1e-7: body=body+rib

    # Continuous first-layer longerons join the nose, battery floor, and
    # tail pad to the arm-root junction and the perimeter shell.
    for rail_y in (-10.0,10.0):
        rail=box(-21.2,rail_y,0,245.6,p.payload_rail_width_mm,wall)
        body=body+(rail & outer_hull)

    # R6: ribbed battery tray with a continuous bed skin, fixed 2.0 mm
    # seating plane and retaining sills. Spar drains do not pierce this bay.
    bx,by,bz=(v*1000 for v in placements['battery'])
    dx,dy,dz=(v*1000 for v in LIBRARY['battery'].dims_m)
    battery_clear=box(bx,by,bz,dx+0.6,dy+0.6,dz+0.6)
    body=body-battery_clear
    # Bed skin takes shear; orthogonal ribs transfer battery inertia into
    # the perimeter sills. Rib tops retain the fixed Z=2 seating plane.
    gauge=p.structural_gauge_mm
    tray=box(bx,by,0,dx+2*wall+0.6,dy+2*wall+0.6,gauge)
    for ry in (-dy/2,0.0,dy/2):
        tray=tray+box(bx,by+ry,0,dx+2*wall+.6,gauge,bz)
    count=math.ceil(dx/p.tray_rib_pitch_mm)
    for i in range(count+1):
        rx=-dx/2+dx*i/count
        tray=tray+box(bx+rx,by,0,gauge,dy+2*wall+.6,bz)
    for side in (-1,1):
        tray=tray+box(bx,by+side*(dy/2+.3+wall/2),0,dx+2*wall,wall,bz+3.0)
    tray=tray+box(bx-dx/2-.3-wall/2,by,0,wall,dy+2*wall,bz+3.0)
    body=body+tray

    def ribbed_pad(x,y,dx,dy,seat_z):
        # Continuous bed skin and orthogonal ribs retain the original
        # seating plane and perimeter, with pockets between contact ribs.
        g=p.structural_gauge_mm
        pad=box(x,y,0,dx,dy,g)
        nx=max(1,math.ceil((dx-g)/p.tray_rib_pitch_mm))
        ny=max(1,math.ceil((dy-g)/p.tray_rib_pitch_mm))
        for i in range(nx+1):
            xx=x-(dx-g)/2+(dx-g)*i/nx
            pad=pad+box(xx,y,0,g,dy,seat_z)
        for j in range(ny+1):
            yy=y-(dy-g)/2+(dy-g)*j/ny
            pad=pad+box(x,yy,0,dx,g,seat_z)
        return pad

    # GPS remains in the tail, directly on a real pad under the south ToF.
    gx,gy,gz=(v*1000 for v in placements['gps'])
    body=body-box(gx,gy,gz,22.6,20.6,7.5)
    body=body+ribbed_pad(gx,gy,24.4,23.6,gz)

    # The camera STEP is taller and much shallower radially than the legacy
    # dimensions used by containment. The mount clears their union and
    # supports both the actual PCB edge and the evaluator's conservative box.
    from components import _apply_orientation
    from pathlib import Path
    import components
    camera_path=Path(components.__file__).parent/LIBRARY['pi_camera_3'].step_path
    camera_brep=Path(str(camera_path)+'.brep')
    camera_shape=(b.import_brep(str(camera_brep)) if camera_brep.exists()
                  else b.import_step(str(camera_path)))
    camera_shape=_apply_orientation('pi_camera_3',camera_shape)
    cbb=camera_shape.bounding_box()
    for key,pos in placements.items():
        if not key.startswith('pi_camera_3#'): continue
        cx,cy,cz=(v*1000 for v in pos)
        cdx=max(25.0,cbb.size.X)+.6
        cdy=max(24.0,cbb.size.Y)+.6
        cdz=max(12.0,cbb.size.Z)+.6
        body=body-box(cx,cy,cz,cdx,cdy,cdz)
        body=body+ribbed_pad(cx,cy,cdx+wall*2,cdy+wall*2,cz)
        for side in (-1,1):
            body=body+box(cx-cbb.size.X/2-.3-wall/2,cy+side*8,
                           0,wall,4.0,cz+5.0)

    # Corrugated mounting deck: two outer rails carry the PCB's underside;
    # pitched faces and vertical piers print from the bed without a wide
    # suspended horizontal ceiling. The aft edge stops ahead of the battery.
    fx,fy,fz=(v*1000 for v in placements['fc_esc_stack'])
    deck_z=fz-1.2
    deck_x0=fx-55.3
    deck_x1=fx+55.3
    # A physical corrugated sheet, 1.3 mm normal gauge, with 2 mm ridge flats.
    profile=[]
    for y in (-26.0,-13.0,0.0,13.0,26.0):
        profile.append((y,deck_z-6.71))
        if y<26:
            profile.extend([(y+6.1,deck_z),(y+6.9,deck_z)])
    # V-vault the underside of each narrow ridge bearing land. The
    # old 0.8 mm horizontal bridge retained 1.94 mm of material; a
    # 1.12:1 pointed underside removes its center while leaving >=1.3 mm
    # normal skin and 1.49 mm below the flat PCB contact at the crown.
    # Both sides print from the existing inclined folds, and the bearing
    # surface and the IMU landing remain at their original Z datums.
    lower=[]
    for i in range(len(profile)-1,-1,-1):
        y,z=profile[i]
        lower.append((y,z-1.94))
        if i>0 and abs(z-profile[i-1][1])<1e-9:
            prev_y=profile[i-1][0]
            lower.append(((y+prev_y)/2,z-1.94+1.12*(y-prev_y)/2))
    wire=b.Wire.make_polygon([(deck_x0,fy+y,z) for y,z in profile+lower],close=True)
    deck=b.Solid.extrude(b.Face(wire),(deck_x1-deck_x0,0,0))
    # Keep the two outer load-bearing deck rails; the open center admits
    # underside wiring and access, with broad support along both PCB edges.
    deck=deck-box((deck_x0+deck_x1)/2,fy,0,deck_x1-deck_x0+2,26.0,60)
    # The real battery occupies the aft inner corner below the flight board.
    deck=deck-box(bx,by,bz,dx+.6,dy+.6,dz+.4)
    # Local flat IMU landing over the north deck rail. Its undersides retain
    # the deck's existing pitched faces rather than adding a flat ceiling.
    for pts in ([(13,deck_z-6.71),(19.1,deck_z),(13,deck_z)],
                [(19.9,deck_z),(26,deck_z-6.71),(26,deck_z)]):
        w=b.Wire.make_polygon([(placements['mpu9250'][0]*1000-13,fy+y,z) for y,z in pts],close=True)
        deck=deck+b.Solid.extrude(b.Face(w),(26,0,0))
    # End/side piers receive this sheet and tie it into the four arm roots.
    for x in (deck_x0+wall/2,deck_x1-wall/2):
        for y in (-26.0,26.0):
            deck=deck+box(x,fy+y,0,wall,wall,deck_z-5.7)
    for y in (-26.0,26.0):
        deck=deck+box((deck_x0+deck_x1)/2,fy+y,0,deck_x1-deck_x0,wall,deck_z-5.7)
    # Swept piers carry the edge deck into the arm roots through a deep
    # open web. Broader, taller vaults remove unloaded panel centers while
    # retaining the 2 mm lower flange and >1.2 mm piers at both deck ends.
    # The roofs rise 1.12:1; the small side lean also builds from below.
    for x in (-44.0,-22.0,0.0,22.0,44.0):
        half=p.deck_arch_half_span_mm+1.6
        lean=math.copysign(0.3 if abs(x)>40 else 1.2,x) if x else 0.0
        apex=deck_z-7.7
        shoulder=apex-1.12*half
        arch=b.Wire.make_polygon([(fx+x-half,fy-28,2.0),
            (fx+x+half,fy-28,2.0),(fx+x+half+lean,fy-28,shoulder),
            (fx+x+lean,fy-28,apex),(fx+x-half+lean,fy-28,shoulder)],close=True)
        deck=deck-b.Solid.extrude(b.Face(arch),(0,56,0))
    # Paired rows of normal-cut slots lighten the folded rail between its
    # bed-founded piers. Each opening spans only 2.4 mm along X: its two
    # ends support a short bridge on every print layer as the fold rises.
    # Continuous ridge bearing lands, 1.4 mm inner/outer edge flanges and
    # 4.1 mm transverse ties retain the saddle's shear path. Keep the full
    # IMU bearing ridges; the flat landing receives the same short vents.
    for side in (-1,1):
        for i in range(15):
            x=-45.5+6.5*i
            # Slot ends bridge only 2.4 mm; continuous side flanges and
            # the central bearing ridge support the complete IMU footprint.
            for y in (16.0,23.0):
                pts=[(fx+x-1.2,fy+side*y-1.5),
                     (fx+x+1.2,fy+side*y-1.5),
                     (fx+x+1.2,fy+side*y+1.5),
                     (fx+x-1.2,fy+side*y+1.5)]
                imu_x=placements['mpu9250'][0]*1000
                if abs(fx+x-imu_x)<14.2:
                    # The flat IMU landing keeps vertical service holes.
                    deck=deck-prism(pts,0,deck_z+1.0)
                else:
                    # Cut normal to the folded sheet. Vertical slot walls
                    # shaved triangular feather edges from its underside;
                    # these perpendicular returns retain the full normal
                    # sheet gauge at both ends of every opening.
                    grade=1.10 if y<19.5 else -1.10
                    top=deck_z-1.10*(abs(y-19.5)-.4)
                    pl=b.Plane(origin=(fx+x,fy+side*y,top-.97),
                               x_dir=(1,0,0),z_dir=(0,-side*grade,1))
                    # Point both ends: their inclined edge faces rise
                    # from the 2.4 mm side jambs instead of closing with
                    # a downward-facing straight lintel along the fold.
                    outline=[(-1.2,-1.75),(0,-2.45),(1.2,-1.75),
                             (1.2,1.75),(0,2.45),(-1.2,1.75)]
                    w=b.Wire.make_polygon([(u,v,-2.0) for u,v in outline],close=True)
                    tool=pl*b.Solid.extrude(b.Face(w),(0,0,4.0))
                    deck=deck-tool
    # Do not introduce a deck wall through the original arm wiring galleries.
    for cavity in arm_cavities: deck=deck-cavity
    body=body+deck
    sh=p.stack_spacing_mm/2
    for x,y in ((sh,sh),(-sh,sh),(-sh,-sh),(sh,-sh)):
        body=body+b.Pos(x,y,0)*b.Cylinder(p.stack_standoff_dia_mm/2,fz,
            align=(b.Align.CENTER,b.Align.CENTER,b.Align.MIN))
        hole=b.Pos(x,y,-1)*b.Cylinder(p.stack_hole_dia_mm/2,fz+3,
             align=(b.Align.CENTER,b.Align.CENTER,b.Align.MIN))
        body=body-hole
    # Clear the full PCBA service envelope, including the two aft upper
    # corners where the hip roof approaches the CM4 component envelope.
    # Keep the mounting contact plane; the deck and standoffs lie below it.
    body=body-box(fx,fy,fz,112.0,56.0,24.2)
    # IMU pose is U6 on the flight board, with the shared deck beneath it.

    # Forward service portal above the camera/ToF ring admits the PCBA.
    # It terminates below the continuous roof, preserving the canopy load path.
    # The front service opening was cut in the shell before adding mounts.

    # Camera apertures use the measured lens origins, never metadata boxes.
    for pose in camera_poses.values():
        ox,oy,oz=(v*1000 for v in pose['origin_m'])
        hh=math.tan(math.radians(pose['hfov_deg']/2+1.0))
        vh=math.tan(math.radians(pose['vfov_deg']/2+1.0))
        def rect(x):
            w=hh*(x-ox)+.5; h=vh*(x-ox)+.5
            return b.Wire.make_polygon([(x,oy-w,oz-h),(x,oy+w,oz-h),
                (x,oy+w,oz+h),(x,oy-w,oz+h)],close=True)
        body=body-b.Solid.make_loft([rect(ox+.5),rect(125.0)],ruled=True)

    # Restore the fixed through motor holes on the resectioned arms.
    for mx,my in p.motor_positions():
        hs=p.motor_hole_spacing_mm/2
        for dx,dy in ((hs,hs),(-hs,hs),(-hs,-hs),(hs,-hs)):
            body=body-b.Pos(mx+dx,my+dy,-1)*b.Cylinder(p.motor_hole_dia_mm/2,120,
                align=(b.Align.CENTER,b.Align.CENTER,b.Align.MIN))
        body=body-b.Pos(mx,my,-1)*b.Cylinder(p.motor_center_hole_dia_mm/2,120,
            align=(b.Align.CENTER,b.Align.CENTER,b.Align.MIN))
    # Bed-facing drains remain outside the restored battery floor.
    for mx,my in p.motor_positions():
        arm_sweep_sign=-1.0 if mx*my>0 else 1.0
        ang=math.degrees(math.atan2(my,mx))
        for x,ro,ri,h in ((45.0,3.2,1.6,3.2),(115.0,2.3,1.0,2.5)):
            collar=b.Pos(x,sweep_center(x),0)*b.Cylinder(ro,h,
                align=(b.Align.CENTER,b.Align.CENTER,b.Align.MIN))
            throat=b.Pos(x,sweep_center(x),-.2)*b.Cylinder(ri,h+.8,
                align=(b.Align.CENTER,b.Align.CENTER,b.Align.MIN))
            body=body+collar.rotate(b.Axis.Z,ang)
            body=body-throat.rotate(b.Axis.Z,ang)
    # Drain every trapped cradle/spar vault through a small vertical port.
    # Connected internal galleries produce one watertight STL boundary.
    for boundary in list(body.shells()):
        enclosed=b.Solid(boundary)
        if 150 < abs(enclosed.volume) < 5000:
            c=enclosed.center()
            vent=b.Pos(c.X,c.Y,-.2)*b.Cylinder(1.0,c.Z+1.0,
                align=(b.Align.CENTER,b.Align.CENTER,b.Align.MIN))
            body=body-vent
        elif 0<abs(enclosed.volume)<=150:
            body=body+enclosed
    # Regularize micron-scale Boolean wire gaps at folded saddle/spar
    # intersections. This is 1/1200 of the minimum wall gauge; it removes
    # numerical sliver faces before STEP meshing without changing the
    # printable load paths, optical openings or component clearances.
    from OCP.ShapeFix import ShapeFix_Wireframe
    wireframe=ShapeFix_Wireframe(body.wrapped)
    wireframe.SetPrecision(0.001)
    wireframe.SetMaxTolerance(0.001)
    wireframe.ModeDropSmallEdges=True
    wireframe.FixSmallEdges()
    wireframe.FixWireGaps()
    body=b.Part(wireframe.Shape()).clean()
    if not body.is_valid or len(body.solids()) != 1:
        raise ValueError(f'Chassis must be one valid solid: valid={body.is_valid}, solids={[(round(s.volume,2),tuple(s.center())) for s in body.solids()]}')
    return b.Part(b.Part(children=body.solids()).wrapped)

if __name__ == '__main__':
    p=ChassisParams()
    part=build_chassis(p)
    print(f'volume {part.volume:.1f} mm^3; PETG mass {part.volume*0.00124:.2f} g')
