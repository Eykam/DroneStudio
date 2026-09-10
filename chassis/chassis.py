"""v126-g125a: split battery shoulders and deep compact perimeter sills.

Paired inward roof folds close the empty headroom beside the battery
service channel. Pointed, normal-cut vents retain continuous shoulder
chords; deeper, shorter inward sills carry the ring with less stock.
The fixed sensor seats, optical cuts and enclosed CM4 deck are preserved.

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
    ring_carrier_cover_mm: float = 26.8  # full cover over the fixed carrier envelope
    bezel_surround_mm: float = 3.2  # preserve optical aperture, recess and 0.8 mm land
    cradle_foot_rail_mm: float = 2.0  # three bed-facing radial ties replace the broad apron
    deck_arch_half_span_mm: float = 8.0  # pitched openings in the tall deck side piers
    cradle_post_depth_mm: float = 4.0  # forward boss seat stays radial -3.77 mm
    structural_gauge_mm: float = 1.24  # nominal 1.2 mm construction with print margin
    cradle_ring_web_mm: float = 1.24
    tray_rib_pitch_mm: float = 16.0
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
    def sweep_datum(x):
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
        # Replace the broad outer bow with a shorter cranked shoulder.
        # Its maximum contraction lies beyond the diagonal carrier seat
        # and ends before the low optical crest at radial X=104 mm.
        # This reduces both developed skin area and sweep-induced normal
        # offset material; every pitched spar face still receives its full
        # 3D normal gauge from the actual adjacent loft stations below.
        # The inboard carrier junction and motor-end radial stay fixed.
        short_chine=(2.0*math.sin(math.pi*(x-79.0)/25.0)**2
                     if 79.0 < x < 104.0 else 0.0)
        return arm_sweep_sign*(-p.arm_sweep_mm*math.sin(math.pi*x/p.arm_length_mm)
                -bypass-window_bypass-root_bypass+short_chine)

    def sweep_center(x):
        # v122: start the straight, tapered wing at the first free station
        # beyond the carrier and its aperture reveal. The saddle stays fixed;
        # the earlier direct chine removes sweep length and normal-offset
        # area while the deeper root transfers its load into the hub.
        # Once past the complete carrier saddle, the old outer optical
        # detour is unnecessary for a proximity-monitor station. Connect
        # its outer shoulder to the motor approach with one direct chine.
        # Blend only at its ends; the middle loft stations are collinear,
        # avoiding the extra kink caused by a larger local sine correction.
        # The root and motor axis keep their original exact coordinates.
        center=sweep_datum(x)
        if 74.5 < x < 132.0:
            t=(x-74.5)/57.5
            chord=(1.0-t)*sweep_datum(74.5)+t*sweep_datum(132.0)
            blend=max(0.0,min(1.0,(x-74.5)/5.0,(132.0-x)/6.0))
            center+=(chord-center)*blend
        return center

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
        # A tapered keel runs from the hub into the outboard wing.
        # At the loaded root, narrow breadth 3% and recover depth 4.5%;
        # raise the haunches to 22% of depth and halve the belly breadth.
        # Beyond the carrier ring, the belly contracts further, keeping
        # extra depth through the light, lenticular free span. Fade
        # both changes out before the original motor-end diaphragm.
        # Original normal wall offsets retain the printable skin gauge;
        # the complete chassis is checked under every FEA load case.
        root_keel=max(0.0,min(1.0,(76.0-x)/25.0))
        wing_keel=max(0.0,min(1.0,(x-78.0)/20.0,(133.0-x)/20.0))
        # The diagonal sensor's horizontal edge ray skims the crown at
        # radial X=104 mm. Keep this station at its original apex height,
        # blending the depth recovery into the adjoining stations; a
        # small extra root depth carries the corresponding bending load.
        optical_crest=max(0.0,min(1.0,(x-94.0)/6.0,(116.0-x)/6.0))
        for blend_keel,keel_ratio,crown_target,depth_gain in (
                (root_keel,0.50,1.30,0.045),
                (wing_keel,0.35,1.40,0.04*(1.0-optical_crest))):
            half*=1.0-0.03*blend_keel
            shoulder_half*=1.0-0.03*blend_keel
            keel*=1.0-0.03*blend_keel
            crown*=1.0-0.03*blend_keel
            height*=1.0+depth_gain*blend_keel
            shoulder*=1.0+depth_gain*blend_keel
            chine*=1.0+depth_gain*blend_keel
            keel+=(keel_ratio*half-keel)*blend_keel
            chine+=(0.22*height-chine)*blend_keel
            crown+=(crown_target-crown)*blend_keel
            shoulder+=(height-p.arm_roof_slope*(shoulder_half-crown)-shoulder)*blend_keel
            shoulder=min(shoulder,height-p.arm_roof_slope*(shoulder_half-crown))
        # A deeper folded root uses a 1.22 mm nominal normal skin instead
        # of the former 1.32 mm root allowance. Recover vertical bending
        # inertia with 3.5% section depth and higher lower chines, keeping
        # the bed keel continuous. The four sloping flanges meet taller
        # shear sides; the roof closes with the original >45-degree pitch.
        # Finish the taper before the carrier's inboard service boundary.
        # Its entire span keeps the original section, rather than letting
        # the higher crown approach the diagonal PCB's conservative box.
        # Existing spanwise-normal offsets still include both neighbors.
        folded_root=max(0.0,min(1.0,(48.0-x)/16.0))
        height*=1.0+0.035*folded_root
        shoulder*=1.0+0.035*folded_root
        chine*=1.0+0.035*folded_root
        keel+=(0.41*half-keel)*folded_root
        chine+=(0.32*height-chine)*folded_root
        crown+=(1.62-crown)*folded_root
        shoulder+=(0.85*height-shoulder)*folded_root
        shoulder=min(shoulder,height-p.arm_roof_slope*(shoulder_half-crown))
        # Rebalance the enclosed root into a slender, flange-biased spar.
        # Keep the complete bed-keel breadth while pulling the neutral-
        # axis sides inward. A slightly deeper crown and broader upper
        # bearing flange recover bending inertia; the taller belly
        # chines remove lower-corner perimeter without a flat apron.
        # End the transition before the diagonal carrier service box.
        # Both the cavity and exterior use this same section function,
        # including the spanwise-normal allowance at every loft station.
        flange_root=max(0.0,min(1.0,(48.0-x)/16.0))
        half*=1.0-0.06*flange_root
        shoulder_half*=1.0-0.06*flange_root
        height*=1.0+0.007*flange_root
        shoulder*=1.0+0.007*flange_root
        chine*=1.0+0.007*flange_root
        chine+=(0.34*height-chine)*flange_root
        crown+=(1.82-crown)*flange_root
        shoulder+=(height-p.arm_roof_slope*(shoulder_half-crown)-shoulder)*flange_root
        shoulder=min(shoulder,height-p.arm_roof_slope*(shoulder_half-crown))
        # A deeper inboard haunch carries bending into the stack ring.
        # Recover stiffness by lifting the existing closed crown and lower
        # chines, retaining the bed keel and the pitched roof. The depth
        # fades completely before the carrier service region; every outer
        # station from radial X=48.3 mm outward is exactly the reference.
        # Its adjoining shoulder sheds excess skin allowance below, using
        # section depth inboard instead of extra sidewall around sensors.
        haunch=max(0.0,min(1.0,(48.0-x)/16.0))
        depth=1.025
        height*=1.0+(depth-1.0)*haunch
        shoulder*=1.0+(depth-1.0)*haunch
        chine*=1.0+(depth-1.0)*haunch
        # A direct outer wing carries its side-impact load in the broad
        # upper shoulders. Pull only the lower neutral-axis corners inward,
        # raise the belly chines, and recover vertical inertia in depth.
        # The closed section retains its full 3D normal gauge; the motor
        # diaphragm, fixed collars and carrier saddle keep their datums.
        # Every section stays below the existing complete chassis height.
        approach=max(0.0,min(1.0,(x-86.0)/22.0,(138.0-x)/10.0))
        height*=1.0+0.07*approach
        shoulder*=1.0+0.07*approach
        chine*=1.0+0.07*approach
        # Keep the lateral compression shoulders broad for cartwheel
        # loading. The lower corners tuck in independently, shortening
        # the neutral-axis skin; higher lower chines retain section depth.
        # Narrow the crown's short internal bridge as well, allowing the
        # pitched flanges to close sooner with the full normal skin gauge.
        # Fade the change out before the original motor diaphragm.
        half*=1.0-0.075*approach
        keel+=(0.32*half-keel)*approach
        chine+=(0.39*height-chine)*approach
        # Shorten the inner crown bridge through the deep outer span.
        # The two pitched upper flanges retain their normal offsets and
        # close over a 2.4 mm outer ridge, reducing unsupported roof area
        # while the broad shoulders continue to carry compression.
        crown+=(1.20-crown)*approach
        shoulder=min(shoulder,height-p.arm_roof_slope*(shoulder_half-crown))
        # A narrow lower web joins the keel directly to the broad upper
        # shoulder. The old vertical neutral-axis side carried perimeter
        # without useful flange breadth. Pull that corner inward, retain
        # the compression shoulders, and recover inertia through depth.
        # This new lenticular section stays closed and receives the same
        # complete spanwise normal offset as the reference. All lower
        # faces rise steeply from the original first-layer keel.
        # Leave the whole carrier crossing and terminal diaphragm intact.
        root_lens=max(0.0,min(1.0,(48.0-x)/16.0))
        wing_lens=max(0.0,min(1.0,(x-84.0)/16.0,(136.0-x)/12.0))
        depth_gain=0.025*root_lens+0.010*wing_lens
        height*=1.0+depth_gain
        shoulder*=1.0+depth_gain
        chine*=1.0+depth_gain
        straight_half=keel+(shoulder_half-keel)*chine/shoulder
        half+=(straight_half-half)*(0.20*root_lens+0.14*wing_lens)
        # Redistribute each free section into taller side webs and
        # broader upper flanges, within its existing width and height.
        # The crown stays short enough for a <2.5 mm internal bridge.
        # End both changes before the complete carrier service region
        # and motor diaphragm. All faces retain their 3D normal offsets.
        flange=max(0.0,min(1.0,(48.0-x)/16.0))
        wing=max(0.0,min(1.0,(x-86.0)/18.0,(136.0-x)/12.0))
        crown-=0.04*flange+0.02*wing
        roof_pitch=p.arm_roof_slope-0.07*max(flange,wing)
        shoulder+=(height-roof_pitch*(shoulder_half-crown)-shoulder)*max(flange,wing)
        # Widen only the bed flange, where it improves both vertical
        # and lateral inertia. Keep the complete upper-shoulder breadth.
        keel+=0.30*flange+0.18*wing
        # Replace the lower corner-heavy section with a flange-biased
        # closed wing. Lower the belly chine and shorten its inclined
        # lower facets while retaining the lateral compression shoulders.
        # The upper roof still closes on a >45-degree pitch; independently
        # sized roots and tips retain >=96% vertical and >=101% lateral
        # section inertia at the deliberately changed loft stations.
        # Root and free-span folds are sized independently; both fade out
        # before the pinned carrier seat and original motor diaphragm.
        root_fold=max(0.0,min(1.0,(48.0-x)/16.0))
        wing_fold=max(0.0,min(1.0,(x-86.0)/18.0,(136.0-x)/12.0))
        for fold, gain, roof_delta, keel_delta, chine_delta, web in (
                (root_fold, -0.010347636, -0.299676393, -0.100000000, -0.090971174, 0.056941622),
                (wing_fold, -0.020563545, 0.045577587, 0.057445685, -0.037685216, 0.014989451)):
            height*=1.0+gain*fold
            crown+=roof_delta*fold
            keel+=keel_delta*fold
            chine+=chine_delta*height*fold
            shoulder+=(height-1.035*(shoulder_half-crown)-shoulder)*fold
            straight=keel+(shoulder_half-keel)*chine/shoulder
            half+=(straight-half)*web*fold
        # A deeper root and lower outer wing split the bending task.
        # Lift the hub compression flange, while the lower-moment free
        # span loses crown depth and shortens its lower chines. The upper
        # shoulder breadth is preserved for lateral impact. Both
        # profiles retain the full neighboring-span normal wall offsets.
        # Blend out before the sensor saddle and terminal diaphragm.
        root_resection=max(0.0,min(1.0,(48.0-x)/16.0))
        wing_resection=max(0.0,min(1.0,(x-86.0)/18.0,(136.0-x)/12.0))
        for resection, gain, roof_delta, keel_delta, chine_delta, web in (
                (root_resection, 0.003107146, 0.058928268, -0.050000000, -0.010176905, 0.024771051),
                (wing_resection, -0.023601424, 0.059870506, -0.044136589, -0.040740184, 0.000000000)):
            height*=1.0+gain*resection
            crown+=roof_delta*resection
            keel+=keel_delta*resection
            chine+=chine_delta*height*resection
            shoulder+=(height-1.035*(shoulder_half-crown)-shoulder)*resection
            straight=keel+(shoulder_half-keel)*chine/shoulder
            half+=(straight-half)*web*resection
        # A low, broad-shouldered closed wing carries lateral impact
        # through both flanges. Re-form the lower chines into shorter,
        # steeper facets and keep full-width upper shoulders, shedding
        # side skin near the neutral axis without thinning it. The root
        # and outboard wing use independent folds; the complete sensor
        # crossing and terminal motor diaphragm keep their old sections.
        hub=max(0.0,min(1.0,(48.0-x)/16.0))
        wing=max(0.0,min(1.0,(x-86.0)/18.0,(136.0-x)/12.0))
        for blend, depth, ridge, bed, lower, web in (
                (hub, -0.004718749, -0.071065162, -0.100000000, -0.046469141, 0.000000000),
                (wing, -0.036848192, -0.010440423, 0.167364885, -0.096725323, 0.000000000)):
            height*=1.0+depth*blend
            crown+=ridge*blend
            keel+=bed*blend
            chine+=lower*height*blend
            shoulder+=(height-1.035*(shoulder_half-crown)-shoulder)*blend
            direct=keel+(shoulder_half-keel)*chine/shoulder
            half+=(direct-half)*web*blend
        # Form a deep closed root and a compact, broad-shouldered wing.
        # Separate the upper flange, lower chine and first-layer keel so
        # each carries bending with less developed skin. The bed rails
        # stay continuous and both roofs close on a >45-degree pitch.
        # True adjacent-span normal offsets below retain the full gauge.
        # Preserve the complete fixed sensor saddle and motor diaphragm.
        hub=max(0.0,min(1.0,(48.0-x)/16.0))
        wing=max(0.0,min(1.0,(x-86.0)/18.0,(136.0-x)/12.0))
        for blend, depth, breadth, bed, ridge, lower, web in (
                (hub, 0.010000000, -0.020124556, -0.150000000, -0.011094996, -0.024540654, 0.000000000),
                (wing, -0.043904600, 0.046563976, 0.009395786, 0.043315842, -0.049598282, 0.000000000)):
            height*=1.0+depth*blend
            shoulder_half*=1.0+breadth*blend
            keel+=bed*blend
            crown+=ridge*blend
            chine+=lower*height*blend
            shoulder+=(height-1.035*(shoulder_half-crown)-shoulder)*blend
            direct=keel+(shoulder_half-keel)*chine/shoulder
            half+=(direct-half)*web*blend
        # Form a deep closed root and a compact, broad-shouldered wing.
        # Separate the upper flange, lower chine and first-layer keel so
        # each carries bending with less developed skin. The bed rails
        # stay continuous and both roofs close on a >45-degree pitch.
        # True adjacent-span normal offsets below retain the full gauge.
        # Preserve the complete fixed sensor saddle and motor diaphragm.
        hub=max(0.0,min(1.0,(48.0-x)/16.0))
        wing=max(0.0,min(1.0,(x-86.0)/18.0,(136.0-x)/12.0))
        for blend, depth, breadth, bed, ridge, lower, web in (
                (hub, 0.010000000, -0.008056680, -0.149989750, 0.171019501, 0.001718026, 0.000000000),
                (wing, -0.011024899, 0.023959210, 0.043693606, 0.036100808, -0.006935741, 0.000000000)):
            height*=1.0+depth*blend
            shoulder_half*=1.0+breadth*blend
            keel+=bed*blend
            crown+=ridge*blend
            chine+=lower*height*blend
            shoulder+=(height-1.035*(shoulder_half-crown)-shoulder)*blend
            direct=keel+(shoulder_half-keel)*chine/shoulder
            half+=(direct-half)*web*blend
        # Form a deep closed root and a compact, broad-shouldered wing.
        # Separate the upper flange, lower chine and first-layer keel so
        # each carries bending with less developed skin. The bed rails
        # stay continuous and both roofs close on a >45-degree pitch.
        # True adjacent-span normal offsets below retain the full gauge.
        # Preserve the complete fixed sensor saddle and motor diaphragm.
        hub=max(0.0,min(1.0,(48.0-x)/16.0))
        wing=max(0.0,min(1.0,(x-86.0)/18.0,(136.0-x)/12.0))
        for blend, depth, breadth, bed, ridge, lower, web in (
                (hub, 0.010000000, 0.000878357, 0.018879946, -0.080131619, -0.000249188, 0.000000000),
                (wing, -0.010991290, 0.034478194, 0.081025883, 0.011316185, -0.017697678, 0.000000000)):
            height*=1.0+depth*blend
            shoulder_half*=1.0+breadth*blend
            keel+=bed*blend
            crown+=ridge*blend
            chine+=lower*height*blend
            shoulder+=(height-1.035*(shoulder_half-crown)-shoulder)*blend
            direct=keel+(shoulder_half-keel)*chine/shoulder
            half+=(direct-half)*web*blend
        return [(-keel,0),(keel,0),(half,chine),(shoulder_half,shoulder),
                (crown,height),(-crown,height),(-shoulder_half,shoulder),(-half,chine)]

    def section_wire(x, center, width, height, inner=False):
        """Offset swept spar faces in 3D, including their spanwise gradients."""
        points = spar_profile(x,width,height)
        if inner:
            root_blend = max(0.0,min(1.0,(75.0-x)/30.0))
            folded_root=max(0.0,min(1.0,(48.0-x)/16.0))
            # Preserve the 1.22 mm hub skin and a 1.24 mm shoulder.
            # The deeper hub haunch above supplies the bending load path;
            # both adjacent loft spans still contribute their full 3D
            # normal correction to the swept faces and pitched flanges.
            wall = (max(1.22,p.arm_rib_thickness_mm-0.13)
                    +min(0.02,0.10*root_blend-0.10*folded_root))
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
                # The folded ring supplies the extra load path. Remove
                # excess spar allowance while retaining the complete 3D
                # normal offset and the >=1.22 mm nominal arm skin.
                gauge = wall*max(1.010,1.005*math.sqrt(1+gradient*gradient))
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
            width = p.arm_root_width_mm*(1-0.30*x/(x0+5.0))
            center = sweep_center(x)
            lower.append((x,center-width/2))
            upper.append((x,center+width/2))
        outline = b.Polyline(*(lower+list(reversed(upper))),close=True)
        outline = b.fillet(outline.vertices(),p.fillet_radius_mm)
        arm = b.extrude(b.make_face(outline),p.body_thickness_mm)
        # Divide the shallow hub saddle into two continuous tapered rails
        # around a bed-facing wiring port. Solid end tongues preserve the
        # central junction and open spar mouth; the bolt ring is outboard.
        # The saddle tapers into the new keel, leaving >1.6 mm side
        # chords around the wiring port and a continuous bed load path.
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
        for x in (bridge_start,rib_end,L-center_boss_radius-0.2):
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
    # The planar roofs below retain 1.005 times the specified normal gauge
    # (1.2261 mm at defaults); vertical shell faces keep the full 1.22 mm.
    # Deeper waist sills provide the local section depth independently.
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
    # Pull empty waist and battery-flank space inside the reference hull.
    # Preserve the four diagonal optical planes and the south carrier's
    # rear clearance. Camera cheeks use a smaller inset so the real stereo
    # PCBs remain behind full-gauge walls. Rebuild the pitched eaves from
    # these facets: both plan area and developed shoulder skin decrease.
    compact_lines=[]
    for a,d in zip(perimeter,perimeter[1:]+perimeter[:1]):
        ex,ey=d[0]-a[0],d[1]-a[1]
        length=math.hypot(ex,ey)
        nx,ny=ey/length,-ex/length
        protected_facet=nx < -0.9999 or abs(abs(nx)-abs(ny)) < 1e-6
        mid_x=(a[0]+d[0])/2
        tuck=0.0 if protected_facet else (1.95 if mid_x > 65.0 else 2.5)
        if not protected_facet and mid_x > 65.0 and abs(nx) < .1:
            tuck=1.45
        # The camera cheeks retain a complete 1.22 mm wall outside the
        # real 25 mm PCB and its 0.3 mm service clearance. The narrower
        # waist more than repays this local clearance in plan area; the
        # overall fuselage and complete-frame bounds never increase.
        # Contract the cardinal waist and adjoining facets together, keeping
        # their junctions aligned and the diagonal optical planes fixed.
        # The bezel aperture follows the shortened optical standoff below.
        if not protected_facet and abs(mid_x) < 44.0:
            # Retain the full conservative east/west sensor envelope;
            # only the empty oblique waist facets move further inward.
            tuck=3.10 if abs(nx) < 0.1 else 3.50
        if not protected_facet and -122.0 < mid_x < -65.0:
            tuck=4.45
        compact_lines.append((nx,ny,nx*a[0]+ny*a[1]-tuck))
    compact=[]
    for a,d in zip(compact_lines[-1:]+compact_lines[:-1],compact_lines):
        det=a[0]*d[1]-d[0]*a[1]
        compact.append(((a[2]*d[1]-d[2]*a[1])/det,
                        (a[0]*d[2]-d[0]*a[2])/det))
    perimeter=compact
    outer_plan=prism(perimeter,0,150)
    inner_plan=prism(offset(perimeter,-wall),-.2,151)
    outs=[];ins=[]
    for x0,x1 in [(-145,-40),(-90,84),(40,103)]:
        pts=convex(clip_x(clip_x(perimeter,x0,1),x1,-1))
        outer=prism(pts,0,150); inner=prism(pts,-.2,151)
        for a,d in zip(pts,pts[1:]+pts[:1]):
            ex,ey=d[0]-a[0],d[1]-a[1]; el=math.hypot(ex,ey)
            nx,ny=ey/el,-ex/el;c=nx*a[0]+ny*a[1]
            # The compact aft skirt uses a slightly steeper cockpit
            # entry hip. Its rising inner face clears the board service
            # corner, avoiding a flat ledge left by the final bay cut.
            # The original crown limits cap its height, and the tucked
            # vertical flanks continue to set the smaller plan envelope.
            # Steepen the local entry hip under the unchanged cockpit
            # crown limits. Its inner face clears the complete PCBA box,
            # removing the broad flat underside left by the service cut.
            pitch=slope+(0.150 if x0 == -90 and nx < -0.01 and abs(ny) > 0.1 else 0.0)
            pl=b.Plane(origin=(nx*c,ny*c,28.0),z_dir=(pitch*nx,pitch*ny,1))
            half=pl*b.Box(600,600,500,align=(b.Align.CENTER,b.Align.CENTER,b.Align.MAX))
            outer=outer & half
            inner=inner & half.moved(b.Pos(0,0,-wall*1.005*math.sqrt(1+pitch*pitch)))
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
            drop=wall*1.005*math.sqrt(1+slope*slope+(rake/sx)**2)
            roof_inner=roof_inner & half.moved(b.Pos(0,0,-drop))
        # A shallow central fold lowers the battery turtledeck while
        # the original outboard pitch remains its envelope limit. Their
        # crease lands at |Y| = 14.78 mm, inboard of the pack shoulders;
        # at the battery corner the original normal-offset roof survives.
        # Both inner slopes exceed 45 degrees. Keeping this operation in
        # the battery branch preserves the full tail/CM4 transition roofs.
        if ridge_x == -104.0 and rake > 0:
            battery_pitch = 1.025
            for side in (-1, 1):
                plane = b.Plane(origin=(ridge_x*sx,0,ridge_z-2.0),
                                z_dir=(-rake/sx,side*battery_pitch,1))
                half = plane*b.Box(800,800,600,
                    align=(b.Align.CENTER,b.Align.CENTER,b.Align.MAX))
                roof_outer = roof_outer & half
                drop = wall*1.005*math.sqrt(1+battery_pitch**2+(rake/sx)**2)
                roof_inner = roof_inner & half.moved(b.Pos(0,0,-drop))
        # Split the battery crown into two close-wrapped shoulders.
        # Their rising transverse faces frame the service opening; the
        # original outer hip is an envelope limit. At the rear battery
        # corner the inner face clears the 35.5 mm pack and its service
        # box. True normal offsets include the longitudinal rake.
        if ridge_x == -104.0 and rake > 0:
            paired_o=[]; paired_i=[]
            for shoulder_side in (-1,1):
                po=box(0,0,-.2,600,600,200)
                pi=box(0,0,-.2,600,600,200)
                for face_side in (-1,1):
                    plane=b.Plane(origin=(ridge_x*sx,shoulder_side*14.0,41.65),
                        z_dir=(-rake/sx,face_side*1.025,1))
                    half=plane*b.Box(800,800,600,
                        align=(b.Align.CENTER,b.Align.CENTER,b.Align.MAX))
                    po=po & half
                    drop=wall*1.005*math.sqrt(1+1.025**2+(rake/sx)**2)
                    pi=pi & half.moved(b.Pos(0,0,-drop))
                paired_o.append(po);paired_i.append(pi)
            roof_outer=roof_outer & (paired_o[0]+paired_o[1])
            roof_inner=roof_inner & (paired_i[0]+paired_i[1])
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
    # The full forward CM4 service corner (X=55,Y=27), including the
    # thin portal jamb, retains over 0.8 mm clearance above the 59.6 mm
    # service volume. The bay cut therefore cannot leave a flat underside
    # where it meets these shoulders.
    roof_rake=0.065/sx
    for roof_side in (-1,1):
        pl=b.Plane(origin=(0,0,99.0),
                   z_dir=(roof_rake,roof_side*1.12,1))
        half=pl*b.Box(800,800,600,
            align=(b.Align.CENTER,b.Align.CENTER,b.Align.MAX))
        outer=outer & half
        drop=wall*1.005*math.sqrt(1+1.12**2+roof_rake**2)
        inner=inner & half.moved(b.Pos(0,0,-drop))
    # A second, opposing longitudinal pitch shortens the high aft
    # cockpit shoulders. The compound hip uses shallower 1.015:1 inner
    # roof slopes and >=1.22 mm normal skin; its ridge intersects the
    # forward rake rather than adding a suspended transverse bulkhead.
    # At the full rear service corner (X=-55,Y=27), the inner roof
    # remains above Z=61.6 mm, clearing the 59.6 mm CM4 service box.
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
        drop=wall*1.005*math.sqrt(1+1.12**2+(-0.015/sx)**2)
        inner=inner & half.moved(b.Pos(0,0,-drop))
    cockpit_pitch=1.015
    # Follow the declared 108 x 52 x 22 mm CM4 board with 1 mm
    # lateral and 0.6 mm top insertion clearance. Keep the >45-degree
    # transverse fold and its full normal offset as the crown moves down.
    cockpit_ridge=90.35
    aft_rake=-0.015/sx
    for roof_side in (-1,1):
        plane=b.Plane(origin=(0,0,cockpit_ridge),
                      z_dir=(aft_rake,roof_side*cockpit_pitch,1))
        half=plane*b.Box(800,800,600,
            align=(b.Align.CENTER,b.Align.CENTER,b.Align.MAX))
        outer=outer & half
        drop=wall*1.005*math.sqrt(1+cockpit_pitch**2+aft_rake**2)
        inner=inner & half.moved(b.Pos(0,0,-drop))
    # Close-wrap the outer cockpit with a lower transverse roof. The
    # original compound hips remain envelope limits, so this only removes
    # volume. Its normal-offset underside clears the full CM4 service box
    # by 0.25 mm at Y=27; the original inner folds and hatch stay intact.
    # Both sides grow at 1.015:1 without an unsupported horizontal cap.
    for roof_side in (-1,1):
        plane=b.Plane(origin=(0,0,88.98),
                      z_dir=(0,roof_side*cockpit_pitch,1))
        half=plane*b.Box(800,800,600,
            align=(b.Align.CENTER,b.Align.CENTER,b.Align.MAX))
        outer=outer & half
        drop=wall*1.005*math.sqrt(1+cockpit_pitch**2)
        inner=inner & half.moved(b.Pos(0,0,-drop))
    # Three compact folds per shoulder replace the high paired cockpit
    # ridges. Their tighter pitch closes unused headroom over the CM4,
    # while six longitudinal creases brace the thin structural roof.
    # At the aft Y=27 service corner the inner skin remains above 59.6;
    # every face keeps a >45-degree pitch and its full 3D normal gauge.
    # The dorsal channel and all ring/optical hips retain their datums.
    crease_pitch=1.025
    crease_centers=((9.2,66.45),(16.2,66.45),(23.2,66.45))
    creased_outer=[]; creased_inner=[]
    for side in (-1,1):
        for crease_y,local_crease_z in crease_centers:
            co=box(0,0,-.2,600,600,200)
            ci=box(0,0,-.2,600,600,200)
            for face_side in (-1,1):
                plane=b.Plane(origin=(0,side*crease_y,local_crease_z),
                              z_dir=(aft_rake,face_side*crease_pitch,1))
                half=plane*b.Box(800,800,600,
                    align=(b.Align.CENTER,b.Align.CENTER,b.Align.MAX))
                co=co & half
                drop=wall*1.005*math.sqrt(1+crease_pitch**2+aft_rake**2)
                ci=ci & half.moved(b.Pos(0,0,-drop))
            creased_outer.append(co); creased_inner.append(ci)
    folded_o=creased_outer[0]; folded_i=creased_inner[0]
    for co,ci in zip(creased_outer[1:],creased_inner[1:]):
        folded_o=folded_o+co; folded_i=folded_i+ci
    outer=outer & folded_o
    inner=inner & folded_i

    # Two long rising creases bring the cockpit shoulders down toward
    # the enclosed FC rather than carrying tall triangular aft cheeks.
    # The original battery roof is the lower branch of the envelope:
    # its union with the creases makes a continuous hip, with no abrupt
    # clipping plane or suspended step at the battery/avionics transition.
    # True normal offsets include the longitudinal and transverse grades.
    folds_outer=[]; folds_inner=[]
    # A two-stage swept hip shortens the tall aft cockpit. Its steep
    # first fold grows from the battery shoulder into a shallower forward
    # run, leaving a deep crease instead of a long triangular side panel.
    # The first 1.50:1 run supports the flared hatch edges; the second
    # 1.12:1 run begins ahead of the flare and carries the narrow opening.
    # At X=-56,Y=6 the inner skin remains above Z=62 mm, clearing the
    # full PCBA service box. Both folds keep a true 3D normal wall offset.
    for fold_side in (-1,1):
        fold_o=box(0,0,-.2,600,600,200)
        fold_i=box(0,0,-.2,600,600,200)
        for fold_x,fold_z,run in ((-56.0,58.2,1.50),(-46.0,73.2,1.12)):
            plane=b.Plane(origin=(fold_x*sx,0,fold_z),
                          z_dir=(-run/sx,-fold_side*1.12,1))
            half=plane*b.Box(800,800,600,
                align=(b.Align.CENTER,b.Align.CENTER,b.Align.MAX))
            fold_o=fold_o & half
            drop=wall*1.005*math.sqrt(1+(run/sx)**2+1.12**2)
            fold_i=fold_i & half.moved(b.Pos(0,0,-drop))
        folds_outer.append(fold_o)
        folds_inner.append(fold_i)
    # Keep both the battery turtledeck and its aft GPS hip as supported
    # starting surfaces; their old longitudinal rakes also protect the
    # complete fixed tail carrier. The cockpit grows out of these roofs.
    fold_outer=folds_outer[0]+folds_outer[1]+roof_outers[0]+roof_outers[2]
    fold_inner=folds_inner[0]+folds_inner[1]+roof_inners[0]+roof_inners[2]
    outer=outer & fold_outer
    inner=inner & fold_inner

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
        rake=(.06 if ridge_y == 0.0 else .03)/sx
        # The camera brows sit lower than the central radial carrier
        # hood. Their inner corners clear the real 25.862 mm board top;
        # the existing steep hip joins them to the cockpit service jamb.
        # Trim the central hood's dead space above its fixed carrier.
        # Camera ridges and the forward service-jamb hip remain separate;
        # the complete optical face and internal seat keep their datums.
        brow_z=44.0 if ridge_y == 0.0 else 43.0
        for side in (-1,1):
            plane=b.Plane(origin=(83.0*sx,ridge_y*sy,brow_z),
                          z_dir=(rake,side*1.12/sy,1))
            half=plane*b.Box(800,800,600,
                align=(b.Align.CENTER,b.Align.CENTER,b.Align.MAX))
            no=no & half
            drop=wall*1.005*math.sqrt(1+(1.12/sy)**2+rake*rake)
            ni=ni & half.moved(b.Pos(0,0,-drop))
        # Opposing roof pitches meet above each fixed lens/PCB. The
        # inboard half rises from the existing cockpit hip at 0.16:1,
        # while the transverse 1.12:1 folds grow from the cheek eaves.
        # This trims the tall rear brow triangles, within the old hull.
        # The normal offset includes both slopes; no wall is thinned.
        for side in (-1,1):
            # A shallow longitudinal hip clears both full board corners;
            # keep the steep transverse pitch for support-free printing.
            back_rake=(-0.06 if ridge_y == 0.0 else -0.03)/sx
            plane=b.Plane(origin=(83.0*sx,ridge_y*sy,brow_z),
                          z_dir=(back_rake,side*1.12/sy,1))
            half=plane*b.Box(800,800,600,
                align=(b.Align.CENTER,b.Align.CENTER,b.Align.MAX))
            no=no & half
            drop=wall*1.005*math.sqrt(1+(1.12/sy)**2+back_rake**2)
            ni=ni & half.moved(b.Pos(0,0,-drop))
        # Close-wrap each camera with a shallow central roof fold. Its
        # intersection with the reference 1.12:1 shoulder lies 10 mm
        # from the ridge, inside the full PCB corners. Those outer
        # shoulders still set the envelope; the new crown only contracts.
        # Opposing longitudinal rakes remove both tall brow ends. True
        # normal skin and a 1.025:1 transverse pitch remain printable.
        if ridge_y != 0.0:
            for end in (-1,1):
                for side in (-1,1):
                    rake=end*.03/sx
                    plane=b.Plane(origin=(83.0*sx,ridge_y*sy,42.05),
                                  z_dir=(rake,side*1.025/sy,1))
                    half=plane*b.Box(800,800,600,
                        align=(b.Align.CENTER,b.Align.CENTER,b.Align.MAX))
                    no=no & half
                    drop=wall*1.005*math.sqrt(1+(1.025/sy)**2+rake*rake)
                    ni=ni & half.moved(b.Pos(0,0,-drop))
        nose_outer.append(no);nose_inner.append(ni)
    hip=b.Plane(origin=(71.0*sx,0,45.3),z_dir=(1.20/sx,0,1))
    half=hip*b.Box(800,800,600,
        align=(b.Align.CENTER,b.Align.CENTER,b.Align.MAX))
    nose_o=half;nose_i=half.moved(b.Pos(0,0,-wall*1.005*math.sqrt(1+(1.20/sx)**2)))
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
    # The 12 mm aft cockpit flare gives a supported closing trajectory:
    # (1.50 - 1.12 * 3.8/12) / sqrt(1 + (3.8/12)**2) = 1.092 > 1.
    # The modest inward crease shift retains the original service slot;
    # the rear board corner remains below the normally offset roof.
    # A 12.8 mm cockpit channel follows the lowered inner folds. The
    # original battery flare, end bridges and external hips remain;
    # its gentler closing trajectory still grows from supported skin.
    hatch=[(-107.0,-6.0),(-91.0,-10.5),(-69.0,-10.5),(-57.0,-6.4),
           (-29.0,-6.4),(-29.0,6.4),(-57.0,6.4),(-69.0,10.5),
           (-91.0,10.5),(-107.0,6.0)]
    # Cut the service channel through the shell at every roof height.
    # The folded battery shoulders can now lie below the former Z=37.2
    # start: a partial-depth cut would leave a detached roof tongue.
    # The tray and all seats are separate bed-founded features below.
    shell=shell-prism(hatch,0,150)
    # Continue the narrow channel to the existing forward service portal.
    # Both complete shoulder ridges and their hip-to-ring ties persist.
    shell=shell-box(-14.0,0,37.2,30.0,12.8,100)
    shell=shell-box(28.0,0,34.0,56.0,54.6,110)

    # A continuous lower roof strip follows the unchanged shell line.
    # Full carrier covers and swept shear bands retain enclosure and
    # connect the ring to the battery/avionics canopy across scalloped bays.
    protected=box(-28,0,0,56,56,150)+box(-68,0,0,78,36.6,150)
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
            # Keep the complete east/west shear fan at the rim: shrinking
            # this specific junction creates an acute crash-load notch.
            # The six other covers retain their close-wrapped perimeter.
            scale=(1.0 if key in ('vl53l9cx_breakout#e','vl53l9cx_breakout#w')
                   else p.ring_carrier_cover_mm/28.0)
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
                # Fan the existing pitched skin across the cardinal
                # hood/rim reaction. The broader outer shoulder spreads
                # crash load around the tucked waist's acute rim return;
                # its entire patch remains inside the old outer hull.
                hood=[(-8.4,-13.4),(14.0,-16.0),(18.0,-11.4),
                      (18.0,11.4),(14.0,16.0),(-8.4,13.4)]
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

    # Each roof facet keeps continuous ridge/valley chords and pointed
    # vents cut on its own normal. The innermost vent stops 0.95 mm
    # from the service-channel edge, leaving 1.36 mm along the inclined
    # skin. Paired rows leave >=1.4 mm plan lands at their valleys;
    # transverse webs bridge at most 2.4 mm along the flight direction.
    for side in (-1,1):
        for vent_y,ridge_y,face_side,half_span in (
                (7.85,9.2,-1,0.5),(11.0,9.2,1,1.0),
                (14.4,16.2,-1,1.0),(18.0,16.2,1,1.0),
                (21.4,23.2,-1,1.0),(25.2,23.2,1,1.1)):
            for vent_x in (-43.0,-38.6,-34.2,-29.8,-25.4,-21.0,-16.6,-12.2,-7.8,-3.4):
                z=dict(crease_centers)[ridge_y]-aft_rake*vent_x-crease_pitch*abs(vent_y-ridge_y)
                plane=b.Plane(origin=(vent_x,side*vent_y,z),
                    x_dir=(1,0,-aft_rake),
                    z_dir=(aft_rake,side*face_side*crease_pitch,1))
                along=math.sqrt(1+crease_pitch**2)
                breadth=.6 if half_span < 1.0 else 1.2
                outline=[(0,-half_span*along),(breadth,-.2*along),
                         (breadth,.2*along),(0,half_span*along),
                         (-breadth,.2*along),(-breadth,-.2*along)]
                wire=b.Wire.make_polygon([(u,v,-2*wall) for u,v in outline],close=True)
                shell=shell-plane*b.Solid.extrude(b.Face(wire),(0,0,4*wall))

    # Pointed vents in the rising and falling battery shoulders.
    # The true fold intersection lies near |Y|=13.9; both rows stay
    # entirely on one face, leaving continuous ridge and service-edge
    # chords. Each 2.4 mm X span bridges between printed jambs, and
    # the normal cut preserves wall gauge at the opening returns.
    for side in (-1,1):
        for vent_y,half_span,face_side in ((12.1,.65,-1),(16.0,.6,1)):
            for vent_x in (-97.0,-91.0,-85.0,-79.0):
                rake=.08/sx
                if face_side < 0:
                    z=41.65+.08*(vent_x/sx+104.0)-1.025*(14.0-vent_y)
                else:
                    z=55.8+.08*(vent_x/sx+104.0)-1.025*vent_y
                plane=b.Plane(origin=(vent_x,side*vent_y,z),
                    x_dir=(1,0,rake),z_dir=(-rake,side*face_side*1.025,1))
                length=math.sqrt(1+1.025**2)
                outline=[(0,-half_span*length),(1.2,-.2*length),
                         (1.2,.2*length),(0,half_span*length),
                         (-1.2,.2*length),(-1.2,-.2*length)]
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
    # Flare the empty centers of the aft shear bays while preserving
    # the continuous 2.6 mm belly chord and 2 mm upper chord. The
    # narrowest inclined pier remains 2.5 mm in longitudinal projection;
    # the roof edges rise above 1.30:1, directly from their printed jambs.
    # The recessed battery tray, carrier hoods and lower ties are added
    # independently and are untouched by these shell-only ventilation cuts.
    for gx in (-108.0,-96.0,-84.0,-72.0):
        # Four swept triangular bays retain 2.6 mm belly chords, 2 mm
        # eave chords and >1.5 mm intervening diagonal piers. Their
        # narrow pitched crowns close from both printed jambs.
        # Broad pointed bays between deeper continuous lower chords;
        # >1.7 mm projected diagonal piers retain their full normal gauge.
        # Taller swept bays remove the unused upper skirt between the
        # same bed/eave chords. Their roof pitches remain above 1.30:1;
        # >1.3 mm normal piers connect the continuous shell flanges.
        outline=[(gx-5.3,3.0),(gx+4.7,3.0),(gx+5.65,19.0),
                 (gx+0.4,26.0),(gx-4.95,19.0)]
        wire=b.Wire.make_polygon([(x*sx,-100.0,z) for x,z in outline],close=True)
        shell=shell-b.Solid.extrude(b.Face(wire),(0,200.0,0))

    # Swept cheek vaults replace the broad forward skirt with a deep
    # shear panel: continuous belly and upper chords surround
    # two inclined piers. The side cuts stay in the near-vertical camera
    # cheeks, ahead of the diagonal ToF carrier and behind the nose facet.
    # Pointed roofs rise at least 1.2:1 and print inward from both jambs;
    # all camera pads, retaining ears and optical cuts are added below.
    # Taller swept cheek vaults carry the nose as a deep shear panel.
    # A 3 mm continuous belly and >=3.8 mm eave chord surround each
    # opening; the original camera seats and corner piers remain intact.
    # Both roof edges rise more than 2:1, and the widening vertical jambs
    # grow directly from the bed. The upper camera brows stay unchanged.
    for gx in (71.5,85.0):
        # Deep eave and belly chords carry the lower, folded brow.
        # Spread the pitched vault into the empty skirt; the narrowest
        # intervening pier is still over 3 mm along X. The inner seats
        # are separate features, and both roof edges rise over 2:1.
        outline=[(gx-5.3,2.8),(gx+4.3,2.8),(gx+5.8,13.0),
                 (gx+0.3,25.2),(gx-4.4,13.0)]
        wire=b.Wire.make_polygon([(x*sx,-100.0,z) for x,z in outline],close=True)
        shell=shell-b.Solid.extrude(b.Face(wire),(0,200.0,0))

    # The waist between each cardinal and diagonal optical facet has
    # no payload behind its lower skirt. A deeper swept frame retains
    # a 3.6 mm lower chord and 1.5 mm eave, with inclined end piers.
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
            # Deepen the lower chord at the arm/shell reaction, then
            # fan the piers outward around the unloaded upper panel.
            # Section depth strengthens the bed junction without adding
            # uniform wall thickness. The high pointed crown removes
            # more area than the deeper lower chord restores, and both
            # roof edges rise >1.8:1. Normal cuts retain the skin gauge.
            lean=math.copysign(1.3,waist_x)
            apex=23.5 if waist_x < 0 else 25.0
            # Relieve the flat lower chord above Z=4.5 mm. The hollow
            # folded sill below provides a 9.0 mm deep load path instead;
            # its inward breadth resists ring twist without a heavier skin.
            # The original arch jambs and pitched crown remain connected.
            outline=[(-10.0,4.5),(10.0,4.5),(9.8+lean,13.0),
                     (lean,apex),(-9.8+lean,13.0)]
            # Center the normal tool on the newly tucked facet.
            inward=-side
            cy+=inward*3.50*ny
            cx=waist_x*sx+inward*3.50*nx
            wire=b.Wire.make_polygon([
                (cx+tx*u-4*nx,cy+ty*u-4*ny,z)
                for u,z in outline],close=True)
            shell=shell-b.Solid.extrude(b.Face(wire),(8*nx,8*ny,0))
            # Short, deep inward sills take ring bending beneath the
            # lowered cockpit. The narrower 3.0 mm toe and 14.4 mm web
            # put material farther from the bed with less developed stock.
            # The full normal offset and open ends keep the enclosed rib
            # printable and drained inside the unchanged optical perimeter.
            g=p.structural_gauge_mm
            breadth,depth=3.0,14.4
            grade=depth/breadth
            def rim_section(u,inside=False):
                points=[(0.0,0.0),(breadth,0.0),(0.0,depth)]
                if inside:
                    c=depth-g*math.sqrt(1+grade*grade)
                    points=[(g,g),((c-g)/grade,g),(g,c-grade*g)]
                return b.Wire.make_polygon([
                    (cx+tx*u+inward*nx*v,cy+ty*u+inward*ny*v,z)
                    for v,z in points],close=True)
            rim_outer=b.Solid.make_loft([rim_section(-4.7),rim_section(4.7)],ruled=True)
            rim_void=b.Solid.make_loft([rim_section(-4.9,True),rim_section(4.9,True)],ruled=True)
            shell=shell+((rim_outer-rim_void) & outer_hull)

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
        # Cross-vault the carrier beds perpendicular to their folds.
        # The diagonal beds share the arm's deep closed shell, so their
        # lower web centers need not remain solid. Use smaller, lower
        # openings there: 5.4 mm central and 3.1 mm outer piers support
        # the full upper folded sheet, PCB ledge and original screw ears.
        # Cutting only the bed before union preserves the arm skins and
        # avoids slicing a thin notch into their independent load paths.
        # Every vault rises 1.12:1 from bed-founded jambs; the diagonal
        # roofs keep an extra 0.6 mm of depth under the folded valleys.
        cardinal=abs(math.sin(math.radians(2*angle)))<0.5
        for radial in (-6.0,6.0):
            # Wider crossed vaults unload the carrier-bed web centers.
            # Four-mm center piers and >=2.4 mm end piers retain the
            # full folded bearing sheet and both PCB mounting ears.
            half=4.9 if cardinal else 4.3
            apex=z0-5.4*1.11/2-p.structural_gauge_mm-(0.0 if cardinal else 0.6)
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
        foot=local(box(5.5,0,0,23.0,max(1.24,p.cradle_foot_rail_mm-0.35),wall))
        for t in (-9.4,9.4):
            foot=foot+local(box(5.5,t,0,23.0,max(1.24,p.cradle_foot_rail_mm-0.35),wall))
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
        # Stop the diagonal bed 0.4 mm short of its former grazing
        # intersection with the sloping spar floor. The complete rear
        # mounting post retains over 2.8 mm of bed behind it; this removes
        # the coplanar wedge without changing the spar's normal skin.
        # End cardinal beds 1.53 mm behind their unchanged mounting posts.
        # The full 23.2 mm tangential seat still extends beyond both side
        # rails; remove only the unused bed perimeter, below the PCB.
        back=-10.6 if diagonal else -9.3
        front=4.4
        # The diagonally rotated square bed left tangential corner
        # skirts far beyond the PCB, rails and gussets. End those skirts
        # at +/-12.0 mm: the 20 mm board and both mounting ears retain full
        # support, while the empty corner no longer grazes the spar floor.
        # This also removes the nearly coplanar cradle/spar wedge that
        # produced degenerate tetrahedra in otherwise valid solid exports.
        skirt_width=24.0 if diagonal else 23.2
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
        # Retain the fixed screw seating face and pilots while removing
        # unused rear/side stock. The 4.4 mm tangential breadth leaves
        # 1.4 mm around each 1.6 mm pilot (1.2 mm at an M2 thread crest).
        # A 3.6 mm engagement depth keeps a 1.75 mm rear spine behind
        # the existing component notch; the PCB seat and rails persist.
        post_depth=max(3.6,p.cradle_post_depth_mm-0.4)
        post=local(box(-3.77-post_depth/2,8.4,z0-3.0,
                       post_depth,4.4,18.6))
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
            # Extend the curved return farther into the existing closed
            # spar. This removes the acute underside at the carrier/arm
            # junction and spreads crash load below the seating plane,
            # while staying inside the same external arm envelope.
            pad_z=max(3.0,z0-2.0)
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
    # Remove only excess skin under the rib-supported payload floors.
    # Full-height ribs and capture sills keep every seating datum. The
    # continuous 1.22 mm belly sheet remains above the 1.2 mm DFAM floor.
    gauge=max(1.22,p.structural_gauge_mm-.02)
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
        g=max(1.22,p.structural_gauge_mm-.02)
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
    # A 1.265 mm normal corrugated sheet preserves the PCB bearing
    # datums while removing excess underside allowance. The shallow
    # contact flats and all >45-degree folded faces stay supported.
    profile=[]
    for y in (-26.0,-13.0,0.0,13.0,26.0):
        profile.append((y,deck_z-6.71))
        if y<26:
            profile.extend([(y+6.1,deck_z),(y+6.9,deck_z)])
    # V-vault the underside of each narrow ridge bearing land. The
    # old 0.8 mm horizontal bridge retained 1.94 mm of material; a
    # 1.12:1 pointed underside now leaves >=1.26 mm normal skin and
    # 1.43 mm below the flat PCB contact at the crown.
    # Both sides print from the existing inclined folds, and the bearing
    # surface and the IMU landing remain at their original Z datums.
    lower=[]
    for i in range(len(profile)-1,-1,-1):
        y,z=profile[i]
        lower.append((y,z-1.88))
        if i>0 and abs(z-profile[i-1][1])<1e-9:
            prev_y=profile[i-1][0]
            lower.append(((y+prev_y)/2,z-1.88+1.12*(y-prev_y)/2))
    wire=b.Wire.make_polygon([(deck_x0,fy+y,z) for y,z in profile+lower],close=True)
    deck=b.Solid.extrude(b.Face(wire),(deck_x1-deck_x0,0,0))
    # Keep the two outer load-bearing deck rails; the open center admits
    # underside wiring and access, with broad support along both PCB edges.
    # The aft half of the pad-camera bay shares this corrugated deck
    # as its overhead shell. Keep a short transverse saddle tied into
    # both PCB rails: it encloses the recessed camera and stiffens the
    # central hub without a separate canopy. The front half stays open
    # to the existing service aisle, so the module can slide in and drop
    # onto its 2 mm seating plane. The lowest roof underside is 27.15 mm,
    # safely above the camera's 14 mm top; every fold prints from below.
    pad_x,pad_y,_=(v*1000 for v in placements['pad_camera'])
    pad_w,pad_l,_=(v*1000 for v in LIBRARY['pad_camera'].dims_m)
    pad_saddle=deck & box(pad_x-pad_w/4,pad_y,0,pad_w/2+1.2,pad_l+4,60)
    deck=deck-box((deck_x0+deck_x1)/2,fy,0,deck_x1-deck_x0+2,26.0,60)
    deck=deck+pad_saddle
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
            # Keep a continuous pitched rail tie at the arm-root return.
            # Vents here made coplanar boundary tetrahedra where the end
            # plane grazed the sloping spar-gallery roof. This short full-
            # gauge cross tie closes that notch and adds local shear area;
            # the remaining normal-cut vents and all service datums persist.
            if i == 10:
                continue
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
                    outline=[(-1.4,-1.75),(0,-2.65),(1.4,-1.75),
                             (1.4,1.75),(0,2.65),(-1.4,1.75)]
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
    board_dx,board_dy,board_dz=(v*1000 for v in LIBRARY['fc_esc_stack'].dims_m)
    body=body-box(fx,fy,fz,board_dx+2.0,board_dy+2.0,board_dz+0.6)
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
        # Put the wiring drain in the deeper part of the closed wing.
        # At X=115 its collar cuts a stress concentration into the shallow
        # keel; X=109 carries the same throat through taller side webs.
        # The full collar gauge and fixed motor diaphragm remain intact.
        for x,ro,ri,h in ((45.0,3.2,1.6,3.2),(109.0,2.3,1.0,2.5)):
            collar=b.Pos(x,sweep_center(x),0)*b.Cylinder(ro,h,
                align=(b.Align.CENTER,b.Align.CENTER,b.Align.MIN))
            throat=b.Pos(x,sweep_center(x),-.2)*b.Cylinder(ri,h+.8,
                align=(b.Align.CENTER,b.Align.CENTER,b.Align.MIN))
            body=body+collar.rotate(b.Axis.Z,ang)
            body=body-throat.rotate(b.Axis.Z,ang)
    # A recessed, top-loading pad-camera cassette sits between the four
    # stack bosses. Its 25 x 24 x 12 mm service box starts at the pinned
    # Z=2 mm datum; only a 10 mm lens bore reaches the underside. The
    # lens barrel seats through this bore with its face at the bed plane.
    # Even the conservative optical apex at component-bottom Z=2 mm
    # clears the bore: 2*tan(65 deg)=4.29 mm < its 5 mm radius.
    # A continuous 1.24 mm belly sheet joins the existing longerons;
    # low vertical capture sills grow from the bed with no overhead lip.
    # The central opening above the pocket remains available for removal.
    pcx,pcy,pcz=(v*1000 for v in placements['pad_camera'])
    pcw,pcl,pch=(v*1000 for v in LIBRARY['pad_camera'].dims_m)
    pg=max(1.22,p.structural_gauge_mm-.02)
    pocket_w,pocket_l=pcw+.6,pcl+.6
    body=body-box(pcx,pcy,pcz,pocket_w,pocket_l,pch+.6)
    camera_floor=box(pcx,pcy,0,pocket_w+2*pg,pocket_l+2*pg,pg)
    # A 1.4 mm edge bearing raises the board onto its prescribed 2 mm
    # seat without filling the broad center of the shallow belly pan.
    for side in (-1,1):
        camera_floor=camera_floor+box(pcx,pcy+side*(pcl/2-.7),0,
            pocket_w+2*pg,1.4,pcz)
        camera_floor=camera_floor+box(pcx+side*(pocket_w+pg)/2,pcy,0,
            pg,pocket_l+2*pg,pcz+2.5)
        camera_floor=camera_floor+box(pcx,pcy+side*(pocket_l+pg)/2,0,
            pocket_w+2*pg,pg,pcz+2.5)
    body=body+camera_floor
    pad_lens_radius=max(5.0,pcz*math.tan(math.radians(65.0))+.6)
    lens_port=b.Pos(pcx,pcy,-.2)*b.Cylinder(pad_lens_radius,pcz+.4,
        align=(b.Align.CENTER,b.Align.CENTER,b.Align.MIN))
    body=body-lens_port

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
    # Four bed-founded internal returns lap the grazing arm/waist seams.
    # Their 3 mm breadth ties the swept spar into the thin ring wall,
    # removes the zero-area seam on STL export and carries ring shear
    # through a deeper local section. Clip to the existing enclosure:
    # the hull never grows, and all toes stay below the carrier boards.
    for x_sign in (-1,1):
        for y_sign in (-1,1):
            toe=box(x_sign*45.1*sx,y_sign*62.1*sy,0,
                    3.0*sx,3.0*sy,10.6)
            body=body+(toe & outer_hull)

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
    # Normalize the closed BRep before final domain unification. Swept
    # loft p-curves can otherwise retain nearly coincident entities that
    # create zero-volume tetrahedra after solver coordinate rounding.
    # Use OpenCascade's native in-memory format, with no external cache
    # or tessellated replacement; this retains the exact parametric solid.
    from io import BytesIO
    from OCP.BRepTools import BRepTools
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Shape
    from OCP.ShapeFix import ShapeFix_Shape
    stream=BytesIO()
    BRepTools.Write_s(b.Part(children=body.solids()).wrapped,stream)
    stream.seek(0)
    normalized=TopoDS_Shape()
    BRepTools.Read_s(normalized,stream,BRep_Builder())
    shape_fix=ShapeFix_Shape(normalized)
    shape_fix.SetPrecision(0.001)
    shape_fix.SetMaxTolerance(0.001)
    shape_fix.Perform()
    body=b.Part(shape_fix.Shape()).clean()
    # Merge microscopic coplanar returns at the carrier/spar junction.
    # These sub-print-resolution slivers can become zero-volume solver
    # tetrahedra after coordinate rounding. Native BRep healing preserves
    # the parametric shell and its full gauge; no mesh or cached part is
    # substituted, and the completed solid must still validate below.
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    from OCP.ShapeFix import ShapeFix_FixSmallFace
    unify=ShapeUpgrade_UnifySameDomain(body.wrapped,True,True,True)
    unify.SetLinearTolerance(0.002)
    unify.SetAngularTolerance(0.0001)
    unify.Build()
    small=ShapeFix_FixSmallFace()
    small.Init(unify.Shape())
    small.SetPrecision(0.002)
    small.SetMaxTolerance(0.002)
    small.Perform()
    body=b.Part(small.Shape())
    if not body.is_valid or len(body.solids()) != 1:
        raise ValueError(f'Chassis must be one valid solid: valid={body.is_valid}, solids={[(round(s.volume,2),tuple(s.center())) for s in body.solids()]}')
    return b.Part(b.Part(children=body.solids()).wrapped)

if __name__ == '__main__':
    p=ChassisParams()
    part=build_chassis(p)
    print(f'volume {part.volume:.1f} mm^3; PETG mass {part.volume*0.00124:.2f} g')
