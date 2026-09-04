# Design inspiration brief (user-requested, 2026-09-04)

User steering: the mutation loop should take form-factor inspiration from
real-world products - DJI and Anduril - instead of only evolving the classic
flat-plate freestyle frame. Hard constraints still gate everything
(evaluator-first): single watertight body, FDM PETG flat-on-plate with no
supports, walls >= 1.2 mm, prop clearance, FEA pass, sim contract
(arm_length_mm and motor hole pattern are FIXED).

## What the references actually do

### DJI FPV / Avata line
- Fully enclosed body shell: the frame IS the bodywork. Arms are faired
  extensions of the fuselage, not bolted-on plates. Drag-conscious:
  smooth belly, no exposed plate edges, no open bays catching air.
- Swept, tapered arm planform: arms widen toward the root (large filleted
  root junction, load path into the body) and narrow toward the motor.
- Structural skin: stiffness comes from curved shell surfaces and closed
  sections, not raw thickness. Avata 2 teardown note: single-board,
  fully integrated (digitalhabitats.global teardown) - integration is the
  aesthetic, BUT it made the drone nearly unrepairable. For this build the
  stack/battery MUST stay accessible; we take the shell language, not the
  glue-everything trade.
- Ducts (Avata) are OUT of scope: 5" prop ducts in PETG are heavy and the
  sim contract fixes motor positions/prop disk.

### Anduril Bolt / Ghost
- Clean monocoque shell, minimal exposed structure, enclosed payload bays
  with defined interfaces (anduril.com/bolt; Anduril Ghost 4 announcement,
  blog.anduril.com). Bolt is a ~12 lb quad; Ghost is a single-rotor heli,
  so the transferable idea is the AIRFRAME LANGUAGE: one flowing body,
  hardware hidden inside, everything faired.
- Battery/payload treated as a designed-in bay with a defined position,
  not strapped on top of a flat plate.

### Monocoque FFF precedent (academic, directly applicable)
- MDPI Technologies 6(1):8 - "Design of Additively Manufactured
  Monocoque Quadcopter": reengineered a multi-part quad frame into a
  single FFF-printed monocoque. Weight down, assembly time down, flight
  trials fine. Methods: topology-optimization-informed material
  distribution, part consolidation, DFAM rules. This proves a printed
  monocoque quad is viable, not just injection-molded.
  https://www.mdpi.com/2411-9660/6/1/8

## Concrete, parameterized cues for chassis.py mutations
1. Arm shaping: taper arm width root->tip (root 1.5-2x tip width), sweep
   the leading edge back a few degrees, large root fillet (r >= 4 mm) to
   spread crash loads into the deck.
2. Sections: prefer closed/hollow arm sections (twin-wall box, internal
   rib) over open I/T - closed sections are the monocoque trick and print
   flat fine.
3. Body: low-profile faired deck - smooth belly (battery bay recessed
   INTO the deck underside), gently domed top shell over the stack bay
   with ventilation slots, chamfered/radiused perimeter instead of sharp
   plate edges. Keep shell walls 1.6-2.4 mm.
4. Integration without glue: stack bay stays open-bottom or side-access
   for serviceability; use capture pockets/lips printed in, not separate
   fasteners, where the evaluator allows.
5. Drag: keep projected frontal area low; fair the motor-mount pedestals
   into the arm with a fillet rather than a step.
6. Mass discipline: shell area costs grams - any canopy/fairing must pay
   for itself in FEA/aero terms or be thin (<= 1.6 mm). Score still
   rewards lower frame mass with all checks passing.

## Explicitly NOT this
- Not multi-part assemblies (evaluator requires a single body - a
  separate canopy part is a future phase, not a mutation now).
- Not DJI's unrepairable integration: electronics must stay reachable.
- Not props-in-ducts, not changed motor positions/arm length (sim contract).
