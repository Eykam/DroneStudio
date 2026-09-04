
## set_dynamics (added 2026-09-04)

{"cmd":"set_dynamics","path":"<manifest.json|abstract>"}

Loads a dronestudio.chassis/1.1 manifest (CAD auto-researcher output) and
makes the episode drone manifest-driven: mass, diagonal inertia, total max
thrust (4 x per-motor), motor params (arm, lag, drag ratio), and per-axis
flat-plate aero drag (F = 0.5*1.225*Cd*A*v|v|, body axes, assembled in the
world frame). "abstract" restores the built-in QuadNavEnv-parity profile
(1.5 kg, diag 0.040/0.040/0.047, 40 N, no drag) - it stays the default so
run-1/2 results stay reproducible. CAD frame is +X fwd/+Z up; the loader
remaps to the sim body (+Y up): sim.x=cad.y, sim.y=cad.z, sim.z=cad.x.
Inertia cross terms are dropped (Bullet takes the diagonal); the reply
reports cross_term_ratio (0.009 for chassis_v1). Physics body origin stays
at the CoM. Fixture: autoresearch/fixtures/chassis_v1.manifest.json
(mass 0.5467 kg, I diag ~[0.00227,0.00388,0.00211] sim-frame).
