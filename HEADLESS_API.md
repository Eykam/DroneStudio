
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

### Motor model (manifest mode)

With a manifest loaded, collective thrust + PID torque are mixed to 4 motor
thrusts through the manifest motor geometry (4x4 solve precomputed at load;
per-motor clamp with his dynamic minimum max(0.1N, 10% avg); first-order lag
per his updateMotorLag; ktau yaw from manifest drag_ratio). CAD cw/ccw signs
are flipped vs his NED prop_drag_signs (sim body is +Y up = -z_NED).
Abstract mode keeps the direct PID-torque path unchanged. Verified: fresh
rate_step on all three axes converges (roll 1.0001, yaw 1.0002, pitch 1.0005
at 1 rad/s setpoints); chaining rate_step commands without reset inherits
body state and can look unstable - issue set_gains or reset between probes.
