## Run 2 (headless Zig sim inner loop) - 2026-09-04

20 generations, 0.73 h wall, backend=quad (dronestudio-headless ReleaseFast),
trainer=cem, budget cem_iters=4, cem_pop=10, train_episodes=4, eval_episodes=6.

Outcome: best variant g13v140 (gen 13), eval success 33.3% (2/6 episodes),
mean return -0.81, train best return +3.80. Lineage g12v129 -> g13v140 (mutator none).

Best scene distribution:
  obstacle_density 0.018, obstacle_size 0.75 +- 0.08, corridor_width 10,
  scene_extent 22, goal_distance 5, light_intensity 0.92,
  light_direction_entropy 0.35, texture_variety 0.55, dynamics_noise 0.0

Selection dynamics:
- First success at gen 2 (17%), plateau 17% through gens 2-12 while diversity
  oscillated 0.28-0.95 (two diversity spikes at g7/g8 did not convert).
- Jump to 33% at gen 13, then stagnant 3 of the last 6 gens (15, 16, 19);
  diversity collapsed to 0.16 at gen 15 - the archive was exploiting, not
  exploring, when the run ended. A longer run may or may not have escaped;
  the stagnation counter was resetting on near-misses without new success.
- LLM spend: 0 calls (ChatGPT throttle demoted the outer loop to heuristic
  mutation this run - heuristic still found the gen-13 improvement, but the
  last-mile exploration was purely heuristic).

Follow-on: PPO-vs-CEM comparison launched on the g13v140 distribution
(autoresearch/ppo_vs_cem.json, compare.log on the box) - first real test of
whether PPO beats CEM on the discovered distribution before the inner loop
swaps to PPO by default.

## PPO vs CEM on run-2 best dist (g13v140) - 2026-09-04

compare_trainers.py, backend=sim (headless Zig), eval seed 20000, 6 episodes, 250 max steps.

  CEM (iters 6, pop 12): 0.6s wall, train best return +0.81, eval success 0/6, mean return -7.16
  PPO (hidden 128, lr 3e-4, rollout 8 x 80 updates, 62k env steps): 10.1s wall,
    eval success 0/6, mean return -36.03

Reading: 0/6 for both is not conclusive against the 33.3% (2/6) run-2 eval -
small-sample noise (P(0/6 | p=1/3) ~ 9%) plus a different eval seed and
from-scratch retrains. The load-bearing finding is PPO's training curve:
eval mean went -16.9 (update 20) -> -125.3 (40) -> -109.4 (60) -> -203.5 (80).
PPO did not just fail to learn, it diverged monotonically after update 20.
At return magnitudes in the hundreds, lr 3e-4 without return scaling looks
too hot; candidates: lower lr, advantage/return normalization, value-loss
clipping, or smaller epochs-per-rollout. Until that is stable, CEM stays the
default inner-loop trainer; PPO needs a hyperparameter stabilization pass
before the swap mandated in the steering note.

## PPO stabilization attempt: LR sweep on g13v140 dist - 2026-09-04

Same protocol as the comparison (backend=sim, eval seed 20000, 6 eps, 250
steps, rollout 8 x 80 updates, hidden 128, seed 42). Script:
autoresearch/ppo_lr_sweep.py, results autoresearch/ppo_lr_sweep.json.

  lr 3e-5: eval -94.4 / -95.0 / -158.8 / -134.0 (updates 20-80) - never recovers
  lr 1e-4: eval -144.0 / -16.5 / -24.0 / -23.0 - recovers to ~-20, plateaus
  lr 3e-4 (from ppo_vs_cem.json): best -16.9 at update 20, then diverges to -203.5

All three LRs share the same ceiling (best eval ~ -17) and 0/6 success.
Step size is NOT the binding constraint: PPO simply does not learn this task
at auto-researcher budgets (62k env steps, ~2k steps per update = very
high-variance gradients on a sparse-success task). CEM reaches train return
+0.81 in 0.6s with 432 episodes total. At seconds-per-variant budgets,
direct policy search dominates PPO by a wide margin.

Decision: CEM remains the inner-loop trainer. Revisit PPO only with (a) a
per-variant budget of 1M+ env steps, or (b) reward shaping / curriculum that
densifies the success signal. Documented per the steering note so the PPO
swap is an informed choice, not a default.

## Manifest dynamics validation - 2026-09-04

- Evaluator/env_sim now carry a dynamics profile end to end (set_dynamics on
  process spawn; eval loop fixed to use the factory - it was bypassing it,
  so evals silently ran abstract). Run records now include dynamics.
- Rate PID re-tuned against the real airframe (rate_tune_manifest.py):
  the real inertia is ~20x smaller, so the loop is far more responsive -
  roll/z channels kp 0.1 (vs 1.2 abstract), vertical (pitch channel, ktau
  authority only) kp 0.5. All axes: rise 0.1s, overshoot <1%, SS err <0.015.
  SIM-TUNED ONLY, hardware re-validation required. Axis labeling caveat:
  PID "pitch" drives the vertical axis, "yaw" a horizontal one - inherited
  from the headless +Y-up convention; gains now match physics, labels do
  not (documented in rate_tune_manifest.py).
- Comparative smoke (g13v140 dist, cem 3x8, 3 train eps): abstract -5.87 /
  manifest -9.34 mean eval return, both 0/6 at smoke budget - dynamics
  measurably change the learning problem, as expected.

## Manifest-dynamics validation run (2026-09-04)
- 5-gen x 3-child outer loop on the sim backend with the CAD chassis manifest:
  machinery healthy (selection, novelty, elites, restart), but success_rate 0.00
  across all 20 variants vs 33.3% best on the abstract quad backend (run 2).
  See autoresearch/manifest_run_analysis.md.
- Watch channel live: streamer.py flies the best policy on the manifest airframe
  and streams telemetry to the dashboard /watch page (SSE + 2D canvas + per-episode
  metric graphs: time-in-air, RMS jerk, off-ideal-path residual).

## Adaptive stopping (outer loop)

Plateau stop: after a stagnation restart, if the post-restart best success rate
does not improve on the pre-restart best by at least `plateau_min_improvement`
(default 0.02), the run stops instead of restarting again. The fixed generation
count remains an upper bound, not a target. Logged as "plateau stop" with the
gen number, pre-restart best, and current best.

## Demo diversity, DAgger, and trajectory collapse (2026-09-04)

User observation: the first BC policy flew one beeline style and only
succeeded when the spawn matched it. Correct - the seed demos were 24
obstacle-free episodes from a single P-controller teacher at one speed.

What we tried, in order:

1. More of the same (fixed demos, varied speed profiles): REGRESSED
   (2m 75% -> 42%). Mixed speed profiles make the obs->action mapping
   multimodal - same obs, different throttle depending on the teacher's
   chosen speed - and BC averages the modes into mush. Lesson: any teacher
   choice not observable from obs must be held constant across demos.
2. Diverse scenes + obstacle-avoiding teacher (potential field on the
   nearest-obstacle obs channel): also regressed on clean evals. Avoidance
   wiggle demos diluted the clean-flight prior.
3. DAgger (student rolls out on diverse scenes - goals 2-15m, densities
   0-0.2 - and the teacher labels the states the student actually visits;
   aggregate, retrain): WORKED. After 4 iterations: 2m 95.8%, 5m 79.2%
   (also 79.2% WITH obstacles at density 0.1), vs the original BC's
   75/21/4. 10m still 0% - student rarely survives to long-horizon states,
   so the teacher never labels them; expected to lift as the ladder
   warm-chains upward.

Conclusion on record: demo DIVERSITY of states visited matters more than
demo count, and the fix for trajectory collapse is labeling the student's
own state distribution (DAgger), not more teacher-only episodes. Teacher
was the classical scripted pilot (rate-PID cascade + potential-field
avoidance); no sim/binary changes.

Artifacts: autoresearch/diverse_bc.py, autoresearch/dagger.py (box
/workspace), teacher autoresearch/scripted_pilot.py, best policy
/workspace/bc_flat.json (hot-reloaded by the /watch streamer).

## Scripted-teacher decode fix (2026-09-04, post-DAgger10)

The 10m curriculum wall was a TEACHER bug, not a student/sim bug. pilot_act2
decoded obs with constants that only hold for extent-10 scenes: rel_goal is
scaled by 1/scene_extent (extent = max(10, 2*goal_m) = 20 at 10m), and velocity
damping used obs-raw values (true vel / 10) making the loop nearly undamped.
Symptoms: orbiting the goal, sign-flipping rel vector, 33% teacher success at
10m d0.0. Fix: decode rel/obstacle vectors with scene_extent; damp against 0.5x
true velocity (gain swept over 24 configs x real sim episodes; 0.1x orbits,
1.0x over-brakes). Fixed teacher (16 eps each, real sim): 2m/5m/10m = 100%
(d0.1 included), 15m d0.2 = 87.5%, 25m d0.1 = 87.5%. DAgger10 rerun excludes
the old buggy-teacher long-range labels. Same /10 hardcode fixed in
scripted_pilot.pilot_act (takes ext=10.0 default now).

## 2026-09-05 - SIM next-steps track (user-approved 2026-09-04 21:08)

T8 (83101be, 5d40877): CAD chassis manifest 1.2 consumption. Loader
(core/ChassisManifest.zig, 1.1-compatible), IMUSensor lever-arm correction
(a_imu = a_com + alpha x r + omega x (omega x r), alpha finite-differenced
at the 1kHz sample rate - pos_body existed but was never applied before),
noise densities to MPU-9250 datasheet typ (gyro 0.01 dps/rtHz, accel
300 ug/rtHz), Drone.zig env-gated manifest consumption
(DRONE_CHASSIS_MANIFEST): mass/inertia/IMU pose/stereo poses+FOV, 56mm
baseline (was 75mm hardcode), Pi Cam 3 Standard module (was Wide 102deg).
Open: v22 chassis.glb needs KHR_mesh_quantization+EXT_meshopt_compression
(unsupported by core/GLTF.zig - CAD asked to export uncompressed);
Z-up->Y-up mapping needs visual verification on a desktop run. See
docs/MANIFEST_SIM.md.

T9 (fd2f40d): motor_v2 prop fidelity. Axial inflow (advance ratio, ~16%
thrust loss at 5 m/s climb, J0=1.2 ESTIMATE) + ground effect
(1/(1-(R/4z)^2), z clamp >= 0.6R, max +21%). Hover validates at analytic
throttle -0.742 (kf fit, 0.526kg fixture). motor_v2 still env-gated OFF.
BLOCKER for m2 training: scripted teacher porpoises under m2s slower EM
response (tuned for the 40ms lag plant) - needs m2 gain retune first.

T10: per-episode dynamics tolerance jitter (autoresearch/dynamics_jitter.py,
env-gated AUTORESEARCH_DYN_JITTER). mass +/-5%, inertia +/-8%, per-motor
thrust +/-4% independent, lag +/-25%, drag +/-15%, CoM +/-3mm. Deterministic
per episode seed (own rng stream, atomic file writes for 24-worker safety).
Teacher flies jittered airframes (goto succ 29 steps, seed 80000). Ranges
are anchors pending bench sysid.
