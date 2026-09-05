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

### T11 - Track 2 dag6j run 1 + anchored jitter fix (2026-09-04 ~9:20 PM)

dag6j run 1 (full-strength jitter, 100% of rollout episodes): all 8 rounds
floors=False, no promotion, t4_best untouched. Final round mean 0.597
(goto_t0 0.562 / hover_hold_t0 0.938 / land_t0 0.562) vs non-jittered dag5
0.799 promoted at r2. Hover cells held all campaign (0.875-0.938); goto/land
base-airframe precision collapsed. Full-strength domain randomization buys
robustness at the cost of the precision the floors measure.

Fix (dag6j2): standard anchor approach - AUTORESEARCH_DYN_JITTER_SCALE=0.5
(half-strength draws) + AUTORESEARCH_DYN_JITTER_MIX=0.5 (deterministic
per-seed flip; 51% jittered over 200 seeds, half-scale mass delta 0.57% vs
full 2.02%). dynamics_jitter.py: should_jitter() + scale env. env_sim.py:
mix gating in make_sim_factory. t4_dagger6j2.py: seeds 500000+, labels
t4-dag6j2, outputs t4_dag6j2_r{k}.json / t4_dag6j2_best.json, t4_best.json
still untouched, eval stays base-airframe for comparability.

### T11b - motor_v2 teacher retune (2026-09-04 ~9:30 PM)

Problem (found in T9): the t4 scripted teacher is hand-tuned for the 40ms
lag plant and porpoises under motor_v2's ~200-300ms EM spin-up. Retune is
env-gated on AUTORESEARCH_MOTOR_V2 (same var the sim reads), so base-plant
behavior stays bit-identical for the t4 lineage.

Sweeps (m2_teacher_tune.py, m2_land_tune*.py, m2_trim_test.py):
- Vertical loop: K_VTHR >= 0.20 porpoises; 0.10 stable. Hover throttle
  -0.742 (T9 analytic, validated).
- Attitude: KP/KD 0.30/0.45 (from 0.40/0.60).
- Land: near-pad cruise offset 1.2 -> 0.6 (old offset found a hover
  equilibrium 0.02m ABOVE the land_descend gate from SoC sag: hover point
  drifts -0.742 -> -0.759 over an episode); terminal vy -0.4 -> -0.25.
- Terminal-commit throttle REJECTED: any pinned value either creeps or
  touches down at -1.1 m/s (limit 0.5); sag makes fixed values fragile.
- Trim integrators: land_descend-scoped I-term (0.0008/step, clamp 0.12)
  fixes the near-ground creep stall (touchdowns at -0.05 m/s); hover-scoped
  micro-trim (0.0003, clamp 0.05). Global trim sank drones (oscillation
  rectifies into the integrator) - rejected.

n=16 tier-0 teacher yields under m2 (m2_validate.py): goto 0.938, land
0.438, hover_hold 0.312-0.375. Workable for demo collection (pipeline
keeps successes only; run more seeds). Known limitations: hover vertical
loop marginal under SoC sag; occasional transit crashes into obstacles
under actuation lag (avoidance gains tuned to fast plant).

Also: CAD sim GLB (chassis b171a6e, chassis.sim.glb) verified structurally:
GLB v2, zero required/used extensions (GLTF.zig-compatible), single mesh,
no materials (as documented), Y-up confirmed numerically (Y span 49mm vs
X/Z 262/235mm). Runtime parse + visual frame check still open for a
desktop run.

## dag10 (2026-09-05): yardstick change - land skim penalty

**YARDSTICK CHANGE: dag10+ land numbers are NOT comparable to dag7-9.**
Sim change (Studio/src/headless_main.zig): in the land scenario, holding a
ground-skim hover (pos.y < GROUND_Y+0.15) for >=60 policy steps (~3s at
20Hz) without touching down now ends the episode at -3.0. Previously
parking at skim height cost only -0.01/step, which the m2 land probe
showed was the learned false equilibrium (11/16 student census episodes
parked at alt~0.00 vy~0 and never committed to touchdown; parking strictly
dominated a -5 failed touchdown under the old economics).

Teacher lever (terminal centering gate) REJECTED after bounded iteration:
four gate variants (hold-until-dxz<=0.5r, stronger horizontal loop
0.9/+-0.5, tightened entry 1.0r, latched hysteresis 1.5r) all scored 0/16
on the teacher census. Step-trace diagnosis (diag_teacher.py): heldout
tier-0 land cells use success_radius 0.5m, while the teacher's terminal
horizontal loop limit-cycles at +-1m around the pad - any tight gate is
inside the oscillation's dead zone; unlatched gates also mode-flickered
(alt 1.4 boundary), alternating corridor descent (vy_des ~ -1.2) with the
terminal hold into vertical porpoise crashes. Hysteresis stopped the
crashes (1/16 off-pad) but the loop still cannot converge inside 0.5m, so
episodes die to the new skim penalty while holding. Teacher terminal
accuracy is a position-loop redesign problem of its own; the dag7-9
integrator pilot remains the label source.

## dag10 PPO experiment (2026-09-05): residual RL vs the new yardstick - NULL

Experiment (approved 12:37 PM, own artifacts only): ppo6b residual recipe
(frozen champion + bounded 0.1 additive residual, zero-init R2) warm-started
from dag10 r14 (best round, mean 0.354 new yardstick), obs v4 + motor v2,
skim penalty live. 60 updates, no floor abort (trend was the signal).

Result: land_t0 = 0.0 in ALL 60 updates. The residual re-triggered the
ppo1-5 precision collapse: hover_hold_t0 0.25 -> 0.0 by u41, goto_t0
0.688 -> 0.25 by u60, while mean return crept -4 -> +1.8 (return-chasing
trading precision, exactly the failure the residual was built to avoid -
the 0.1 action bound is not small enough to protect a marginal champion).

Conclusion: the skim-equilibrium lever fails under BOTH imitation (dag10:
reward-blind) and residual RL (this run: collapse dominates before land
precision can be explored). Land is blocked by terminal precision, full
stop: 0.5m heldout radius vs the teacher loop's +-1m limit cycle. The
teacher position-loop redesign is now THE land blocker. Open question to
quantify during the redesign: is 0.5m reachable at all under m2 actuation
latency, or does tier-0 land radius need to move?

## Teacher position-loop redesign (2026-09-05 PM): land teacher 6/16 -> 11/16

Follow-up to the dag10 teacher-lever rejection. Reachability study FIRST
(m2_teacher_reach.py): spawned at alt 1.4m, offsets to 2m, a position PD
through the teacher's own attitude pipeline settles to sustained dxz
0.02-0.09m in 1.6-3.6s - the 0.5m tier-0 radius is REACHABLE with 5-10x
margin; the teacher's +-1m limit cycle was a tuning bug (velocity-saturated
loop rectifying plant lag), not a physics limit. Best gains kp=1.2, kd=2.0
(higher gains oscillate MORE - latency-limited).

Redesign (t4_pilot.py, land_descend only unless noted):
1. Terminal horizontal: direct position PD (kp 1.2, kd 2.0, accel clip
   +-2.5) replacing the v_des-saturation loop.
2. land_descend latched with hysteresis (unlatched gate mode-flickered
   into porpoise crashes).
3. Center-first descent gate: vy_des = -0.25 only when dxz <= max(0.25,
   0.5r); while centering, POSITION-hold at 1.2m (kv 0.25) - a velocity
   hold sagged through SoC drift faster than the trim integrator winds.
4. Low+in-gate commit: alt<0.35 and dxz<=r -> vy_des=-0.35.
5. SoC feedforward now applied in integrator mode too (was stateless-only;
   FF carries the sag, integrator mops residual). Also touches hover_hold.
6. Cushion-arrest fast-wind: the descent arrests quasi-statically at the
   max-GE equilibrium (~0.09m); trim integrator winds 10x (0.008/step) when
   arrested low with a descent command - breakthrough in ~1-2s. THIS was
   the last blocker: the old teacher's 6/16 successes relied on slow
   integrator wind-through over 10-15s, which the 60-step skim-kill
   preempted (the yardstick change broke the old teacher's own mechanism).

Sim skim-clock refinement (headless_main.zig): the clock now ticks only
when |vy| < 0.05 (a true park), not while the vertical loop is actively
working through the cushion. Yardstick note: dag10 itself trained against
the v1 clock; v1->v2 changes only WHEN the clock ticks, and dag10's null
result (BC reward-blind) is unaffected.

Teacher census (16 heldout tier-0 land cells, m2): 11/16 success (was
6/16). Remaining 5 are APPROACH-phase, not terminal: 1 off-pad transit
miss (dxz 3.25), 2 transit timeouts (alt 1.3-2.2, never reached the gate),
2 slow-centering timeouts at alt 0.8-0.9. Terminal landing accuracy is
fixed; approach-phase robustness is the next teacher item if land labels
need more yield.
