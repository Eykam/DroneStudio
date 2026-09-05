# Night Progress Log (auto-researcher, 2026-09-03/04)

Running log, newest last. All work on branch `auto-researcher`.

## 22:30-23:00 PDT - infrastructure + first loop
- Railway box provisioned (smallest), zig 0.14.1, 7 compat patches committed (db8a5ef).
- Repo on box with deploy key (volume-backed after v1 key loss on redeploy).
- QuadNavEnv: numpy port of his flight model (680a867). CEM trainer + evaluator.
- Run 1 (6 gens x 3 children): best g2v15 succ 0.17 - the loop discovered the
  curriculum direction (density 0.12, corridor 7.0, goal 12m vs base 25m).
- Spend cap + run_generations.py (d4beb4d). PPO trainer landed (ca1f32a).

## 23:00-23:10 PDT - ChatGPT outer loop
- Codex CLI installed on box, device-code auth against his ChatGPT account.
- Mutator wired to Codex (283a239), throttled (f8234e5: sleep through rate
  waits instead of silently demoting to heuristic - run-1 lesson).
- Loop flipped from heuristic to his ChatGPT; verified live.
- RENDERER_FALLBACK.md (28abae4): bindless texture requirement makes headless
  rendering impossible on CPU; physics-only episodes mandated; software
  rasterizer recommended for vision later.
- Archive intelligence (b316878): novelty, multi-elite selection, diversity
  bonus, trajectory recording, stagnation restart (outer_loop v2).

## 23:09-23:30 PDT - dashboard
- dashboard/: React+shadcn+Tailwind+react-router+react-query, Hono-on-Bun,
  single-user password auth (hashed env var, HttpOnly+Secure+SameSite=Lax
  session cookie), bearer-token ingest (a53343d).
- Railway second service. Deploy saga: no repo trigger -> built main (no
  dashboard/ there); Railpack + bun 1.2 can't read bun 1.4 lockfile
  (lockfileVersion 2). Fixed: committed Dockerfile pinned to oven/bun:1.4
  (09aeb42, df28727), deployed via railway up.
- LIVE: https://dronestudio-dashboard-production.up.railway.app - verified
  via cloud-browser login + screenshot; poster on box pushes state every 30s.
- Poster run card now reports current generation (0a6aba8).

## 23:18-23:35 PDT - ZIG BINARY MANDATE (his 23:18 steering)
- headless_main.zig (Studio/src): physics-only episode runner - real Bullet
  rigid body (mass 1.5, inertia 0.040/0.040/0.047 from Drone.zig), his
  RateController/PIDController verbatim at 500Hz/20Hz, JSON-lines stdio.
  No GL/renderer/ECS. `zig build headless` -> dronestudio-headless (c79ea5e).
- PARITY VERIFIED (parity_report.json): 3.1e-5 m max position divergence over
  a 61-step episode vs QuadNavEnv, returns match to 5 decimals. QuadNavEnv
  was a faithful port; the Zig stack flies identically.
- Design calls (delegated): sphere collider r=0.3 (exact env parity),
  Python-sampled scenes passed on reset, 500Hz/20Hz, actuation v1 = PID
  torque + collective thrust (motor mixer = v1.1 upgrade).
- evaluator backend="sim" wired; SimBinaryEnv same contract; sim_smoke.json
  (sim ~11x faster than quad end-to-end). Default backend flips to sim when
  the binary exists (709d9e6). compare_trainers backend-parameterized +
  --dist-json (8401e9f).
- No-overclaiming pass: README/env_quad/HEADLESS_API carry explicit
  "gains lifted verbatim from firmware, tuning status unknown" caveats.
- RATE_CONTROLLER_ASSESSMENT.md (9f9ef12): 6 findings (derivative-on-error
  mislabeled as on-measurement; unverified gains look soft for I=0.040;
  debug.print spam in 500Hz paths; accel gating commented out; unguarded dt
  divisor; rate-sp clamp disabled) + 5-test sim verification plan.

## 23:35-23:37 PDT - rate tuning on the real binary
- headless binary: rate_step probe (constant setpoint, 500Hz telemetry) +
  set_gains (runtime swap, no rebuild) (2eb77c3, 904e206).
- rate_tuning.py: MEASURED firmware gains - roll/pitch rise 860ms, 0%
  overshoot; yaw never reaches 3 rad/s in 1.5s (22% SS error). Assessment
  finding 2 confirmed with measurements.
- rate_tune_sweep.py: first sweep - sim-tuned candidates roll/pitch
  (1.2, 0.05, 0.012) = 150ms rise / 0.14% overshoot (6x faster), yaw
  (0.3, 0.02, 0.005) = 390ms rise. Rise floors at the 1.0 Nm torque limit.
  Labeled SIM-TUNED ONLY, hardware validation required. Firmware untouched.

## Run 2 (in progress at 23:35)
20 gens x 6 children, multi-elite loop, fully ChatGPT-driven. At gen 9:
best succ 0.17, diversity recovered to 0.94 after stagnation restart.
Completion -> report commit + PPO-vs-CEM on best variant (armed wake).

## 2026-09-04 obs v2 (yaw-relative) + scenario sampler, sim side

- headless_main.zig: scenario params (`success_radius`, `scenario`,
  `hold_s`, `max_touchdown_vs`, `shaping_v2`) with goto-preserving
  defaults; hover_hold (4s hold, drift shaping) and land (touchdown
  classification: horizontal pad dist + |vertical speed|) scenarios;
  `obs_v2` cmd: 19-dim obs with rel_goal/vel/obstacle rotated into the
  yaw frame + scenario one-hot + success_radius/extent.
- WHY yaw-frame: obs v1 was yaw-blind (world-frame rel_goal/vel/obstacle,
  yaw-invariant g_body) while spawn yaw is always identity and the
  teacher implicitly assumes yaw=0; ktau/gyro yaw drift silently broke
  the frame mapping mid-flight (user-observed "nav wonky, always points
  one way"). In the yaw frame the teacher's mapping is exact at any
  heading and the policy gets working nav regardless of yaw.
- Parity: bit-identical consumed trajectories vs the previous binary
  under goto defaults. Land/hover/obs_v2 functionally validated.

## 2026-09-04 com offset + v14 thrust rebase

- headless_main.zig: set_dynamics now loads the manifest com and offsets
  motor lever arms (position - com) so mix torques are about the true com
  (Bullet already integrates the body frame as the com frame). Abstract
  profile unchanged (com=0); goto-default trajectories bit-identical.
- v14_g13.fixture: CAD snapshot carried STALE per-motor max_thrust_n=17.27
  / kv=2400 (the old pre-adjudication numbers) - rebased to 11.0 / 2300
  per the RS2205 bench calibration; upstream CAD fix routed via parent.
  Found via non-monotonic open-loop thrust probes (hover "froze" mid-air
  because 0.0765 x 69.1N accidentally equals the v14 weight).
- Teacher re-validated under v14-g13 + com + 11N: 1.0/1.0/1.0 (held-out,
  n=16); land touchdown mean 0.46 m/s (limit 0.5).

## T13 dag7m2 - motor_v2-fidelity DAgger (2026-09-04 late)

- Baseline first: t4_best (0.799 on v1 plant) = **0.083 mean under motor_v2**
  (goto_t0 0.19 / hover_t0 0.06 / land_t0 0.13). The v1 deployment lineage
  does not transfer to the calibrated RS2205 EM/battery plant. Series:
  t4best-m2-baseline.
- dag7m2: init t4_best, m2-native teacher anchor (20.8k samples, v1 demo
  anchors dropped - v1 labels fight the m2 trim), 8 rounds x 30 eps,
  m2-relative floors goto>=0.8/hover>=0.3/land>=0.35 (parent-approved
  re-scope; v1 floors unreachable under m2 - even the teacher yields only
  goto 0.938 / land 0.438 / hover ~0.35 there).
- Result: NO round passed floors -> no best saved. goto_t0 climbed
  0.375->0.812 (floor met by r5), hover_t0 0.125->0.375 (met), but
  land_t0 stayed 0.0-0.06 across ALL rounds. Best mean 0.340 (r8).
- Leading hypothesis: the m2 teacher land fix uses a land-scoped TRIM
  INTEGRATOR (dc867e0) - internal state, actions non-Markovian in obs.
  A feedforward MLP on obs v3 cannot imitate integrator memory from
  (obs, action) pairs. Land fails because the imitable part of the teacher
  is not what lands the drone under sag.
- Options: (a) expose trim/SoC estimate in obs; (b) stateless m2 land
  teacher (sag-compensated feedforward); (c) recurrent student; (d) more
  rounds (weak - the representability gap is structural, not data volume).
- Side fix: t4_dag6j2_best.json had been copied from the wrong campaign
  prefix (run-1 r3, full-jitter, floors-failing); corrected to the true
  dag6j2 r3 (mean 0.750). dag6j2.py copy line noted.


## T15 - dag8c: land curriculum (near-pad terminal-descent spawns) - 2026-09-04 late

dag8b closed the volume hypothesis (24 rounds, correct integrator labels, obs v4:
land_t0 = 0.0 EVERY round; goto 0.938 / hover 0.375 climb fine). Probe showed a
false equilibrium: student parks vy~=0, 0.1-0.5m above pad, zero crashes.

dag8c = dag8b recipe + spawn curriculum: 3/10 collection episodes spawn in
terminal descent (0.3-3m above pad, +-1.5m offset; env_sim scenario_spec key
"spawn_rel_goal"). Integrator teacher lands 5/8 from these spawns (checked).

RESULT (24 rounds, no champion - floors never met): land finally moved off zero
- land_t0 0.062 at r15/r18/r19/r21, land_t2 0.25 at r21 - but collapsed back to
0.0 by r22-24. Best mean 0.403 (r24) vs dag8b 0.361. goto_t0 0.875-0.938,
hover_t0 0.31-0.44. READ: curriculum put terminal-descent data in the mix and
land flickered, but 3/10 density and 24 rounds did not break the equilibrium.
Next: dag9 combines the curriculum with IC v1 (below) and raises near-pad
density; if land still stalls, the lever after that is a land reward shaping
change (descent-progress shaping inside 1m), not more DAgger volume.

## T16 - initial-condition randomization (IC v1) - user steering 2026-09-04

User: "none of the training iterations/scenes start from the drone at rest -
want real variety in initial conditions: starting position, velocity,
acceleration, trajectory." Correct: spawn was FIXED at [0,2,0], velocity always
zero, only goal/scene varied.

Design (AUTORESEARCH_IC_V1=1, sampler-owned, deterministic per seed):
- Every episode: spawn jitter +-3m horizontal, +-1m vertical (floor 0.5m).
- 35% exact rest starts (zero velocity).
- 65% in-motion: log-uniform speed 0.5-4 m/s, directed along the spawn->goal
  leg (+-60deg yaw spread, vertical component +-0.5 of speed). "Acceleration/
  trajectory" variety falls out of velocity x attitude x the controller's
  transient; no separate accel knob (non-physical as an IC).
- Heldout eval cells use fixed seeds, so ICs are deterministic there too -
  but the eval DISTRIBUTION changes when IC is on. Floors stay gated on the
  standard (rest-start) cells for cross-campaign comparability; IC eval is
  reported as a secondary read in dag9.

Implementation: scenario_sampler.sample_initial_conditions (own rng stream
0x1C01), env_quad._sample_scene hook (gated on env flag), env_sim passes
scene.spawn_vel, headless_main.zig Scene.spawn_vel (default zero - old scenes
bit-identical), bullet.cbtBodySetLinearVelocity at reset. Verified: reset obs
velocity channel = 2.0 m/s / 10 with spawn_vel [2,0,0] vs 0 without; velocity
persists through steps. Binary swapped at /workspace/zig-out/bin/
(previous preserved as dronestudio-headless.pre-icv1).

## 2026-09-05 ~06:15 UTC - SI single-port matrix: run5 exonerated, run4 the corrupt sweep; v6 verify visuals regenerated clean

The 12x12 single-port matrix (one port excited per run, 13 runs incl. a
run5 re-run) closed the SI harness question:

- Matrix == run5 BIT-IDENTICAL on 11/12 ports (port 0 differs at numerical
  noise level). run5 (ems/simulation_run5_full) is CLEAN.
- run4 (ems/simulation_sweep1, source of the original v6 SI visuals) is the
  sole corrupt sweep: bottom-port block 2.147-2.168 passivity, port 10 at
  2.47; its clean-looking top-port 1.03 values are ALSO wrong (cross-checked
  bit-level, not by plausibility).
- Top board-edge ports carry a DETERMINISTIC +0.4 passivity excess
  (1.36-1.42), bit-identical across all 13 runs - a board-edge port artifact
  of the harness, not a board defect. Root-causing it is an open experiment.
- Bottom ports carry ~+0.2 excess from an open stub in the harness.

Dashboard ee-cam3 v6 Tests tab: all four SI visuals (sdd_diff, z_diff,
diff_delay, s11_smith) regenerated from the clean matrix only, plus a new
si_passivity panel carrying the reworked HARNESS caveat naming both
artifacts. Uniform-trace ports only, connector/fanout excluded, 550-1500 MHz
characterized band (sweep-edge artifacts >1.9 GHz excluded).

gerber2ems bug report (drafted, covers the run4 corruption signature) is with
the user for review before filing upstream.

## 2026-09-05 ~06:16 UTC - dag9 (land curriculum + IC v1): land still 0.0, honest negative

24 rounds, integrator labels, obs v4, IC v1 on all training episodes,
near-pad density 4/10, dual eval (standard rest-start floors + IC read).

Result: NO champion, floors never met. land_t0 = 0.0 in EVERY round again -
neither the denser terminal-descent data (4/10) nor in-motion spawn variety
moved land off zero. Best standard-eval mean 0.368 (r20), below dag8c best
0.403. IC read runs 0.28-0.31 vs standard 0.31-0.37: a modest robustness
gap on goto/hover, not a collapse.

Interpretation: the false equilibrium (park vy~=0 just above pad) is not an
initial-condition artifact. Next suspects: (a) integrator-teacher labels
themselves encode the park behavior in terminal descent, (b) the land
success criterion is unreachable from near-pad spawns under m2 dynamics,
(c) student capacity. Recommend probing teacher behavior in terminal
descent before more DAgger volume.

## 2026-09-05 ~19:12 UTC - m2 land probe (dag9 follow-up): the blocker is the skim equilibrium, not imitation

Probe: m2_land_probe.py + m2_land_probe_c.py, 16 heldout land t0 cells, m2 plant.

Phase A (integrator teacher census): 6/16 success, 6/16 OFF-PAD touchdowns
(dxz 2.5-3.6 vs radius 1.0-2.0 - horizontal terminal accuracy is the
teacher's real weakness under m2), 3/16 porpoise timeouts (vy 0.2-0.7
sustained), 1/16 park. The dag8-era "teacher stalls at the descend gate"
is NOT the dominant t0 failure mode.

Phase B (forced terminal descent law): changed nothing structural - failures
are upstream of the gate region (transit/centering), except it broke one
teacher success (99011) by fighting its horizontal correction.

Phase C (dag8c r24 student): thrust gap vs teacher along the teacher's own
traces is negligible (terminal region mean -0.009, p90 |gap| < 0.04) -
imitation fidelity is NOT the problem. Student solo census: 11/16 PARK at
alt ~= 0.00, vy ~= 0 indefinitely; 2 off_pad, 2 porpoise, 1 off-pad
touchdown. The student skims at ground level and never crosses the
touchdown plane (sim requires y < GROUND_Y with dxz <= radius, |vs| <= 0.5).

Interpretation: land_t0 = 0.0 across dag7m2/8/9 was never a data or capacity
problem. Under m2 ground-effect the hover equilibrium sits at ground-skim
height; parking there costs -0.01/step while a failed touchdown costs -5,
so the policy rationally refuses to commit. Teacher labels can't fix it -
the teacher itself misses the pad 37% of the time on t0.

Attack for dag10 (proposed): (1) teacher terminal horizontal centering
(tighten land_descend horizontal loop, the off-pad source), (2) reward-side
commit pressure: sustained skim below ~0.15m without touchdown ends the
episode with a penalty worse than a good-faith touchdown attempt, or a
shaping bonus for plane-crossing within radius. Touching reward/scenario
logic only - 500Hz physics untouched.
