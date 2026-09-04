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
