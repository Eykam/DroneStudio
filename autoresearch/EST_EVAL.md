# Estimator-in-the-loop policy eval (2026-09-06)

Policy obs rebuilt from ESKF state (fed ONLY by simulated MPU-9250 @500Hz via
headless fast_telemetry, AK8963 mag + VL53L9CX altimeter @20Hz). goto scenario,
cell_dist/sample_spec eval cells, seeds 10000+, 30 eps/arm, identical seeds
across arms. Harness: autoresearch/eval_estimated.py. Rows: results/est_eval/.

| policy | arm | success | collision | return | final dist |
|---|---|---|---|---|---|
| v2 champion (19d) | GT | 96.7% | 0.0% | +18.8 | 2.2m |
| v2 champion | EST vision-denied | 26.7% | 40.0% | -29.9 | 41.2m |
| v2 champion | EST + synthetic VO | 53.3% | 16.7% | +7.0 | 8.1m |
| v1 (15d) | GT | 86.7% | 6.7% | +15.6 | 3.8m |
| v1 | EST vision-denied | 33.3% | 46.7% | -16.7 | 28.5m |
| v1 | EST + synthetic VO | 43.3% | 13.3% | +2.9 | 10.9m |

Synthetic VO: chained increments, per-episode scale N(1,3%), yaw-walk
0.3deg/step, 2cm white floor, growing R (0.25^2*step). Anchored to measured
full-pilot VO ATE 5-17m / 50-60m. NOT rendered VO.

Zig: headless_main.zig gains fast_telemetry (default-off; per-fast-step GT
omega/quat/filtered_thrust in step reply; physics untouched).

## ppo_est: estimator-in-the-loop PPO retrain (2026-09-06)

40 updates x 32 eps, warm start bc_ppo_v2_best, all episodes under estimated
obs (synthetic-VO tier), selection on est-obs heldout cells with GT-goto
regression floor (u0 0.938, floor 0.887). Wall ~9 min.

- Heldout EST-obs: u0 goto 0.438 / hover 0 / land 0 -> best goto ~0.50,
  hover/land 0.0 throughout. best_est_mean 0.188 (barely above u0 0.146).
- Apples-to-apples eval cells (30 eps, seeds 10000+, EST+VO):
  bc_ppo_est_best goto success 63.3% vs champion 53.3% (+10pts),
  collision 16.7% (unchanged), final dist 9.4m.
- HONEST NEGATIVE: light PPO (log_std 0.05, lr 1e-4, 40x32) barely moves
  est-obs performance and actively erodes GT skill during training
  (gt_goto 0.938 -> 0.688 on later updates; floors protected the ckpt).
  hover/land 0% under est obs means zero reward signal there - needs
  denser shaping, DAgger, or GT->est curriculum, not more of the same.
