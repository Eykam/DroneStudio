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

## Observability diagnostic (2026-09-06)

diag_est.py: 6 arms, scenarios {goto, hover_hold, land} x {est-obs, GT passthrough (estimator passive)}, 12 eps each, eval cells, seeds 10000+.

Champion (bc_ppo_v2_best, GT-trained):
- goto: est-obs 66.7% vs GT 91.7% (estimator costs ~25pts; pos err 1.49 vs 1.08m)
- hover_hold: est-obs 0% (33% collision) vs GT 50% (0% collision); pos err ~2.3-2.5m, p90 4.5m in BOTH arms (drift is estimator-side, trajectory-independent)
- land: est-obs 0% (75% collision) vs GT 41.7% (50% collision); pos err ~1.3m both arms

Read: est obs are NOT control-sufficient for hover/land - the champion holds 50%/42% on GT obs and collapses to 0%/0% on est obs. Hover drift ~2x goto (weak VO aiding at low translation). Residual GT-obs skill gap also real (50%/42%, land collisions 50% even on GT).

Confound note: bc_ppo_est_best cannot hover/land even on GT obs (8.3%/0%) - ppo_est training atrophied GT hover/land skill; its earlier diagnostic (diag_observability_estbest.json) is policy-confounded.

## Noise-ramp curriculum (2026-09-06, ppo_est_ramp.py) - NEGATIVE on hover/land

Parent GO 17:41. From champion bc_ppo_v2_best; noise scale 0.25x -> 1.0x over u1-u20, hold to u40; 24 est-obs (8/8/8) + 8 GT hover/land rehearsal per update; dense shaped reward; eval at 1.0x on heldout cells (16 eps/scenario).

- EST hover/land: 0.0% at ALL 40 updates (0/640 eval eps), including ns=0.25-0.55 where hover drift is sub-meter. Hover is not learnable under est obs even at quarter noise: the wall is structural, not noise magnitude.
- GT rehearsal failed to defend: GT hover 68.8% -> 6.2% (monotonic decay), GT land noisy 18.8-43.8%, GT goto held 93.8% to u39 (87.5% at u40). floors_ok=False on 38/40 updates.
- What worked: est goto 43.8% -> 62.5% with GT goto intact. Best ckpt results/bc_ppo_est_ramp_best.json (mean 0.208, floors-ok update). Log: results/est_eval/ppo_est_ramp.log.
- Five recipes now at exactly 0% est hover/land: ppo_est, anneal, dagger, shaped, ramp. Next levers proposed to parent: est-obs vector redesign (uncertainty/innovation channels), GT specialist distillation, or block on real VO/depth track.
