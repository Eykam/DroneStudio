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

## ToF altimeter mount bug + fix (2026-09-06 evening) - all prior est results were IMU+mag+VO only

Two-layer bug: SimToF default mount = identity (FoV forward-looking) and the EstEnv/fusion nadir gate checked RAW sensor-frame dirs, bypassing mount.rot. update_ground_range never fired. Fixed: boresight-down mount + mounted-dir gate (4735402). Verified: tof_valid 0.4%->96.7%, reading accuracy 1.5cm. Effect on UNRETRAINED champion under v3 obs: est goto 43.8->87.5%, est land 0->6.2%, hover alt probe R2 -0.07->0.974. fusion_chained_g has the same gate bug (ToF inert in all fusion scorecards) - patch pending.

## DAgger v3 run1 (inert ToF) + run2 (live ToF) - hover/land still ~0%

run1: est hover/land 0% all 12 iters, GT arm collapsed i1 (bc_train aggregation overwrites warm start). run2 (post-fix): est hover blips 6.2% (i4-6), est land 0% after iter0, GT hover/land 0-18.8%/0% all iters, floors_ok=False x12. Best = iter0 (0.312). Teacher check: pilot_act3 on GT = goto 100%, hover 87.5%, land 100% (eval cells) - teacher is NOT the cap. Chain: observability fixed + teacher competent + student still fails = policy-class wall (memoryless MLP + bc_train recipe). Log: results/est_eval/dagger_est_v3_run2.log.

## v61-g60a dynamics re-derivation (2026-09-06) - conclusions survive

fixtures/v61_g60a.manifest.json (schema 1.2, from CAD v61-g60a): mass 0.5398->0.5201kg, ixx +6.3% iyy -3.9% izz +0.6%, aero z-area +81%, IMU real pose + offset_from_com. Champion eval under BOTH manifests (16 eps/cell): GT arm IDENTICAL (93.8/68.8/37.5 both), est_v3 within n=16 noise (goto 87.5/81.2, land 6.2/12.5, hover 0/0). All est-track conclusions (ToF fix, 0% hover/land wall, probe) survive the re-derivation. Data: results/est_eval/manifest_delta_eval.json.

## Tracks launched (parent call 20:37): A = GT-obs hover/land specialists (dagger_gt_specialist.py), B = history-stacked K=4 v3 est-obs policy (dagger_est_hist.py)

## Track A: GT-obs specialists (2026-09-06) - SUCCESS

dagger_gt_specialist.py, DAgger from champion, teacher pilot_act3, GT obs, v14-era dynamics (pre-flip import; GT numbers measured identical under v61).
- hover_hold specialist: 68.8% -> 87.5% = TEACHER PARITY (results/bc_gt_hover_hold_best.json). Oscillated 68.8-81.2% live, best-checkpoint discipline held.
- land specialist: 37.5% -> 50.0% (teacher 100%; results/bc_gt_land_best.json).
Both are deep specialists (goto/land collapse on the hover specialist, goto/hover ~0 on the land specialist) - dispatch per-scenario; champion remains the generalist.

## Track B: history-stacked K=4 policy under est obs - NEGATIVE

dagger_est_hist.py (100-dim, champion newest-frame block, 12 iters): est hover 0.0% all 12 iters, est land one 6.2% blip, GT arm eroded (hover 62.5->6.2%). Memory does NOT change the outcome. Across v3-run1, v3-run2, hist: the common failure is the DAgger recipe itself - 1500-iter bc_train on aggregated student-visited (crashing) states overwrites the warm start in ONE iteration, every variant. Est-obs hover/land final tally: six recipe families, 0%.

## Fusion scorecards re-run with LIVE ToF (2026-09-06, post gate fix) - earlier v11 conclusion REVERSED

Gate fix (mounted dirs + boresight-down mount) applied to fusion_chained_g.py; inert-ToF scorecards preserved as *_tof_inert.json.
- GT-depth + live ToF: ATE 36.2 (max 133.4), yRMSE 3.65, att 26.5, RPE 0.80
- v11-depth + live ToF: ATE 38.6 (max 92.4), yRMSE 1.63, att 22.4, RPE 0.88
With the altimeter live, predicted-depth fusion MATCHES GT-depth fusion (better yRMSE/att, no catastrophic scene). The inert-ToF "v11 degrades fusion" read was an artifact. Note: ToF worsened GT-depth ATE vs inert (30.2->36.2) - the fusion gates were tuned without ToF; retuning queued behind training tracks.
