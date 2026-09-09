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

## DAgger v4 (2026-09-06/07) - recipe fix test: lr 1e-3, 300 iters, 8/32 GT-anchor episodes

Recipe hypothesis: prior DAgger variants destroyed the BC warm start in one iteration
(lr 3e-3 x 1500). v4 lowered lr to 1e-3, cut train iters to 300, and mixed 8 GT-anchor
passthrough episodes into each 32-episode aggregation round. 14 iterations, wall 661s.

Result: NEGATIVE on the headline, partial on the mechanism.
- EST arm: hover_hold 0.0 and land 0.0 on ALL 14 iterations (goto 0.50-0.88).
  Est-obs hover/land remains unlocked after seven recipe families
  (ppo_est, anneal, dagger v1-v3, shaped, ramp, v3/hist, v4-recipe-fix).
- GT arm: no catastrophic collapse (hover 0.125-0.625 band, land up to 0.75 at i10),
  vs prior variants collapsing hover to ~0.188 in one iter. The gentle recipe DOES
  preserve skill. But floors_ok=False on every iteration - never held all three GT
  arms above the gt0-0.05 floors simultaneously (goto dipped below 0.887 most iters).
- best_mean=0.292 (iter1). Best checkpoint: results/bc_est_dag_v4_best.json.

Conclusion: recipe collapse is fixable but is NOT the wall for est hover/land.
With GT-anchor data, a working altimeter (ToF fix), uncertainty channels (v3 obs),
history stacking (K=4, negative), and a competent teacher (100/87.5/100), the est-obs
student still gets 0% on hover/land. Remaining hypotheses: (a) est-obs hover/land
needs closed-loop correction authority the BC/DAgger action space does not express
(residual/hybrid control), (b) the EKF estimate distribution under hover/land has
shift that BC cannot mimic from aggregated state-action pairs (needs on-policy RL
with a reward that survives estimation noise), or (c) eval-cell geometry for
hover/land is out-of-distribution for the aggregated dataset.

## Fusion gate sweep (2026-09-07) - 9 configs, 12 scenes, GT depth, live ToF

Full results: results/fusion_gate_sweep.json. Baseline (inno25, tgate2, safloor.15,
reanchor20): ATE 39.4 / max 141.9 / yRMSE 3.89 / att 27.4 / RPE 0.824.

| config        | ATE   | maxATE | yRMSE | att  | RPE  |
|---------------|-------|--------|-------|------|------|
| baseline      | 39.40 | 141.87 | 3.887 | 27.4 | 0.824 |
| inno15        | 30.56 |  68.95 | 2.032 | 23.7 | 0.889 |
| inno40        | 34.49 |  90.09 | 3.221 | 27.0 | 0.892 |
| tgate1.0      | 36.98 | 126.15 | 4.252 | 26.0 | 0.850 |
| tgate4.0      | 31.34 |  67.40 | 1.362 | 21.7 | 0.738 |
| safloor.05    | 41.01 | 139.21 | 4.670 | 27.4 | 0.840 |
| safloor.30    | 32.27 |  90.74 | 1.326 | 22.4 | 0.750 |
| reanchor12    | 36.00 |  88.72 | 1.995 | 23.8 | 0.838 |
| reanchor30    | 25.75 |  38.65 | 1.315 | 19.9 | 0.701 |

reanchor30 dominates on every metric (ATE -35%, max ATE -73%, no metric worse).
NOT adopted as default - awaiting parent decision.

## Diagnostic (c) 2026-09-07: is eval-cell geometry OOD for the aggregated dataset? REFUTED.

Two-level check (est_ood_diag.py, full log results/est_ood_diag.log):

A) Spec parity: held-out eval blocks (hover 88000+, land 99000+) are drawn from
IDENTICAL distributions as training seeds - success_radius (hover 0.50-1.50 train
vs 0.60-1.37 eval; land 0.30-0.60 vs 0.33-0.58), hold_s (2.0-59.8 vs 2.4-54.1),
goal_distance {2,5,10,15,25}, density {0,0.05,0.1,0.2}. No spec-level OOD.

B) Student (v4 best) visitation on 24 TRAINING hover/land cells: 0/24 success,
but it REACHES the target region - min_dist down to 0.03m (median ~1.1m), then
drifts off (final_dist up to 25m). Steps within success radius: 0-3.9%.
The student transits the success region; it cannot hold.

C) Teacher on EVAL cells: 32/32 success, 8-93% of steps within radius.

D) Coverage: 6387 teacher success-region states vs 12208 student-visited states
(normalized 10-dim pos/vel/rel/dist space): NN median 1.11 sigma, p90 1.74,
only 0.8% beyond 2 sigma. Student cloud covers the teacher success-region
manifold, including velocities (teacher hold |v| <= 2.2 m/s is inside the
student range). Coverage is THIN near the hold manifold (student spends 0-4%
of steps there vs teacher 8-93%) but not absent.

Conclusion: eval geometry is fair; hypothesis (c) rejected. The wall is
behavioral - the policy cannot convert target transits into station-keeping
under estimated obs. Proceeding to hypothesis (a) per parent direction:
closed-loop correction authority. First measurement: teacher (pilot_act3)
driven by ESTIMATED v3 obs instead of GT - quantifies how much estimation
noise alone degrades a controller with full authority.

## Hypothesis (a) gate measurement 2026-09-07: teacher on ESTIMATED obs. NEGATIVE.

pilot_act3 (full-authority scripted pilot, GT-obs eval: goto 100 / hover 87.5 /
land 100) driven by estimated v3 obs on the same held-out cells:
  goto 87.5% | hover_hold 0.0% | land 0.0%
(hover hold-speed under est obs: 1.63 m/s mean - it believes it is holding while
drifting). Result: est_teacher_eval.py, results/est_teacher_eval.log, series
est_teacher_* posted.

Consequence: estimation noise in the terminal phase does not just handicap the
BC student - it defeats a controller with FULL closed-loop authority. Authority
(hypothesis a) is therefore not sufficient on the current estimator; the binding
constraint for est hover/land is terminal-phase estimation quality (horizontal
drift over multi-second holds; vertical is ToF-aided, alt R2 0.974). Note the
control-time EKF is still the UNGATED estimator - the fusion innovation-gate /
reanchor work (reanchor30 now default) has only been applied to the trajectory
pipeline, never to the control loop.

## Fusion port to control loop (2026-09-07, parent dir) - NEGATIVE, destabilizing.

Ported fusion_chained_gs innovation gate + chain re-anchoring into EstEnvs
control-time estimator behind an opt-in flag (fusion_gated=True; default False,
ungated path untouched). Teacher-on-est gate (pilot_act3, est v3 obs):
  goto 81.2% (vs 87.5% ungated) | hover 0.0% | land 0.0%
Diagnostics: hover pos_err 21.6m mean, land pos_err 502.7m mean, att_err ~8.5deg,
vo rejects nearly zero (14-70 vs 11k-29k accepts).

Why it failed: the trajectory-pipeline gate works because ICP rotation increments
are gated against GYRO integration - two independent rotation measurements. The
control loops synthetic VO is a position chain

## Fusion port to control loop (2026-09-07, parent dir) - NEGATIVE, destabilizing.

Ported fusion_chained_g's innovation gate + chain re-anchoring into EstEnv's
control-time estimator behind an opt-in flag (fusion_gated=True; default False,
ungated path untouched). Teacher-on-est gate (pilot_act3, est v3 obs):
  goto 81.2% (vs 87.5% ungated) | hover 0.0% | land 0.0%
Diagnostics: hover pos_err 21.6m mean, land pos_err 502.7m mean, att_err ~8.5deg,
vo rejects nearly zero (14-70 vs 11k-29k accepts).

Why it failed: the trajectory-pipeline gate works because ICP rotation increments
are gated against GYRO integration - two independent rotation measurements. The
control loop's synthetic VO is a position chain; the only thing to gate it
against is the filter itself (self-referential), so the gate accepts ~everything
(toothless), while the reanchor snaps the chain to the filter during rejection
windows - if the filter has run away on IMU drift, the chain anchors to garbage
(positive feedback -> 503m mean land pos_err). Direct port does not transfer.

Actual terminal-phase error budget (from hold analysis): during holds the drone
is near-stationary, so VO increments are ~2cm/step noise random-walk (~0.7m over
a 60s hold - exactly success-radius scale) plus 0.3deg/step yaw walk. The filter
ingests that wander every step and the teacher chases it (1.63 m/s believed-hold).
The honest lever for THIS failure mode is velocity-domain damping (e.g.
zero-velocity updates when IMU+ToF indicate stationary), not position-chain
gating. Ungated control estimator remains the default.

## ZUPT (2026-09-07, parent-approved) - partial fix, hover/land still 0%.

Zero-velocity updates in EstEnv (zupt flag, default False) + ESKF.update_velocity.
Detector (sensor-side only): |gyro|<0.15 rad/s AND |a|~g AND VO step <0.04m AND
ToF alt steady. Teacher-on-est (est v3 obs):
  goto 87.5% (pos_err 0.86m) | hover 0.0% (pos_err 4.2m, hold speed 2.08m/s,
  zupt 499 fires) | land 0.0% (pos_err 2.8m, zupt 157)
vs ungated 87.5/0/0 (hold speed 1.63) and gated 81.2/0/0 (21.6m/502.7m pos_err).

ZUPT stabilizes the estimate (no runaway, pos_err 4.2m vs gated 21.6m) but does
not reach the ~1m hover precision needed. Root cause is structural: horizontal
position has NO absolute channel in the sim sensor suite - IMU integrates and
wanders, VO random-walks, mag is attitude-only, ToF is vertical-only. Nothing
anchors x/z. The placed hardware (placement-effective.json) includes GPS at
[-0.1165, 0, 0.002] - the sim simply does not simulate it. Next honest lever:
add a u-blox-class GPS channel (~1.5m CEP, 5-10Hz) so GPS anchors the absolute
frame while VO supplies smooth relative motion; re-gate teacher-on-est.

## 2026-09-07 GPS channel (u-blox-class) + ZUPT, teacher-on-est gate
Parent-greenlit fidelity fix: placement-effective.json has GPS at [-0.1165,0,0.002] but sim never modeled it. Added 5Hz GPS: OU bias (tau 120s, 1m stationary std/axis) + white noise (1.27m h, 2.5m v per-axis), fused via kf.update_position; sampled from TRUE pose. GPS=1 env flag.
Teacher (pilot_act3 GT-trained) on est v3 obs, GPS+ZUPT, 16 ep/phase:
- goto 87.5% (pos_err 0.889)
- hover 6.2% (1/16, pos_err 2.085, att_err 5.98, zupt=865 fires, gps=6659 fixes)
- land 6.2% (1/16, pos_err 1.834, att_err 9.76)
vs ZUPT-only: pos_err hover 4.2->2.09m, land 2.8->1.83m. Estimate fidelity now sits at the GPS bias floor (~1-2m); hover/land success criterion is tighter than that floor, so headline stays ~0. Estimation is no longer the obvious bottleneck at this magnitude - remaining gap is bias-floor vs criterion plus behavioral (att_err rose with GPS coupling).
Next lever: est-obs policy training on the GPS+ZUPT stack (DAgger v4 was blocked by garbage estimate; distribution now much closer to GT).

## 2026-09-07 Terminal relative-nav attempt: GPS bias-as-state (18-state ESKF)
Parent-greenlit architecture probe: GPS gets you there, relative sensing puts
you down. Added bgps OU states (tau 120s, 1m stationary) + update_gps H=[I|I].
Synthetic static test: bias converges to ~0.1m horizontal. In-episode: NEGATIVE.
Single-episode trace (hover seed 88000): bias_err 0.49 at t=0 -> grows with
vo_drift (0.05 -> 1.5m); the VO-chain random walk (2cm/step) pollutes the bias
split through the VO position-update channel. Teacher gate with bias state:
goto 93.8/hover 6.2/land 0.0, pos_err 2.61/2.62 (vs plain GPS+ZUPT 2.09/1.83,
hover/land 6.2/6.2). Reanchor30-on-GPS variant STRONGLY NEGATIVE (hover pos_err
17.1m, land 24.5m: fresh small-R VO noise injected at 20Hz kicked the filter).
Conclusion: VO chain drift ~= GPS bias dynamics, so the bias split redistributes
rather than reduces error; terminal relative-nav does NOT close land 0.3-0.6m
on this sensor suite. Arrival offset (~|bias| ~1.25m mean) is preserved by any
relative frame. Kept behind gps_bias_state flag, default OFF. Plain GPS+ZUPT
remains the stack. Est-obs PPO (lever b) launched on GPS+ZUPT.

## 2026-09-07 Est-obs on-policy PPO (lever b) on GPS+ZUPT stack: NEGATIVE
ppo_est.py (PPO fine-tune, warm start bc_ppo_v2_best, estimator-in-loop,
heldout-cell selection, GT-goto floor) patched to zupt=True gps=True.
40 updates x 32 eps (~10min): best_est_mean 0.3125 set at u4 (goto 87.5/
hover 0/land 6.2) and never beaten; hover_hold 0% at the selected checkpoint.
Late updates degraded GT-goto to 0.69-0.75 (floor 0.887). On-policy RL at this
budget does not move hover/land on the GPS+ZUPT est stack.
WALL STATUS (all levers measured): (a) authority/hybrid refuted; (b) on-policy
PPO negative; (c) OOD refuted; DAgger v4 negative; ZUPT estimate-stable but
headline 0-6%; GPS channel halves pos_err (2.1/1.8m) but bias floor > criterion;
terminal relative-nav (bias state) negative. Land 0.3-0.6m is structurally
beyond single-band GPS + this IMU/VO/mag/ToF suite under GT-frame scoring.

## 2026-09-08 Pad-marker channel (parent-greenlit): THE WALL CRACKS
Downward camera tracks an AprilTag-style marker ON THE GROUND below the
scenario target (65-deg half-angle cone, <5m, 10Hz, sigma = 1cm + 2cm/m,
sampled from TRUE pose), fused via kf.update_marker (body-frame relative
vector, H on position+attitude). Terminal relative nav: ties the filter to
the TARGET, bypassing the GPS bias floor. Marker at goal altitude was wrong
(6 fixes/hover, no signal); marker on the ground below is the right geometry.
Teacher (pilot_act3 GT-trained) on est v3 obs, GPS+ZUPT+MARKER, 16 ep/phase:
- goto 93.8% (pos_err 0.738)
- hover_hold 50.0% (8/16, pos_err 1.451, att_err 5.66, marker=1434 fixes)
- land 62.5% (10/16, pos_err 0.893, att_err 9.20, marker=952 fixes)
vs GPS+ZUPT: hover 6.2 -> 50.0, land 6.2 -> 62.5. Estimation is no longer
the binding constraint; remaining gap is behavioral (att_err, hold
stability). Next: est-obs PPO on the full stack (lever b retry, marker on).

## 2026-09-08 Est-obs PPO on GPS+ZUPT+MARKER: NEGATIVE (warm start stands)
ppo_est.py (marker=True): best_est_mean 0.3958 = the u0/u1 warm-start snapshot
(goto 81.2/hover 12.5/land 25.0 est-obs); 40 updates never beat it, GT-goto
eroded to 0.688 (floor 0.887). Same shape as the GPS+ZUPT run: PPO at this
budget does not close the behavioral gap (teacher gate hover 50/land 62.5 vs
policy-on-est 12.5/25). Next lever: DAgger v5 (v4 recipe, marker stack) - v4
failed on the OLD est distribution; the marker stack puts est obs much closer
to GT, which is exactly what imitation needs. v5 launched 6:38 PM.

## 2026-09-08 DAgger v5 (v4 recipe on GPS+ZUPT+MARKER): PARTIAL
14 iters, 672s. best_mean 0.479 at iter6: EST goto 87.5/hover 12.5/land 43.8
(warm start was 81.2/12.5/25.0). Land climbed 25 -> 43.8% (real imitation gain
under the marker channel); hover oscillated 0-25% and never settled. Still
under the teacher gate (50/62.5). Best checkpoint /workspace/bc_est_dag_v5_best.json.
Housekeeping: v5 inherited v4s OUT prefix and overwrote v4s /workspace
checkpoints during the run; v4 artifacts restored from results/, v5 renamed.
Remaining behavioral gap is hover-dominant: land is learning, hover station-
keeping is not (hold-speed oscillation in the student under est noise).

## DAgger v6 (2026-09-08 ~19:11-19:29 PDT) - hover-heavy mix from v5 best: MARGINAL
- Recipe: v5 (lr 1e-3 x300, GT-anchor 8/32), iteration mix 70/20/10 hover/land/goto, 25 iters, warm from bc_est_dag_v5_best.json. Log: results/dagger_est_v6.log
- Result: best_mean=0.500 at iter21 (EST goto 87.5 / hover 18.8 / land 43.8) vs v5 best 0.479 (goto 87.5 / hover 12.5 / land 43.8). +0.021, all of it hover, land unchanged.
- Hover still oscillates 0-31% across iters and never holds; teacher GT hover improved under the hover-heavy data (0.75-0.88 mid-run) so the data is there - the student is not absorbing it. Hover-heavy mix alone does not crack hover.
- Best checkpoint: /workspace/bc_est_dag_v6_best.json (iter21).
- Launch incidents (fixed pre-flight): 25-dim warm-start must bypass warm_start_25() (19-dim only); /proc-kill idiom bug left a duplicate racing the first launch - relaunched clean.

## Marker range 5m -> 8m + DAgger v7 (2026-09-08 ~20:05-20:40 PDT): mechanism confirmed, training still flat
- Root cause found by marker-visibility probe (results/marker_vis_probe.*): hover goals sit 0.5-6.9m altitude; 5m cap dead by design in 3/16 cells; marker valid 49.5% overall, 96.7% steady vs 45.2% correcting, 78% of invalidity range-out. eval_estimated.py cap raised to 8m (a6a48db).
- Teacher gate at 8m (results + /workspace/est_teacher_marker8m.log): scores FLAT (goto 100 / hover 50.0 / land 62.5) but hover est pos_err 1.451 -> 0.679, marker fixes 3.9x. Teacher hover failures are behavioral, not estimation.
- Probe at 8m: valid 49.5 -> 52.4% overall, corrections 45.2 -> 48.7%; residual invalidity is wander beyond any cap (p99 range 16.7m).
- DAgger v7 (v6 recipe, hover-heavy, from v6 best, 25 iters): best_mean=0.521 = the warm start itself (iter0 87.5/25.0/43.8; hover 12.5->25.0 from the sensor change alone). No iteration beat it under floors. BUT hover EST reached 56.2% at iter25 (v6 max was 31.2%) with goto collapsing to 62.5 (floors_ok=False) - the BC recipe oscillates between phases instead of converging; hover is now trainable signal, the aggregator is the limiter.
- Best checkpoint: /workspace/bc_est_dag_v7_best.json (= v6 best numbers; training could not improve on it).
