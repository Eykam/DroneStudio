# T4 retrain on fixed flight stack (post bodyOmega body-frame fix dc67941)

Chain: run_t4_chain.sh (pilot -> pilot2 -> bc -> bc_eval -> dagger -> dagger2),
completed 2026-09-05T00:29:57Z, wall ~5 min. Teacher dc7c80a (wall-aware goto
+ bearing-seeking yaw +-0.04, |yaw_err|<2 rad gate).

Demo health: T4PILOT land_t2 teacher kept only 7/48 (gap-wall land is hard
for the scripted pilot); T4PILOT2 land top-up kept 84/500 on land_t2.
Land remains the weakest behavior.

## Selection (best-with-floors: goto_t0 >= 0.9 AND hover_hold_t0 >= 0.6)

| candidate | goto t0/t1/t2/t3 | hover t0/t2 | land t0/t2 | mean |
|---|---|---|---|---|
| dag2_r3 | 1.0 .875 .625 .375 | .875 .688 | .312 .062 | 0.602 |
| dag2_r4 | .938 .875 .625 .750 | .812 .688 | .312 .062 | **0.633** |
| dag2_r5 | .938 .875 .562 .562 | .812 .688 | .250 .250 | 0.617 |
| dag2_r8 | 1.0 .812 .688 .375 | .938 .625 | .188 .062 | 0.586 |

SELECTED: t4_dag2_r4 -> /workspace/t4_best.json (mean 0.633; pre-fix best
t4_dag3_r1 was 0.562, backup at /workspace/t4_best_pre_omegafix.json).
Policy JSONs live on the /workspace volume (repo convention: no policy
blobs in git).

Champion bc_ppo_v2_best untouched; promotion flip frozen pending user go.
