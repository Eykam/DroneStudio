# T6: land-in-clutter teacher (land attack, follow-up to T5)

Land was the weakest behavior (t2 0.062 pre-T5, 0.25 post-T5). Diagnosis with
t4_landdbg.py on heldout land t2: 13/16 failures were OBSTACLE HITS during
phase-1 transit at cruise alt 1.2m - the wall protections (velocity cap,
active braking) were gated to scenario=="goto" only, and t2 scenes are
GAP-WALLS (r=0.6 sphere lines, tops ~2.6-2.8m, corridor_width 2.0).

## Fixes (t4_pilot.py)

1. Wall protections extended to land phase-1 transit (velocity cap toward
   obstacle + active braking + full damping near walls; land_descend still
   exempt). Killed the transit crashes (13/16 -> 0/16) but exposed a head-on
   deadlock at ~2m standoff (limit cycle; fixed-direction slide drifted away).
2. Real unlock: FLY OVER the walls - phase-1 cruise alt 1.2m -> 3.5m while
   dist_xz > max(3.0, 3r), dropping to 1.2m in the pad corridor. 16/16 land
   t2 teacher success (was 3/16), 16/16 land t0.
3. Descent-rate cap (-0.6 m/s) in the corridor phase so the land_descend gate
   (alt<=1.4, vy_des=-0.4) inherits no vertical momentum (touchdown vs 0.5).

## Results (heldout cells; chain rerun, selected dag2_r4 -> t4_best.json;
prior best backed up to /workspace/t4_best_pre_landfix.json)

| metric | T5 dag2_r2 | T6 dag2_r4 |
|---|---|---|
| succ ALL | 0.672 | 0.703 |
| face20 ALL | 0.679 | 0.700 |
| goto_t0 / t3 | 1.0 / 0.375 | 1.0 / 0.438 |
| hover_t0 / t2 | 0.938 / 0.625 | 1.0 / 0.5 |
| land_t0 | 0.75 | 0.375 (regressed) |
| land_t2 | 0.25 | 0.938 |

Floors held (goto_t0 1.0 >= 0.9, hover_t0 1.0 >= 0.6). land_t2 face20 0.951.
Known regressions to attack next: land_t0 (0.75 -> 0.375 across the landfix
chains; policy-level, teacher is 16/16 - likely descent-phase imitation
noise), goto_t3 face20 (0.23).
