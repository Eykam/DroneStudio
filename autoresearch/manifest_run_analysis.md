# Manifest-dynamics validation run (5 gens x 3 children, seed 11)

Report: autoresearch/generations_report_manifest.json
Archive: autoresearch/archive_manifest.jsonl (20 records)
Backend: sim (real Zig headless binary), dynamics: fixtures/chassis_v1.manifest.json
(CAD chassis v1, 0.5143 kg, EMAX 2207 2400KV motor model, aero flat-plate drag)

## Result

- All 20 variants: success_rate 0.00. Best by return: g1v5 (mean_return 6.75,
  mean_steps 22.5 of 200 - episodes end early, mostly crashes).
- Selection behaved sanely: codex mutators produced novelty, elites tracked the
  best-return variants, restart fired at gen 4 after stagnation. The outer loop
  machinery works identically on the manifest airframe; the environment is
  simply much harder than the abstract model (run 2 best: 33.3% on quad backend).

## Interpretation

- The validation goal is met: the full pipeline (evaluator, trainer, outer
  loop, archive, records) now runs end-to-end on the CAD airframe dynamics.
- The difficulty gap (0% vs 33%) is expected to be some mix of: real inertia /
  mass / motor saturation / aero drag (manifest), vs the abstract point model
  (quad backend). PID gains were re-tuned for the manifest airframe
  (rate_tune_manifest_report.json, SIM-TUNED ONLY).
- A/B diagnostic (same dists, sim backend, abstract vs manifest dynamics)
  isolates how much of the gap is the airframe vs the sim backend itself.
