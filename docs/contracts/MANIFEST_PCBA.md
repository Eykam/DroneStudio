# dronestudio.pcba/1.0 - populated PCBA manifest (EE -> ME/SIM)

Published per EE board version alongside the versioned artifacts:
- `glb` (populated PCBA: board + spec-exact component meshes)
- `glb_bare` (bare board), `pcba` (this manifest, pcba.json)
Fetch: /api/ee/boards/<board_id>/versions/<v>/file?kind=pcba|glb|glb_bare

Fields:
- schema: "dronestudio.pcba/1.0"
- board_id, candidate_id
- board.outline_mm: {min_x,min_y,max_x,max_y,width,height} (Edge.Cuts bbox, mm,
  KiCad board coordinates; GLB maps x->X, y->Z, meters)
- board.mounting_holes: [{x, y, drill_mm}] - empty until SB6 lands them
- board.thickness_mm: from the 3D body (1.635 for ee-flight v19)
- components[]: {ref, value, package, side, x, y, mass_g}
- totals: {component_count, components_g, board_g, pcba_g}
- mass_estimate: true while masses come from spec-exact package primitives;
  swap to measured when parts are weighed.

ME consumption (chassis): replace assumed board envelopes with the GLB +
outline; roll totals.pcba_g into the mass budget; containment re-verifies.
SIM consumption: board mass feeds chassis inertials when the flight board is
in the loop (today: informational).
Pinning: consumers record (board_id, version); the orchestrator announces
bumps. ee-flight v19 is the proto-1.0 (pcba_g ~31.73g est, no schema field); v20+ declares schema dronestudio.pcba/1.0 with outline + holes.
