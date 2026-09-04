# CAD / Mechanicals Ingest API (contract for the CAD researcher agent)

Audience: the CAD researcher agent (box #2, `chassis` branch). The dashboard
service (dronestudio-dashboard-production.up.railway.app) owns storage and
display. This document is the contract - if you change the payload shape,
update this file in the same commit. Coordinated through the repo: CAD agent
pushes code to `chassis`, dashboard code lives on `auto-researcher`; this doc
is visible on both.

## Auth

Machine-to-machine: the same bearer `INGEST_TOKEN` used by the sim box
poster. NOT in the repo. It is set as `INGEST_TOKEN` (+ `DASHBOARD_URL`) in
your Railway project's service variables (set 2026-09-04 by the dashboard
agent). Read them from your environment.

    Authorization: Bearer $INGEST_TOKEN

## Two-phase push (metrics now, geometry when ready)

### Phase 1 - snapshot record (metrics, lineage, GLB path)

POST a record through the EXISTING generic ingest endpoint. Dedup is by
record `id`, kind-agnostic - CAD records coexist with run records.

    POST $DASHBOARD_URL/api/ingest
    Authorization: Bearer $INGEST_TOKEN
    Content-Type: application/json

```json
{
  "records": [
    {
      "id": "chassis-v003",
      "kind": "cad.chassis.snapshot",
      "parent_id": "chassis-v002",
      "name": "Chassis v3",
      "ts": "2026-09-04T07:00:00Z",
      "glb_path": "/workspace/cad/out/chassis-v003.glb",
      "metrics": {
        "mass_g": 142.5,
        "inertia": {"ixx": 0.0012, "iyy": 0.0012, "izz": 0.0018},
        "printability": {"material": "PLA-CF", "wall_mm": 1.6, "supports_needed": false},
        "fea": {"max_stress_mpa": 41.2, "safety_factor": 2.3}
      },
      "notes": "Freed 8g from the arm roots."
    }
  ]
}
```

Rules:
- `kind` MUST be exactly `cad.chassis.snapshot` - that is how the dashboard
  routes it to the CAD section (records without a kind are run variants).
- `id`: unique per design, `[A-Za-z0-9._-]`. Re-POST the same id to update
  (e.g. when FEA lands). Use a NEW id for a new design.
- `parent_id`: id of the design this derives from (drives the lineage view).
  Omit/null for a root.
- `metrics`: free-form; `mass_g`, `inertia`, `printability`, `fea` render as
  named sections, everything else under "other metrics".

### Phase 2 - GLB binary upload

    POST $DASHBOARD_URL/api/cad/designs
    Authorization: Bearer $INGEST_TOKEN
    Content-Type: multipart/form-data

Fields:
- `file` (required): the GLB binary, self-contained (no external refs).
  Max 64 MB. Plain GLB - Draco compression NOT supported by the viewer.
- `meta` (required): JSON string, same shape as the phase-1 record minus
  `kind`/`glb_path`: `{"id": "chassis-v003", "parent_id": "chassis-v002",
  "metrics": {...}, "name": "...", "notes": "..."}`

The `id` links the phases: the CAD page merges snapshot metrics with the GLB
of the same id. Until a GLB lands, the design shows "geometry pending" with
your `glb_path` so the user can see where the bytes live.

Response: `{"ok": true, "id": "...", "glb_bytes": 12345}`.
Errors: 401 bad token, 400 missing/invalid fields, 413 over size limit.

You MAY skip phase 1 and push everything in one multipart call (meta carries
metrics) - phase 1 exists so designs are visible before geometry is ready.

### curl example (phase 2)

```bash
curl -X POST "$DASHBOARD_URL/api/cad/designs" \
  -H "Authorization: Bearer $INGEST_TOKEN" \
  -F "file=@chassis-v003.glb;type=model/gltf-binary" \
  -F 'meta={"id":"chassis-v003","parent_id":"chassis-v002","metrics":{"mass_g":142.5}}'
```

## Read side (for reference; the dashboard UI consumes these)

    GET /api/cad/designs            (session cookie) -> {"designs": [...meta + glb_url...]}
    GET /api/cad/designs/:id/glb    (session cookie) -> model/gltf-binary

## Geometry guidance for the viewer

- Units: meters preferred, Y-up. The viewer auto-centers; be consistent
  across designs so lineage comparisons make sense.
- Self-contained GLB only.
