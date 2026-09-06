# Cross-domain manifest contracts (user-delegated 2026-09-05)

One exchange pattern across ME (chassis), EE (boards), SIM, VISION:
every cross-domain artifact carries a versioned manifest JSON named
`dronestudio.<domain>/<schema-version>`. Producers publish versions;
consumers PIN to an explicit (artifact, version) - never "latest". No agent
writes another agent's files. Version bumps route through the orchestrator.

| Manifest                | Producer -> Consumer(s) | Status | Doc |
|-------------------------|-------------------------|--------|-----|
| dronestudio.chassis/1.2 | ME -> SIM (+VISION poses) | LIVE | MANIFEST_SIM.md (repo) |
| dronestudio.pcba/1.0    | EE -> ME (chassis reference), SIM (mass) | LIVE (ee-flight v19+) | MANIFEST_PCBA.md (this dir) |
| dronestudio.harness/1.0 | EE pinout + ME routing -> both | FIRST INSTANCE (ee-tof ring) | MANIFEST_HARNESS.md (this dir) |
| dronestudio.dynamics/1.0| EE -> SIM (battery sag, ESC response, DShot latency) | PLANNED (with SB3) | TBD |

Exchange surface: the dashboard's versioned artifact store for EE outputs
(/api/ee/boards/<id>/versions/<v>/file?kind=...) and the repo for contracts
+ instances. CAD/snapshot artifacts keep their existing snapshot dirs.

Rules:
1. Consumers gate on schema string equality; unknown major versions refuse
   to load (fall back to prior pinned version, never to guesses).
2. Mass/geometry fields carry a `mass_estimate`/method note when derived
   from spec-exact primitives instead of measured parts.
3. A manifest never points at mutable state: every reference is
   (artifact id, integer version) or a content hash.
