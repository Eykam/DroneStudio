# Sensor abstraction library (dronestudio.sensor/1)

One spec-driven model per sensor part. Three consumers read the SAME file:

- **sim dynamics** (box1): `dynamics` block - noise densities, bias walks,
  sample rate, latency. Loaded by `Studio/src/core/SensorSpec.zig`, env-gated
  (`DRONE_IMU_SPEC`, `DRONE_TOF_SPEC`); unset = compiled-in values unchanged.
- **CAD** (box2): `physical` + `orientation` blocks - envelope bbox, shape,
  mount pattern, mass (+ provenance), die/lens axis conventions. Positions are
  NOT in this schema: placement stays CAD-side pins (placement.json).
- **EE** (box3): `procurement` + `ee` blocks - PLM OPN, MPN, KiCad footprint,
  bus interface. A part swap updates the OPN/footprint here and the EE tracks
  redesign around it.

A part swap is: new `<part_id>.json` + repoint the consumer. No dynamics-code
edits, no CAD-library edits, no EE schematic archeology.

## Field reference

```
schema                  "dronestudio.sensor/1"
part_id                 snake_case key, matches CAD components.py entry
category                imu | tof | gps | camera | baro | mag
procurement.plm_opn     OSP-xx-xxxx in /work/plm (null = not yet registered)
procurement.mpn         manufacturer part number
procurement.status      active | eol-replacement-pending
physical.mass_g         grams; null when unknown
physical.mass_provenance    measured | datasheet | estimate | unverified
physical.dims_m         bbox {x,y,z} meters
physical.dims_provenance    same enum
physical.shape          box | cylinder | pcb-module
physical.mount.style    smd | breakout-screws | solder-pads | stack-standoff
physical.mount.pattern  free-form pattern key (e.g. qfn-24_3x3_p0.4)
orientation.*           explicit axis conventions - the silent-drift risk.
                        sensor/optical frame axes, die-vs-lens offsets, and
                        any mount-rotation convention live HERE, not in code.
ee.kicad_footprint      KiCad lib:footprint (null = EE to fill)
ee.interface            bus, address, extra pins (xshut etc.)
dynamics.<category>     sim noise/latency model parameters (sim-owned)
```

Provenance fields are mandatory: known-weak numbers stay visible as weak
(e.g. vl53l9cx_breakout mass 2.0g is an unmeasured estimate) instead of
silently hardening into "data".
