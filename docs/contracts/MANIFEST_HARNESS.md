# dronestudio.harness/1.0 - wire harness contract (EE pinout + ME routing)

One instance per board-to-board (or board-to-sensor-ring) harness. Split of
ownership: EE authors the electrical pinout + wire spec; ME owns the physical
route, lengths, and containment inside the chassis. Both gate on schema
equality; changes bump the instance's `version` and route via the orchestrator.

Fields:
- schema, id, version
- endpoints[]: {board, connector, ref, pins: [{pos, name, net, wire_awg}]}
- runs[]: {from_pin, to_pin, net, wire_awg, length_budget_mm}
- routing: {style (star|daisy-chain), notes} - physical waypoints added by ME
- em: noise notes / bus-speed limits that constrain the route (e.g. keep off
  ESC phase nodes)

First instance: harness/ee-flight_to_ee-tof_xshut_ring.json (8-carrier
VL53L9CX ring on the XSHUT address chain - user-locked 2026-09-05: no hub
board, no mux).
