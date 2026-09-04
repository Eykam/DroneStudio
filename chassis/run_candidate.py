"""One evaluator-first candidate run: build -> evaluate -> export -> snapshot."""
import sys, os, json, dataclasses
import trimesh
from chassis import ChassisParams, build_chassis
import evaluate as ev
import export_manifest as em
import snapshot as sn

def run(variant_id, parent_id, generation, params: ChassisParams, out_base):
    part = build_chassis(params)
    import build123d as b
    b.export_stl(part, out_base + ".stl")
    m = trimesh.load(out_base + ".stl", force='mesh')
    checks = ev.check_sanity(m) + [ev.check_overhang(m), ev.check_wall_thickness(m)]
    ok_c, adj, need = params.check_prop_clearance()
    checks.append(("prop_clearance", ok_c, f"{adj:.0f} mm vs {need:.0f} mm needed", 0.0 if ok_c else 0.5))
    props, _ = ev.mass_properties(m, params.motor_positions(), params.arm_length_mm)
    hover_ok = props["hover_thrust_frac"] < 0.65
    checks.append(("hover_margin", hover_ok, f"hover at {props['hover_thrust_frac']*100:.0f}% max thrust", 0.0 if hover_ok else 0.4))
    score = ev.score(checks)
    fea_result = None
    if os.environ.get("RUN_FEA") == "1":
        import fea
        fea_result = fea.evaluate_fea(out_base + ".step", params.motor_positions(), params.stack_spacing_mm,
                                      out_base + "_fea")
        checks.append(("fea", fea_result["passed"], json.dumps(fea_result), 0.0 if fea_result["passed"] else 0.5))
        score = ev.score(checks)
    em.export(params, out_base)
    rec = sn.make_record(variant_id, parent_id, generation, dataclasses.asdict(params), checks, score, props)
    rec["fea"] = fea_result
    d = sn.save_snapshot(rec, out_base)
    print(f"variant={variant_id} score={score:.3f} snapshot={d}")
    for n, p, det, _ in checks:
        print(f"  [{'PASS' if p else 'FAIL'}] {n}: {det}")
    return rec

if __name__ == "__main__":
    rec = run("v1-baseline", None, 0, ChassisParams(), "/home/sandbox/cad-researcher/chassis_v1")
    sn.post_records([rec])
