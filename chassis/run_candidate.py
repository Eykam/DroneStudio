"""One evaluator-first candidate run: build -> evaluate -> export -> snapshot."""
import sys, os, json, dataclasses
import trimesh
from chassis import ChassisParams, build_chassis
import evaluate as ev
import progress
import export_manifest as em
import snapshot as sn

def run(variant_id, parent_id, generation, params: ChassisParams, out_base):
    progress.set_stage("building", f"{variant_id}: generating geometry", design_id=f"cad-chassis-{variant_id}")
    part = build_chassis(params)
    import build123d as b
    b.export_stl(part, out_base + ".stl")
    progress.set_stage("rendering", f"{variant_id}: exporting STEP/GLB/manifest")
    em.export(params, out_base)  # STEP/GLB/manifest first: FEA consumes the STEP
    m = trimesh.load(out_base + ".stl", force='mesh')
    checks = ev.check_sanity(m) + [ev.check_overhang(m), ev.check_wall_thickness(m)]
    ok_c, adj, need = params.check_prop_clearance()
    checks.append(("prop_clearance", ok_c, f"{adj:.0f} mm vs {need:.0f} mm needed", 0.0 if ok_c else 0.5))
    import containment
    checks.append(containment.check_containment(part))
    checks.append(ev.check_camera_fov(m))
    checks.append(ev.check_imu_lever_arm(m))
    checks.append(ev.check_dfam(m))
    props, _ = ev.mass_properties(m, params.motor_positions(), params.arm_length_mm)
    hover_ok = props["hover_thrust_frac"] < 0.65
    checks.append(("hover_margin", hover_ok, f"hover at {props['hover_thrust_frac']*100:.0f}% max thrust", 0.0 if hover_ok else 0.4))
    score = ev.score(checks)
    progress.set_stage("evaluating", f"{variant_id}: geometry checks")
    fea_result = None
    if os.environ.get("RUN_FEA") == "1":
        progress.set_stage("FEA", f"{variant_id}: gmsh + CalculiX hover-max/crash")
        import fea
        fea_result = fea.evaluate_fea(out_base + ".step", params.motor_positions(), params.stack_spacing_mm,
                                      out_base + "_fea")
        checks.append(("fea", fea_result["passed"], json.dumps(fea_result), 0.0 if fea_result["passed"] else 0.5))
        score = ev.score(checks)
    rec = sn.make_record(variant_id, parent_id, generation, dataclasses.asdict(params), checks, score, props, fea=fea_result)
    d = sn.save_snapshot(rec, out_base)
    if os.environ.get("SKIP_PUBLISH") != "1":
        sn.publish(rec, d)
    print(f"variant={variant_id} score={score:.3f} snapshot={d}")
    for n, p, det, _ in checks:
        print(f"  [{'PASS' if p else 'FAIL'}] {n}: {det}")
    return rec

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="v1-baseline")
    ap.add_argument("--parent", default=None)
    ap.add_argument("--gen", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--source", default=None,
                    help="path to an alternative chassis.py (multi-candidate gens)")
    args = ap.parse_args()
    if args.source:
        import importlib.util
        spec = importlib.util.spec_from_file_location("chassis_alt", args.source)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ChassisParams, build_chassis = mod.ChassisParams, mod.build_chassis
    out = args.out or os.path.join(os.getcwd(), args.variant)
    rec = run(args.variant, args.parent, args.gen, ChassisParams(), out)
    print("RESULT_JSON " + json.dumps({"variant": args.variant, "score": rec["score"], "parent": args.parent}))
