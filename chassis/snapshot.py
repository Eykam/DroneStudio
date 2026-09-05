"""Stable on-disk snapshot + dashboard poster for chassis candidates.

Layout (stable schema, backfill = one command):
  snapshots/<variant_id>/chassis.glb      - geometry, meters, binary glTF
  snapshots/<variant_id>/chassis.step     - mm, for CAD/print interchange
  snapshots/<variant_id>/manifest.json    - dronestudio.chassis/1.1 physics manifest
  snapshots/<variant_id>/metrics.json     - the record below

Dashboard contract: CAD_INGEST_API.md (repo root, auto-researcher branch).
Phase 1: record via POST /api/ingest (dedup by id). Phase 2: GLB binary via
POST /api/cad/designs (multipart). Auth: bearer $INGEST_TOKEN (Railway var).
"""
import os, json, subprocess, urllib.request, datetime, uuid

SNAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")

def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None

def make_record(variant_id, parent_id, generation, params, eval_checks, score, mass_props, fea=None):
    checks = [{"name": n, "passed": bool(p), "detail": str(d)} for n, p, d, _ in eval_checks]
    failed = [c["name"] for c in checks if not c["passed"]]
    metrics = {
        "mass_g": round(mass_props["frame_mass_g"], 1),
        "all_up_mass_kg": mass_props["all_up_mass_kg"],
        "inertia": {"ixx": mass_props["inertia_kgm2"][0][0],
                    "iyy": mass_props["inertia_kgm2"][1][1],
                    "izz": mass_props["inertia_kgm2"][2][2]},
        "printability": {c["name"]: c["detail"] for c in checks if c["name"] in ("overhang", "wall_thickness", "prop_clearance")},
        "fea": fea,
        "score": score,
        "hover_thrust_frac": mass_props["hover_thrust_frac"],
    }
    return {
        "id": f"cad-chassis-{variant_id}",
        "kind": "cad.chassis.snapshot",
        "parent_id": f"cad-chassis-{parent_id}" if parent_id else None,
        "name": f"Chassis {variant_id}",
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "glb_path": f"snapshots/{variant_id}/chassis.glb",
        "metrics": metrics,
        "notes": ("failing: " + ", ".join(failed)) if failed else "all checks pass",
        # richer detail beyond the documented shape (dashboard ignores unknown keys):
        "variant": {"id": variant_id, "parent_id": parent_id,
                    "generation": generation, "params": params, "git_commit": git_commit()},
        "score": score,
        "evaluator": checks,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

def save_snapshot(record, out_base):
    d = os.path.join(SNAP_DIR, record["variant"]["id"])
    os.makedirs(d, exist_ok=True)
    import shutil
    for src, dst in ((out_base + ".glb", "chassis.glb"), (out_base + ".step", "chassis.step"),
                     (out_base + ".sim.glb", "chassis.sim.glb"),
                     (out_base + ".manifest.json", "manifest.json")):
        if os.path.exists(src):
            shutil.copy(src, os.path.join(d, dst))
    with open(os.path.join(d, "metrics.json"), "w") as f:
        json.dump(record, f, indent=2)
    return d

def _env():
    return os.environ.get("DASHBOARD_URL", "").rstrip("/"), os.environ.get("INGEST_TOKEN", "")

def post_records(records):
    url, tok = _env()
    if not url or not tok:
        print("DASHBOARD_URL/INGEST_TOKEN unset - snapshots kept on disk only", flush=True)
        return None
    req = urllib.request.Request(url + "/api/ingest", data=json.dumps({"records": records}).encode(),
        headers={"Authorization": "Bearer " + tok, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())

def upload_glb(record, glb_path):
    """Phase 2: multipart GLB upload. Returns response or None."""
    url, tok = _env()
    if not url or not tok or not os.path.exists(glb_path):
        return None
    meta = {k: record[k] for k in ("id", "parent_id", "name", "metrics", "notes") if k in record}
    boundary = uuid.uuid4().hex
    glb = open(glb_path, "rb").read()
    body = b""
    body += ("--%s\r\nContent-Disposition: form-data; name=\"meta\"\r\n\r\n%s\r\n" % (boundary, json.dumps(meta))).encode()
    body += ("--%s\r\nContent-Disposition: form-data; name=\"file\"; filename=\"%s.glb\"\r\nContent-Type: model/gltf-binary\r\n\r\n" % (boundary, record["id"])).encode()
    body += glb + ("\r\n--%s--\r\n" % boundary).encode()
    req = urllib.request.Request(url + "/api/cad/designs", data=body,
        headers={"Authorization": "Bearer " + tok, "Content-Type": "multipart/form-data; boundary=" + boundary})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"GLB upload failed (kept on disk): {e}", flush=True)
        return None

def publish(record, snap_dir):
    resp = post_records([record])
    up = upload_glb(record, os.path.join(snap_dir, "chassis.glb"))
    return resp, up

def backfill():
    records = []
    for p in sorted(os.listdir(SNAP_DIR)) if os.path.isdir(SNAP_DIR) else []:
        mp = os.path.join(SNAP_DIR, p, "metrics.json")
        if os.path.exists(mp):
            records.append((json.load(open(mp)), os.path.join(SNAP_DIR, p)))
    if records:
        post_records([r for r, _ in records])
        for r, d in records:
            upload_glb(r, os.path.join(d, "chassis.glb"))

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "backfill":
        backfill()
