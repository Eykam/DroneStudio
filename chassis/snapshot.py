"""Stable on-disk snapshot + dashboard poster for chassis candidates.

Layout (stable schema, backfill = one command):
  snapshots/<variant_id>/chassis.glb      - geometry, meters, binary glTF
  snapshots/<variant_id>/chassis.step     - mm, for CAD/print interchange
  snapshots/<variant_id>/manifest.json    - dronestudio.chassis/1 physics manifest
  snapshots/<variant_id>/metrics.json     - the record below

The record is also POSTed to the dashboard ingest endpoint when
DASHBOARD_URL + INGEST_TOKEN are set (server dedups by record id, resends harmless).
Record kind discriminator: "cad.chassis.snapshot".
"""
import os, json, subprocess, urllib.request, datetime

SNAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")

def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None

def make_record(variant_id, parent_id, generation, params, eval_checks, score, mass_props):
    return {
        "id": f"cad-chassis-{variant_id}",
        "kind": "cad.chassis.snapshot",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "variant": {"id": variant_id, "parent_id": parent_id,
                    "generation": generation, "params": params, "git_commit": git_commit()},
        "score": score,
        "evaluator": [{"name": n, "passed": bool(p), "detail": str(d)} for n, p, d, _ in eval_checks],
        "mass": mass_props,
        "fea": None,  # filled once gmsh+CalculiX stage runs on the box
        "artifacts": {"glb": f"snapshots/{variant_id}/chassis.glb",
                      "step": f"snapshots/{variant_id}/chassis.step",
                      "manifest": f"snapshots/{variant_id}/manifest.json",
                      "metrics": f"snapshots/{variant_id}/metrics.json"},
    }

def save_snapshot(record, out_base):
    d = os.path.join(SNAP_DIR, record["variant"]["id"])
    os.makedirs(d, exist_ok=True)
    import shutil
    for src, dst in ((out_base + ".glb", "chassis.glb"), (out_base + ".step", "chassis.step"),
                     (out_base + ".manifest.json", "manifest.json")):
        if os.path.exists(src):
            shutil.copy(src, os.path.join(d, dst))
    with open(os.path.join(d, "metrics.json"), "w") as f:
        json.dump(record, f, indent=2)
    return d

def post_records(records):
    url = os.environ.get("DASHBOARD_URL", "").rstrip("/")
    tok = os.environ.get("INGEST_TOKEN", "")
    if not url or not tok:
        print("DASHBOARD_URL/INGEST_TOKEN unset - snapshots kept on disk only", flush=True)
        return None
    req = urllib.request.Request(url + "/api/ingest", data=json.dumps({"records": records}).encode(),
        headers={"Authorization": "Bearer " + tok, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())

def backfill():
    records = []
    for p in sorted(os.listdir(SNAP_DIR)) if os.path.isdir(SNAP_DIR) else []:
        mp = os.path.join(SNAP_DIR, p, "metrics.json")
        if os.path.exists(mp):
            records.append(json.load(open(mp)))
    if records:
        print(post_records(records))

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "backfill":
        backfill()
