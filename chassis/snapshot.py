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

def render_preview(stl_path, png_path, manifest_path=None):
    """Orthographic side/top/front vertex-projection preview of a variant ->
    PNG smoke-check image in the snapshot dir. Numeric gates cannot catch
    geometry that is obviously wrong to a human (the 2026-09-04 camera
    orientation bug passed every gate); eyeball these at adoption/milestone.
    Camera markers show the measured lens apex AND a 25mm arrow along the lens
    axis - the arrow must point toward the nose (+X). Component poses come from
    the snapshot manifest when available (retro-renders), else live components."""
    import json as _json
    import numpy as np, trimesh
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = trimesh.load(stl_path)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.dump()))
    V = np.asarray(m.vertices, dtype=float)
    scale = 1000.0 if np.ptp(V, axis=0).max() < 1.0 else 1.0  # glb is meters, step/stl mm
    V = V * scale
    views = [(0, 2, "side (x=nose, z=up)"), (0, 1, "top (x=nose, y=lateral)"), (1, 2, "front (y=lateral, z=up)")]
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    step = max(1, len(V) // 60000)
    for ax, (i, j, title) in zip(axes, views):
        ax.scatter(V[::step, i], V[::step, j], s=0.3, c="k")
        ax.set_aspect("equal"); ax.set_title(title); ax.grid(True, alpha=0.3)
    overlay_note = ""
    try:
        cams, imu_p, motors = None, None, []
        if manifest_path and os.path.exists(manifest_path):
            man = _json.load(open(manifest_path))
            cams = [(c["lens_origin_m"], c["lens_axis"]) for c in man.get("cameras", [])]
            imu_p = (man.get("imu") or {}).get("position_m")
            motors = [mt["position_m"] for mt in man.get("motors", [])]
        else:
            from components import camera_lens_poses, imu_pose
            cams = [(p["origin_m"], p["axis"]) for p in camera_lens_poses().values()]
            imu_p = imu_pose()["position_m"]
        for o_m, a in cams or []:
            o = np.array(o_m, dtype=float) * 1000.0
            a = np.array(a, dtype=float)
            bad = a[0] < 0.5
            for ax, (i, j, _t) in zip(axes, views):
                ax.scatter([o[i]], [o[j]], c=("r" if not bad else "m"), s=60, marker="x", zorder=5)
                ax.annotate("", xy=(o[i] + a[i] * 25, o[j] + a[j] * 25), xytext=(o[i], o[j]),
                            arrowprops=dict(arrowstyle="->", color=("r" if not bad else "m"), lw=2.5), zorder=6)
            if bad:
                overlay_note += " CAMERA %s AXIS NOT +X!" % str(o_m)
        if imu_p:
            io = np.array(imu_p, dtype=float) * 1000.0
            for ax, (i, j, _t) in zip(axes, views):
                ax.scatter([io[i]], [io[j]], c="g", s=60, marker="D", zorder=5)
        for mo in motors:
            mv = np.array(mo, dtype=float) * 1000.0
            for ax, (i, j, _t) in zip(axes, views):
                ax.scatter([mv[i]], [mv[j]], c="b", s=45, marker="^", zorder=5)
    except Exception as e:
        overlay_note = " overlay failed: %s" % e
    fig.text(0.01, 0.02, "red X+arrow: camera lens apex + axis (MUST point to nose/+X); green D: IMU; blue ^: motors." + overlay_note,
             color=("r" if "NOT +X" in overlay_note else "k"))
    if "NOT +X" in overlay_note:
        print("SMOKE CHECK FAIL:" + overlay_note, flush=True)
    fig.tight_layout(); fig.savefig(png_path, dpi=110); plt.close(fig)


def save_snapshot(record, out_base):
    d = os.path.join(SNAP_DIR, record["variant"]["id"])
    os.makedirs(d, exist_ok=True)
    import shutil
    for src, dst in ((out_base + ".glb", "chassis.glb"), (out_base + ".step", "chassis.step"),
                     (out_base + ".sim.glb", "chassis.sim.glb"),
                     (out_base + ".manifest.json", "manifest.json")):
        if os.path.exists(src):
            shutil.copy(src, os.path.join(d, dst))
    try:
        render_preview(out_base + ".stl", os.path.join(d, "preview.png"))
    except Exception as e:
        print(f"preview render failed (non-fatal): {e}", flush=True)
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
