#!/usr/bin/env python3
"""Posts live auto-researcher state to the dashboard API.

Runs as a loop on the research box. Reads the archive + reports from disk
(decoupled from any runner - works for runs already in flight) and POSTs
snapshots to the Hono ingest endpoint. Server-side dedup by record id, so
re-sends are harmless. All config via env:

  DASHBOARD_URL   - e.g. https://dronestudio-dashboard.up.railway.app
  INGEST_TOKEN    - shared bearer token (also a Railway env var on the
                    dashboard service)
  POST_INTERVAL_S - default 30
"""
import os, json, time, glob, urllib.request

DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "").rstrip("/")
INGEST_TOKEN = os.environ.get("INGEST_TOKEN", "")
INTERVAL = float(os.environ.get("POST_INTERVAL_S", "30"))
HERE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = os.path.join(HERE, "archive.jsonl")

def _current_generation():
    """Latest 'gen N' line in the newest genrun log (last 8KB tail scan)."""
    import re
    logs = sorted(glob.glob("/workspace/genrun*.log"),
                  key=os.path.getmtime, reverse=True)
    if not logs:
        return None
    try:
        with open(logs[0], "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 8192))
            tail = f.read().decode(errors="replace")
        gens = re.findall(r"gen (\d+)", tail)
        return int(gens[-1]) if gens else None
    except Exception:
        return None

def run_status():
    """Detect a live runner via /proc cmdline scan (no ps dependency)."""
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().decode(errors="replace").replace("\0", " ")
            if "run_generations.py" in cmd or "compare_trainers.py" in cmd:
                return {"status": "running", "pid": int(pid),
                        "generations": _current_generation(),
                        "detail": cmd.strip()[-160:]}
        except Exception:
            continue
    return {"status": "idle"}

def collect():
    records = []
    if os.path.exists(ARCHIVE):
        with open(ARCHIVE) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
    reports = []
    for p in sorted(glob.glob(os.path.join(HERE, "generations_report*.json"))):
        try:
            with open(p) as f:
                r = json.load(f)
            r["file"] = os.path.basename(p)
            reports.append(r)
        except Exception:
            pass
    return {"run": run_status(), "records": records, "reports": reports}

def post(payload):
    req = urllib.request.Request(
        DASHBOARD_URL + "/api/ingest",
        data=json.dumps(payload).encode(),
        headers={"Authorization": "Bearer " + INGEST_TOKEN,
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())

def main():
    if not DASHBOARD_URL or not INGEST_TOKEN:
        raise SystemExit("DASHBOARD_URL and INGEST_TOKEN required")
    print(f"posting to {DASHBOARD_URL} every {INTERVAL}s", flush=True)
    while True:
        try:
            resp = post(collect())
            print(f"posted {resp.get('records')} records", flush=True)
        except Exception as e:
            print(f"post failed: {e}", flush=True)
        time.sleep(INTERVAL)

if __name__ == "__main__":
    main()
