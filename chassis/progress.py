"""Live loop-progress publisher for the /cad dashboard banner.
POSTs stage transitions + a 60s heartbeat to /api/cad/progress.
Never raises: progress is best-effort, the research loop must not break on it."""
import json, os, threading, urllib.request

_URL = os.environ.get("DASHBOARD_URL", "").rstrip("/")
_TOKEN = os.environ.get("INGEST_TOKEN", "")
_state = {"status": "idle", "design_id": None, "stage": "idle", "detail": ""}
_stop = threading.Event()
_hb = None

def _post():
    if not _URL or not _TOKEN:
        return
    try:
        req = urllib.request.Request(
            _URL + "/api/cad/progress",
            data=json.dumps(_state).encode(),
            headers={"Authorization": "Bearer " + _TOKEN,
                     "Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(req, timeout=10).read()
    except Exception:
        pass

def _beat():
    while not _stop.wait(60):
        if _state["status"] == "working":
            _post()

def start_heartbeat():
    global _hb
    if _hb is None:
        _hb = threading.Thread(target=_beat, daemon=True)
        _hb.start()

def set_stage(stage, detail="", design_id=None):
    if design_id:
        _state["design_id"] = design_id
    _state.update(status="working", stage=stage, detail=detail)
    _post()

def idle(detail=""):
    _state.update(status="idle", stage="idle", detail=detail)
    _stop.set()
    _post()
