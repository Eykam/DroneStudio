"""T4 stage 2b: eval-only for t4_bc.json (training completed; eval crashed on
a missing env_sim import in t4_common pool workers)."""
import sys, json, os
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
import t4_common as P
flat = json.load(open("/workspace/t4_bc.json"))
res = P.eval_all(flat, "t4-bc")
print("T4BC_EVAL " + json.dumps({f"{sc}_t{t}": round(v, 3) for (sc, t), v in res.items()}), flush=True)
P.post_status({"training": {"status": "idle", "name": "t4_bc",
    "note": "T4 HID-64 bootstrap stage 2 (BC) done: " +
            " ".join(f"{sc}t{t}={v:.2f}" for (sc, t), v in res.items())}})
print("T4BC_EVAL_DONE", flush=True)
