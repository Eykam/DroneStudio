"""Training-follower for the /watch "t4-training" source.

Copies the newest T4-lineage training checkpoint to /workspace/t4_live.json;
the streamer hot-reloads that file on mtime change, so /watch follows the
training loop round-by-round. Convention: any training script that wants
live streaming writes flat policy checkpoints matching one of the GLOBS
below (dag rounds do this already). JSON-validated before publish so a
half-written checkpoint is never served.
"""
import glob, json, os, time

GLOBS = ["/workspace/t4_dag*_r*.json", "/workspace/t4live_*.json"]
OUT = "/workspace/t4_live.json"
POLL = 10

def newest():
    best, bm = None, -1.0
    for g in GLOBS:
        for p in glob.glob(g):
            try:
                m = os.path.getmtime(p)
            except OSError:
                continue
            if m > bm:
                best, bm = p, m
    return best, bm

cur = None
print("t4_watch: following", GLOBS, "->", OUT, flush=True)
while True:
    p, m = newest()
    if p and m != cur:
        try:
            json.load(open(p))  # skip partial writes
            tmp = OUT + ".tmp"
            with open(p, "rb") as fi, open(tmp, "wb") as fo:
                fo.write(fi.read())
            os.replace(tmp, OUT)
            cur = m
            print("t4_watch: published", p, flush=True)
        except Exception as e:
            print("t4_watch: skip", p, e, flush=True)
    time.sleep(POLL)
