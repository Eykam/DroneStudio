#!/usr/bin/env python3
"""Multi-source /watch streamer supervisor.

Launches one streamer process per experiment source in
/workspace/stream_sources.json, restarts on crash, reaps zombies
(the box PID 1 does not). Each source is an experiment/hypothesis that
can be streamed to the dashboard /watch picker.
"""
import json, os, subprocess, time

CFG_PATH = "/workspace/stream_sources.json"
ENV_FILE = "/workspace/.dashboard_env"

# the supervisor is launched from a bare shell: load dashboard url/token
for _line in open(ENV_FILE):
    if "=" in _line:
        k, v = _line.strip().split("=", 1)
        os.environ.setdefault(k, v)
STREAMER = "/workspace/DroneStudio/autoresearch/streamer.py"
PY = "/workspace/venv/bin/python"

def launch(src):
    env = dict(os.environ)
    env["STREAM_SOURCE"] = src["id"]
    env["STREAM_LABEL"] = src.get("label", src["id"])
    env["STREAM_POLICY_FLAT"] = src["policy"]
    env["STREAM_POLICY_OBS"] = src.get("obs", "v1")
    if src.get("arch"):
        env["STREAM_ARCH"] = src["arch"]
    if src.get("obs_v3"):
        env["AUTORESEARCH_OBS_V3"] = "1"
    elif src.get("obs_v2"):
        env["AUTORESEARCH_OBS_V2"] = "1"
    if src.get("scenarios"):
        env["STREAM_SCENARIOS"] = "1"
    if src.get("dynamics"):
        env["STREAM_DYNAMICS"] = src["dynamics"]
    if src.get("eval"):
        env["STREAM_EVAL"] = json.dumps(src["eval"])
    log = open(f"/workspace/streamer_{src['id']}.log", "ab", buffering=0)
    p = subprocess.Popen([PY, STREAMER], env=env, stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True, stdin=subprocess.DEVNULL)
    print(f"launched {src['id']} pid={p.pid}", flush=True)
    return p

def main():
    cfg = json.load(open(CFG_PATH))
    procs = {}
    while True:
        for src in cfg:
            p = procs.get(src["id"])
            if p is None or p.poll() is not None:
                if p is not None:
                    print(f"{src['id']} exited rc={p.returncode}, restarting", flush=True)
                procs[src["id"]] = launch(src)
        try:
            while True:
                pid, _ = os.waitpid(-1, os.WNOHANG)
                if pid <= 0:
                    break
        except ChildProcessError:
            pass
        time.sleep(5)

if __name__ == "__main__":
    main()
