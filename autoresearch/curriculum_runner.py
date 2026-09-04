"""Curriculum runner: climb goal_distance with warm-chained CEM.

Stage 1 warm-starts from the BC policy (bc_flat.json, cloned from the
scripted pilot). Each later stage warm-starts from the previous stage's
best policy. A stage advances when held-out success >= ADVANCE_AT; if a
stage stays below the bar after its attempts, the ladder stops and reports
(a stuck stage is the interesting signal).

Every stage result AND working status is POSTed to the dashboard
/api/curriculum/progress (bearer INGEST_TOKEN). The best policy so far is
atomically swapped into /workspace/bc_flat.json, which the /watch streamer
hot-reloads - so the live viewer always flies the best available policy.
"""
import json, os, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from scene_schema import SceneDistribution
from env_sim import make_sim_factory, SimBinaryEnv
from policy import cem_train_parallel, MLP

MANIFEST = os.environ.get("CURR_DYNAMICS", os.path.join(HERE, "fixtures", "chassis_v1.manifest.json"))
DASHBOARD = os.environ.get("DASHBOARD_URL", "").rstrip("/")
TOKEN = os.environ.get("INGEST_TOKEN", "")
POLICY_OUT = os.environ.get("CURR_POLICY_OUT", "/workspace/bc_flat.json")
STAGES = [float(x) for x in os.environ.get("CURR_STAGES", "2,5,10,15,25").split(",")]
ADVANCE_AT = float(os.environ.get("CURR_ADVANCE_AT", "0.5"))
EVAL_EPS = int(os.environ.get("CURR_EVAL_EPS", "24"))
ITERS = int(os.environ.get("CURR_ITERS", "40"))
POP = int(os.environ.get("CURR_POP", "64"))
MAX_STEPS = 400

def post(payload):
    if not (DASHBOARD and TOKEN):
        return
    try:
        req = urllib.request.Request(
            DASHBOARD + "/api/curriculum/progress",
            data=json.dumps(payload).encode(),
            headers={"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10).read()
    except Exception as e:
        print("dashboard post failed:", e, flush=True)

def stage_dist(gd):
    return SceneDistribution(obstacle_density=0.0, corridor_width=10.0,
        scene_extent=max(10.0, gd * 2), goal_distance=gd, light_direction_entropy=0.0,
        texture_variety=0.0, dynamics_noise=0.0)

def eval_policy(policy, gd, n=EVAL_EPS, seed0=90000):
    factory = make_sim_factory(stage_dist(gd), max_steps=MAX_STEPS, dynamics=MANIFEST)
    succ, rets, lens = [], [], []
    for i in range(n):
        env = factory(seed0 + i)
        obs = env.reset()
        total = 0.0
        for _ in range(MAX_STEPS):
            obs, r, done = env.step(policy.act(obs))
            total += r
            if done:
                break
        succ.append(bool(env.succeeded)); rets.append(total); lens.append(env.steps)
        env.close()
    return float(np.mean(succ)), float(np.mean(rets)), float(np.mean(lens))

def publish_policy(policy):
    tmp = POLICY_OUT + ".tmp"
    json.dump(policy.get_flat().tolist(), open(tmp, "w"))
    os.replace(tmp, POLICY_OUT)  # atomic: streamer hot-reloads on mtime

def main():
    policy = MLP(SimBinaryEnv.OBS_DIM, SimBinaryEnv.ACT_DIM)
    policy.set_flat(np.array(json.load(open(POLICY_OUT)), dtype=np.float64))
    report = {"stages": [], "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for gd in STAGES:
        post({"status": "working", "current_stage": gd,
              "note": f"warm-chained CEM, advance at {ADVANCE_AT:.0%}"})
        dist = stage_dist(gd)
        spec = ("sim", dist.to_json(), MAX_STEPS, MANIFEST)
        # attempt 0: the incoming policy itself - never publish worse than what we started with
        s0, r0, l0 = eval_policy(policy, gd)
        print(f"stage {gd}m attempt 0 (init): succ={s0:.3f} ret={r0:.2f} steps={l0:.0f}", flush=True)
        best = (s0, policy)
        for attempt in range(1, 4):
            t0 = time.time()
            cand, train_ret = cem_train_parallel(
                spec, SimBinaryEnv.OBS_DIM, SimBinaryEnv.ACT_DIM,
                iters=ITERS, pop=POP, episodes_per_eval=8,
                seed=int(gd * 100 + attempt), n_jobs=12,
                init_mean=policy.get_flat(), init_std=0.05, fitness="progress")
            s, r, l = eval_policy(cand, gd)
            print(f"stage {gd}m attempt {attempt}: succ={s:.3f} ret={r:.2f} steps={l:.0f} train={train_ret:.2f} wall={time.time()-t0:.0f}s", flush=True)
            post({"status": "working", "current_stage": gd,
                  "stage_result": {"goal_m": gd, "trainer": f"cem warm a{attempt}",
                                   "success_rate": s, "mean_return": r, "mean_steps": l,
                                   "wall_s": round(time.time() - t0, 1),
                                   "eval_episodes": EVAL_EPS, "budget": f"{ITERS}x{POP} warm"}})
            if best is None or s > best[0]:
                best = (s, cand)
            if s >= ADVANCE_AT:
                break
        s, cand = best
        report["stages"].append({"goal_m": gd, "success": s})
        if s > 0:
            policy = cand
            publish_policy(policy)  # /watch flies this now
            print(f"stage {gd}m -> published policy (succ {s:.3f})", flush=True)
        if s < ADVANCE_AT:
            print(f"STUCK at {gd}m (best {s:.3f} < {ADVANCE_AT}); stopping ladder", flush=True)
            post({"status": "idle", "note": f"stuck at {gd}m stage (best {s:.1%}); ladder stopped"})
            break
    json.dump(report, open("/workspace/curriculum_report.json", "w"), indent=2)
    done = len(report["stages"]) == len(STAGES)
    post({"status": "idle", "note": "ladder complete" if done else "ladder stopped early"})
    print("CURR_DONE", json.dumps(report), flush=True)

if __name__ == "__main__":
    main()
