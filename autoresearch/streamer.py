#!/usr/bin/env python3
"""Live watch-channel streamer.

Trains a CEM policy on the best-known scene distribution, then flies
real-time episodes on the headless Zig sim (manifest dynamics) and POSTs
telemetry to the dashboard /api/stream/ingest for the /watch page.

Env: DASHBOARD_URL, INGEST_TOKEN (same contract as dashboard_poster.py),
STREAM_DYNAMICS (manifest path), STREAM_REPORT (generations report to pick
the best dist from), STREAM_FPS, STREAM_RETRAIN_EPS.
"""
import os, json, sys, json, time, urllib.request, traceback
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from scene_schema import SceneDistribution
from env_sim import SimBinaryEnv, make_sim_factory
from policy import cem_train, MLP

if os.environ.get("STREAM_SCENARIOS"):
    from scenario_sampler import sample_spec

DASH = os.environ.get("DASHBOARD_URL", "").rstrip("/")
TOKEN = os.environ.get("INGEST_TOKEN", "")
DYNAMICS = os.environ.get("STREAM_DYNAMICS", os.path.join(HERE, "fixtures", "chassis_v1.manifest.json"))
REPORT = os.environ.get("STREAM_REPORT", os.path.join(HERE, "generations_report2.json"))
FPS = float(os.environ.get("STREAM_FPS", "20"))
RETRAIN_EPS = int(os.environ.get("STREAM_RETRAIN_EPS", "8"))
POLICY_FLAT = os.environ.get("STREAM_POLICY_FLAT", "")  # flat-vector json: fly this policy, hot-reload on mtime change
ARCH = os.environ.get("STREAM_ARCH", "")  # "t4" = t4_common HID-64 trunk + wp pathway
MAX_STEPS = 200
# multi-source /watch: this streamer's source id + picker metadata
SOURCE = os.environ.get("STREAM_SOURCE", "default")
LABEL = os.environ.get("STREAM_LABEL", "")
SCENARIOS = bool(os.environ.get("STREAM_SCENARIOS"))  # sample scenario specs per episode (v2 track)
EVAL_JSON = os.environ.get("STREAM_EVAL", "")  # inline JSON eval scores for the picker chip


def post(payload):
    if not DASH or not TOKEN:
        return
    payload = {"source": SOURCE, **payload}
    req = urllib.request.Request(
        DASH + "/api/stream/ingest",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + TOKEN})
    try:
        urllib.request.urlopen(req, timeout=8).read()
    except Exception as e:
        print("ingest:", e, flush=True)


def best_dist():
    rep = json.load(open(REPORT))
    b = rep["best"]
    names = set(SceneDistribution.__dataclass_fields__)
    return b["id"], SceneDistribution(**{k: v for k, v in b["params"].items() if k in names})


class StreamingEnv(SimBinaryEnv):
    """SimBinaryEnv that keeps the per-step info dict for telemetry."""

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        resp = self._call({"cmd": "step", "action": [float(x) for x in a]})
        info = resp.get("info", {})
        self.last_info = info
        self.steps = int(info.get("steps", self.steps + 1))
        self.collided = bool(info.get("collided", False))
        self._succeeded_sim = bool(info.get("succeeded", False))
        return (np.array(resp["obs"], dtype=np.float64),
                float(resp["reward"]), bool(resp["done"]))


def status(state, dist_id):
    post({"type": "status", "status": state, "mode": "watch",
          "dist_id": dist_id, "dynamics": os.path.basename(DYNAMICS),
          "policy": os.path.basename(POLICY_FLAT) if POLICY_FLAT else "archive-best",
          "policy_obs": os.environ.get("STREAM_POLICY_OBS", "")})


def register():
    """Upsert this source into the /watch picker registry."""
    if not DASH or not TOKEN:
        return
    body = {"id": SOURCE,
            "label": LABEL or SOURCE,
            "policy": os.path.basename(POLICY_FLAT) if POLICY_FLAT else "archive-best",
            "policy_obs": os.environ.get("STREAM_POLICY_OBS", "v1"),
            "dynamics": os.path.basename(DYNAMICS),
            "status": "streaming"}
    if EVAL_JSON:
        try:
            body["eval"] = json.loads(EVAL_JSON)
        except Exception:
            pass
    req = urllib.request.Request(
        DASH + "/api/stream/sources",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + TOKEN})
    try:
        urllib.request.urlopen(req, timeout=8).read()
    except Exception as e:
        print("register:", e, flush=True)


def main():
    ep = 0
    dist_id, dist = best_dist()
    status("training", dist_id)
    factory = make_sim_factory(dist, max_steps=MAX_STEPS, dynamics=DYNAMICS)
    policy_mtime = [None]
    obs_dim = [SimBinaryEnv.OBS_DIM]

    def load_policy(force_seed):
        if POLICY_FLAT and os.path.exists(POLICY_FLAT):
            mt = os.path.getmtime(POLICY_FLAT)
            if mt != policy_mtime[0]:
                flat = np.array(json.load(open(POLICY_FLAT)), dtype=np.float64)
                if ARCH == "t4":
                    import t4_common as P
                    from ppo import MLP as PPOMlp
                    actor = PPOMlp(np.random.default_rng(0), P.OBS_DIM, P.HID, P.ACT_DIM)
                    actor.load(P.bc_to_actor_params(flat))
                    wp1, wp2 = P.unpack_wp(flat)
                    class _T4Pol:
                        def act(self, obs):
                            mu, _ = actor.forward(obs[None, :])
                            wpmu, _ = P.wp_forward(obs[None, :], wp1, wp2)
                            return np.tanh(mu[0] + wpmu[0])
                    pol = _T4Pol()
                else:
                    pol = MLP(obs_dim[0], SimBinaryEnv.ACT_DIM)
                    pol.set_flat(flat)
                policy_mtime[0] = mt
                print(f"loaded policy from {POLICY_FLAT}", flush=True)
                return pol
            return None
        pol, train_ret = cem_train(factory, SimBinaryEnv.OBS_DIM, SimBinaryEnv.ACT_DIM,
                                   iters=2, pop=8, episodes_per_eval=3, seed=42 + force_seed)
        print(f"trained on {dist_id}: ret={train_ret:.2f}", flush=True)
        return pol
    policy = None  # built lazily: obs dim depends on obs v1/v2 until first reset
    register()
    while True:
        ep += 1
        ep_id = f"w{ep:05d}"
        seed = 50_000 + ep
        spec = sample_spec(seed) if SCENARIOS else None
        env = StreamingEnv(dist, seed=seed, max_steps=MAX_STEPS, dynamics=DYNAMICS,
                           scenario_spec=spec)
        try:
            obs = env.reset()
            if policy is None:
                obs_dim[0] = len(obs)
                policy = load_policy(0)
            status("streaming", dist_id)
            post({"type": "scene", "episode_id": ep_id, "dist_id": dist_id,
                  "seed": seed,
                  **({"scenario": spec["scenario"],
                      "success_radius": round(spec["success_radius"], 2)} if spec else {}),
                  "spawn": [float(x) for x in env.spawn],
                  "goal": [float(x) for x in env.goal],
                  "obstacles": [[float(c[0]), float(c[1]), float(c[2]), float(r)]
                                for c, r in zip(env.obs_centers, env.obs_radii)],
                  "extent": float(dist.scene_extent)})
            dt = 1.0 / FPS
            done = False
            while not done:
                t0 = time.time()
                obs, r, done = env.step(policy.act(obs))
                info = getattr(env, "last_info", {})
                post({"type": "frame", "episode_id": ep_id, "step": env.steps,
                      "pos": info.get("pos"), "quat": info.get("quat"),
                      "vel": info.get("vel"),
                      "reward": float(r), "done": bool(done)})
                time.sleep(max(0.0, dt - (time.time() - t0)))
            post({"type": "episode_end", "episode_id": ep_id,
                  "succeeded": bool(env.succeeded), "collided": bool(env.collided),
                  "steps": env.steps})
            print(f"{ep_id}: succ={env.succeeded} coll={env.collided} steps={env.steps}", flush=True)
        except Exception:
            traceback.print_exc()
            time.sleep(2)
        finally:
            env.close()
        if ep % RETRAIN_EPS == 0:
            try:
                if POLICY_FLAT:
                    newp = load_policy(ep)
                    if newp is not None:
                        policy = newp
                        print("policy hot-reloaded", flush=True)
                else:
                    dist_id, dist = best_dist()
                    status("retraining", dist_id)
                    factory = make_sim_factory(dist, max_steps=MAX_STEPS, dynamics=DYNAMICS)
                    policy = load_policy(ep) or policy
            except Exception:
                traceback.print_exc()


if __name__ == "__main__":
    main()
