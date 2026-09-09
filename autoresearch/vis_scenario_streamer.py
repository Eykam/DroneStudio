#!/usr/bin/env python3
"""vis_scenario_streamer.py - the learned model watches the policy fly, live.

Flies real episodes on the headless sim (t4_live policy + scenario sampler,
same driving as streamer.py), renders the camera at 128x96 every 4th policy
step (~5Hz), runs the learned depth+seg model on the RGB frame, and posts
GT-vs-prediction to /api/vision/ingest (type=scenario) for the /vision
Scenarios tab. Model checkpoint hot-reloads on mtime change, so retraining
shows up live without a restart.
"""
import json, os, sys, time, traceback, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from scene_schema import SceneDistribution
from env_sim import SimBinaryEnv
from scenario_sampler import sample_spec
from vis_visual import sample_visual

DASH = os.environ.get("DASHBOARD_URL", "").rstrip("/")
TOKEN = os.environ.get("INGEST_TOKEN", "")
DYNAMICS = os.environ.get("STREAM_DYNAMICS", os.path.join(HERE, "fixtures", "chassis_v1.manifest.json"))
REPORT = os.environ.get("STREAM_REPORT", os.path.join(HERE, "generations_report2.json"))
FPS = float(os.environ.get("STREAM_FPS", "20"))
POLICY_FLAT = os.environ.get("STREAM_POLICY_FLAT", "/workspace/t4_live.json")
CKPT = os.environ.get("VIS_CKPT", "/workspace/vision_model/visnet_v1_best.pt")
MAX_STEPS = 200
VW, VH = 128, 96
POLICY_NAME = os.path.basename(POLICY_FLAT)


def post(payload):
    if not DASH or not TOKEN:
        return
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        DASH + "/api/vision/ingest",
        data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + TOKEN})
    t0 = time.time()
    try:
        urllib.request.urlopen(req, timeout=25).read()
        post.ok_n += 1
        if post.ok_n <= 3 or post.ok_n % 50 == 0:
            print(f"post ok #{post.ok_n} ({len(body)}B {time.time()-t0:.1f}s)", flush=True)
    except Exception as e:
        print(f"ingest: {e} after {time.time()-t0:.1f}s ({len(body)}B)", flush=True)
post.ok_n = 0


def best_dist():
    rep = json.load(open(REPORT))
    b = rep["best"]
    names = set(SceneDistribution.__dataclass_fields__)
    return b["id"], SceneDistribution(**{k: v for k, v in b["params"].items() if k in names})


class StreamingEnv(SimBinaryEnv):
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


def load_t4_policy():
    import t4_common as P
    from ppo import MLP as PPOMlp
    flat = np.array(json.load(open(POLICY_FLAT)), dtype=np.float64)
    actor = PPOMlp(np.random.default_rng(0), P.OBS_DIM, P.HID, P.ACT_DIM)
    actor.load(P.bc_to_actor_params(flat))
    wp1, wp2 = P.unpack_wp(flat)

    class _T4Pol:
        def act(self, obs):
            mu, _ = actor.forward(obs[None, :])
            wpmu, _ = P.wp_forward(obs[None, :], wp1, wp2)
            return np.tanh(mu[0] + wpmu[0])
    return _T4Pol()


class Model:
    def __init__(self):
        import torch
        from vis_train import VisNet
        self.torch = torch
        torch.set_num_threads(4)
        self.net = VisNet()
        self.mtime = 0.0
        self.reload()

    def reload(self):
        st = self.torch.load(CKPT, map_location="cpu")
        self.net.load_state_dict(st)
        self.net.eval()
        self.mtime = os.path.getmtime(CKPT)
        print(f"model loaded (mtime {self.mtime:.0f})", flush=True)

    def maybe_reload(self):
        try:
            if os.path.getmtime(CKPT) > self.mtime:
                self.reload()
        except OSError:
            pass

    def infer(self, rgb):
        r = self.torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
        with self.torch.no_grad():
            pd, ps = self.net(r)
        d = np.clip(pd[0].exp().numpy() * 1000.0, 0, 65535).astype(np.uint32).ravel()
        s = ps[0].argmax(0).numpy().astype(np.uint8).ravel()
        return d, s


def main():
    model = Model()
    policy = load_t4_policy()
    dist_id, dist = best_dist()
    ep = 0
    print(f"scenario streamer up: policy={POLICY_NAME} dist={dist_id}", flush=True)
    while True:
        ep += 1
        ep_id = f"v{ep:05d}"
        seed = 50_000 + ep
        spec = sample_spec(seed)
        vis = sample_visual(seed)  # DR visuals per episode scene (rung-2)
        env = StreamingEnv(dist, seed=seed, max_steps=MAX_STEPS, dynamics=DYNAMICS,
                           scenario_spec=spec)
        try:
            obs = env.reset()
            done = False
            dt = 1.0 / FPS
            while not done:
                t0 = time.time()
                obs, _, done = env.step(policy.act(obs))
                info = getattr(env, "last_info", {})
                if env.steps % 4 == 0:
                    try:
                        vf = env._call({"cmd": "render", "width": VW, "height": VH, "visual": vis})
                        rgb = np.array(vf["rgb"], dtype=np.uint32)
                        rgb3 = np.stack([(rgb >> 16) & 255, (rgb >> 8) & 255, rgb & 255],
                                        axis=-1).astype(np.uint8).reshape(VH, VW, 3)
                        pd, psg = model.infer(rgb3)
                        post({"type": "scenario",
                              "episode_id": ep_id,
                              "model": os.path.basename(CKPT),
                              "visual": vis,
                              "scenario": spec["scenario"],
                              "step": env.steps,
                              "pos": info.get("pos"),
                              "policy": POLICY_NAME,
                              "dist_id": dist_id,
                              "w": vf["w"], "h": vf["h"],
                              "frames": [{
                                  "rgb": [int(x) for x in rgb],
                                  "depth": vf["depth"],
                                  "seg": vf["seg"],
                                  "pred_depth": [int(x) for x in pd],
                                  "pred_seg": [int(x) for x in psg],
                              }]})
                    except Exception as ve:
                        print("vision:", ve, flush=True)
                time.sleep(max(0.0, dt - (time.time() - t0)))
            print(f"{ep_id}: {spec['scenario']} succ={env.succeeded} "
                  f"coll={env.collided} steps={env.steps}", flush=True)
        except Exception:
            traceback.print_exc()
            time.sleep(2)
        finally:
            env.close()
        model.maybe_reload()


if __name__ == "__main__":
    main()
