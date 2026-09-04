"""T3 scripted teacher: the v1 cascaded pilot steered at the CURRENT waypoint.

obs v3: [0:3] = final-goal rel (yaw frame, /extent); [19:22] = (cur_wp - goal)
delta same frame/scale. Sum = current-target rel - exactly what the reward
progress tracks. Measures teacher quality per curriculum phase (A-D knobs
from ppo_v39) to find where a T3 demo set can come from.
"""
import sys, json, os
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
os.environ["AUTORESEARCH_OBS_V3"] = "1"
import numpy as np
from scene_schema import SceneDistribution
from scenario_sampler import tier_dist, sample_spec
from env_sim import make_sim_factory

MANIFEST = "/workspace/DroneStudio/autoresearch/fixtures/v14_g13.manifest.json"

PHASES = {"A": {"lat": 0.05, "dens": 0.0,  "corr": 4.0},
          "B": {"lat": 0.10, "dens": 0.02, "corr": 4.0},
          "C": {"lat": 0.20, "dens": 0.05, "corr": 4.0},
          "D": {"lat": 0.35, "dens": 0.10, "corr": 2.0}}

def t3_dist(seed, knobs):
    base = tier_dist(seed, 3)
    d = SceneDistribution.from_vector(base.to_vector())
    d.waypoint_lat = knobs["lat"]; d.obstacle_density = knobs["dens"]
    d.corridor_width = knobs["corr"]
    return d

def teacher_act(obs, ext):
    rel = (obs[0:3] + obs[19:22]) * ext   # current-target rel, meters
    vel = obs[3:6]                        # keep obs scale (/10): gains tuned for it
    gb = obs[6:9]
    vmax = 2.0
    v_des = np.clip(0.5 * rel, -vmax, vmax)
    a_des = np.clip(1.2 * (v_des - vel), -2.0, 2.0)
    gx_des = np.clip(a_des[0] / 9.81, -0.20, 0.20)
    gz_des = np.clip(a_des[2] / 9.81, -0.20, 0.20)
    rates = obs[9:12]                     # obs scale (/10), as the v1 pilot used
    kp, kd = 0.4, 0.6
    act0 = kp * (gz_des - gb[2]) - kd * rates[0]
    act2 = -kp * (gx_des - gb[0]) - kd * rates[2]
    vy_des = np.clip(0.8 * rel[1], -1.5, 1.5)
    thr = -0.756 + 0.15 * (vy_des - vel[1])
    return np.clip(np.array([act0, 0.0, act2, thr]), -1, 1)

def run_phase(name, knobs, episodes=16, seed0=40000):
    succ, wpfrac, lens = [], [], []
    for i in range(episodes):
        seed = seed0 + i
        dist = t3_dist(seed, knobs)
        spec = sample_spec(seed, force_scenario="goto")
        ext = float(dist.scene_extent)
        env = make_sim_factory(dist, max_steps=600, dynamics=MANIFEST,
                               scenario_spec=spec)(seed)
        obs = env.reset()
        nwp = len(getattr(env, "waypoints", []))
        for _ in range(600):
            obs, r, done = env.step(teacher_act(obs, ext))
            if done: break
        wp = int(env.last_info.get("wp", 0))
        succ.append(bool(env.succeeded))
        wpfrac.append(wp / max(1, nwp))
        lens.append(env.steps)
        env.close()
    out = {"phase": name, "success": round(float(np.mean(succ)), 3),
           "wp_frac": round(float(np.mean(wpfrac)), 3),
           "steps": round(float(np.mean(lens)), 1)}
    print("T3TEACHER", json.dumps(out), flush=True)
    return out

if __name__ == "__main__":
    for name in ("A", "B", "C", "D"):
        run_phase(name, PHASES[name])
