import sys, json
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
import numpy as np
from scene_schema import SceneDistribution
from env_sim import make_sim_factory

MANIFEST = "/workspace/DroneStudio/autoresearch/fixtures/chassis_v1.manifest.json"

def pilot_act(obs, ext=10.0):
    rel = obs[0:3] * ext          # world rel goal, m (obs = rel / scene_extent)
    vel = obs[3:6]                 # world vel, m/s
    gb  = obs[6:9]                 # gravity dir in body frame (~thrust.x=gb.x, thrust.z=gb.z)
    # position -> velocity
    vmax = 1.5
    v_des = np.clip(0.5 * rel, -vmax, vmax)
    # velocity -> accel
    a_des = np.clip(1.2 * (v_des - vel), -2.0, 2.0)
    # accel -> desired tilt (thrust horizontal components are g_body.x / g_body.z)
    gx_des = np.clip(a_des[0] / 9.81, -0.20, 0.20)
    gz_des = np.clip(a_des[2] / 9.81, -0.20, 0.20)
    # tilt -> rate commands (act0+ raises gb.z; act2+ lowers gb.x), with rate damping
    rates = obs[9:12]
    kp, kd = 0.4, 0.6
    act0 = kp * (gz_des - gb[2]) - kd * rates[0]
    act2 = -kp * (gx_des - gb[0]) - kd * rates[2]
    # vertical: throttle PI-ish on vertical velocity around hover -0.6
    vy_des = np.clip(0.8 * rel[1], -1.5, 1.5)
    thr = -0.6 + 0.15 * (vy_des - vel[1])
    return np.clip(np.array([act0, 0.0, act2, thr]), -1, 1)

def run_stage(gd, episodes=24, max_steps=400, seed0=30000):
    dist = SceneDistribution(obstacle_density=0.0, corridor_width=10.0,
        scene_extent=max(10.0, gd*2), goal_distance=gd, light_direction_entropy=0.0,
        texture_variety=0.0, dynamics_noise=0.0)
    factory = make_sim_factory(dist, max_steps=max_steps, dynamics=MANIFEST)
    succ, rets, lens = [], [], []
    for i in range(episodes):
        env = factory(seed0 + i)
        obs = env.reset()
        total = 0.0
        for _ in range(max_steps):
            obs, r, done = env.step(pilot_act(obs))
            total += r
            if done: break
        succ.append(bool(env.succeeded)); rets.append(total); lens.append(env.steps)
        env.close()
    return float(np.mean(succ)), float(np.mean(rets)), float(np.mean(lens))

if __name__ == "__main__":
    for gd in (2.0, 5.0, 10.0):
        s, r, l = run_stage(gd)
        print("PILOT_RESULT", json.dumps({"goal_m": gd, "success": s, "ret": round(r,2), "steps": round(l,1)}), flush=True)
