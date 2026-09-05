"""T4 hybrid policy: learned t4_best flies goto/hover/approach, the verified
teacher position-PD module (t4_pilot.teacher_act land branch) runs terminal
descent as a scripted precision-land controller.

Handoff boundary (user decision 2026-09-05, option (a)+(b) after dag11 null):
  engage  when scenario=="land" and alt <= 1.4 and dxz <= max(0.75, 1.5*r)
  release when alt > 3.0 or dxz > max(3.0, 3.0*r)   (same hysteresis the
          redesigned teacher uses internally; teacher state starts cold at
          handoff - its land_descend latch + integrator live in tstate)
The learned policy is byte-identical to the t4_best streamer path
(t4_common.bc_to_actor_params + wp_forward, tanh(mu+wpmu)); nothing about
t4_best itself changes - this is an actuator-level composition, not a new
checkpoint. Promotion/tile-flip questions are untouched.
"""
import sys, os, json
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
os.environ.setdefault("AUTORESEARCH_OBS_V3", "1")
os.environ.setdefault("AUTORESEARCH_MOTOR_V2", "1")
import numpy as np
from scenario_sampler import sample_spec, tier_dist, hover_max_steps
from env_sim import make_sim_factory
import t4_pilot
import t4_common as P
from ppo import MLP as PPOMlp

POLICY = os.environ.get("HYBRID_POLICY", "/workspace/t4_best.json")


def load_student(path):
    flat = np.array(json.load(open(path)), dtype=np.float64)
    actor = PPOMlp(np.random.default_rng(0), P.OBS_DIM, P.HID, P.ACT_DIM)
    actor.load(P.bc_to_actor_params(flat))
    wp1, wp2 = P.unpack_wp(flat)

    def act(obs):
        mu, _ = actor.forward(obs[None, :])
        wpmu, _ = P.wp_forward(obs[None, :], wp1, wp2)
        return np.tanh(mu[0] + wpmu[0])
    return act


class Hybrid:
    def __init__(self, student_act):
        self.student = student_act
        self.engaged = False
        self.tstate = {}
        self.handoffs = 0

    def act(self, obs, ext, scenario, radius):
        if scenario != "land" or os.environ.get("HYBRID_OFF"):
            return self.student(obs)
        rel = (obs[0:3] + obs[19:22]) * ext
        alt = -float(rel[1])
        dxz = float(np.hypot(rel[0], rel[2]))
        # eval3: widening the funnel (alt<=2.0, dxz<=1.5) engaged the teacher
        # OUTSIDE its approach-FSM envelope (cold latch at low alt + lateral
        # speed) -> engage/release flapping, land 0.438 < 0.625. Reverted to
        # the teacher's own latch geometry; post-handoff success is 10/10
        # there. The residual misses are student approach precision - that is
        # what the land-specific training experiment (b) is for.
        if not self.engaged and alt <= 1.4 and dxz <= max(0.75, 1.5 * radius):
            self.engaged = True
            self.tstate = {}          # cold teacher state at handoff
            self.handoffs += 1
        elif self.engaged and (alt > 3.0 or dxz > max(3.0, 3.0 * radius)):
            self.engaged = False
        if self.engaged:
            return t4_pilot.teacher_act(obs, ext, "land", radius, state=self.tstate)
        return self.student(obs)


def run_ep(seed, sc, ms, student):
    dist = tier_dist(seed, 0)
    spec = sample_spec(seed, force_scenario=sc)
    if sc != "goto":
        dist.n_waypoints = 0.0
    ext = float(dist.scene_extent)
    env = make_sim_factory(dist, max_steps=ms, dynamics=t4_pilot.MANIFEST, scenario_spec=spec)(seed)
    obs = env.reset()
    hy = Hybrid(student)
    for _ in range(ms):
        a = hy.act(obs, ext, sc, float(spec["success_radius"]))
        obs, r, done = env.step(a)
        if done:
            break
    ok = bool(env.succeeded)
    rel = (obs[0:3] + obs[19:22]) * ext
    env.close()
    return ok, -float(rel[1]), float(np.hypot(rel[0], rel[2])), hy.handoffs


def main():
    student = load_student(POLICY)
    print(f"hybrid student={os.path.basename(POLICY)}", flush=True)
    for sc, ms in (("hover_hold", None), ("goto", 400), ("land", 1200)):
        outs = []
        for j in range(16):
            m = ms if ms else hover_max_steps(4.0, 0)
            outs.append(run_ep(620000 + j, sc, m, student))
        sr = float(np.mean([o[0] for o in outs]))
        fails = [o for o in outs if not o[0]]
        hand = [o[3] for o in outs]
        print(f"{sc} t0 n=16: succ {sr:.3f} handoffs={sum(1 for h in hand if h > 0)}/16", flush=True)
        for o in fails:
            print(f"   FAIL alt={o[1]:.2f} dxz={o[2]:.2f} handoffs={o[3]}", flush=True)
    print("HYBRID_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
