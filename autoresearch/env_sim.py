"""SimBinaryEnv: QuadNavEnv-compatible env driving the headless Zig binary.

Same contract as env_quad.QuadNavEnv (factory(seed) -> env, reset(), step()
-> (obs, reward, done), OBS_DIM/ACT_DIM, succeeded) but episodes run inside
the real DroneStudio physics: Bullet rigid body + his RateController/PID
code path in the dronestudio-headless binary.

Concrete scenes are sampled with the SAME generator as QuadNavEnv (we
subclass it and forward the sampled spawn/goal/obstacles), so backends run
identical scenarios - the comparison is dynamics-only.
"""
import os, json, subprocess
import numpy as np
from env_quad import QuadNavEnv

BINARY = os.environ.get(
    "AUTORESEARCH_SIM_BIN",
    "/workspace/zig-out/bin/dronestudio-headless")


class SimBinaryEnv(QuadNavEnv):
    def __init__(self, distribution, seed=0, max_steps=200, binary=BINARY, dynamics=None):
        super().__init__(distribution, seed=seed, max_steps=max_steps)
        self.seed = seed
        self.binary = binary
        self.dynamics = dynamics  # manifest path, "abstract", or None
        self.proc = None

    def _ensure_proc(self):
        if self.proc is None or self.proc.poll() is not None:
            self.proc = subprocess.Popen(
                [self.binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                text=True, bufsize=1)
            if self.dynamics:
                self._call({"cmd": "set_dynamics", "path": self.dynamics})
            if os.environ.get("AUTORESEARCH_MOTOR_V2"):
                self._call({"cmd": "motor_v2", "on": True})
            self._call({"cmd": "ping"})

    def _call(self, msg):
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("headless sim died: " + repr(self.proc.stderr))
        resp = json.loads(line)
        if "error" in resp:
            raise RuntimeError("headless sim error: " + resp["error"])
        return resp

    def reset(self):
        super().reset()  # samples the concrete scene into self.spawn/goal/obs_*
        self._ensure_proc()
        scene = {
            "spawn": [float(x) for x in self.spawn],
            "goal": [float(x) for x in self.goal],
            "obstacles": [[float(c[0]), float(c[1]), float(c[2]), float(r)]
                          for c, r in zip(self.obs_centers, self.obs_radii)],
            "extent": float(self.dist.scene_extent),
            "max_steps": int(self.max_steps),
            "dynamics_noise": float(self.dist.dynamics_noise),
        }
        resp = self._call({"cmd": "reset", "seed": int(self.seed), "scene": scene})
        # mirror bookkeeping used by env_quad.succeeded
        self.pos = self.spawn.copy()
        self.steps = 0
        self.collided = False
        self._succeeded_sim = False
        return np.array(resp["obs"], dtype=np.float64)

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        resp = self._call({"cmd": "step", "action": [float(x) for x in a]})
        info = resp.get("info", {})
        self.steps = int(info.get("steps", self.steps + 1))
        self.collided = bool(info.get("collided", False))
        self._succeeded_sim = bool(info.get("succeeded", False))
        return (np.array(resp["obs"], dtype=np.float64),
                float(resp["reward"]), bool(resp["done"]))

    @property
    def succeeded(self):
        return self._succeeded_sim

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                self._call({"cmd": "close"})
            except Exception:
                pass
            self.proc.terminate()
        if self.proc is not None:
            try:
                self.proc.wait(timeout=5)  # reap: no zombie accumulation
            except Exception:
                try:
                    self.proc.kill()
                    self.proc.wait(timeout=5)
                except Exception:
                    pass
        self.proc = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def make_sim_factory(distribution, max_steps=200, binary=BINARY, dynamics=None):
    def factory(seed):
        return SimBinaryEnv(distribution, seed=seed, max_steps=max_steps, binary=binary,
                            dynamics=dynamics)
    return factory
