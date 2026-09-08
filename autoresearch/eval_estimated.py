"""Estimator-in-the-loop policy eval on the real headless physics binary.

GT arm: policy sees binary GT obs (v1 15-dim or v2 19-dim per flag).
EST arm: binary physics untouched; per-fast-step GT telemetry (omega, quat,
filtered_thrust @500Hz) feeds the MPU-9250 SimIMU -> ESKF predict; AK8963 mag
+ VL53L9CX downward ToF (ground+sphere cast) update at the 20Hz policy rate;
policy obs rebuilt from EKF state (v1/v2 layout parity). No GT in the filter.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
from env_quad import FAST_DT
from env_sim import SimBinaryEnv
from scenario_sampler import sample_spec
from eval_scenarios import cell_dist
from policy import MLP
from sensors.ekf import ESKF, R_of, quat_from_rotvec, quat_mul
from sensors.imu import SimIMU
from sensors.tof import SimToF
from sensors.specs.mpu9250 import MPU9250_SPEC
from sensors.specs.vl53l9cx import VL53L9CX_SPEC
from sensors.base import SimEnvironment

MASS = 0.539805          # v14_g13 manifest total_mass_kg
MAX_THRUST = 40.0        # headless_main.zig default
MANIFEST = "/workspace/DroneStudio/autoresearch/fixtures/v14_g13.manifest.json"
B_WORLD = np.array([20.0, 40.0, 5.0])       # uT
HARD_IRON = np.array([1.2, -0.8, 0.5])      # uT, calibrated
NOISE = dict(gyro_nd=np.deg2rad(0.01), accel_nd=300e-6*9.81,
             gyro_bias_rw=np.deg2rad(1.0)/np.sqrt(300.0),
             accel_bias_rw=0.5e-3*9.81/np.sqrt(300.0))
G_VEC = np.array([0.0, -9.81, 0.0])

def yaw_frame(v, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([v[0]*c - v[2]*s, v[1], v[0]*s + v[2]*c])

class EstEnv(SimBinaryEnv):
    def __init__(self, *a, estimated=False, obs_v2=True, est_seed=0, vo_aided=False,
                 passthrough=False, noise_scale=1.0, obs_v3=False, fusion_gated=False,
                 zupt=False, gps=False, gps_rate_hz=5.0, **kw):
        super().__init__(*a, **kw)
        self.estimated, self.obs_v2, self.est_seed = estimated, obs_v2, est_seed
        self.vo_aided = vo_aided
        # fusion_gated: port of fusion_chained_g's innovation gate + chain
        # re-anchoring into the control-time estimator (parent dir 2026-09-07).
        self.fusion_gated = fusion_gated
        self.zupt = zupt  # zero-velocity updates when detector says stationary (parent 2026-09-07)
        # GPS (parent 2026-09-07): placed hardware (placement-effective.json
        # [-0.1165,0,0.002]) that the sim never modeled - the only absolute
        # horizontal channel on the real frame. u-blox-class: ~1.5m CEP
        # horizontal (per-axis ~1.27m), vertical ~2.5m, OU bias (tau 120s,
        # stationary std 1m) + white noise. Fused as slow absolute anchor;
        # VO keeps relative smoothness, ToF owns vertical via Kalman weights.
        self.gps = gps
        self.gps_dt = 1.0 / gps_rate_hz
        # GPS bias-as-state (terminal relative-nav attempt, 2026-09-07):
        # measured NEUTRAL-NEGATIVE in the teacher gate (bias split polluted
        # by VO-chain drift; hover/land 6.2/0.0 vs 6.2/6.2 plain). Kept behind
        # a flag, default off. Reanchor30 variant strongly negative (17m).
        self.gps_bias_state = kw.get("gps_bias_state", False)
        self.passthrough = passthrough   # run estimator + diagnostics, return GT obs
        self.noise_scale = noise_scale  # curriculum knob: scales VO/mag measurement noise
        self.obs_v3 = obs_v3            # 25-dim: + tof_alt/valid, sigma_p/v/att, vo_aid_std

    def _ensure_proc(self):
        fresh = self.proc is None or self.proc.poll() is not None
        super()._ensure_proc()
        if fresh and self.estimated:
            self._call({"cmd": "fast_telemetry", "on": True})

    def reset(self):
        obs = super().reset()
        self.last_gt_obs = obs
        if not self.estimated:
            return obs
        self.imu = SimIMU(MPU9250_SPEC, seed=self.est_seed)
        self.tof = SimToF(VL53L9CX_SPEC, mode="wake", seed=self.est_seed + 1)
        self.tof.spec = {**self.tof.spec, "rate_hz": 20.0}
        # altimeter mount: sensor boresight (+x) points DOWN (-y body).
        # Default identity mount left the FoV forward-looking -> the 15-deg
        # nadir gate never passed and update_ground_range never fired (found
        # 2026-09-06: tof_valid 0.4% in hover cells). Real design is an
        # 8-sensor ring (CAD poses pending); single nadir altimeter is the
        # harness's documented abstraction until the ring sim lands.
        self.tof.mount.rot = np.array([[0.0, 1.0, 0.0],
                                       [-1.0, 0.0, 0.0],
                                       [0.0, 0.0, 1.0]])
        self.senv = SimEnvironment()
        self.kf = ESKF(NOISE)
        self.kf.p = self.spawn.copy()
        self.kf.v = np.array(getattr(self, "spawn_vel", np.zeros(3)), dtype=float)
        rng = np.random.default_rng(self.est_seed + 2)
        self.kf.q = quat_mul(np.array([0., 0., 0., 1.]),
                             quat_from_rotvec(rng.normal(0, np.deg2rad(1.2), 3)))
        self.kf.P[6:9, 6:9] = np.eye(3) * (np.deg2rad(3.0)) ** 2
        self.t = 0.0
        self.gyro_last = np.zeros(3)
        self.accel_last = np.array([0.0, 9.81, 0.0])
        self.vo_p_prev = None
        self.tof_alt_prev = None
        self.zupt_fires = 0
        self.pos_errs, self.att_errs = [], []
        self.vel_errs, self.rate_errs = [], []
        # synthetic VO chain: correlated scale + yaw-walk + white floor,
        # anchored to measured full-pilot VO ATE (5-17m over 50-60m) and
        # fusion v1's growing-R convention
        self.vo_p = self.spawn.copy()
        self.vo_yaw = 0.0
        self.vo_scale = float(rng.normal(1.0, 0.03 * self.noise_scale))
        self.vo_n = 0
        self.vo_drift = []
        # gated-fusion anchors: last ACCEPTED chain/filter correspondence
        self.vo_anchor = self.spawn.copy()
        self.p_anchor_filter = self.spawn.copy()
        self.vo_rejects = 0
        self.vo_accepts = 0
        self.tof_alt = 0.0
        self.tof_alt_t = -1e9
        self.gps_bias = np.zeros(3)
        self.gps_last_t = -1e9
        self.gps_fixes = 0
        self.gps_bias_err = []
        self.p_prev = self.spawn.copy()
        return self._est_obs()

    def step(self, action):
        if not self.estimated:
            return super().step(action)
        if self.passthrough:
            obs, r, done = self._est_step(action)
            if self.obs_v3:
                return self._last_gt_v3_obs, r, done
            return self.last_gt_obs, r, done
        return self._est_step(action)

    def _est_step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        resp = self._call({"cmd": "step", "action": [float(x) for x in a]})
        self.last_gt_obs = np.array(resp["obs"], dtype=np.float64)
        info = resp.get("info", {})
        self.last_info = info
        self.steps = int(info.get("steps", self.steps + 1))
        self.collided = bool(info.get("collided", False))
        self._succeeded_sim = bool(info.get("succeeded", False))
        for row in resp.get("fast", []):
            om = np.array(row[0:3]); q = np.array(row[3:7]); fthr = float(row[7])
            self.t += FAST_DT
            ws = dict(omega=om, alpha=np.zeros(3), quat=q,
                      thrust_world=R_of(q) @ np.array([0.0, fthr, 0.0]), mass=MASS)
            meas = self.imu.sample(self.t, FAST_DT, ws, self.senv,
                                   throttle=float(np.clip(fthr / MAX_THRUST, 0, 1)))
            if meas is not None:
                self.gyro_last = meas.channels["gyro"]
                self.accel_last = meas.channels["accel"]
                self.kf.predict(meas.channels["gyro"], meas.channels["accel"], FAST_DT)
        # policy-rate aiding from TRUE pose (sensor sim must never read the filter)
        q_t = np.array(info["quat"], dtype=float); p_t = np.array(info["pos"], dtype=float)
        rng = self.imu.rng
        b_meas = R_of(q_t).T @ B_WORLD + HARD_IRON + rng.normal(0, 0.5 * self.noise_scale, 3)
        self.kf.update_mag(b_meas - HARD_IRON, B_WORLD, 0.7)
        centers, radii = self.obs_centers, self.obs_radii
        def cast(o, d):
            best = None
            if d[1] < -1e-6:
                tt = -o[1] / d[1]
                if tt > 0: best = tt
            if len(centers):
                oc = centers - o
                b = oc @ d
                disc = b*b - (np.sum(oc*oc, 1) - radii*radii)
                hit = disc > 0
                if np.any(hit):
                    tt = np.where(hit, b - np.sqrt(np.maximum(disc, 0)), np.inf)
                    tt = np.where(tt > 0, tt, np.inf)
                    j = int(np.argmin(tt))
                    if np.isfinite(tt[j]) and (best is None or tt[j] < best):
                        best = float(tt[j])
            return best
        if self.vo_aided:
            dp = p_t - self.p_prev
            self.vo_yaw += float(rng.normal(0, np.deg2rad(0.3 * self.noise_scale)))
            c, sn = np.cos(self.vo_yaw), np.sin(self.vo_yaw)
            Rz = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1.0]])
            self.vo_p = self.vo_p + self.vo_scale * (Rz @ dp) + rng.normal(0, 0.02 * self.noise_scale, 3)
            self.vo_n += 1
            if not self.fusion_gated:
                self.kf.update_position(self.vo_p.copy(), np.eye(3) * (0.25 ** 2 * self.vo_n))
            else:
                # INNOVATION GATE (port of fusion_chained_g): chain displacement
                # since last accepted anchor vs filter displacement over the same
                # interval. Reject when inconsistent -> coast on IMU+ToF+mag.
                # REANCHOR: chain is an integrated reference and leaks without
                # bound; after 2s of consecutive rejects, reset chain to filter.
                n_br = self.vo_n - getattr(self, "_vo_n_anchor", 0)
                chain_d = self.vo_p - self.vo_anchor
                filt_d = self.kf.p - self.p_anchor_filter
                inno = float(np.linalg.norm(chain_d - filt_d))
                gate = 0.25 * np.sqrt(max(n_br, 1)) * 1.5 + 0.05 * float(np.linalg.norm(filt_d))
                if inno <= gate:
                    self.kf.update_position(self.vo_p.copy(), np.eye(3) * (0.25 ** 2 * self.vo_n))
                    self.vo_anchor = self.vo_p.copy()
                    self.p_anchor_filter = self.kf.p.copy()
                    self._vo_n_anchor = self.vo_n
                    self.vo_rejects = 0
                    self.vo_accepts += 1
                else:
                    self.vo_rejects += 1
                    if self.vo_rejects >= 40:  # ~2s of policy-rate rejects
                        self.vo_p = self.kf.p.copy()
                        self.vo_anchor = self.kf.p.copy()
                        self.p_anchor_filter = self.kf.p.copy()
                        self._vo_n_anchor = self.vo_n
                        self.vo_rejects = 0
            self.vo_drift.append(float(np.linalg.norm(self.vo_p - p_t)))
        self.p_prev = p_t.copy()
        tm = self.tof.scan(self.t, dict(quat=q_t, origin=p_t), self.senv, cast)
        if tm is not None:
            rc = tm.channels["ranges"].ravel(); st = tm.channels["status"].ravel()
            ok = np.where(st == 0)[0]
            if len(ok):
                Rk = R_of(self.kf.q)
                dirs = np.stack([self.tof.mount.rot @ d for d in self.tof._dirs_sensor])
                j = ok[int(np.argmin([(Rk @ dirs[m])[1] for m in ok]))]
                d_w = Rk @ dirs[j]
                if np.arccos(np.clip(-d_w[1], 0.0, 1.0)) <= np.deg2rad(15):
                    r_true = float(rc[j])
                    sig = max(self.tof.spec["sigma_base_mm"]/1000.0
                              + self.tof.spec["sigma_range2"]*r_true*r_true, 0.005)
                    self.kf.update_ground_range(r_true, dirs[j], sig)
                    self.tof_alt = r_true
                    self.tof_alt_t = self.t
        # GPS aiding: sampled from TRUE pose (sensor sim never reads the filter)
        if self.gps and (self.t - self.gps_last_t) >= self.gps_dt:
            self.gps_last_t = self.t
            # OU bias: tau=120s, stationary std 1.0m per axis
            a_ou = np.exp(-self.gps_dt / 120.0)
            self.gps_bias = a_ou * self.gps_bias + np.sqrt(1 - a_ou ** 2) * rng.normal(0, 1.0 * self.noise_scale, 3)
            z = p_t + self.gps_bias + rng.normal(0, [1.27 * self.noise_scale,
                                                     2.5 * self.noise_scale,
                                                     1.27 * self.noise_scale])
            Rg = np.diag([1.27 ** 2, 2.5 ** 2, 1.27 ** 2]) * (self.noise_scale ** 2)
            if self.gps_bias_state:
                self.kf.update_gps(z, Rg)
                self.gps_bias_err.append(float(np.linalg.norm(self.kf.bgps - self.gps_bias)))
            else:
                self.kf.update_position(z, Rg)
            self.gps_fixes += 1
        # ZUPT (parent 2026-09-07): during holds the VO random walk (~2cm/step)
        # is the dominant terminal-phase error. Detector uses only sensor-side
        # quantities: rates low, specific force ~ g, VO increment near zero,
        # ToF altitude steady. Cruise/descent cannot false-fire: cruise has
        # ~0.1m/step VO increments, descent moves the altimeter.
        if self.zupt:
            vo_step = float(np.linalg.norm(self.vo_p - self.vo_p_prev)) if self.vo_p_prev is not None else 1.0
            tof_steady = (self.tof_alt_prev is not None
                          and abs(self.tof_alt - self.tof_alt_prev) < 0.03)
            if (np.linalg.norm(self.gyro_last) < 0.15
                    and abs(float(np.linalg.norm(self.accel_last)) - 9.81) < 0.5
                    and vo_step < 0.04 and tof_steady):
                self.kf.update_velocity(np.zeros(3), 0.1)
                self.zupt_fires += 1
        self.vo_p_prev = self.vo_p.copy()
        self.tof_alt_prev = self.tof_alt
        # diagnostics
        self.pos_errs.append(float(np.linalg.norm(self.kf.p - p_t)))
        dq = quat_mul(q_t, np.array([-self.kf.q[0], -self.kf.q[1], -self.kf.q[2], self.kf.q[3]]))
        self.att_errs.append(float(np.rad2deg(2*np.arccos(np.clip(abs(dq[3]), 0, 1)))))
        self.vel_errs.append(float(np.linalg.norm(self.kf.v - np.array(info["vel"], dtype=float))))
        self.rate_errs.append(float(np.rad2deg(np.linalg.norm(self.gyro_last - np.array(row[0:3])))) if len(resp.get("fast", [])) else 0.0)
        if self.obs_v3:
            self._last_gt_v3_obs = self._gt_v3_obs(q_t, p_t, np.array(info["vel"], dtype=float))
        return self._est_obs(), float(resp["reward"]), bool(resp["done"])

    def _v3_channels(self, ext):
        P = self.kf.P
        sp = float(np.sqrt(max(np.trace(P[0:3, 0:3]) / 3.0, 0.0)))
        sv = float(np.sqrt(max(np.trace(P[3:6, 3:6]) / 3.0, 0.0)))
        sa = float(np.sqrt(max(np.trace(P[6:9, 6:9]) / 3.0, 0.0)))
        tof_valid = 1.0 if (self.t - self.tof_alt_t) <= 0.5 else 0.0
        vo_std = 0.25 * float(np.sqrt(max(self.vo_n, 1)))
        return np.array([min(self.tof_alt, 20.0) / ext, tof_valid,
                         min(sp, 5.0) / 5.0, min(sv, 2.0) / 2.0,
                         min(sa, 0.35) / 0.35, min(vo_std, 2.0) / 2.0])

    def _gt_v3_obs(self, q_t, p_t, v_t):
        # GT-state channels + real sensor channels: the v3 GT arm for
        # regression floors (differs from est arm ONLY in state source).
        ext = max(self.dist.scene_extent, 1.0)
        fwd = R_of(q_t) @ np.array([1.0, 0.0, 0.0])
        yaw = np.arctan2(-fwd[2], fwd[0])
        YF = lambda v: yaw_frame(v, yaw)
        rel_goal = YF(self.goal - p_t) / ext
        v = YF(np.asarray(v_t, dtype=float)) / 10.0
        g_body = (R_of(q_t).T @ G_VEC) / 9.81
        rates = self.gyro_last / 10.0
        rel = np.zeros(3)
        if len(self.obs_centers):
            dvec = self.obs_centers - p_t
            rel = YF(dvec[int(np.argmin(np.linalg.norm(dvec, axis=1)))]) / ext
        base = np.concatenate([rel_goal, v, g_body, rates, rel])
        spec = self.scenario_spec or {}
        oh = {"goto": [1,0,0], "hover_hold": [0,1,0], "land": [0,0,1]}[spec.get("scenario", "goto")]
        base = np.concatenate([base, oh, [spec.get("success_radius", 2.0) / ext]])
        return np.concatenate([base, self._v3_channels(ext)])

    def _est_obs(self):
        ext = max(self.dist.scene_extent, 1.0)
        fwd = R_of(self.kf.q) @ np.array([1.0, 0.0, 0.0])
        yaw = np.arctan2(-fwd[2], fwd[0])
        YF = (lambda v: yaw_frame(v, yaw)) if self.obs_v2 else (lambda v: v)
        rel_goal = YF(self.goal - self.kf.p) / ext
        v = YF(self.kf.v) / 10.0
        g_body = (R_of(self.kf.q).T @ G_VEC) / 9.81
        rates = self.gyro_last / 10.0
        rel = np.zeros(3)
        if len(self.obs_centers):
            dvec = self.obs_centers - self.kf.p
            rel = YF(dvec[int(np.argmin(np.linalg.norm(dvec, axis=1)))]) / ext
        base = np.concatenate([rel_goal, v, g_body, rates, rel])
        if self.obs_v2:
            spec = self.scenario_spec or {}
            oh = {"goto": [1,0,0], "hover_hold": [0,1,0], "land": [0,0,1]}[spec.get("scenario", "goto")]
            base = np.concatenate([base, oh, [spec.get("success_radius", 2.0) / ext]])
        if self.obs_v3:
            base = np.concatenate([base, self._v3_channels(ext)])
        return base

def load_policy(path):
    flat = np.array(json.load(open(path)), dtype=np.float64)
    for od in (15, 19, 26, 27):
        for h in (32, 64, 128):
            if MLP.param_count(od, 4, h) == len(flat):
                net = MLP(od, 4, hidden=h)
                net.set_flat(flat)
                return net, od
    raise ValueError(f"no shape fits {len(flat)}")

def run_arm(policy_path, estimated, n_episodes=30, base_seed=10_000,
            scenario="goto", max_steps=400, vo_aided=False):
    net, od = load_policy(policy_path)
    obs_v2 = od >= 19
    if obs_v2: os.environ["AUTORESEARCH_OBS_V2"] = "1"
    else: os.environ.pop("AUTORESEARCH_OBS_V2", None)
    rows = []
    for k in range(n_episodes):
        seed = base_seed + k
        dist = cell_dist(seed)
        spec = sample_spec(seed, force_scenario=scenario)
        env = EstEnv(dist, seed=seed, max_steps=max_steps, dynamics=MANIFEST,
                     scenario_spec=spec, estimated=estimated, obs_v2=obs_v2,
                     est_seed=seed + 777, vo_aided=vo_aided)
        obs = env.reset()
        total = 0.0
        while True:
            obs, r, done = env.step(net.act(obs))
            total += r
            if done: break
        row = dict(ok=bool(env.succeeded), collided=env.collided, steps=env.steps,
                   ret=total,
                   final_dist=float(np.linalg.norm(env.goal - np.array(env.last_info["pos"]))))
        if estimated:
            row["pos_err"] = float(np.mean(env.pos_errs))
            row["att_err"] = float(np.mean(env.att_errs))
            if vo_aided:
                row["vo_drift"] = float(np.mean(env.vo_drift))
        rows.append(row)
        env.close()
    agg = dict(policy=policy_path.split("/")[-1], obs_dim=od, estimated=estimated,
               vo_aided=vo_aided,
               scenario=scenario, n=n_episodes,
               success=float(np.mean([r["ok"] for r in rows])),
               collision=float(np.mean([r["collided"] for r in rows])),
               ret=float(np.mean([r["ret"] for r in rows])),
               final_dist=float(np.mean([r["final_dist"] for r in rows])))
    if estimated:
        agg["pos_err"] = float(np.mean([r["pos_err"] for r in rows]))
        agg["att_err"] = float(np.mean([r["att_err"] for r in rows]))
        if vo_aided:
            agg["vo_drift"] = float(np.mean([r["vo_drift"] for r in rows]))
    return agg, rows

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    pols = sys.argv[2:] or ["/workspace/bc_ppo_v2_best.json", "/workspace/bc_flat.json"]
    out = []
    for pol in pols:
        for est, vo in ((False, False), (True, False), (True, True)):
            agg, rows = run_arm(pol, est, n_episodes=n, vo_aided=vo)
            out.append(agg)
            print(json.dumps(agg), flush=True)
            tag = "gt" if not est else ("est_vo" if vo else "est")
            json.dump(rows, open(f"/workspace/vision_model/est_eval_{agg['policy']}_{tag}.json", "w"))
    json.dump(out, open("/workspace/vision_model/est_in_loop_eval.json", "w"), indent=1)
