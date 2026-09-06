"""VL53L9CX-class multizone dToF sensor simulation.

Backend-injected: the environment provides a ray-cast callable
    cast(origin_world(3,), dir_world(3,)) -> range_m or None (no return)
so the same sensor runs against the analytic ray-caster, rendered depth maps
(reprojected), or future mesh backends.

Sensor frame matches the sim camera convention: +X forward, FoV spans
horizontal (55 deg) x vertical (42 deg), zone (row, col) maps to a direction
with uniform angular spacing.

Noise: per-zone Gaussian sigma = base + k*r^2, scaled by ambient light and
junction temperature; Bernoulli dropout rising from 80% of mode max range;
status per zone: 0 ok, 1 over-range, 2 no-return (dropout).
"""
import numpy as np
from .base import Mount, Measurement, SimEnvironment, SimSensor

class SimToF(SimSensor):
    def __init__(self, spec, mount=None, mode="room_mapping", seed=0,
                 ambient_scale=1.0):
        super().__init__(spec, mount or Mount(), name="tof", seed=seed)
        m = spec["modes"][mode]
        self.mode_name = mode
        self.rows, self.cols = m["grid"]
        self.rate_hz = float(m["fps"])
        self.max_m = m["max_m"]
        self.min_m = spec["range_min_m"][m["mode"]]
        self.ambient_scale = ambient_scale
        self.spec = {**self.spec, "rate_hz": self.rate_hz}  # base.due() key
        self._dirs_sensor = self._zone_dirs()

    def _zone_dirs(self):
        """(rows*cols, 3) unit dirs in sensor frame, +X forward."""
        hh = np.deg2rad(self.spec["fov_h_deg"]) / 2
        hv = np.deg2rad(self.spec["fov_v_deg"]) / 2
        us = np.linspace(-np.tan(hh), np.tan(hh), self.cols)
        vs = np.linspace(-np.tan(hv), np.tan(hv), self.rows)
        uu, vv = np.meshgrid(us, vs)
        d = np.stack([np.ones_like(uu), vv, uu], -1)   # (1, v, u): cam convention
        return (d / np.linalg.norm(d, axis=-1, keepdims=True)).reshape(-1, 3)

    def zone_dirs_world(self, quat):
        from .ekf import quat_rotate
        return np.stack([quat_rotate(quat, self.mount.rot @ d) for d in self._dirs_sensor])

    def ideal(self, ws, env):
        raise NotImplementedError("ToF needs a cast backend; use scan()")

    def scan(self, t, ws, env, cast, throttle=0.0):
        """One frame. ws: quat + origin (body world pos). cast: ray backend."""
        if not self.due(t):
            return None
        origin = ws["origin"] if "origin" in ws else np.zeros(3)
        dirs_w = self.zone_dirs_world(ws["quat"])
        origin_w = origin + self.mount.pos
        n = len(dirs_w)
        ranges = np.full(n, np.nan)
        status = np.full(n, 2, np.uint8)
        T = env.temperature(t)
        sig_scale = self.ambient_scale * (
            1.0 + self.spec["temp_sigma_scale_per_30c"] * max(0.0, T - 25.0) / 30.0)
        for i in range(n):
            r = cast(origin_w, dirs_w[i])
            if r is None or r > self.max_m:
                status[i] = 1 if r is not None else 2
                continue
            if r < self.min_m:
                status[i] = 2
                continue
            sigma_m = (self.spec["sigma_base_mm"] +
                       self.spec["sigma_range2"] * r * r * 1000.0) / 1000.0
            sigma_m *= sig_scale
            # dropout past 80% of mode max
            frac = r / self.max_m
            p_drop = 0.0
            fs = self.spec["dropout_start_frac"]
            if frac > fs:
                p_drop = min(1.0, (frac - fs) / (1.0 - fs)) ** self.spec["dropout_slope"] * 4
                p_drop = min(1.0, p_drop)
            if self.rng.random() < p_drop:
                status[i] = 2
                continue
            ranges[i] = r + self.rng.normal(0, sigma_m)
            status[i] = 0
        return Measurement(self.name, t, self.rate_hz, {
            "ranges": ranges.reshape(self.rows, self.cols),
            "status": status.reshape(self.rows, self.cols),
        })

def self_test(seed=1):
    """Ground plane y=0, sensor at 2m pointing straight down: center zone
    must return ~2.0m with modeled sigma and ~zero dropout."""
    from .specs.vl53l9cx import VL53L9CX_SPEC
    tof = SimToF(VL53L9CX_SPEC, mode="room_mapping", seed=seed)
    env = SimEnvironment()
    q_down = np.array([0.0, 0.0, -np.sin(np.pi/4), np.cos(np.pi/4)])  # +X -> -Y (rot about Z)
    ws = dict(quat=q_down, origin=np.array([0.0, 2.0, 0.0]))
    def cast(o, d):
        if d[1] >= -1e-6:
            return None
        t = -o[1] / d[1]
        return float(t) if t > 0 else None
    mid = None
    rs = []
    for k in range(60):
        meas = tof.scan(k / 30.0, ws, env, cast)
        if meas is None:
            continue
        r = meas.channels["ranges"]; s = meas.channels["status"]
        c = (r.shape[0]//2, r.shape[1]//2)
        if s[c] == 0:
            rs.append(r[c])
    rs = np.array(rs)
    assert len(rs) > 45, f"too many dropouts: {len(rs)}/60"
    err = abs(rs.mean() - 2.0)
    print(f"tof self_test: center-zone mean {rs.mean():.4f}m (truth 2.0), "
          f"std {rs.std()*1000:.2f}mm, n={len(rs)}/60")
    assert err < 0.01, err
    print("PASS")
    return True

if __name__ == "__main__":
    self_test()
