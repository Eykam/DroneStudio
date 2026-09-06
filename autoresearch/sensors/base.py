"""Sensor simulation library - shared contracts.

Separations: IDEAL measurement (physics truth at the sensor) vs ERROR model
(datasheet-anchored corruption). Specs carry datasheet numbers; sensor
classes carry generic machinery; mounts carry airframe placement.
"""
import numpy as np
from dataclasses import dataclass, field

@dataclass
class Mount:
    """Sensor placement on the airframe, body frame."""
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))   # m
    rot: np.ndarray = field(default_factory=lambda: np.eye(3))     # body->sensor

@dataclass
class Measurement:
    sensor: str
    t: float                      # timestamp (with jitter)
    rate_hz: float
    channels: dict                # name -> np.ndarray
    latency_s: float = 0.0

class SimEnvironment:
    """Shared airframe-level channels: temperature + vibration state."""
    def __init__(self, temp_ambient_c=25.0, seed=0):
        self.t0 = temp_ambient_c
        self.rng = np.random.default_rng(seed)

    def temperature(self, t):
        return self.t0 + 8.0 * (1.0 - np.exp(-t / 90.0)) + 1.5 * np.sin(2 * np.pi * t / 600.0)

class SimSensor:
    """Base: spec (datasheet dict) + mount + rate-gated sampling."""
    def __init__(self, spec, mount=None, seed=0, name=None):
        self.spec = spec
        self.mount = mount or Mount()
        self.name = name or spec["part"]
        self.rng = np.random.default_rng(seed)
        self._next_t = 0.0

    def due(self, t):
        if t + 1e-12 >= self._next_t:
            self._next_t = t + 1.0 / self.spec["rate_hz"]
            return True
        return False

    def ideal(self, world_state, env):
        raise NotImplementedError

    def corrupt(self, ideal, env, t, dt):
        raise NotImplementedError

    def sample(self, t, dt, world_state, env):
        if not self.due(t):
            return None
        ideal = self.ideal(world_state, env)
        out = self.corrupt(ideal, env, t, dt)
        return Measurement(self.name, t + self.rng.normal(0, 1e-4),
                           self.spec["rate_hz"], out)
