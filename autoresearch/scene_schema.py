"""Scene-distribution parameters: the mutation target of the outer loop.

Each parameter is a knob the procedural scene generator (sim side) will
consume. The outer loop never touches policy weights or sim code - it only
moves this distribution, and the evaluator scores what that does to policy
training.
"""
from dataclasses import dataclass, fields, asdict
import json

@dataclass
class SceneDistribution:
    obstacle_density: float = 0.3       # 0..1, fraction of scene cells occupied
    obstacle_size_mean: float = 2.0     # meters
    obstacle_size_spread: float = 0.5   # 0..1 relative spread
    corridor_width: float = 4.0         # meters, min free path width
    scene_extent: float = 40.0          # meters, square side length
    goal_distance: float = 25.0         # meters from spawn
    light_intensity: float = 0.8        # 0.2..1
    light_direction_entropy: float = 0.3  # 0..1, variation across episodes
    texture_variety: float = 0.5        # 0..1, procedural material diversity
    dynamics_noise: float = 0.05        # 0..0.5, actuation disturbance std
    n_waypoints: float = 0.0            # T3: 0=off, else int count of slalom waypoints

    BOUNDS = {
        "obstacle_density": (0.0, 1.0),
        "obstacle_size_mean": (0.5, 5.0),
        "obstacle_size_spread": (0.0, 1.0),
        "corridor_width": (1.0, 10.0),
        "scene_extent": (10.0, 100.0),
        "goal_distance": (5.0, 80.0),
        "light_intensity": (0.2, 1.0),
        "light_direction_entropy": (0.0, 1.0),
        "texture_variety": (0.0, 1.0),
        "dynamics_noise": (0.0, 0.5),
        "n_waypoints": (0.0, 6.0),
    }

    def to_vector(self):
        return [getattr(self, f.name) for f in fields(self)]

    @classmethod
    def from_vector(cls, vec):
        names = [f.name for f in fields(cls)]
        kwargs = {}
        for n, v in zip(names, vec):
            lo, hi = cls.BOUNDS[n]
            kwargs[n] = min(max(float(v), lo), hi)
        return cls(**kwargs)

    def to_json(self):
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s):
        return cls(**json.loads(s))

    @classmethod
    def names(cls):
        return [f.name for f in fields(cls)]
