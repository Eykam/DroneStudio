"""Scenario sampler: per-episode task spec drawn deterministically from the
episode seed, so training rollouts and held-out eval blocks are stable and
reproducible.

Scenarios (mix): goto 50% / hover_hold 25% / land 25%.
success_radius is sampled per episode, range by scenario:
  goto 1.5-3.0m (nav legs), hover_hold 0.5-1.5m, land 0.3-0.6m (precision).
Log-uniform-ish: uniform in log space so tight radii are well represented.

Held-out eval blocks (never trained on): goto 77000+, hover_hold 88000+,
land 99000+, n=16 per cell. Eval forces the scenario and samples the rest
of the spec from the block seed, keeping cells comparable across runs.

Sim side: headless_main.zig consumes scenario/success_radius/hold_s/
max_touchdown_vs in the reset scene JSON (defaults = old goto behavior).
"""
import numpy as np

SCENARIOS = ("goto", "hover_hold", "land")
MIX = (0.50, 0.25, 0.25)
RADIUS_RANGE = {"goto": (1.5, 3.0), "hover_hold": (0.5, 1.5), "land": (0.3, 0.6)}
HOLD_S = 4.0
# Variable hover hold time (user ask 2026-09-04: "navigate to target and wait
# for 1 min"). Sampled per-episode on its own stream so scenario/radius draws
# stay put; log-uniform so short holds (cheap) dominate but 60s holds appear.
HOLD_S_RANGE = (2.0, 60.0)
MAX_TOUCHDOWN_VS = 0.5

HELDOUT_BASE = {"goto": 77000, "hover_hold": 88000, "land": 99000}
HELDOUT_N = 16

# Harder-scenes difficulty axis (user directive 2026-09-04). Phase 1 shipped
# T0-T2; Phase 2 adds T3 (waypoint slalom, alternating 0.25-0.45 leg-length
# lateral offsets force 120-180 deg turns) on obs v3 (26-dim). Dense scenes
# remain partially observable (nearest-obstacle vector only), so per-tier
# numbers must not be read as regressions.
TIERS = (0, 1, 2, 3)
TIER_MIX = (0.4, 0.3, 0.2, 0.1)   # approved 40/30/20/10; T3 needs obs v3 (26-dim)
HELDOUT_TIER_STRIDE = 1000


def _rng(seed):
    # scenario stream must not shift when scene sampling changes
    return np.random.default_rng(np.uint64(seed) ^ np.uint64(0x5CE0))


def hover_max_steps(hold_s, tier):
    # transit budget (700 policy steps @20Hz covered 4s holds) + the extra hold
    # time with 25% margin; 60s hold -> ~2100 steps
    return int(700 + max(0.0, hold_s - 4.0) * 25)


def sample_spec(seed, force_scenario=None, hold_s=None):
    """Deterministic scenario spec for an episode seed."""
    rng = _rng(seed)
    if force_scenario is None:
        scenario = SCENARIOS[rng.choice(len(SCENARIOS), p=MIX)]
    else:
        scenario = force_scenario
    lo, hi = RADIUS_RANGE[scenario]
    # uniform in log space
    radius = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    spec = {
        "scenario": scenario,
        "success_radius": radius,
    }
    if scenario == "hover_hold":
        hrng = np.random.default_rng(np.uint64(seed) ^ np.uint64(0x401D))
        lo, hi = HOLD_S_RANGE
        spec["hold_s"] = hold_s if hold_s is not None else float(np.exp(hrng.uniform(np.log(lo), np.log(hi))))
    if scenario == "land":
        spec["max_touchdown_vs"] = MAX_TOUCHDOWN_VS
    return spec


def heldout_cells():
    """{scenario: [seeds]} fixed held-out blocks for comparable evals."""
    return {s: [HELDOUT_BASE[s] + i for i in range(HELDOUT_N)] for s in SCENARIOS}


def sample_tier(seed):
    """Deterministic difficulty tier per episode seed (own stream)."""
    rng = np.random.default_rng(np.uint64(seed) ^ np.uint64(0x7E12))
    return int(rng.choice(len(TIERS), p=TIER_MIX))


def tier_dist(seed, tier, base_rng=None):
    """SceneDistribution for a tier. T0 = today's open field; T1 = dense
    clutter; T2 = gap-walls (corridor_width 2.0 triggers the generator)."""
    from scene_schema import SceneDistribution
    r = base_rng or np.random.default_rng(np.uint64(seed) ^ np.uint64(0xD1A9))
    gd = float(r.uniform(2.0, 25.0))
    if tier == 2:
        gd = max(gd, 10.0)  # need leg room for walls
    if tier == 3:
        gd = max(gd, 12.0)  # need leg room for the slalom
    return SceneDistribution(
        obstacle_density=(float(r.uniform(0.05, 0.15)) if tier == 3 else
                          float(r.choice([0.0, 0.05, 0.1, 0.2])) if tier == 0
                          else float(r.uniform(0.25, 0.40)) if tier == 1
                          else float(r.choice([0.05, 0.1]))),
        corridor_width=4.0 if tier < 2 else 2.0,
        obstacle_size_mean=2.0 if tier < 1 else float(r.uniform(2.0, 3.0)),
        scene_extent=max(10.0, gd * 2), goal_distance=gd,
        n_waypoints=(float(r.integers(2, 5)) if tier == 3 else 0.0),
        light_direction_entropy=0.3, texture_variety=0.0, dynamics_noise=0.0)


def heldout_cells_tiered():
    """{(scenario, tier): [seeds]} fixed held-out blocks per difficulty tier.
    Tier cells live at base + 1000*tier; tier 0 == heldout_cells()."""
    out = {}
    for s, base in HELDOUT_BASE.items():
        for t in TIERS:
            out[(s, t)] = [base + HELDOUT_TIER_STRIDE * t + i for i in range(HELDOUT_N)]
    return out
