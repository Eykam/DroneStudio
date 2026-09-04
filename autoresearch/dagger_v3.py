"""DAgger run 3: parallel collection + harder-scene difficulty tiers (Phase 1).

Approved design (2026-09-04): new per-episode difficulty axis from
scenario_sampler (T0 open / T1 dense clutter / T2 2m gap-walls; T3 slalom
waits on obs v3 - the obs carries only the NEAREST obstacle vector, so dense
scenes are partially observable and per-tier numbers are not regressions).

vs dagger_land: episode collection AND held-out eval run over a fork pool
(parallel_rollout.parallel_episodes) - collection was the serial bottleneck
(~1 vCPU of 48). Held-out cells per (scenario x tier), fixed seed blocks
(base + 1000*tier, n=16), so per-tier progress is measurable and stable.

T2 is eval-only in Phase 1 (teacher cannot fly it - see module docstring
history); tiers T0/T1 train. Warm start: bc_ppo_v2_best.json. Checkpoints
every iteration; accept gate on tier-0 floors so harder-scene training can
never silently regress the base scenarios -> /workspace/bc_flat_v3.json.
"""
import sys, json, os, time
sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/DroneStudio/autoresearch")
os.environ["AUTORESEARCH_OBS_V2"] = "1"
import numpy as np
from env_sim import make_sim_factory
from policy import MLP
from diverse_bc import pilot_act3, bc_train
from scenario_sampler import (sample_spec, sample_tier, tier_dist,
                              heldout_cells_tiered, TIERS)
from eval_scenarios import post_series, cell_dist
from parallel_rollout import parallel_episodes

MANIFEST = "/workspace/DroneStudio/autoresearch/fixtures/v14_g13.manifest.json"
START = os.environ.get("V3_START", "/workspace/bc_ppo_v2_best.json")
ITERS = int(os.environ.get("V3_ITERS", "8"))
EPS_PER_ITER = int(os.environ.get("V3_EPS", "32"))
SCEN_MIX = ("land", "land", "hover_hold", "goto")  # land stays oversampled


def _scenario_for(seed):
    rng = np.random.default_rng(np.uint64(seed) ^ np.uint64(0x5C11))
    return SCEN_MIX[rng.integers(0, len(SCEN_MIX))]


def _collect_one(flat, seed):
    """One student rollout with teacher labels (worker)."""
    net = MLP(19, 4, seed=0)
    net.set_flat(np.array(flat, dtype=np.float64))
    sc = _scenario_for(seed)
    # teacher sanity (smoke): T2 gap-walls are beyond the reactive teacher
    # (land 0.0) - training on those labels would inject noise. T2 stays
    # eval-only until obs v3 / a gap-aware teacher; map T2 draws to T0/T1.
    tier = sample_tier(seed)
    if tier == 2:
        tier = seed % 2
    spec = sample_spec(seed, force_scenario=sc)
    dist = tier_dist(seed, tier)
    max_steps = 400 if sc == "goto" else 700
    env = make_sim_factory(dist, max_steps=max_steps, dynamics=MANIFEST,
                           scenario_spec=spec)(seed)
    obs = env.reset()
    traj = []
    for _ in range(max_steps):
        a = pilot_act3(obs, 1.75, dist.scene_extent)
        traj.append((obs.copy(), a.copy()))
        obs, r, done = env.step(net.act(obs))
        if done:
            break
    env.close()
    return traj


def _eval_one(flat, scenario, tier, seed):
    """One held-out episode (worker). Tier-0 cells reuse the legacy
    cell_dist so success_<scenario> stays comparable across runs."""
    net = MLP(19, 4, seed=0)
    net.set_flat(np.array(flat, dtype=np.float64))
    if tier == 0:
        dist = cell_dist(seed)
    else:
        rng = np.random.default_rng(np.uint64(seed) ^ np.uint64(0xE7A1))
        dist = tier_dist(seed, tier, base_rng=rng)
    spec = sample_spec(seed, force_scenario=scenario)
    max_steps = 400 if scenario == "goto" else 700
    env = make_sim_factory(dist, max_steps=max_steps, dynamics=MANIFEST,
                           scenario_spec=spec)(seed)
    obs = env.reset()
    for _ in range(max_steps):
        obs, r, done = env.step(net.act(obs))
        if done:
            break
    ok = bool(env.succeeded)
    env.close()
    return scenario, tier, ok


def eval_tiered(flat):
    cells = heldout_cells_tiered()
    args = [(flat, sc, t, seed) for (sc, t), seeds in cells.items() for seed in seeds]
    out = {}
    for sc, t, ok in parallel_episodes(_eval_one, args):
        out.setdefault((sc, t), []).append(ok)
    return {k: float(np.mean(v)) for k, v in out.items()}


def _teach_one(_flat, sc, tier, seed):
    """Teacher-flown held-out episode (module-level: pool pickles fns)."""
    if tier == 0:
        dist = cell_dist(seed)
    else:
        rng = np.random.default_rng(np.uint64(seed) ^ np.uint64(0xE7A1))
        dist = tier_dist(seed, tier, base_rng=rng)
    spec = sample_spec(seed, force_scenario=sc)
    env = make_sim_factory(dist, max_steps=400 if sc == "goto" else 700,
                           dynamics=MANIFEST, scenario_spec=spec)(seed)
    obs = env.reset()
    done = False
    while not done:
        obs, r, done = env.step(pilot_act3(obs, 1.75, dist.scene_extent))
    ok = bool(env.succeeded)
    env.close()
    return sc, ok


def teacher_sanity():
    """Teacher success per tier before training: if the teacher cannot fly a
    tier, its DAgger labels there are noise - report, don't train blind."""
    for t in TIERS:
        args = [(None, sc, t, seed)
                for sc in ("goto", "hover_hold", "land")
                for seed in heldout_cells_tiered()[(sc, t)][:8]]
        res = parallel_episodes(_teach_one, args)
        by_sc = {}
        for (sc, _), (s2, ok) in zip([(a[1], a[2]) for a in args], res):
            by_sc.setdefault(s2, []).append(ok)
        line = {k: round(float(np.mean(v)), 3) for k, v in by_sc.items()}
        print(f"TEACHER_T{t}: " + json.dumps(line), flush=True)
        for sc, v in line.items():
            post_series(f"success_{sc}_t{t}", v, label="teacher")


def main():
    t0 = time.time()
    net = MLP(19, 4, seed=0)
    net.set_flat(np.array(json.load(open(START)), dtype=np.float64))
    print(f"DAGGERV3 start={os.path.basename(START)} iters={ITERS} eps/iter={EPS_PER_ITER}", flush=True)
    teacher_sanity()
    flat = list(net.get_flat())
    res = eval_tiered(flat)
    print("DAGGERV3 iter0 (start net): " + json.dumps({f"{s}t{t}": round(v, 3) for (s, t), v in sorted(res.items())}), flush=True)
    for (sc, t), v in sorted(res.items()):
        post_series(f"success_{sc}_t{t}" if t else f"success_{sc}", v,
                    label=f"dagger-v3 i0{tier_note(t)}")
    X, Y = [], []
    best_flat = list(flat)
    best_score = np.mean([res[("goto", 0)], res[("hover_hold", 0)], res[("land", 0)]])
    cur = list(flat)
    for it in range(1, ITERS + 1):
        tc = time.time()
        # collect against the CURRENT net each iteration
        args = [(list(cur), 9000 + it * 100 + j) for j in range(EPS_PER_ITER)]
        trajs = parallel_episodes(_collect_one, args)
        for tr in trajs:
            for o, a in tr:
                X.append(o); Y.append(a)
        if len(X) > 24000:
            X, Y = X[-24000:], Y[-24000:]
        net = bc_train(np.array(X), np.clip(np.array(Y), -0.95, 0.95),
                       iters=1500, obs_dim=19, init_flat=cur)
        cur = list(net.get_flat())
        json.dump(cur, open(f"/workspace/bc_flat_v3_i{it}.json", "w"))
        res = eval_tiered(cur)
        line = {f"{s}t{t}": round(v, 3) for (s, t), v in sorted(res.items())}
        print(f"DAGGERV3_ITER {it}: n={len(X)} collect+eval wall={time.time()-tc:.0f}s " + json.dumps(line), flush=True)
        for (sc, t), v in sorted(res.items()):
            post_series(f"success_{sc}_t{t}" if t else f"success_{sc}", v,
                        label=f"dagger-v3 i{it}{tier_note(t)}")
        score = np.mean([res[("goto", 0)], res[("hover_hold", 0)], res[("land", 0)]])
        if (score > best_score and res[("goto", 0)] >= 0.9 and res[("hover_hold", 0)] >= 0.8):
            best_score = score
            best_flat = list(cur)
            json.dump(best_flat, open("/workspace/bc_flat_v3.json", "w"))
            print(f"DAGGERV3_SAVE iter={it} t0mean={score:.3f}", flush=True)
    print(f"DAGGERV3_DONE best_t0mean={best_score:.3f} wall={time.time()-t0:.0f}s", flush=True)


def tier_note(t):
    return "" if t == 0 else " (partial obs)"


if __name__ == "__main__":
    main()
