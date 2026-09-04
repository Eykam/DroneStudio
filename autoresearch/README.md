# DroneStudio Auto-Researcher

An outer loop that optimizes the simulator itself: it mutates the procedural
scene distribution, trains a navigation policy under each variant, measures
held-out performance, keeps an archive, and selects elites - generation over
generation, the sim becomes a better teacher.

## Architecture (evaluator-first)

```
outer_loop.py   driver: mutate -> train -> evaluate -> archive -> select
mutate.py       mutation target = SceneDistribution (never policy weights or
                sim code). LLM mutator when AUTORESEARCH_LLM_API_KEY is set,
                heuristic mutator otherwise.
scene_schema.py the distribution: obstacle density/size, corridor width,
                extent, goal distance, lighting, texture variety, dynamics noise
evaluator.py    THE source of truth: variant score = held-out success rate of
                a policy trained from scratch under that variant
policy.py       tiny MLP (10->32->32->2) + CEM trainer (numpy, CPU-scale)
env.py          StubNavEnv (analytic placeholder, kept for A/B)
env_quad.py     QuadNavEnv (DEFAULT): 3D quadrotor on DroneStudio's own flight
                model - his mass/inertia (Drone.zig), his rate-PID gains and
                thrust/rate filters (FlightController.zig), thrust along body
                +Y with torque plant (PhysicsThread.zig). Policy emits desired
                body rates + collective thrust at 20 Hz; his PID tracks them
                at 500 Hz - the same classical-fast-loop + learned-nav-policy
                split the full system will use.
archive.py      JSONL variant archive - the loop's memory + audit trail
run_e2e.py      single-cycle end-to-end test (this is the CI gate)
```

## Running

    python3 -m venv /workspace/venv && /workspace/venv/bin/pip install numpy
    /workspace/venv/bin/python autoresearch/run_e2e.py            # one full cycle
    /workspace/venv/bin/python autoresearch/run_e2e.py --quick    # smaller budget

## What run_e2e proves (and doesn't)

PROVES: the loop machinery closes - mutation, training, held-out evaluation,
archive writes, and elite selection all execute to budget on CPU, and metrics
are sane.
DOES NOT PROVE: rendering/vision (no pixels - obs are state-based),
sim-exact dynamics (quad env replicates his flight model in numpy with
semi-implicit Euler, not Bullet), RL quality (CEM, not PPO), LLM outer loop
(heuristic without an API key).

## LLM outer loop

Set AUTORESEARCH_LLM_API_KEY (+ optional AUTORESEARCH_LLM_BASE_URL,
AUTORESEARCH_LLM_MODEL). Keys come from the environment only, never the repo.

## Next milestones

1. Sim-side headless episode API (reset/seed/step clock), launched under xvfb.
   Renderer blocker found 2026-09-03: fragment shaders use
   GL_ARB_bindless_texture, unsupported by llvmpipe - headless RENDERING on
   CPU needs a non-bindless fallback path, or the GPU box. Dynamics-level
   training is unaffected.
2. Replace StubNavEnv with SimBinaryEnv against that API.
3. torch CPU + PPO (or the transformer policy) once loop is proven at scale.
4. VIO fast control loop under the learned nav policy.
