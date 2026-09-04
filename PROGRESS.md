## Run 2 (headless Zig sim inner loop) - 2026-09-04

20 generations, 0.73 h wall, backend=quad (dronestudio-headless ReleaseFast),
trainer=cem, budget cem_iters=4, cem_pop=10, train_episodes=4, eval_episodes=6.

Outcome: best variant g13v140 (gen 13), eval success 33.3% (2/6 episodes),
mean return -0.81, train best return +3.80. Lineage g12v129 -> g13v140 (mutator none).

Best scene distribution:
  obstacle_density 0.018, obstacle_size 0.75 +- 0.08, corridor_width 10,
  scene_extent 22, goal_distance 5, light_intensity 0.92,
  light_direction_entropy 0.35, texture_variety 0.55, dynamics_noise 0.0

Selection dynamics:
- First success at gen 2 (17%), plateau 17% through gens 2-12 while diversity
  oscillated 0.28-0.95 (two diversity spikes at g7/g8 did not convert).
- Jump to 33% at gen 13, then stagnant 3 of the last 6 gens (15, 16, 19);
  diversity collapsed to 0.16 at gen 15 - the archive was exploiting, not
  exploring, when the run ended. A longer run may or may not have escaped;
  the stagnation counter was resetting on near-misses without new success.
- LLM spend: 0 calls (ChatGPT throttle demoted the outer loop to heuristic
  mutation this run - heuristic still found the gen-13 improvement, but the
  last-mile exploration was purely heuristic).

Follow-on: PPO-vs-CEM comparison launched on the g13v140 distribution
(autoresearch/ppo_vs_cem.json, compare.log on the box) - first real test of
whether PPO beats CEM on the discovered distribution before the inner loop
swaps to PPO by default.

## PPO vs CEM on run-2 best dist (g13v140) - 2026-09-04

compare_trainers.py, backend=sim (headless Zig), eval seed 20000, 6 episodes, 250 max steps.

  CEM (iters 6, pop 12): 0.6s wall, train best return +0.81, eval success 0/6, mean return -7.16
  PPO (hidden 128, lr 3e-4, rollout 8 x 80 updates, 62k env steps): 10.1s wall,
    eval success 0/6, mean return -36.03

Reading: 0/6 for both is not conclusive against the 33.3% (2/6) run-2 eval -
small-sample noise (P(0/6 | p=1/3) ~ 9%) plus a different eval seed and
from-scratch retrains. The load-bearing finding is PPO's training curve:
eval mean went -16.9 (update 20) -> -125.3 (40) -> -109.4 (60) -> -203.5 (80).
PPO did not just fail to learn, it diverged monotonically after update 20.
At return magnitudes in the hundreds, lr 3e-4 without return scaling looks
too hot; candidates: lower lr, advantage/return normalization, value-loss
clipping, or smaller epochs-per-rollout. Until that is stable, CEM stays the
default inner-loop trainer; PPO needs a hyperparameter stabilization pass
before the swap mandated in the steering note.
