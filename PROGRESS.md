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
