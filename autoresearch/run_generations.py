#!/usr/bin/env python3
"""Multi-generation auto-researcher run. Writes a JSON report when done so it
can run unattended (nohup) and be collected later.

Usage: python run_generations.py [--generations N] [--children K] [--seed S]
                                 [--quick] [--report PATH]
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from outer_loop import run
from mutate import spend_status

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generations", type=int, default=6)
    p.add_argument("--children", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--report", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "generations_report.json"))
    a = p.parse_args()
    budget = dict(cem_iters=3, cem_pop=8, train_episodes=3, eval_episodes=5, max_steps=200) if a.quick \
        else dict(cem_iters=4, cem_pop=10, train_episodes=4, eval_episodes=6, max_steps=250)
    t0 = time.time()
    best = run(generations=a.generations, children=a.children, seed=a.seed,
               budget=budget,
               archive_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "archive.jsonl"))
    report = {
        "elapsed_s": round(time.time() - t0, 1),
        "generations": a.generations, "children": a.children, "seed": a.seed,
        "budget": budget, "best": best, "llm_spend": spend_status(),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(a.report, "w") as f:
        json.dump(report, f, indent=2)
    print("RUN_COMPLETE", a.report)
    return 0

if __name__ == "__main__":
    sys.exit(main())
