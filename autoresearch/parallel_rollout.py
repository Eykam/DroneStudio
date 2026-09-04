"""Parallel episode execution for training loops.

The sim is a physics-only headless subprocess per episode, so episodes are
independent and CPU-bound - on the 48-vCPU research box, serial collection
uses ~1 core while the rest idles. This module runs episodes over a
fork-context Pool (default 24 workers; AUTORESEARCH_WORKERS overrides).

fork (not spawn) so workers inherit the already-imported module state and
we never re-execute training-script module-level code.

Zombie hygiene: each episode MUST close() its env (reaps the sim child);
the pool itself is context-managed so workers are joined on exit.
"""
import multiprocessing as mp
import os

DEFAULT_WORKERS = int(os.environ.get("AUTORESEARCH_WORKERS", "24"))


def parallel_episodes(fn, args_list, workers=None):
    """starmap fn over args_list in a fork Pool; returns results in order."""
    if not args_list:
        return []
    w = min(workers or DEFAULT_WORKERS, len(args_list))
    if w <= 1:
        return [fn(*a) for a in args_list]
    ctx = mp.get_context("fork")
    with ctx.Pool(w) as pool:
        return pool.starmap(fn, args_list)
