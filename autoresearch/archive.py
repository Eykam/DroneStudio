"""Variant archive: JSONL, append-only. Every evaluated distribution lands
here - the outer loop's memory and the user's audit trail.

Intelligence layer (all schema additions are optional fields, old records
stay valid):
  - novelty: distance of a candidate to the nearest archived variant, in
    bounds-normalized parameter space. Selection pays a small novelty
    bonus so the loop does not collapse onto one hill.
  - elites(k): top-k distinct variants (multi-parent selection instead of
    single-lineage hill climbing).
  - diversity(): mean pairwise distance across recent records - the
    archive's health metric; falling diversity = premature convergence.
  - trajectory(): per-generation best success rate, for stagnation
    detection and the night report.
  - richer summary(): this text is the LLM mutator's only view of its own
    history, so it carries elite params, diversity, and mutator win rates.
"""
import json, time, os
import itertools

class Archive:
    def __init__(self, path):
        self.path = path
        self.records = []
        if os.path.exists(path):
            with open(path) as f:
                self.records = [json.loads(l) for l in f if l.strip()]

    # ---- schema ----------------------------------------------------------
    def add(self, generation, parent_id, dist, metrics, mutator, novelty=None):
        rec = {
            "id": f"g{generation}v{len(self.records)}",
            "parent": parent_id,
            "generation": generation,
            "params": json.loads(dist.to_json()),
            "metrics": metrics,
            "mutator": mutator,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if novelty is not None:
            rec["novelty"] = round(float(novelty), 4)
        self.records.append(rec)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    # ---- distances -------------------------------------------------------
    @staticmethod
    def _norm_vec(params):
        from scene_schema import SceneDistribution
        v = []
        for n in SceneDistribution.names():
            lo, hi = SceneDistribution.BOUNDS[n]
            span = (hi - lo) or 1.0
            v.append((params.get(n, lo) - lo) / span)
        return v

    def novelty(self, dist):
        """Min normalized Euclidean distance to any archived variant."""
        if not self.records:
            return 1.0
        import math
        v = self._norm_vec(json.loads(dist.to_json()))
        best = None
        for r in self.records:
            w = self._norm_vec(r["params"])
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(v, w)))
            best = d if best is None else min(best, d)
        return best

    def diversity(self, last_n=20):
        """Mean pairwise normalized distance over recent records."""
        import math
        recs = self.records[-last_n:]
        if len(recs) < 2:
            return 0.0
        vecs = [self._norm_vec(r["params"]) for r in recs]
        ds = []
        for a, b in itertools.combinations(vecs, 2):
            ds.append(math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))))
        return sum(ds) / len(ds)

    # ---- selection -------------------------------------------------------
    @staticmethod
    def _score(r):
        m = r["metrics"]
        return (m.get("success_rate", 0.0), m.get("mean_return", float("-inf")))

    def best(self):
        if not self.records:
            return None
        return max(self.records, key=self._score)

    def elites(self, k=2):
        """Top-k variants, deduped by rounded params (distinct hills)."""
        out = []
        seen = set()
        for r in sorted(self.records, key=self._score, reverse=True):
            key = tuple(round(x, 2) for x in self._norm_vec(r["params"]))
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
            if len(out) >= k:
                break
        return out

    def trajectory(self):
        """Per-generation best (success_rate, mean_return) pairs."""
        gens = {}
        for r in self.records:
            g = r["generation"]
            cur = gens.get(g)
            if cur is None or self._score(r) > self._score(cur):
                gens[g] = r
        return {g: {"id": r["id"], "success_rate": r["metrics"]["success_rate"],
                    "mean_return": r["metrics"]["mean_return"],
                    "mutator": r.get("mutator", "?")}
                for g, r in sorted(gens.items())}

    def lineage(self, rec_id):
        by_id = {r["id"]: r for r in self.records}
        chain = []
        cur = by_id.get(rec_id)
        while cur is not None:
            chain.append(cur["id"])
            cur = by_id.get(cur.get("parent"))
        return list(reversed(chain))

    # ---- LLM-facing summary ----------------------------------------------
    def summary(self, n=8):
        lines = []
        traj = self.trajectory()
        if traj:
            lines.append("Per-generation best: " + ", ".join(
                f"g{g}:{t['success_rate']:.2f}({t['mutator']})" for g, t in traj.items()))
            lines.append(f"Archive diversity (recent, 0-1 normalized): {self.diversity():.3f}")
        wins = {}
        for r in self.records:
            wins[r.get("mutator", "?")] = wins.get(r.get("mutator", "?"), 0) + (1 if self._score(r) == self._score(self.best()) else 0)
        for r in self.records[-n:]:
            m = r["metrics"]
            nov = f" nov={r['novelty']:.2f}" if "novelty" in r else ""
            lines.append(f"{r['id']} gen={r['generation']} succ={m['success_rate']:.2f} "
                         f"ret={m['mean_return']:.1f}{nov} params={json.dumps(r['params'])}")
        return "\n".join(lines) or "(empty archive)"
