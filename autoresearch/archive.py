"""Variant archive: JSONL, append-only. Every evaluated distribution lands
here - the outer loop's memory and the user's audit trail."""
import json, time, os

class Archive:
    def __init__(self, path):
        self.path = path
        self.records = []
        if os.path.exists(path):
            with open(path) as f:
                self.records = [json.loads(l) for l in f if l.strip()]

    def add(self, generation, parent_id, dist, metrics, mutator):
        rec = {
            "id": f"g{generation}v{len(self.records)}",
            "parent": parent_id,
            "generation": generation,
            "params": json.loads(dist.to_json()),
            "metrics": metrics,
            "mutator": mutator,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self.records.append(rec)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    def best(self):
        if not self.records:
            return None
        return max(self.records, key=lambda r: (r["metrics"]["success_rate"], r["metrics"]["mean_return"]))

    def summary(self, n=8):
        lines = []
        for r in self.records[-n:]:
            m = r["metrics"]
            lines.append(f"{r['id']} gen={r['generation']} succ={m['success_rate']:.2f} ret={m['mean_return']:.1f}")
        return "\n".join(lines) or "(empty archive)"
