"""Auto-researcher mutation loop: Codex mutates chassis.py -> evaluator gates -> commit + dashboard.

One invocation = one generation (designed to be driven by a cron/supervisor or repeated calls).
State: loop_state.json in this dir (best variant, score, history).
"""
import os, sys, json, subprocess, shutil, dataclasses, time
import progress

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "loop_state.json")
ARCHIVE = os.path.join(HERE, "archive.jsonl")

PROMPT_TEMPLATE = """You are the mutation brain of an auto-researcher optimizing a 3D-printed 5-inch FPV quad chassis.

Context:
- chassis.py in this directory holds the parametric model (build123d). ChassisParams + build_chassis(p) -> Part MUST keep their signatures.
- evaluate.py scores candidates: watertight, single body, FDM overhang, wall thickness, prop clearance, hover margin, and FEA (gmsh+CalculiX: hover-max and crash load cases; pass needs max von Mises < 30 MPa AND arm-tip displacement < 5 mm).
- The sim contract (DroneStudio): quad-X, motor arm length 0.15 m, 17.27 N/motor max (EMAX ECO II 2207 2400KV on 4S). Do NOT change arm_length_mm or motor hole pattern; the sim depends on them. Everything else is fair game: arm cross-section, ribs, fillets, plate shapes, cutouts, material distribution.
- DESIGN LANGUAGE: read INSPIRATION.md in this directory first. The user wants the chassis to move beyond flat-plate freestyle frames toward DJI/Anduril-style form factors - faired enclosed body, swept/tapered arms with filleted roots, closed arm sections, recessed battery bay - while staying single-body, printable, and serviceable.
- Print process: FDM, PETG (rho 1240 kg/m3, E 2.1 GPa, yield 50 MPa), no supports, flat on the plate.

Current best: variant {best_variant}, score {best_score:.3f}
Its evaluation failures: {failures}
History of tried mutations and scores: {history}

Your job: edit chassis.py (and ONLY chassis.py) to make the chassis pass ALL checks with the lowest frame mass. Stiffness is the binding constraint: add ribs/depth/taper rather than uniform thickness where you can. Push the DESIGN LANGUAGE too: each generation should visibly advance the DJI/Anduril-inspired form factor from INSPIRATION.md (faired shell, tapered swept arms, closed sections, recessed bays) - incremental but real steps, never just a parameter nudge. Keep it printable (no support-requiring overhangs, walls >= 1.2 mm).
After editing, verify the model still builds: run `python3 -c "from chassis import ChassisParams, build_chassis; p=ChassisParams(); part=build_chassis(p); print(round(part.volume,1))"`.
End by printing one line: MUTATION_SUMMARY <one sentence describing what you changed>."""

def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=HERE, **kw)

def load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"generation": 0, "best_variant": "v1-baseline", "best_score": 0.5, "best_commit": None, "history": []}

def save_state(s):
    json.dump(s, open(STATE, "w"), indent=2)

def run_generation():
    st = load_state()
    gen = st["generation"] + 1
    variant = f"v{gen+1}-g{gen}"
    parent = st["best_variant"]
    # snapshot parent code for rollback
    parent_backup = f"/tmp/chassis_parent_{variant}.py"
    shutil.copy(os.path.join(HERE, "chassis.py"), parent_backup)
    parent_metrics = os.path.join(HERE, "snapshots", parent, "metrics.json")
    failures = "unknown"
    if os.path.exists(parent_metrics):
        pm = json.load(open(parent_metrics))
        failures = json.dumps([c for c in pm["evaluator"] if not c["passed"]], indent=2)
    hist = "; ".join(f"{h['variant']}={h['score']:.3f} ({h.get('summary','')[:60]})" for h in st["history"][-8:]) or "none yet"
    prompt = PROMPT_TEMPLATE.format(best_variant=parent, best_score=st["best_score"], failures=failures, history=hist)
    progress.set_stage("codex editing", f"gen {gen}: codex mutating {parent} -> {variant}", design_id=f"cad-chassis-{variant}")
    print(f"[gen {gen}] asking Codex for mutation of {parent}...", flush=True)
    r = subprocess.run(["codex", "exec", "--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox", "-o", "/tmp/codex_last.md", prompt],
                       capture_output=True, text=True, cwd=HERE, timeout=1800)
    summary = ""
    if os.path.exists("/tmp/codex_last.md"):
        for line in open("/tmp/codex_last.md"):
            if "MUTATION_SUMMARY" in line:
                summary = line.split("MUTATION_SUMMARY", 1)[1].strip()
    if r.returncode != 0:
        print(f"[gen {gen}] codex failed: {r.stderr[-300:]}", flush=True)
        return None
    # evaluate candidate
    progress.set_stage("evaluating", f"gen {gen}: building + evaluating {variant}")
    r2 = subprocess.run([sys.executable, "run_candidate.py", "--variant", variant, "--parent", parent, "--gen", str(gen)],
                        capture_output=True, text=True, cwd=HERE, env={**os.environ, "RUN_FEA": "1"}, timeout=1800)
    print(r2.stdout[-800:], flush=True)
    score = None
    for line in r2.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            score = json.loads(line[len("RESULT_JSON "):])["score"]
    if score is None:
        print(f"[gen {gen}] candidate failed to evaluate; rolling back", flush=True)
        shutil.copy(parent_backup, os.path.join(HERE, "chassis.py"))
        return None
    cand_mass = None
    try:
        pm = json.load(open(os.path.join(HERE, "snapshots", variant, "metrics.json")))
        cand_mass = (pm.get("metrics") or {}).get("mass_g") or (pm.get("mass") or {}).get("frame_mass_g")
    except Exception:
        pass
    best_mass = st.get("best_mass_g")
    improved = score > st["best_score"] or (score == st["best_score"] and cand_mass and (best_mass is None or cand_mass < best_mass))
    if not improved:
        shutil.copy(parent_backup, os.path.join(HERE, "chassis.py"))
        print(f"[gen {gen}] {variant} score {score:.3f} <= best {st['best_score']:.3f}; rolled back chassis.py (record kept)", flush=True)
    else:
        st["best_variant"], st["best_score"] = variant, score
    st["generation"] = gen
    st["history"].append({"variant": variant, "parent": parent, "score": score, "improved": improved, "summary": summary})
    # commit candidate artifacts + (possibly rolled-back) code state
    sh('git add -A && git -c user.name="Instinct Chassis Researcher" -c user.email="instinct-chassis-researcher@users.noreply.github.com" commit -q -m "gen %d: %s score %.3f (parent %s)%s - %s" && git push -q origin chassis' % (
        gen, variant, score, parent, " NEW BEST" if improved else " rolled-back", summary[:120] or "no summary"))
    with open(ARCHIVE, "a") as f:
        f.write(json.dumps({"id": f"cad-gen-{gen}", "kind": "cad.chassis.generation", "variant": variant,
                            "parent": parent, "score": score, "improved": improved, "summary": summary,
                            "ts": time.time()}) + "\n")
    save_state(st)
    print(f"[gen {gen}] done: {variant} score {score:.3f} ({'NEW BEST' if improved else 'no improvement'})", flush=True)
    return score

if __name__ == "__main__":
    progress.start_heartbeat()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    for _ in range(n):
        run_generation()
    progress.idle(f"batch done: best {load_state()['best_variant']}")
