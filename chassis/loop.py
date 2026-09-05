"""Auto-researcher mutation loop: Codex mutates chassis.py -> evaluator gates -> commit + dashboard.

One invocation = one generation (designed to be driven by a cron/supervisor or repeated calls).
State: loop_state.json in this dir (best variant, score, history).
"""
import os, sys, json, subprocess, shutil, dataclasses, time, glob
import progress

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "loop_state.json")
ARCHIVE = os.path.join(HERE, "archive.jsonl")
def codex_model_args():
    """Model is file-driven: /work/codex_model.txt holds the model name; empty or 'default' = codex default.
    The astra recovery probe flips this file live when astra passes a realistic long-session probe."""
    try:
        m = open("/work/codex_model.txt").read().strip()
    except OSError:
        m = ""
    return [] if m in ("", "default") else ["-m", m]


LETTERS = ("a", "b")  # candidates per generation (dropped 3->2: shorter astra sessions, hang mitigation)

PROMPT_TEMPLATE = """You are the mutation brain of an auto-researcher optimizing a 3D-printed 5-inch FPV quad chassis.

Context:
- chassis.py in this directory holds the parametric model (build123d). ChassisParams + build_chassis(p) -> Part MUST keep their signatures.
- evaluate.py scores candidates: watertight, single body, FDM overhang, wall thickness, prop clearance, hover margin, CONTAINMENT, CAMERA FOV, IMU LEVER ARM, and FEA.
- CAMERA FOV (hard gate, user requirement): both Pi cameras mount lens-FORWARD (+X) and see out the nose apertures: no frame material inside the FOV pyramid (66.3h x 41.6v deg). chassis.py cuts the sight-line voids from components.camera_lens_poses() - keep that coupling; cameras in placement.json stay lens-forward at the nose.
- IMU LEVER ARM (gate): the mpu9250 must sit within 60 mm of the frame CoM (on/near the FC stack). The manifest exports its full transform (position + rotation + offset from CoM, schema 1.2) for the estimator.
- CONTAINMENT (hard gate): every placed component (stack, battery, Pi, cameras, GPS, IMU) must sit FULLY INSIDE the frame envelope plus 2mm service clearance. The frame must keep an enclosed fuselage/deck cavity tall enough to swallow them OR recess bays; components.py placement() says where they sit (placement.json may be edited to move them - x/y/z in meters, z = component bottom). Enclose like DJI/Avata: battery and stack INSIDE the body, GPS in a rear fin/perch cavity, camera behind the nose shell with only the lens aperture forward.
- FEA (gmsh+CalculiX, all must pass): hover_max (4x 11.0N up), crash (3x), torsion (diagonal pairs +/-1.5x yaw torque), pullout (single motor pad 1x), cartwheel (2x lateral side impact on one motor pad), battery_eject (30g x 0.18kg forward at the tray), modal (first eigenmode >= 120 Hz), buckling (factor >= 2.0), fatigue screen. Directional allowables (FFF anisotropy): vertical-load cases 0.6*0.55*yield = 16.5 MPa, torsion 0.6*0.8*yield = 24 MPa; arm-tip displacement < 5 mm.
- The sim contract (DroneStudio): quad-X, motor arm length 0.15 m, 11.0 N/motor max (AKK RS2205 2300KV bench peak on 5045-class props, 4S). Do NOT change arm_length_mm or motor hole pattern; the sim depends on them. Everything else is fair game: arm cross-section, ribs, fillets, plate shapes, cutouts, material distribution.
- DESIGN LANGUAGE: read INSPIRATION.md in this directory first. The user wants the chassis to move beyond flat-plate freestyle frames toward DJI/Anduril-style form factors - faired enclosed body, swept/tapered arms with filleted roots, closed arm sections, recessed battery bay - while staying single-body, printable, and serviceable.
- Print process: FDM, PETG (rho 1240 kg/m3, E 2.1 GPa, yield 50 MPa), no supports, flat on the plate.

Current best: variant {best_variant}, score {best_score:.3f}
Its evaluation failures: {failures}
History of tried mutations and scores: {history}

Your job: propose TWO DISTINCT candidate mutations of the current design, each aimed at passing ALL checks with the lowest frame mass. Stiffness is the binding constraint: add ribs/depth/taper rather than uniform thickness where you can. Keep every candidate printable (no support-requiring overhangs, walls >= 1.2 mm).
Write each candidate as a COMPLETE drop-in replacement of chassis.py (same interface: ChassisParams dataclass with the same constructor, build_chassis(p) -> Part) at these exact paths:
  /tmp/candidates/{base_variant}a.py
  /tmp/candidates/{base_variant}b.py
Do NOT edit chassis.py itself - treat it as the read-only reference implementation of the current best. Read it first.
The two candidates must take genuinely different design directions (e.g. one reworks the fuselage/shell strategy, the other reworks arm sections/roots or mass distribution) - NOT two parameter nudges of one idea. Each should visibly advance the DJI/Anduril-inspired form factor from INSPIRATION.md - incremental but real steps.
placement.json may be edited to move components (the edit is shared by all three candidates; x/y/z in meters, z = component bottom).
Verify EACH candidate builds before finishing. For each file F in /tmp/candidates/{base_variant}a.py /tmp/candidates/{base_variant}b.py run:
  python3 -c "import importlib.util,sys; s=importlib.util.spec_from_file_location(sys.argv[1],sys.argv[2]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); p=m.ChassisParams(); part=m.build_chassis(p); print(round(part.volume,1))" F F
(a file named e.g. v18-g17a.py is not importable as a module name - the importlib form above is required)
End by printing two lines:
MUTATION_SUMMARY_A <one sentence describing candidate a>
MUTATION_SUMMARY_B <one sentence describing candidate b>"""

def kill_codex(proc=None):
    """Kill only OUR codex session's process group (Popen uses start_new_session=True).
    Global /proc sweeps are unsafe now: the astra probe daemon shares this box."""
    if proc is not None:
        try:
            os.killpg(proc.pid, 9)
            return
        except Exception:
            pass
        try:
            proc.kill()
        except Exception:
            pass

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
    base = f"v{gen+1}-g{gen}"
    parent = st["best_variant"]
    parent_backup = f"/tmp/chassis_parent_{base}.py"
    shutil.copy(os.path.join(HERE, "chassis.py"), parent_backup)
    parent_metrics = os.path.join(HERE, "snapshots", parent, "metrics.json")
    failures = "unknown"
    if os.path.exists(parent_metrics):
        pm = json.load(open(parent_metrics))
        failures = json.dumps([c for c in pm["evaluator"] if not c["passed"]], indent=2)
    hist = "; ".join(f"{h['variant']}={h['score']:.3f} ({h.get('summary','')[:60]})" for h in st["history"][-8:]) or "none yet"
    os.makedirs("/tmp/candidates", exist_ok=True)
    for L in LETTERS:
        p = f"/tmp/candidates/{base}{L}.py"
        if os.path.exists(p):
            os.remove(p)
    prompt = PROMPT_TEMPLATE.format(best_variant=parent, best_score=st["best_score"],
                                    failures=failures, history=hist, base_variant=base)
    progress.set_stage("codex editing", f"gen {gen}: codex drafting 2 candidates from {parent}", design_id=f"cad-chassis-{base}")
    print(f"[gen {gen}] asking Codex for 2 candidate mutations of {parent}...", flush=True)
    codex_cmd = ["codex", "exec", *codex_model_args(), "--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox", "-o", "/tmp/codex_last.md", prompt]
    try:
        # Popen + poll instead of blocking run(): adds a stall watchdog. Two live
        # hangs showed the same signature - codex finishes tool work (candidates
        # written + validated), then the model response never arrives and the
        # process sits silent for 40+ minutes. Treat 20 min without any file
        # activity in /tmp/candidates or the repo top level as a hang: kill and
        # take the timeout path (which salvages written candidates).
        # stdout/stderr to FILES, not PIPE: an undrained 64KB pipe buffer blocks codex
        # mid-session (zero file activity -> watchdog kill). Latent hang source.
        codex_out = open(f"/tmp/codex_gen{gen}.out", "w")
        codex_err = open(f"/tmp/codex_gen{gen}.err", "w")
        proc = subprocess.Popen(codex_cmd, stdout=codex_out, stderr=codex_err, text=True, cwd=HERE, start_new_session=True)
        hard_deadline = time.time() + 3600
        gen_start = time.time()
        while proc.poll() is None:
            now = time.time()
            if now > hard_deadline:
                kill_codex(proc); proc.wait()
                raise subprocess.TimeoutExpired(codex_cmd, 3600)
            latest = gen_start  # a session that never writes runs to the hard timeout
            # codex session rollout files count as activity too: sessions can sit
            # 15+ min pre-rollout at startup (API stall), then work fine - killing
            # them at the 20-min mark wastes the work that just started.
            for rf in glob.glob("/root/.codex/sessions/*/*/*/*.jsonl"):
                try:
                    latest = max(latest, os.path.getmtime(rf))
                except OSError:
                    pass
            for root, _, files in os.walk("/tmp/candidates"):
                for fn in files:
                    try:
                        latest = max(latest, os.path.getmtime(os.path.join(root, fn)))
                    except OSError:
                        pass
            for fn in os.listdir(HERE):
                if fn.endswith((".py", ".json")):
                    try:
                        latest = max(latest, os.path.getmtime(os.path.join(HERE, fn)))
                    except OSError:
                        pass
            if now - latest > 1200:
                print(f"[gen {gen}] codex stalled: no file activity for 20 min; killing session", flush=True)
                kill_codex(proc); proc.wait()
                raise subprocess.TimeoutExpired(codex_cmd, 3600)
            time.sleep(30)
        codex_out.close(); codex_err.close()
        r = subprocess.CompletedProcess(codex_cmd, proc.returncode, "", open(f"/tmp/codex_gen{gen}.err").read())
    except subprocess.TimeoutExpired:
        kill_codex(proc)  # process-group kill: takes the node orphans with it
        salvaged = [p for p in (f"/tmp/candidates/{base}{L}.py" for L in LETTERS) if os.path.exists(p)]
        if salvaged:
            # codex hung AFTER writing candidates (seen live: 40+ min no file activity) -
            # evaluate what it left rather than burning the whole generation.
            print(f"[gen {gen}] codex timed out after 3600s; salvaging {len(salvaged)} written candidates", flush=True)
            progress.set_stage("codex timeout", f"gen {gen}: codex timed out, salvaging {len(salvaged)} candidates", design_id=f"cad-chassis-{base}")
            summaries = {}
            r = subprocess.run(["true"], capture_output=True, text=True)  # rc 0: fall through to evaluation
        else:
            print(f"[gen {gen}] codex timed out after 3600s; skipping generation", flush=True)
            progress.set_stage("codex timeout", f"gen {gen}: codex timed out, skipping", design_id=f"cad-chassis-{base}")
            return None
    summaries = {}
    if os.path.exists("/tmp/codex_last.md"):
        for line in open("/tmp/codex_last.md"):
            for L in LETTERS:
                tag = "MUTATION_SUMMARY_" + L.upper()
                if tag in line:
                    summaries[L] = line.split(tag, 1)[1].strip()
    if r.returncode != 0:
        print(f"[gen {gen}] codex failed: {r.stderr[-300:]}", flush=True)
        return None
    cand_paths = {L: f"/tmp/candidates/{base}{L}.py" for L in LETTERS}
    cand_paths = {L: p for L, p in cand_paths.items() if os.path.exists(p)}
    if not cand_paths:
        print(f"[gen {gen}] codex produced no candidate files in /tmp/candidates; skipping generation", flush=True)
        return None
    # evaluate candidates SEQUENTIALLY: each eval already runs 8 ccx jobs in
    # parallel, and 8 concurrent ccx is the proven memory ceiling on this box.
    results = {}
    for L, path in cand_paths.items():
        variant = base + L
        progress.set_stage("evaluating", f"gen {gen}: building + evaluating {variant}", design_id=f"cad-chassis-{variant}")
        try:
            r2 = subprocess.run([sys.executable, "-u", "run_candidate.py", "--variant", variant, "--parent", parent,
                                 "--gen", str(gen), "--source", path],
                                capture_output=True, text=True, cwd=HERE,
                                env={**os.environ, "RUN_FEA": "1", "SKIP_PUBLISH": "1"}, timeout=1800)
        except subprocess.TimeoutExpired:
            print(f"[gen {gen}] {variant} evaluation timed out after 1800s; skipped", flush=True)
            continue
        print(r2.stdout[-800:], flush=True)
        score = None
        for line in r2.stdout.splitlines():
            if line.startswith("RESULT_JSON "):
                score = json.loads(line[len("RESULT_JSON "):])["score"]
        if score is None:
            print(f"[gen {gen}] {variant} failed to evaluate; skipped", flush=True)
            continue
        check_lines = [l for l in r2.stdout.splitlines() if l.strip().startswith(("[PASS]", "[FAIL]"))]
        all_pass = bool(check_lines) and all("[PASS]" in l for l in check_lines)
        mass = None
        try:
            pm = json.load(open(os.path.join(HERE, "snapshots", variant, "metrics.json")))
            mass = (pm.get("metrics") or {}).get("mass_g") or (pm.get("mass") or {}).get("frame_mass_g")
        except Exception:
            pass
        results[L] = {"variant": variant, "score": score, "all_pass": all_pass, "mass_g": mass}
    if not results:
        print(f"[gen {gen}] all candidates failed to evaluate; generation lost", flush=True)
        return None
    best_all_pass = st.get("best_all_pass", False)
    best_mass = st.get("best_mass_g")
    # winner among candidates: gates first, then score, then lowest mass
    winner = max(results.values(), key=lambda c: (c["all_pass"], c["score"], -(c["mass_g"] or 1e9)))
    w = winner
    # then the winner must still beat the incumbent (same gates-first rule)
    improved = (w["all_pass"] and not best_all_pass) or                (w["all_pass"] == best_all_pass and (w["score"] > st["best_score"] or
                (w["score"] == st["best_score"] and w["mass_g"] and (best_mass is None or w["mass_g"] < best_mass))))
    tally = " | ".join(f"{c['variant']}={c['score']:.3f}{' PASS' if c['all_pass'] else ' fail'}{(' %.1fg' % c['mass_g']) if c['mass_g'] else ''}" for c in results.values())
    if improved:
        wL = w["variant"][-1]
        shutil.copy(cand_paths[wL], os.path.join(HERE, "chassis.py"))
        st["best_variant"], st["best_score"] = w["variant"], w["score"]
        st["best_all_pass"] = w["all_pass"]
        if w["mass_g"]:
            st["best_mass_g"] = w["mass_g"]
        # all candidates suppressed publish during eval; publish only the winner
        try:
            import snapshot as sn_mod
            mpath = os.path.join(HERE, "snapshots", w["variant"], "metrics.json")
            sn_mod.publish(json.load(open(mpath)), os.path.dirname(mpath))
        except Exception as e:
            print(f"[gen {gen}] winner publish failed: {e}", flush=True)
    else:
        shutil.copy(parent_backup, os.path.join(HERE, "chassis.py"))  # defensive; codex never edits it now
        print(f"[gen {gen}] no candidate beat incumbent; tally: {tally}", flush=True)
    wsum = summaries.get(w["variant"][-1], "")[:100]
    st["generation"] = gen
    st["history"].append({"variant": w["variant"], "parent": parent, "score": w["score"],
                          "improved": improved, "summary": (wsum + " [" + tally + "]")[:200]})
    # commit candidate artifacts + adopted code state
    sh("git add -A && git -c user.name=\"Instinct Chassis Researcher\" -c user.email=\"instinct-chassis-researcher@users.noreply.github.com\" commit -q -m \"gen %d: %s (parent %s, %s) - %s\" && git push -q origin chassis" % (
        gen, tally, parent, "NEW BEST " + w["variant"] if improved else "incumbent held", wsum[:80] or "no summary"))
    with open(ARCHIVE, "a") as f:
        f.write(json.dumps({"id": f"cad-gen-{gen}", "kind": "cad.chassis.generation", "variant": w["variant"],
                            "parent": parent, "score": w["score"], "improved": improved,
                            "candidates": tally, "summary": wsum, "ts": time.time()}) + "\n")
    save_state(st)
    print(f"[gen {gen}] done: {w['variant']} score {w['score']:.3f} ({'NEW BEST' if improved else 'no improvement'}) [{tally}]", flush=True)
    return w["score"]

if __name__ == "__main__":
    # single-instance lock: a second loop.py exits immediately (stale starts,
    # double restarts). flock is released by the OS on process death.
    import fcntl
    _lock_fd = open("/work/loop.lock", "w")
    try:
        fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another loop.py holds /work/loop.lock; exiting", flush=True)
        sys.exit(0)
    progress.start_heartbeat()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    for _ in range(n):
        run_generation()
    progress.idle(f"batch done: best {load_state()['best_variant']}")
