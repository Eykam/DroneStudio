"""Structural FEA stage: gmsh mesh -> CalculiX ccx -> pass/fail.

Load cases (fixed at the 30.5 mm stack hole ring, thrust at motor pads):
  hover_max: 4x max thrust (11.0 N, AKK RS2205 2300KV bench peak on 5045-class) upward
  crash:     3x that (hard-impact proxy)
  torsion:   diagonal pairs at differential thrust [+1.5,-1.5,+1.5,-1.5] x max,
             net yaw torque - frames fail in torsion before bending
  modal:     *FREQUENCY, first 8 eigenmodes - resonance screening vs control
             band (FC gyro filters ~100-250Hz) and blade-pass excitation
Material: PETG, E=2100 MPa, nu=0.36, yield 50 MPa, rho 1.24e-9 tonne/mm3.
Anisotropy: FFF layer adhesion is weaker across layers - vertical-load cases
(hover/crash bend arms across layer lines) gated at 0.55x yield-allowable,
torsion (in-plane shear) at 0.8x. Pass: each case's max von Mises under its
directional allowable AND arm-tip displacement < 5 mm AND first eigenmode
>= MIN_FIRST_MODE_HZ.
Node sets are built from coordinates (robust to gmsh surface-id churn).
"""
import os, math, subprocess, numpy as np

EXTRA_NSETS = {}  # populated by _emit_base (e.g. NBATT)

E_MPA, NU, YIELD_MPA = 2100.0, 0.36, 50.0
MAX_THRUST_N = 11.0  # AKK RS2205 2300KV (user motor 2026-09-04): bench peak 989-1155 g on 5045-class props, 4S (oscarliang/EMAX thrust stand); fitted 9.7 N full-throttle equilibrium (sim MOTOR_V2.md)
TIP_DISP_LIMIT_MM = 5.0
STRESS_FOS = 0.6
RHO_T_PER_MM3 = 1.24e-9  # PETG, tonne/mm^3 (N-mm-s consistent)
ANISO_VERTICAL = 0.55    # FFF Z-layer strength fraction vs in-plane
ANISO_SHEAR = 0.8
MIN_FIRST_MODE_HZ = 120.0  # first elastic mode floor (control-band margin)
BUCKLE_MIN_FACTOR = 2.0   # first buckling eigenvalue must exceed 2x the crash preload
PETG_ENDURANCE_MPA = 10.0  # ~0.2x yield, 1e7-cycle endurance for FFF PETG
FATIGUE_RIPPLE = 0.3      # cruise thrust ripple as fraction of hover_max stress
N_MODES = 8

def mesh_step(step_path, inp_path, mesh_size=2.5):
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.occ.importShapes(step_path)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size * 0.5)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
    gmsh.model.mesh.generate(3)
    gmsh.write(inp_path.replace(".inp", ".msh"))
    gmsh.finalize()
    import meshio
    m = meshio.read(inp_path.replace(".inp", ".msh"))
    tets = [c for c in m.cells if c.type == "tetra"]
    if not tets:
        raise RuntimeError("no tetra elements in mesh")
    meshio.Mesh(m.points, tets).write(inp_path, file_format="abaqus")
    return inp_path

def _emit_base(inp_path, motor_positions_mm, stack_spacing_mm):
    """Shared mesh/nset/material cards. Returns (lines, fix_count, load_id_lists)."""
    import meshio
    m = meshio.read(inp_path)
    pts = m.points
    sh = stack_spacing_mm / 2
    stack_xy = np.array([(sh, sh), (-sh, sh), (-sh, -sh), (sh, -sh)])
    fix = np.zeros(len(pts), bool)
    for (sx, sy) in stack_xy:
        fix |= (np.hypot(pts[:,0]-sx, pts[:,1]-sy) < 4.0)
    loads = []
    for (mx, my) in motor_positions_mm:
        sel = np.hypot(pts[:,0]-mx, pts[:,1]-my) < 12.0
        loads.append(np.where(sel)[0])
    out = []
    out.append("*HEADING")
    out.append("chassis FEA")
    out.append("*NODE")
    for i, p_ in enumerate(pts):
        out.append("%d,%.8e,%.8e,%.8e" % (i + 1, p_[0], p_[1], p_[2]))
    eid = 1
    for blk in [c for c in m.cells if c.type == "tetra"]:
        out.append("*ELEMENT,TYPE=C3D4,ELSET=EALL")
        for tet in blk.data:
            out.append("%d,%d,%d,%d,%d" % (eid, tet[0]+1, tet[1]+1, tet[2]+1, tet[3]+1))
            eid += 1
    def nset(name, ids):
        out.append("*NSET,NSET=%s" % name)
        for i in range(0, len(ids), 12):
            out.append(",".join(str(x) for x in ids[i:i+12]))
    nset("NFIX", np.where(fix)[0] + 1)
    for mi, sel in enumerate(loads):
        nset("NLOAD%d" % (mi+1), sel + 1)
    # battery-tray node set for ejection-load case (from component placement)
    global EXTRA_NSETS
    EXTRA_NSETS = {}
    try:
        from components import placement
        bp = placement().get("battery")
        if bp:
            bx, by, bz = bp[0]*1000, bp[1]*1000, bp[2]*1000
            sel = np.where((np.abs(pts[:,0]-bx) < 40.0) & (np.abs(pts[:,1]-by) < 25.0)
                           & (pts[:,2] > bz - 20.0) & (pts[:,2] < bz + 5.0))[0]
            if len(sel) > 0:
                nset("NBATT", sel + 1)
                EXTRA_NSETS["NBATT"] = list(sel + 1)
    except Exception:
        pass
    out.append("*MATERIAL,NAME=PETG")
    out.append("*ELASTIC")
    out.append("%.1f,%.3f" % (E_MPA, NU))
    out.append("*DENSITY")
    out.append("%.3e" % RHO_T_PER_MM3)
    out.append("*SOLID SECTION,ELSET=EALL,MATERIAL=PETG")
    out.append("*BOUNDARY")
    out.append("NFIX,1,3,0.0")
    return out, int(fix.sum()), [list(s + 1) for s in loads]

def build_job(inp_path, job_name, motor_positions_mm, stack_spacing_mm, motor_scales,
              extra_loads=()):
    """Static job; motor_scales = signed per-motor multiplier of MAX_THRUST_N (dir 3).
    extra_loads = [(nset_name, direction(1|2|3), total_N)] for non-motor loads."""
    out, nfix, loads = _emit_base(inp_path, motor_positions_mm, stack_spacing_mm)
    out.append("*STEP")
    out.append("*STATIC")
    out.append("*CLOAD")
    for mi, sel in enumerate(loads):
        n = max(len(sel), 1)
        per = MAX_THRUST_N * motor_scales[mi]
        for nid in sel:
            out.append("%d,3,%.6f" % (nid, per / n))
    import meshio as _m
    _pts = None
    for (ns_name, direction, total) in extra_loads:
        if _pts is None:
            _pts = _m.read(inp_path).points
        # nset by name: NBATT is built in _emit_base_extra
        ids = EXTRA_NSETS.get(ns_name)
        if ids is None or len(ids) == 0:
            continue
        n = len(ids)
        for nid in ids:
            out.append("%d,%d,%.6f" % (nid, direction, total / n))
    out.append("*NODE FILE")
    out.append("U")
    out.append("*EL FILE")
    out.append("S")
    out.append("*END STEP")
    with open(job_name + ".inp", "w") as f:
        f.write(chr(10).join(out) + chr(10))
    return job_name + ".inp", nfix, [len(s) for s in loads]

def _build_lateral_motor_case(inp_path, job_name, motor_positions_mm, stack_spacing_mm,
                              motor_idx, force_n, direction=1):
    out, nfix, loads = _emit_base(inp_path, motor_positions_mm, stack_spacing_mm)
    out.append("*STEP")
    out.append("*STATIC")
    out.append("*CLOAD")
    sel = loads[motor_idx]
    n = max(len(sel), 1)
    for nid in sel:
        out.append("%d,%d,%.6f" % (nid, direction, force_n / n))
    out.append("*NODE FILE")
    out.append("U")
    out.append("*EL FILE")
    out.append("S")
    out.append("*END STEP")
    with open(job_name + ".inp", "w") as f:
        f.write(chr(10).join(out) + chr(10))
    return job_name + ".inp"

def build_freq_job(inp_path, job_name, motor_positions_mm, stack_spacing_mm):
    out, nfix, loads = _emit_base(inp_path, motor_positions_mm, stack_spacing_mm)
    out.append("*STEP")
    out.append("*FREQUENCY")
    out.append("%d" % N_MODES)
    out.append("*NODE FILE")
    out.append("U")
    out.append("*END STEP")
    with open(job_name + ".inp", "w") as f:
        f.write(chr(10).join(out) + chr(10))
    return job_name + ".inp"

def build_buckle_job(inp_path, job_name, motor_positions_mm, stack_spacing_mm, preload_scale=3.0):
    """Linear buckling: static preload step (crash-scale thrust) then *BUCKLE.
    Eigenvalue = load multiplier at which thin sections buckle."""
    out, nfix, loads = _emit_base(inp_path, motor_positions_mm, stack_spacing_mm)
    out.append("*STEP")
    out.append("*STATIC")
    out.append("*CLOAD")
    for mi, sel in enumerate(loads):
        n = max(len(sel), 1)
        per = MAX_THRUST_N * preload_scale
        for nid in sel:
            out.append("%d,3,%.6f" % (nid, per / n))
    out.append("*END STEP")
    out.append("*STEP,BUCKLE")
    out.append("*BUCKLE")
    out.append("2")
    out.append("*NODE FILE")
    out.append("U")
    out.append("*END STEP")
    with open(job_name + ".inp", "w") as f:
        f.write(chr(10).join(out) + chr(10))
    return job_name + ".inp"

def parse_buckle_factors(dat_path):
    """Buckling load factors from ccx .dat: first numeric column of eigen rows."""
    import re
    factors = []
    row = re.compile(r"^\s*(\d+)\s+(\S+)\s+\S*\s*$")
    in_buckle = False
    for line in open(dat_path):
        if "BUCKLING" in line.upper() or "EIGENVALUE" in line.upper():
            in_buckle = True
            continue
        if in_buckle:
            m = row.match(line)
            if m:
                try:
                    factors.append(float(m.group(2)))
                except ValueError:
                    pass
            elif line.strip() and not line.startswith(" "):
                in_buckle = False
    return factors

def run_ccx(job_base, ccx="ccx"):
    r = subprocess.run([ccx, "-i", job_base], capture_output=True, text=True, timeout=1800)
    return r.returncode == 0 and os.path.exists(job_base + ".frd")

def parse_frd(frd_path):
    """max von Mises (MPa) and max |U| (mm) from a ccx .frd text file."""
    mode, max_vm, max_u = None, 0.0, 0.0
    for line in open(frd_path):
        s = line.lstrip()
        if s.startswith("-4"):
            if "STRESS" in line:
                mode = "S"
            elif "DISP" in line:
                mode = "U"
            else:
                mode = None
        elif s.startswith("-1") and mode in ("S", "U"):
            rest = s[2:]
            body = rest[10:]
            vals = []
            for i in range(0, len(body), 12):
                c = body[i:i+12].strip()
                if c:
                    try:
                        vals.append(float(c))
                    except ValueError:
                        pass
            if mode == "S" and len(vals) >= 6:
                sxx, syy, szz, sxy, syz, sxz = vals[:6]
                vm = math.sqrt(0.5*((sxx-syy)**2+(syy-szz)**2+(szz-sxx)**2)+3*(sxy**2+syz**2+sxz**2))
                if vm > max_vm:
                    max_vm = vm
            elif mode == "U" and len(vals) >= 3:
                u = math.sqrt(vals[0]**2 + vals[1]**2 + vals[2]**2)
                if u > max_u:
                    max_u = u
    return max_vm, max_u

def parse_freqs(dat_path):
    """Eigenfrequencies (Hz, cycles/time) from a ccx .dat after a *FREQUENCY step.
    Table rows: <mode> <eigenvalue> <rad/time> <cycles/time> <imaginary>; ends at PARTICIPATION FACTORS."""
    import re
    freqs = []
    row = re.compile(r"^\s*(\d+)\s+\S+\s+\S+\s+(\S+)\s+\S+\s*$")
    for line in open(dat_path):
        if "PARTICIPATION" in line:
            break
        m = row.match(line)
        if m:
            try:
                freqs.append(float(m.group(2)))
            except ValueError:
                pass
    return freqs

def evaluate_fea(step_path, motor_positions_mm, stack_spacing_mm, workdir, ccx="ccx"):
    """Parallel FEA: build every ccx job up front, run them concurrently.

    Each ccx job is a single-threaded subprocess with its own job files, so a
    thread pool of subprocesses saturates idle cores; results are identical to
    the serial version (same jobs, same parsers)."""
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(workdir, exist_ok=True)
    inp = os.path.join(workdir, "mesh.inp")
    mesh_step(step_path, inp)
    out = {}
    # real failure modes, not just paper cases:
    # hover_max/crash/torsion as before; cartwheel = lateral arm-snap (side impact
    # on one motor pad, in-plane); pullout = single pad full-throttle (mount boss);
    # battery_eject = 30g forward jolt at the battery tray (0.18 kg pack).
    cases = (("hover_max", [1.0]*4, ANISO_VERTICAL, ()),
             ("crash", [3.0]*4, ANISO_VERTICAL, ()),
             ("torsion", [1.5, -1.5, 1.5, -1.5], ANISO_SHEAR, ()),
             ("pullout", [1.0, 0.0, 0.0, 0.0], ANISO_VERTICAL, ()))
    jobs = {}  # name -> (job_base, limit_mpa)
    for name, scales, aniso, extra in cases:
        job = os.path.join(workdir, name)
        build_job(inp, job, motor_positions_mm, stack_spacing_mm, scales, extra)
        jobs[name] = (job, STRESS_FOS * YIELD_MPA * aniso)
    # cartwheel: lateral (dir 1) side impact on motor pad 1, 2x max thrust
    job = os.path.join(workdir, "cartwheel")
    _build_lateral_motor_case(inp, job, motor_positions_mm, stack_spacing_mm,
                              motor_idx=0, force_n=2.0 * MAX_THRUST_N)
    jobs["cartwheel"] = (job, STRESS_FOS * YIELD_MPA)  # in-plane: no aniso derate
    # battery ejection: 30g x 0.18 kg forward (dir 1) at the tray
    job = os.path.join(workdir, "battery_eject")
    build_job(inp, job, motor_positions_mm, stack_spacing_mm, [0.0]*4,
              extra_loads=[("NBATT", 1, 30.0 * 9.81 * 0.18)])
    jobs["battery_eject"] = (job, STRESS_FOS * YIELD_MPA)
    build_freq_job(inp, os.path.join(workdir, "modal"), motor_positions_mm, stack_spacing_mm)
    build_buckle_job(inp, os.path.join(workdir, "buckle"), motor_positions_mm, stack_spacing_mm)

    names = list(jobs) + ["modal", "buckle"]
    results = {}
    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        for name, ok in pool.map(lambda n: (n, run_ccx(os.path.join(workdir, n), ccx)), names):
            results[name] = ok

    for name, (job, limit) in jobs.items():
        if results.get(name):
            vm, u = parse_frd(job + ".frd")
            out[name] = {"max_von_mises_mpa": round(vm,1), "max_disp_mm": round(u,2),
                         "limit_mpa": round(limit,1),
                         "passed": vm < limit and u < TIP_DISP_LIMIT_MM}
        else:
            out[name] = {"passed": False, "error": "ccx failed"}
    job = os.path.join(workdir, "modal")
    if results.get("modal"):
        freqs = parse_freqs(job + ".dat")
        f1 = freqs[0] if freqs else 0.0
        out["modal"] = {"modes_hz": [round(f,1) for f in freqs],
                        "first_mode_hz": round(f1,1),
                        "min_first_mode_hz": MIN_FIRST_MODE_HZ,
                        "passed": bool(freqs) and f1 >= MIN_FIRST_MODE_HZ}
    else:
        out["modal"] = {"passed": False, "error": "ccx failed"}
    job = os.path.join(workdir, "buckle")
    if results.get("buckle"):
        factors = parse_buckle_factors(job + ".dat")
        f1 = factors[0] if factors else 0.0
        out["buckling"] = {"load_factors": [round(f,2) for f in factors],
                           "first_factor": round(f1,2), "min_factor": BUCKLE_MIN_FACTOR,
                           "passed": bool(factors) and f1 >= BUCKLE_MIN_FACTOR}
    else:
        out["buckling"] = {"passed": False, "error": "ccx failed"}
    # fatigue screen: arm-root stress amplitude at cruise vs PETG endurance limit.
    # Screening assumption: oscillatory thrust ripple at cruise ~0.3x hover_max stress.
    hvm = out.get("hover_max", {}).get("max_von_mises_mpa")
    if hvm is not None:
        amp = FATIGUE_RIPPLE * hvm
        out["fatigue"] = {"stress_amplitude_mpa": round(amp,1),
                          "endurance_limit_mpa": PETG_ENDURANCE_MPA,
                          "assumption": "cruise ripple 0.3x hover_max stress; endurance ~0.2x yield (1e7 cycles)",
                          "passed": amp < PETG_ENDURANCE_MPA}
    else:
        out["fatigue"] = {"passed": False, "error": "no hover_max stress"}
    out["passed"] = all(v.get("passed") for v in out.values())
    return out
