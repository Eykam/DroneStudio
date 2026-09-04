"""Structural FEA stage: gmsh mesh -> CalculiX ccx -> pass/fail.

Load case 1 (hover-max): 4x max thrust (10 N each) upward at motor pads,
fixed at the 30.5 mm stack hole ring. Load case 2 (crash): 3x that.
Material: PETG, E=2100 MPa, nu=0.36, yield 50 MPa. Pass: max von Mises
< 0.6*yield AND arm-tip displacement < 5 mm, both cases.
Node sets are built from coordinates (robust to gmsh surface-id churn).
"""
import os, math, subprocess, numpy as np

E_MPA, NU, YIELD_MPA = 2100.0, 0.36, 50.0
MAX_THRUST_N = 10.0
TIP_DISP_LIMIT_MM = 5.0
STRESS_FOS = 0.6

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
    # keep tetra only, write a clean abaqus/ccx mesh via meshio
    import meshio
    m = meshio.read(inp_path.replace(".inp", ".msh"))
    tets = [c for c in m.cells if c.type == "tetra"]
    if not tets:
        raise RuntimeError("no tetra elements in mesh")
    meshio.Mesh(m.points, tets).write(inp_path, file_format="abaqus")
    return inp_path

def build_job(inp_path, job_name, motor_positions_mm, stack_spacing_mm, thrust_scale=1.0):
    import meshio
    m = meshio.read(inp_path)
    pts = m.points  # mm
    # stack fixation: nodes within 4 mm of any stack hole axis
    sh = stack_spacing_mm / 2
    stack_xy = np.array([(sh, sh), (-sh, sh), (-sh, -sh), (sh, -sh)])
    fix = np.zeros(len(pts), bool)
    for (sx, sy) in stack_xy:
        fix |= (np.hypot(pts[:,0]-sx, pts[:,1]-sy) < 4.0)
    # motor pad load nodes: within pad radius of each motor axis
    loads = []
    for (mx, my) in motor_positions_mm:
        sel = np.hypot(pts[:,0]-mx, pts[:,1]-my) < 12.0
        loads.append(np.where(sel)[0])
    lines = [l for l in open(inp_path).read().splitlines() if not l.startswith("*")]
    head = ["*HEADING", "chassis auto-researcher job"]
    # reuse nodes/elements blocks from gmsh inp verbatim
    raw = open(inp_path).read()
    keep = []
    in_keep = False
    for l in raw.splitlines():
        u = l.upper()
        if u.startswith("*NODE") or u.startswith("*ELEMENT"):
            in_keep = True
        elif l.startswith("*") and in_keep and not (u.startswith("*NODE") or u.startswith("*ELEMENT")):
            in_keep = False
        if in_keep or l.startswith("*NODE") or l.startswith("*ELEMENT"):
            keep.append(l)
    with open(job_name + ".inp", "w") as f:
        f.write("*HEADING\nchassis FEA\n")
        f.write("\n".join(keep) + "\n")
        f.write("*NSET,NSET=NFIX\n")
        idx = np.where(fix)[0] + 1
        for i in range(0, len(idx), 12):
            f.write(",".join(str(x) for x in idx[i:i+12]) + "\n")
        for mi, sel in enumerate(loads):
            f.write(f"*NSET,NSET=NLOAD{mi+1}\n")
            ids = sel + 1
            for i in range(0, len(ids), 12):
                f.write(",".join(str(x) for x in ids[i:i+12]) + "\n")
        # element set: all
        f.write("*ELSET,ELSET=EALL,GEN\n1,%d\n" % sum(len(c.data) for c in m.cells))
        f.write("*MATERIAL,NAME=PETG\n*ELASTIC\n%.1f,%.3f\n" % (E_MPA, NU))
        f.write("*SOLID SECTION,ELSET=EALL,MATERIAL=PETG\n")
        f.write("*BOUNDARY\nNFIX,1,3,0.0\n")
        f.write("*STEP\n*STATIC\n")
        per = MAX_THRUST_N * thrust_scale
        f.write("*CLOAD\n")
        for mi, sel in enumerate(loads):
            n = max(len(sel), 1)
            for nid in (sel + 1):
                f.write("%d,3,%.6f\n" % (nid, per / n))
        f.write("*NODE FILE\nU\n*EL FILE\nS\n*END STEP\n")
    return job_name + ".inp", int(fix.sum()), [int(len(s)) for s in loads]

def run_ccx(job_base, ccx="ccx"):
    r = subprocess.run([ccx, "-i", job_base], capture_output=True, text=True, timeout=1800)
    return r.returncode == 0 and os.path.exists(job_base + ".frd")

def parse_frd(frd_path):
    """max von Mises (MPa) and max |U| (mm) from a ccx .frd (text)."""
    mode, max_vm, max_u = None, 0.0, 0.0
    sxx=syy=szz=sxy=syz=sxz=None
    for line in open(frd_path):
        if line.startswith("    1PSTEP"):
            mode = None
        elif "-4" in line[:13] and "STRESS" in line:
            mode = "S"; cnt = 0
        elif "-4" in line[:13] and "DISP" in line:
            mode = "U"; cnt = 0
        elif line.startswith(" -1") and mode == "S":
            v = [float(line[13+i*12:25+i*12]) for i in range(6)]
            sxx,syy,szz,sxy,syz,sxz = v
        elif line.startswith(" -2") and mode == "S" and sxx is not None:
            vm = math.sqrt(0.5*((sxx-syy)**2+(syy-szz)**2+(szz-sxx)**2)+3*(sxy**2+syz**2+sxz**2))
            max_vm = max(max_vm, vm); sxx=None
        elif line.startswith(" -1") and mode == "U":
            v = [float(line[13+i*12:25+i*12]) for i in range(3)]
            max_u = max(max_u, math.sqrt(sum(x*x for x in v)))
    return max_vm, max_u

def evaluate_fea(step_path, motor_positions_mm, stack_spacing_mm, workdir, ccx="ccx"):
    os.makedirs(workdir, exist_ok=True)
    inp = os.path.join(workdir, "mesh.inp")
    mesh_step(step_path, inp)
    out = {}
    for name, scale in (("hover_max", 1.0), ("crash", 3.0)):
        job = os.path.join(workdir, name)
        inp_path, nfix, nloads = build_job(inp, job, motor_positions_mm, stack_spacing_mm, scale)
        if run_ccx(job, ccx):
            vm, u = parse_frd(job + ".frd")
            ok = vm < STRESS_FOS * YIELD_MPA and u < TIP_DISP_LIMIT_MM
            out[name] = {"max_von_mises_mpa": round(vm,1), "max_disp_mm": round(u,2),
                         "limit_mpa": STRESS_FOS*YIELD_MPA, "passed": ok,
                         "fix_nodes": nfix, "load_nodes": nloads}
        else:
            out[name] = {"passed": False, "error": "ccx failed"}
    out["passed"] = all(v.get("passed") for v in out.values())
    return out

if __name__ == "__main__":
    from chassis import ChassisParams
    p = ChassisParams()
    r = evaluate_fea("/home/sandbox/cad-researcher/chassis_v1.step", p.motor_positions(), p.stack_spacing_mm,
                     "/home/sandbox/cad-researcher/fea_v1", ccx="/tmp/CalculiX/ccx_2.22/src/ccx_2.22")
    import json; print(json.dumps(r, indent=2))
