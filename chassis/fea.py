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
    out.append("*MATERIAL,NAME=PETG")
    out.append("*ELASTIC")
    out.append("%.1f,%.3f" % (E_MPA, NU))
    out.append("*SOLID SECTION,ELSET=EALL,MATERIAL=PETG")
    out.append("*BOUNDARY")
    out.append("NFIX,1,3,0.0")
    out.append("*STEP")
    out.append("*STATIC")
    out.append("*CLOAD")
    per = MAX_THRUST_N * thrust_scale
    for mi, sel in enumerate(loads):
        n = max(len(sel), 1)
        for nid in (sel + 1):
            out.append("%d,3,%.6f" % (nid, per / n))
    out.append("*NODE FILE")
    out.append("U")
    out.append("*EL FILE")
    out.append("S")
    out.append("*END STEP")
    with open(job_name + ".inp", "w") as f:
        f.write(chr(10).join(out) + chr(10))
    return job_name + ".inp", int(fix.sum()), [int(len(s)) for s in loads]

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
            body = rest[10:]  # skip 10-char node id; values are fixed 12-char fields
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
        elif s.startswith("-2") and mode == "S":
            pass  # stress fits on the -1 line; ignore continuations
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
