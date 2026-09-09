"""Prescription scan for the sensor findings (Eyad ask 2026-09-09 ~1AM):
 L1a: diagonal ToF relocated OUTBOARD radially (aperture follows) -> coverage per mm
 L1b: diagonal ToF raised VERTICALLY over the arm sweep
 L1c: diagonal ToF azimuth-TILTED off the arm axis (near hits <20mm = aperture-follow artifacts, excluded)
 L3: down-camera candidate belly positions -> clear-down fraction (near hits <20mm excluded)
Runs against the decoded published mesh with the batch MT caster, 24x18=432 rays per cast.
Origin convention: GLB frame. tof_ne empirical emitter origin from poses.json.
"""
import json, math, os, sys
import numpy as np
import trimesh

HERE = "/work/DroneStudio/chassis"
sys.path.insert(0, HERE)
from fov_extract import raycast, frustum_dirs, cone_dirs, ch2glb

MESH = trimesh.load("/tmp/pub_raycast.glb", force="mesh", process=False)
pose = json.load(open(os.path.join(HERE, "coverage", "v97-g96a.poses.json")))
S = {s["id"]: s for s in pose["sensors"]}
TOF_FOV = (55.0, 42.0)

# coarse grid for parametric sweep
import fov_extract
fov_extract.N_AZ, fov_extract.N_EL = 24, 18

def axis_from_quat_glb(axis_list):
    a = np.array(axis_list, float); return a/np.linalg.norm(a)

ne = S["tof_ne"]
O0 = np.array(ne["origin"], float)          # empirical emitter origin (GLB)
AX0 = axis_from_quat_glb(ne["axis"])        # (0.7071, 0, -0.7071)
UP = np.array([0, 1, 0.0])
RAD = np.array([AX0[0], 0, AX0[2]]); RAD /= np.linalg.norm(RAD)  # horizontal radial (GLB)

def hits_all(origin, direction, tmax=0.05):
    """all hit distances along one ray (for march-out)"""
    tri = np.asarray(MESH.triangles)
    v0 = tri[:,0,:]; e1 = tri[:,1,:]-v0; e2 = tri[:,2,:]-v0
    d = np.asarray(direction, float)
    o = np.asarray(origin, float)
    P = np.cross(d, e2); det = (e1*P).sum(-1)
    nz = np.abs(det) > 1e-12
    inv = np.zeros_like(det); inv[nz] = 1.0/det[nz]
    TV = o - v0
    U = (TV*P).sum(-1)*inv
    Q = np.cross(TV, e1)
    V = (d*Q).sum(-1)*inv
    T = (e2*Q).sum(-1)*inv
    ok = nz & (U>=0) & (V>=0) & (U+V<=1) & (T>1e-9) & (T<=tmax)
    return np.sort(T[ok])

def march_out(origin, direction):
    """move origin past the last solid crossing along +direction (aperture follows)"""
    hs = hits_all(origin, direction)
    if len(hs) == 0:
        return np.asarray(origin, float)
    return np.asarray(origin, float) + (hs[-1] + 0.0001) * np.asarray(direction, float)

def cast(origin, axis, fov, near_excl=0.0, rng=8.8):
    dirs = frustum_dirs(axis, fov[0], fov[1])
    hits = raycast(MESH, origin, dirs, rng)
    hits = [h for h in hits if h is None or h >= near_excl*1000]
    clear = sum(1 for h in hits if h is None)
    return clear/len(dirs)*100

def cast_cone(origin, axis, half_angle, near_excl=0.0, rng=8.0):
    dirs = cone_dirs(axis, half_angle)
    hits = raycast(MESH, origin, dirs, rng)
    hits = [h for h in hits if h is None or h >= near_excl*1000]
    clear = sum(1 for h in hits if h is None)
    return clear/len(dirs)*100

print("== L0 baseline (current tof_ne empirical origin, near hits <20mm excluded) ==", flush=True)
print("  coverage %.1f%%" % cast(O0, AX0, TOF_FOV, near_excl=0.02), flush=True)

print("== L1a radial outboard shift (aperture follows sensor) ==", flush=True)
for dr in (0.0, 0.005, 0.010, 0.015, 0.020):
    o = march_out(O0 + dr*RAD, AX0)
    print("  dr +%4.0f mm -> %5.1f%%" % (dr*1000, cast(o, AX0, TOF_FOV)), flush=True)

print("== L1b vertical raise ==", flush=True)
for dz in (0.005, 0.010, 0.015):
    o = march_out(O0 + dz*UP, AX0)
    print("  dz +%4.0f mm -> %5.1f%%" % (dz*1000, cast(o, AX0, TOF_FOV)), flush=True)

print("== L1c azimuth tilt off arm axis (origin fixed, near<20mm excluded) ==", flush=True)
for tilt in (-15, -10, -5, 5, 10, 15):
    a = math.radians(tilt)
    # rotate AX0 about GLB Y (vertical)
    ax = np.array([AX0[0]*math.cos(a) - AX0[2]*math.sin(a), 0.0, AX0[0]*math.sin(a) + AX0[2]*math.cos(a)])
    print("  tilt %+3d deg -> %5.1f%%" % (tilt, cast(O0, ax, TOF_FOV, near_excl=0.02)), flush=True)

print("== L3 down-camera belly candidates (65deg cone, near<20mm excluded) ==", flush=True)
for (cx, cy) in ((0.0,0.0), (0.040,0.0), (-0.040,0.0), (0.0,0.025), (0.0,-0.025), (0.060,0.020)):
    start = ch2glb((cx, cy, 0.005))          # 5mm above belly plane inside cavity
    down = np.array([0.0, -1.0, 0.0])
    hs = hits_all(start, down, tmax=0.05)
    o = start if len(hs)==0 else start + (hs[-1]+0.0001)*down
    print("  belly (%+4.0f,%+4.0f) -> %5.1f%% clear down-cone" % (cx*1000, cy*1000, cast_cone(o, down, 65.0, near_excl=0.02)), flush=True)

print("SCAN-DONE", flush=True)
