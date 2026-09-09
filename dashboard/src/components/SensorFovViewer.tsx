import { Suspense, useMemo, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, useGLTF, Bounds, GizmoHelper, GizmoViewport } from "@react-three/drei";
import * as THREE from "three";
import ErrorBoundary from "./ErrorBoundary";
import type { CadFov, FovSensor } from "@/api";

const PALETTE = ["#22d3ee", "#f472b6", "#a3e635", "#fbbf24", "#818cf8", "#f87171", "#34d399", "#e879f9", "#60a5fa", "#fb923c", "#2dd4bf"];

const DOT_GRID: React.CSSProperties = {
  backgroundColor: "hsl(222 47% 7%)",
  backgroundImage: "radial-gradient(circle, rgba(255,255,255,0.14) 1px, transparent 1.6px)",
  backgroundSize: "22px 22px",
};

type Geom = { lines: THREE.BufferGeometry; points: THREE.BufferGeometry };

// Ray endpoints straight from the coverage file (apex fan + hit cloud).
// Rendering from rays - not from assumed boresight conventions - keeps the
// visualization truthful to whatever CAD raycast.
function buildGeom(s: FovSensor, scale: number): Geom {
  const p = s.mount.pos_mm;
  const n = s.raycast.dirs.length;
  const lp = new Float32Array(n * 6);
  const pp: number[] = [];
  const maxU = (s.range_m?.max ?? 1) * 1000; // file units are mm per contract
  for (let i = 0; i < n; i++) {
    const d = s.raycast.dirs[i];
    const h = s.raycast.hit_mm[i];
    const dist = h ?? maxU;
    const ex = p[0] + d[0] * dist, ey = p[1] + d[1] * dist, ez = p[2] + d[2] * dist;
    lp.set([p[0] * scale, p[1] * scale, p[2] * scale, ex * scale, ey * scale, ez * scale], i * 6);
    if (h != null) pp.push(ex * scale, ey * scale, ez * scale);
  }
  const lines = new THREE.BufferGeometry();
  lines.setAttribute("position", new THREE.BufferAttribute(lp, 3));
  const points = new THREE.BufferGeometry();
  points.setAttribute("position", new THREE.BufferAttribute(new Float32Array(pp), 3));
  return { lines, points };
}

function SensorRays({ s, color, scale }: { s: FovSensor; color: string; scale: number }) {
  const { lines, points } = useMemo(() => buildGeom(s, scale), [s, scale]);
  return (
    <>
      <lineSegments geometry={lines}>
        <lineBasicMaterial color={color} transparent opacity={0.08} depthWrite={false} />
      </lineSegments>
      <points geometry={points}>
        <pointsMaterial color={color} size={3} sizeAttenuation={false} transparent opacity={0.85} depthWrite={false} />
      </points>
    </>
  );
}

function Scene({ glbUrl, fov, enabled }: { glbUrl: string; fov: CadFov; enabled: Record<string, boolean> }) {
  const gltf = useGLTF(glbUrl);
  // Coverage file is mm per contract; GLB may be meters or mm. Pick the scale
  // whose sensor-mount extents land on the model's extents.
  const scale = useMemo(() => {
    const box = new THREE.Box3().setFromObject(gltf.scene);
    const size = new THREE.Vector3();
    box.getSize(size);
    const diag = size.length();
    let maxCoord = 0;
    for (const s of fov.sensors ?? []) for (const v of s.mount.pos_mm) maxCoord = Math.max(maxCoord, Math.abs(v));
    if (!maxCoord || !diag) return 1;
    const ratio = maxCoord / diag;
    if (ratio > 100) return 0.001;
    if (ratio < 0.01) return 1000;
    return 1;
  }, [gltf, fov]);
  return (
    <>
      {/* Bounds fits the MODEL only - 50m camera rays would shrink it */}
      <Bounds fit clip observe margin={1.5}>
        <primitive object={gltf.scene} />
      </Bounds>
      {(fov.sensors ?? []).map((s, i) =>
        enabled[s.id] !== false ? (
          <SensorRays key={s.id} s={s} color={PALETTE[i % PALETTE.length]} scale={scale} />
        ) : null,
      )}
    </>
  );
}

export default function SensorFovViewer({ glbUrl, fov }: { glbUrl: string; fov: CadFov }) {
  const sensors = fov.sensors ?? [];
  const [enabled, setEnabled] = useState<Record<string, boolean>>({});
  return (
    <ErrorBoundary
      fallback={
        <div className="h-full min-h-[200px] rounded-md border border-border bg-muted/30 flex items-center justify-center text-xs text-muted-foreground">
          Sensor coverage view unavailable
        </div>
      }
    >
      <div className="w-full h-full flex gap-2">
        <div className="flex-1 min-w-0">
          <Canvas
            camera={{ position: [0.3, 0.2, 0.4], fov: 45, near: 0.0001, far: 100000 }}
            style={{ touchAction: "none", borderRadius: 8, ...DOT_GRID }}
          >
            <ambientLight intensity={0.7} />
            <directionalLight position={[3, 4, 2]} intensity={1.6} />
            <directionalLight position={[-3, 2, -2]} intensity={0.6} />
            <Suspense fallback={null}>
              <Scene glbUrl={glbUrl} fov={fov} enabled={enabled} />
            </Suspense>
            <OrbitControls makeDefault enablePan enableZoom enableRotate />
            <GizmoHelper alignment="bottom-right" margin={[56, 56]}>
              <GizmoViewport labelColor="white" axisHeadScale={0.8} />
            </GizmoHelper>
          </Canvas>
        </div>
        <div className="w-40 md:w-48 shrink-0 overflow-y-auto space-y-1 pr-0.5">
          {sensors.map((s, i) => {
            const on = enabled[s.id] !== false;
            return (
              <button
                key={s.id}
                onClick={() => setEnabled((m) => ({ ...m, [s.id]: !on }))}
                className={`w-full text-left rounded-md border p-1.5 transition-opacity ${on ? "border-border" : "border-border/40 opacity-45"}`}
              >
                <div className="flex items-center gap-1.5">
                  <span className="h-2 w-2 rounded-full shrink-0" style={{ background: PALETTE[i % PALETTE.length] }} />
                  <span className="text-[11px] font-mono truncate">{s.id}</span>
                  <span className="ml-auto text-[10px] text-muted-foreground shrink-0">
                    {s.coverage_fraction != null ? `${(s.coverage_fraction * 100).toFixed(0)}%` : ""}
                  </span>
                </div>
                <div className="text-[10px] text-muted-foreground truncate mt-0.5">{s.label}</div>
              </button>
            );
          })}
        </div>
      </div>
    </ErrorBoundary>
  );
}
