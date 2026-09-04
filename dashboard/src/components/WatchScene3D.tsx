import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, useGLTF, Line } from "@react-three/drei";
import * as THREE from "three";

export type CamMode = "orbit" | "chase" | "fpv";

type Frame = { pos: number[]; quat: number[]; vel: number[]; step: number };
type Scene = { spawn: number[]; goal: number[]; obstacles: number[][]; extent: number };

const DRONE_VISUAL_SCALE = 2.0; // exaggerate drone size so it reads at scene scale

function Drone({ glbUrl, frameRef, visibleRef }: { glbUrl: string | null; frameRef: React.MutableRefObject<Frame | null>; visibleRef: React.MutableRefObject<boolean> }) {
  const group = useRef<THREE.Group>(null);
  const gltf = glbUrl ? useGLTF(glbUrl) : null;
  const model = useMemo(() => {
    if (!gltf) return null;
    const obj = gltf.scene.clone(true);
    // normalize to ~0.25m real size, then exaggerate for visibility
    const box = new THREE.Box3().setFromObject(obj);
    const size = box.getSize(new THREE.Vector3()).length() || 1;
    const center = box.getCenter(new THREE.Vector3());
    obj.position.sub(center);
    const s = (0.25 * DRONE_VISUAL_SCALE) / size;
    obj.scale.setScalar(s);
    return obj;
  }, [gltf]);
  useFrame(() => {
    const f = frameRef.current;
    if (!f || !group.current) return;
    group.current.visible = visibleRef.current;
    group.current.position.set(f.pos[0], f.pos[1], f.pos[2]);
    group.current.quaternion.set(f.quat[0], f.quat[1], f.quat[2], f.quat[3]);
  });
  return (
    <group ref={group}>
      {model ? (
        <primitive object={model} />
      ) : (
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <coneGeometry args={[0.15 * DRONE_VISUAL_SCALE, 0.4 * DRONE_VISUAL_SCALE, 8]} />
          <meshStandardMaterial color="#ffd166" />
        </mesh>
      )}
    </group>
  );
}

function Trail({ trail3dRef }: { trail3dRef: React.MutableRefObject<[number, number, number][]> }) {
  const [points, setPoints] = useState<[number, number, number][]>([]);
  const counter = useRef(0);
  useFrame(() => {
    if (++counter.current % 12 === 0) setPoints([...trail3dRef.current]);
  });
  if (points.length < 2) return null;
  return <Line points={points} color="#4aa3ff" lineWidth={1} transparent opacity={0.7} />;
}

function Rig({ mode, frameRef, extent }: { mode: CamMode; frameRef: React.MutableRefObject<Frame | null>; extent: number }) {
  const { camera } = useThree();
  const prevMode = useRef<CamMode>("orbit");
  useFrame(() => {
    if (mode === "orbit") {
      // one-time reset when returning from chase/fpv
      if (prevMode.current !== "orbit") {
        camera.position.set(extent * 0.9, extent * 0.8, extent * 0.9);
        camera.up.set(0, 1, 0);
        camera.lookAt(0, 0, 0);
      }
      prevMode.current = mode;
      return;
    }
    prevMode.current = mode;
    const f = frameRef.current;
    if (!f) return;
    const p = new THREE.Vector3(f.pos[0], f.pos[1], f.pos[2]);
    const q = new THREE.Quaternion(f.quat[0], f.quat[1], f.quat[2], f.quat[3]);
    if (mode === "chase") {
      const back = new THREE.Vector3(0, 0, 1).applyQuaternion(q); // body -Z is forward
      camera.position.copy(p.clone().add(back.multiplyScalar(3)).add(new THREE.Vector3(0, 1.2, 0)));
      camera.up.set(0, 1, 0);
      camera.lookAt(p);
    } else if (mode === "fpv") {
      // body frame matches camera frame (both -Z forward); direct copy avoids
      // degenerate lookAt when the drone pitches vertically
      camera.position.copy(p.clone().add(new THREE.Vector3(0, 0.06, 0)));
      camera.quaternion.copy(q);
    }
  });
  return null;
}

export default function WatchScene3D({ sceneRef, frameRef, trail3dRef, mode }:
  { sceneRef: React.MutableRefObject<Scene | null>;
    frameRef: React.MutableRefObject<Frame | null>;
    trail3dRef: React.MutableRefObject<[number, number, number][]>;
    mode: CamMode }) {
  const [glbUrl, setGlbUrl] = useState<string | null>(null);
  const droneVisible = useRef(true);
  droneVisible.current = mode !== "fpv";
  useEffect(() => {
    (async () => {
      try {
        const r = await fetch("/api/cad/designs", { credentials: "same-origin" });
        const d = await r.json();
        const designs = (d.designs || []).filter((x: any) => x.glb_bytes > 0);
        // prefer the design the current sim manifest was built from (chassis_v1), else latest
        const v1 = designs.find((x: any) => /v1/.test(x.id));
        const pick = v1 || designs[designs.length - 1];
        if (pick) setGlbUrl(`/api/cad/designs/${pick.id}/glb`);
      } catch { /* fallback cone */ }
    })();
  }, []);
  const sc = sceneRef.current;
  const ext = sc?.extent ?? 22;
  return (
    <Canvas camera={{ position: [ext * 0.9, ext * 0.8, ext * 0.9], fov: 50, near: 0.01, far: 2000 }}
      style={{ background: "#0b0f14" }}>
      <ambientLight intensity={0.6} />
      <directionalLight position={[10, 20, 10]} intensity={1.2} />
      <gridHelper args={[ext * 2, 20, "#1a2230", "#131a26"]} />
      {sc && (
        <>
          {sc.obstacles.map((o, i) => (
            <mesh key={i} position={[o[0], o[1], o[2]]}>
              <sphereGeometry args={[o[3], 16, 12]} />
              <meshStandardMaterial color="#7a3b3b" transparent opacity={0.85} />
            </mesh>
          ))}
          <mesh position={[sc.goal[0], sc.goal[1], sc.goal[2]]}>
            <sphereGeometry args={[0.45, 16, 12]} />
            <meshStandardMaterial color="#2ecc71" wireframe />
          </mesh>
          <mesh position={[sc.spawn[0], 0.02, sc.spawn[2]]} rotation={[-Math.PI / 2, 0, 0]}>
            <ringGeometry args={[0.3, 0.45, 24]} />
            <meshBasicMaterial color="#888888" />
          </mesh>
        </>
      )}
      <Suspense fallback={null}>
        <Drone glbUrl={glbUrl} frameRef={frameRef} visibleRef={droneVisible} />
      </Suspense>
      <Trail trail3dRef={trail3dRef} />
      <Rig mode={mode} frameRef={frameRef} extent={ext} />
      {mode === "orbit" && <OrbitControls makeDefault enablePan enableZoom enableRotate />}
    </Canvas>
  );
}
