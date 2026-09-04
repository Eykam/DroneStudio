import { Suspense } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, useGLTF, Bounds, GizmoHelper, GizmoViewport } from "@react-three/drei";

function Model({ url }: { url: string }) {
  const gltf = useGLTF(url);
  return <primitive object={gltf.scene} />;
}

export default function CadViewer({ url }: { url: string }) {
  return (
    <Canvas
      camera={{ position: [0.3, 0.2, 0.4], fov: 45, near: 0.0001, far: 100000 }}
      style={{ touchAction: "none", background: "hsl(222 47% 7%)", borderRadius: 8 }}
    >
      <ambientLight intensity={0.7} />
      <directionalLight position={[3, 4, 2]} intensity={1.6} />
      <directionalLight position={[-3, 2, -2]} intensity={0.6} />
      <Suspense fallback={null}>
        {/* Bounds auto-fits the camera to the model - works whether the GLB
            is exported in meters or millimeters */}
        <Bounds fit clip observe margin={1.5}>
          <Model url={url} />
        </Bounds>
      </Suspense>
      <OrbitControls makeDefault enablePan enableZoom enableRotate />
      <GizmoHelper alignment="bottom-right" margin={[56, 56]}>
        <GizmoViewport labelColor="white" axisHeadScale={0.8} />
      </GizmoHelper>
    </Canvas>
  );
}
