import { Suspense, useRef, useState } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls, useGLTF, Bounds, GizmoHelper, GizmoViewport } from "@react-three/drei";
import { Crosshair, Maximize2, X, ZoomIn, ZoomOut } from "lucide-react";
import ErrorBoundary from "./ErrorBoundary";

function Model({ url }: { url: string }) {
  const gltf = useGLTF(url);
  return <primitive object={gltf.scene} />;
}

// White/grayish dot grid behind the model. The WebGL canvas is transparent
// (R3F default), so this CSS on the wrapper shows through as the background.
const DOT_GRID: React.CSSProperties = {
  backgroundColor: "hsl(222 47% 7%)",
  backgroundImage: "radial-gradient(circle, rgba(255,255,255,0.14) 1px, transparent 1.6px)",
  backgroundSize: "22px 22px",
};

type ZoomApi = { zoom: (f: number) => void };

// Lives inside the Canvas; hands the toolbar a way to dolly the camera.
function ZoomBridge({ api }: { api: React.MutableRefObject<ZoomApi | null> }) {
  const camera = useThree((s) => s.camera);
  const controls = useThree((s) => s.controls) as { update?: () => void } | null;
  api.current = {
    zoom: (f: number) => {
      camera.position.multiplyScalar(f);
      camera.updateProjectionMatrix();
      controls?.update?.();
    },
  };
  return null;
}

function ToolBtn({ title, onClick, children }: { title: string; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      title={title}
      aria-label={title}
      onClick={onClick}
      className="h-7 w-7 rounded-md border border-border bg-background/85 backdrop-blur flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
    >
      {children}
    </button>
  );
}

function ViewerCanvas({
  url,
  expanded,
  onFullscreen,
  onClose,
}: {
  url: string;
  expanded?: boolean;
  onFullscreen?: () => void;
  onClose?: () => void;
}) {
  // Bumping fitKey remounts Bounds, which re-fits/centers the camera.
  const [fitKey, setFitKey] = useState(0);
  const zoomApi = useRef<ZoomApi | null>(null);

  return (
    <div className="relative w-full h-full">
      <Canvas
        camera={{ position: [0.3, 0.2, 0.4], fov: 45, near: 0.0001, far: 100000 }}
        style={{ touchAction: "none", borderRadius: expanded ? 0 : 8, ...DOT_GRID }}
      >
        <ambientLight intensity={0.7} />
        <directionalLight position={[3, 4, 2]} intensity={1.6} />
        <directionalLight position={[-3, 2, -2]} intensity={0.6} />
        <Suspense fallback={null}>
          {/* Bounds auto-fits the camera to the model - works whether the GLB
              is exported in meters or millimeters */}
          <Bounds key={fitKey} fit clip observe margin={1.5}>
            <Model url={url} />
          </Bounds>
        </Suspense>
        <OrbitControls makeDefault enablePan enableZoom enableRotate />
        <GizmoHelper alignment="bottom-right" margin={[56, 56]}>
          <GizmoViewport labelColor="white" axisHeadScale={0.8} />
        </GizmoHelper>
        <ZoomBridge api={zoomApi} />
      </Canvas>
      <div className="absolute top-2 right-2 flex gap-1.5">
        <ToolBtn title="Center / reset view" onClick={() => setFitKey((k) => k + 1)}>
          <Crosshair className="h-3.5 w-3.5" />
        </ToolBtn>
        {expanded && (
          <>
            <ToolBtn title="Zoom in" onClick={() => zoomApi.current?.zoom(0.75)}>
              <ZoomIn className="h-3.5 w-3.5" />
            </ToolBtn>
            <ToolBtn title="Zoom out" onClick={() => zoomApi.current?.zoom(1.33)}>
              <ZoomOut className="h-3.5 w-3.5" />
            </ToolBtn>
          </>
        )}
        {onFullscreen && (
          <ToolBtn title="Fullscreen" onClick={onFullscreen}>
            <Maximize2 className="h-3.5 w-3.5" />
          </ToolBtn>
        )}
        {onClose && (
          <ToolBtn title="Close" onClick={onClose}>
            <X className="h-3.5 w-3.5" />
          </ToolBtn>
        )}
      </div>
    </div>
  );
}

export default function CadViewer({ url }: { url: string }) {
  const [full, setFull] = useState(false);
  return (
    <ErrorBoundary
      fallback={
        <div className="h-full min-h-[200px] rounded-md border border-border bg-muted/30 flex items-center justify-center text-xs text-muted-foreground">
          3D preview unavailable for this file
        </div>
      }
    >
      <div className="w-full h-full">
        <ViewerCanvas url={url} onFullscreen={() => setFull(true)} />
      </div>
      {full && (
        <div
          className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-3 md:p-8"
          onClick={() => setFull(false)}
        >
          <div
            className="relative w-[92vw] h-[85vh] max-w-6xl rounded-xl border border-border bg-background overflow-hidden shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <ViewerCanvas url={url} expanded onClose={() => setFull(false)} />
          </div>
        </div>
      )}
    </ErrorBoundary>
  );
}
