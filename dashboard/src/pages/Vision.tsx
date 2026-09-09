import { useEffect, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { useSearchParams } from "react-router-dom";

// VISION: live view of the learned depth+seg training run (vision workstream
// Phase 1). Frames: RGB | GT depth | GT seg | pred depth | pred seg from a
// fixed val-scene batch, refreshed whenever the best checkpoint improves.
// Curves: per-epoch loss / depth metrics / mIoU posted by the box1 poster.

const SEG_COLORS: [number, number, number][] = [
  [17, 24, 39],    // sky
  [107, 114, 128], // floor
  [217, 119, 6],   // obstacle
  [34, 197, 94],   // goal pad
];

type FrameRec = { rgb: number[]; depth: number[]; seg: number[]; pred_depth: number[]; pred_seg: number[] };
type TrainEpoch = {
  epoch: number; loss?: number; loss_d?: number; loss_s?: number;
  mae?: number; rmse?: number; d125?: number; miou?: number;
  iou0?: number; iou1?: number; iou2?: number; iou3?: number; ts?: string;
};
type ScenarioState = {
  episode_id?: string; scenario?: string; step?: number;
  pos?: number[]; policy?: string; dist_id?: string;
  w: number; h: number; frames: FrameRec[]; ts: string;
};
type VisionState = {
  meta: { tag?: string; status?: string; params?: number; epochs_target?: number; pid_alive?: boolean; ts?: string } | null;
  epochs: TrainEpoch[];
  frames: { w: number; h: number; epoch: number | null; frames: FrameRec[]; ts: string } | null;
  scenario: ScenarioState | null;
};

function depthGray(d: number): number {
  return d >= 65535 ? 8 : Math.max(24, Math.min(255, Math.round(255 * (1 - d / 20000))));
}

function FramesCanvas({ data, rowH = 130 }: { data: { w: number; h: number; frames: FrameRec[] }; rowH?: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const { w, h, frames } = data;
    const panelW = Math.floor(canvas.width / 5);
    const scale = Math.min((panelW - 8) / w, (rowH - 8) / h);
    const dw = Math.floor(w * scale), dh = Math.floor(h * scale);
    ctx.fillStyle = "#0b0f14";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingEnabled = false;
    const labels = ["RGB (model input)", "GT depth", "GT seg", "pred depth", "pred seg"];
    ctx.fillStyle = "#9ca3af";
    ctx.font = "10px monospace";
    labels.forEach((l, i) => ctx.fillText(l, i * panelW + 6, 12));
    frames.forEach((fr, row) => {
      const imgs = [
        ctx.createImageData(w, h), ctx.createImageData(w, h), ctx.createImageData(w, h),
        ctx.createImageData(w, h), ctx.createImageData(w, h),
      ];
      for (let i = 0; i < w * h; i++) {
        const pk = fr.rgb[i];
        imgs[0].data.set([(pk >> 16) & 255, (pk >> 8) & 255, pk & 255, 255], i * 4);
        const v1 = depthGray(fr.depth[i]);
        imgs[1].data.set([v1, v1, v1, 255], i * 4);
        const c1 = SEG_COLORS[Math.min(fr.seg[i], 3)];
        imgs[2].data.set([c1[0], c1[1], c1[2], 255], i * 4);
        const v2 = depthGray(fr.pred_depth[i]);
        imgs[3].data.set([v2, v2, v2, 255], i * 4);
        const c2 = SEG_COLORS[Math.min(fr.pred_seg[i], 3)];
        imgs[4].data.set([c2[0], c2[1], c2[2], 255], i * 4);
      }
      const y0 = 18 + row * rowH + Math.floor((rowH - 8 - dh) / 2);
      imgs.forEach((img, p) => {
        const off = document.createElement("canvas");
        off.width = w; off.height = h;
        off.getContext("2d")!.putImageData(img, 0, 0);
        ctx.drawImage(off, p * panelW + 4, y0, dw, dh);
      });
    });
  }, [data]);
  return <canvas ref={ref} width={1100} height={18 + (data.frames.length * rowH)} className="w-full rounded-md border border-border" />;
}

type Series = { name: string; color: string; values: (number | undefined)[] };

function Chart({ title, series, epochs, height = 170 }: { title: string; series: Series[]; epochs: number[]; height?: number }) {
  const W = 520, H = height, padL = 40, padR = 8, padT = 10, padB = 20;
  const all = series.flatMap((s) => s.values.filter((v): v is number => v !== undefined && Number.isFinite(v)));
  if (!all.length || epochs.length < 2) {
    return (
      <div className="rounded-md border border-border p-3">
        <div className="text-xs text-muted-foreground mb-2">{title}</div>
        <div className="text-xs text-muted-foreground">waiting for epochs...</div>
      </div>
    );
  }
  let lo = Math.min(...all), hi = Math.max(...all);
  if (hi - lo < 1e-9) { hi = lo + 1; }
  const pad = (hi - lo) * 0.08;
  lo -= pad; hi += pad;
  const x0 = epochs[0], x1 = epochs[epochs.length - 1];
  const X = (e: number) => padL + ((e - x0) / Math.max(1, x1 - x0)) * (W - padL - padR);
  const Y = (v: number) => padT + (1 - (v - lo) / (hi - lo)) * (H - padT - padB);
  return (
    <div className="rounded-md border border-border p-3">
      <div className="flex items-baseline justify-between mb-1">
        <div className="text-xs text-muted-foreground">{title}</div>
        <div className="flex gap-3">
          {series.map((s) => {
            const last = [...s.values].reverse().find((v): v is number => v !== undefined && Number.isFinite(v));
            return (
              <span key={s.name} className="text-[10px] font-mono" style={{ color: s.color }}>
                {s.name} {last !== undefined ? last.toFixed(3) : "-"}
              </span>
            );
          })}
        </div>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="#374151" strokeWidth="1" />
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="#374151" strokeWidth="1" />
        <text x={4} y={Y(hi - pad) + 3} fill="#6b7280" fontSize="9" fontFamily="monospace">{(hi - pad).toFixed(2)}</text>
        <text x={4} y={Y(lo + pad) + 3} fill="#6b7280" fontSize="9" fontFamily="monospace">{(lo + pad).toFixed(2)}</text>
        <text x={padL} y={H - 6} fill="#6b7280" fontSize="9" fontFamily="monospace">ep {x0}</text>
        <text x={W - padR - 30} y={H - 6} fill="#6b7280" fontSize="9" fontFamily="monospace">ep {x1}</text>
        {series.map((s) => {
          const pts = s.values
            .map((v, i) => (v !== undefined && Number.isFinite(v) ? `${X(epochs[i]).toFixed(1)},${Y(v).toFixed(1)}` : null))
            .filter(Boolean)
            .join(" ");
          return <polyline key={s.name} points={pts} fill="none" stroke={s.color} strokeWidth="1.5" />;
        })}
      </svg>
    </div>
  );
}

export default function Vision() {
  const [params, setParams] = useSearchParams();
  const tabParam = params.get("tab");
  const tab = tabParam === "curves" ? "curves" : tabParam === "training" ? "training" : "scenarios";
  const q = useQuery({
    queryKey: ["vision-train"],
    queryFn: async () => (await (await fetch("/api/vision/state", { credentials: "same-origin" })).json()) as VisionState,
    refetchInterval: 2_500,
  });
  const st = q.data;
  const latest = st?.epochs?.length ? st.epochs[st.epochs.length - 1] : null;
  const epochs = (st?.epochs ?? []).map((e) => e.epoch);
  const col = (k: keyof TrainEpoch) => (st?.epochs ?? []).map((e) => e[k] as number | undefined);

  return (
    <div className="space-y-4">
      <div className="flex gap-2 border-b border-border">
        {(["scenarios", "training", "curves"] as const).map((t) => (
          <button key={t} onClick={() => setParams(t === "scenarios" ? {} : { tab: t }, { replace: true })}
            className={`px-3 py-2 text-sm border-b-2 -mb-px ${
              tab === t ? "border-primary text-foreground" : "border-transparent text-muted-foreground hover:text-foreground"}`}>
            {t === "scenarios" ? "Scenarios" : t === "training" ? "Training frames" : "Curves"}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-2 md:grid-cols-6 gap-2">
        {[
          ["status", st?.meta?.status ?? (st ? "no data" : "...")],
          ["epoch", latest ? `${latest.epoch}${st?.meta?.epochs_target ? ` / ${st.meta.epochs_target}` : ""}` : "-"],
          ["train loss", latest?.loss?.toFixed(2) ?? "-"],
          ["val MAE", latest?.mae !== undefined ? `${latest.mae.toFixed(2)} m` : "-"],
          ["delta<1.25", latest?.d125?.toFixed(3) ?? "-"],
          ["mIoU", latest?.miou?.toFixed(3) ?? "-"],
        ].map(([k, v]) => (
          <div key={k} className="rounded-md border border-border px-3 py-2">
            <div className="text-[10px] uppercase tracking-wide text-muted-foreground">{k}</div>
            <div className="text-sm font-mono">{String(v)}</div>
          </div>
        ))}
      </div>

      {tab === "scenarios" ? (
        <div>
          {st?.scenario ? (
            <>
              <div className="text-xs text-muted-foreground mb-1">
                {st.scenario.policy ?? "t4_live"} flying {st.scenario.scenario ?? "episode"} ({st.scenario.episode_id ?? "?"})
                {st.scenario.step !== undefined ? ` - step ${st.scenario.step}` : ""}
                {st.scenario.pos ? ` - pos [${st.scenario.pos.map((v) => v.toFixed(1)).join(", ")}]` : ""}
                {" - model watches every frame live"}
              </div>
              <FramesCanvas data={st.scenario} rowH={190} />
            </>
          ) : (
            <div className="rounded-md border border-border p-6 text-sm text-muted-foreground">
              Waiting for the scenario streamer (model + t4_live flying episodes on the sim)...
            </div>
          )}
        </div>
      ) : tab === "training" ? (
        <div>
          {st?.frames ? (
            <>
              <div className="text-xs text-muted-foreground mb-1">
                fixed val-scene batch, predictions from the best checkpoint (epoch {st.frames.epoch ?? "?"})
              </div>
              <FramesCanvas data={st.frames} />
            </>
          ) : (
            <div className="rounded-md border border-border p-6 text-sm text-muted-foreground">
              Waiting for the first checkpoint + prediction frames from the trainer...
            </div>
          )}
        </div>
      ) : (
        <div className="grid md:grid-cols-2 gap-3">
          <Chart title="train loss" epochs={epochs} series={[
            { name: "depth L1 (log)", color: "#f59e0b", values: col("loss_d") },
            { name: "seg CE", color: "#38bdf8", values: col("loss_s") },
          ]} />
          <Chart title="val depth error (m)" epochs={epochs} series={[
            { name: "MAE", color: "#f59e0b", values: col("mae") },
            { name: "RMSE", color: "#ef4444", values: col("rmse") },
          ]} />
          <Chart title="val delta<1.25 + mIoU" epochs={epochs} series={[
            { name: "d1.25", color: "#34d399", values: col("d125") },
            { name: "mIoU", color: "#a78bfa", values: col("miou") },
          ]} />
          <Chart title="per-class IoU (0 sky 1 ground 2 obstacle 3 pad)" epochs={epochs} series={[
            { name: "sky", color: "#64748b", values: col("iou0") },
            { name: "ground", color: "#22c55e", values: col("iou1") },
            { name: "obstacle", color: "#f59e0b", values: col("iou2") },
            { name: "pad", color: "#84cc16", values: col("iou3") },
          ]} />
        </div>
      )}
    </div>
  );
}