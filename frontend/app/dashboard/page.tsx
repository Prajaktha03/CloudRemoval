"use client";

import { Crosshair, Database, Layers3, ShieldCheck, Workflow } from "lucide-react";
import { useMemo } from "react";
import { MapCard } from "@/components/map-card";
import { MetricRing, PerformanceChart, TrendPoint } from "@/components/charts";
import { SceneImageCard } from "@/components/scene-image-card";
import { Button, ErrorAlert, LoadingState, Panel, SectionTitle } from "@/components/ui";
import { useResource } from "@/hooks/use-resource";
import type { ComparisonReport, Experiment, Health, SceneMetadata } from "@/lib/api";

type TrainingReport = {
  histories: Record<string, { epoch: number[]; val_psnr?: number[]; val_ssim?: number[]; loss_train?: number[] }>;
};

type DatasetInfo = {
  configured: boolean;
  file_count: number;
  files_by_extension: Record<string, number>;
  note: string;
};

export default function DashboardPage() {
  const health = useResource<Health>("/health");
  const experiments = useResource<{ experiments: Experiment[] }>("/experiments?limit=200");
  const report = useResource<ComparisonReport>("/analytics/model-comparison");
  const training = useResource<TrainingReport>("/analytics/training-history");
  const dataset = useResource<DatasetInfo>("/dataset/info");
  const scene = useResource<SceneMetadata>("/dataset/sample");

  const multimodal = report.data?.summary.find((row) => row.model === "C_Multimodal_UNet_CNN");
  const currentHistory = training.data?.histories.multimodal_unet;
  const trend: TrendPoint[] = useMemo(
    () => (currentHistory?.epoch ?? []).map((epoch, index) => ({
      epoch,
      psnr: currentHistory?.val_psnr?.[index],
      ssim: currentHistory?.val_ssim?.[index],
    })),
    [currentHistory],
  );

  const experimentRows = experiments.data?.experiments ?? [];
  const predictions = experimentRows.filter((row) => row.kind === "prediction").length;
  const loading = report.loading && scene.loading;
  const failure = report.error || scene.error || health.error;

  return (
    <>
      <div className="dashboard-heading">
        <div>
          <div className="eyebrow">EARTH OBSERVATION · RESEARCH WORKSPACE</div>
          <h1>Satellite cloud removal</h1>
          <p>Multimodal reconstruction · evidence from your saved test and experiment data</p>
        </div>
        <Button href="/studio">New cloud removal</Button>
      </div>

      {failure && <ErrorAlert message={failure} onRetry={() => { void report.refresh(); void scene.refresh(); void health.refresh(); }} />}

      <div className="dashboard-kpis" aria-label="Research workspace metrics">
        <Kpi icon={<Database size={14} />} label="Saved experiment records" value={experiments.loading ? "…" : experiments.error ? "—" : String(experimentRows.length)} note="Successful requests only" />
        <Kpi icon={<Workflow size={14} />} label="Completed predictions" value={experiments.loading ? "…" : experiments.error ? "—" : String(predictions)} note="Saved in the API ledger" />
        <Kpi icon={<Layers3 size={14} />} label="Mean test SSIM" value={multimodal?.ssim_mean === undefined ? "—" : multimodal.ssim_mean.toFixed(3)} note={multimodal ? `${multimodal.samples ?? "—"} samples · saved test report` : "No recorded evaluation"} />
        <Kpi icon={<ShieldCheck size={14} />} label="Success rate" value="—" note="Failures are not persisted; rate unavailable" />
      </div>

      {loading ? <LoadingState label="Loading saved experiment and scene data…" /> : (
        <>
          <div className="dashboard-top">
            <SceneImageCard scene={scene.data} kind="cloudy_optical" title="Cloudy Sentinel-2" caption="Real held-out test imagery" />
            <SceneImageCard scene={scene.data} kind="sar" title="Sentinel-1 · VV / VH" caption="Real held-out radar composite" />

            <Panel className="metric-panel">
              <div className="panel-heading">
                <div>
                  <h2>Reconstruction quality</h2>
                  <p>Multimodal U-Net · held-out test report</p>
                </div>
                {multimodal && <span className="badge badge-cyan">TEST</span>}
              </div>
              {multimodal ? (
                <div className="metric-split">
                  <div className="metric-list">
                    <MetricRow value={`${multimodal.psnr_mean?.toFixed(2) ?? "—"} dB`} label="PSNR · mean" />
                    <MetricRow value={multimodal.mae_mean?.toFixed(3) ?? "—"} label="MAE · mean" />
                    <MetricRow value={multimodal.rmse_mean?.toFixed(3) ?? "—"} label="RMSE · mean" />
                  </div>
                  <MetricRing value={multimodal.ssim_mean ?? null} display={multimodal.ssim_mean === undefined ? "—" : multimodal.ssim_mean.toFixed(3)} label="SSIM" sublabel={`${multimodal.samples ?? "—"} saved test samples`} />
                </div>
              ) : <div className="empty-state"><h3>No evaluation report</h3><p>Saved held-out model metrics will appear here when the report is available.</p></div>}
            </Panel>

            <Panel className="metric-panel status-panel">
              <div className="panel-heading">
                <div><h2>Processing completion</h2><p>Job success rate</p></div>
              </div>
              <MetricRing value={null} display="N/A" label="Not tracked" sublabel="Failed jobs aren't persisted" />
              <div className={`model-unavailable ${health.data?.model.available ? "status-ready" : ""}`}>
                {health.loading ? "Checking checkpoint…" : health.data?.model.available ? `Model ready · ${health.data.model.device}` : "Inference checkpoint unavailable"}
              </div>
            </Panel>
          </div>

          <div className="dashboard-bottom">
            <Panel className="chart-panel">
              <div className="panel-heading">
                <div>
                  <h2>Validation performance</h2>
                  <p>Multimodal U-Net · saved training history</p>
                </div>
                <span className="chart-legend"><span className="legend-dot" />PSNR · dB</span>
              </div>
              {trend.length > 0 ? (
                <>
                  <PerformanceChart data={trend} metric="psnr" label="Validation PSNR" />
                  <p className="chart-footnote">{trend.length} recorded validation epochs · no interpolation</p>
                </>
              ) : <div className="empty-state"><h3>Training history unavailable</h3><p>The saved report does not contain validation observations for this model.</p></div>}
            </Panel>
            <MapCard scene={scene.data} />
          </div>
          {!dataset.loading && dataset.data && (
            <div className="dashboard-footer">
              <span>{dataset.data.configured ? dataset.data.file_count.toLocaleString() : "No"} dataset files indexed by the API</span>
              <span>{scene.data?.cloudy_optical.width ?? "—"} × {scene.data?.cloudy_optical.height ?? "—"} px · {scene.data?.cloudy_optical.bands ?? "—"} optical bands</span>
              <span>Held-out previews are contrast-stretched for display, not model inputs</span>
            </div>
          )}
        </>
      )}
    </>
  );
}

function Kpi({ icon, label, value, note }: { icon: React.ReactNode; label: string; value: string; note: string }) {
  return (
    <div className="dashboard-kpi">
      <span className="kpi-icon">{icon}</span>
      <div className="dashboard-kpi-text"><span>{label}</span><strong>{value}</strong><small>{note}</small></div>
    </div>
  );
}

function MetricRow({ value, label }: { value: string; label: string }) {
  return <div className="metric-list-row"><span className="metric-pin"><Crosshair size={13} /></span><div><strong>{value}</strong><small>{label}</small></div></div>;
}
