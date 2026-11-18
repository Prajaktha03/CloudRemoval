"use client";

import { useMemo, useState } from "react";
import { Activity, BarChart3, Timer } from "lucide-react";
import { CloudClearChart, MetricRing, ModelBarChart, PerformanceChart, TrendPoint } from "@/components/charts";
import { ErrorAlert, LoadingState, Notice, Panel, SectionTitle, SelectField, StatusBadge } from "@/components/ui";
import { useResource } from "@/hooks/use-resource";
import type { ComparisonReport, Health } from "@/lib/api";

type History = { epoch: number[]; val_psnr?: number[]; val_ssim?: number[] };
type Training = { histories: Record<string, History> };
type Metric = "psnr_mean" | "ssim_mean" | "mae_mean" | "rmse_mean";

const metricNames: Record<Metric, string> = {
  psnr_mean: "PSNR · dB",
  ssim_mean: "SSIM",
  mae_mean: "MAE",
  rmse_mean: "RMSE",
};
const emptyModels: ComparisonReport["summary"] = [];

export default function AnalyticsPage() {
  const report = useResource<ComparisonReport>("/analytics/model-comparison");
  const training = useResource<Training>("/analytics/training-history");
  const health = useResource<Health>("/health");
  const [range, setRange] = useState("all");
  const [model, setModel] = useState("all");
  const [metric, setMetric] = useState<Metric>("psnr_mean");
  const models = report.data?.summary ?? emptyModels;
  const selected = useMemo(() => models.filter((row) => model === "all" || row.model === model), [models, model]);
  const multimodal = models.find((row) => row.model === "C_Multimodal_UNet_CNN");
  const history = training.data?.histories.multimodal_unet;
  const trend: TrendPoint[] = (history?.epoch ?? []).map((epoch, index) => ({
    epoch,
    psnr: history?.val_psnr?.[index],
    ssim: history?.val_ssim?.[index],
  }));
  const rangeUnavailable = range !== "all";
  const failuresTracked = false;
  const error = report.error || training.error || health.error;

  return (
    <>
      <SectionTitle
        eyebrow="EVIDENCE · HELD-OUT EVALUATION"
        title="Model analytics"
        detail="Explore saved evaluation summaries and validation history. Every chart is sourced from files or records already produced by this repository."
        action={<StatusBadge tone="cyan">{report.data?.split ?? "Saved report"}</StatusBadge>}
      />
      <div className="analytics-filters">
        <SelectField label="Evaluation window" value={range} onChange={setRange}>
          <option value="all">Full saved test split</option><option value="7">Last 7 days</option><option value="30">Last 30 days</option><option value="90">Last 90 days</option>
        </SelectField>
        <SelectField label="Model" value={model} onChange={setModel}>
          <option value="all">All reported models</option>{models.map((row) => <option key={row.model} value={row.model}>{row.model.replace(/^[A-Z]_/, "").replace(/_/g, " ")}</option>)}
        </SelectField>
        <SelectField label="Evaluation metric" value={metric} onChange={(value) => setMetric(value as Metric)}>
          <option value="psnr_mean">PSNR</option><option value="ssim_mean">SSIM</option><option value="mae_mean">MAE</option><option value="rmse_mean">RMSE</option>
        </SelectField>
        <SelectField label="Region / scene category" value="unavailable" disabled onChange={() => undefined}>
          <option value="unavailable">Not present in report</option>
        </SelectField>
      </div>
      {error && <ErrorAlert message={error} onRetry={() => { void report.refresh(); void training.refresh(); void health.refresh(); }} />}
      {rangeUnavailable && <Notice tone="warning">The saved comparison is aggregated by test split and contains no per-sample dates. A {range}-day filter cannot be applied to the available records.</Notice>}
      <Notice tone="info">
        Report provenance: <strong>outputs/reports/model_comparison.json</strong> · split “{report.data?.split ?? "—"}” · {report.data?.total_samples ?? "—"} total samples. The report does not include per-scene dates, region labels, SAM, inference time, or failed jobs.
      </Notice>

      {report.loading ? <LoadingState label="Loading saved evaluation evidence…" /> : report.data ? rangeUnavailable ? (
        <Panel className="panel-pad"><div className="empty-state"><h3>No dated evaluation records</h3><p>The available comparison is an aggregate for the full test split. No values are shown for a date window because the source contains no per-sample timestamps.</p><button className="button button-secondary" onClick={() => setRange("all")} type="button">Show full saved test split</button></div></Panel>
      ) : (
        <>
          <div className="panel-grid grid-4 analytics-kpis">
            <Panel className="analytics-kpi"><span>Mean PSNR · multimodal U-Net</span><strong>{multimodal?.psnr_mean?.toFixed(2) ?? "—"} <small>dB</small></strong><p>{multimodal?.samples ?? "—"} evaluated samples</p></Panel>
            <Panel className="analytics-kpi"><span>Mean SSIM · multimodal U-Net</span><strong>{multimodal?.ssim_mean?.toFixed(3) ?? "—"}</strong><p>Held-out test summary</p></Panel>
            <Panel className="analytics-kpi"><span>Mean MAE · multimodal U-Net</span><strong>{multimodal?.mae_mean?.toFixed(3) ?? "—"}</strong><p>Reflectance range [0, 1]</p></Panel>
            <Panel className="analytics-kpi"><span>Processing failures</span><strong>—</strong><p>Failure records are unavailable</p></Panel>
          </div>
          <div className="analytics-grid">
            <Panel className="panel-pad">
              <div className="panel-heading"><div><h2>Model comparison</h2><p>{metricNames[metric]} · {model === "all" ? `${selected.length} reported models` : "selected model"}</p></div><BarChart3 size={16} color="#12d9f5" /></div>
              {selected.length ? <ModelBarChart data={selected} metric={metric} /> : <div className="empty-state"><h3>No values recorded</h3><p>This metric is missing for the selected model.</p></div>}
              <p className="chart-footnote">Bars show saved model means; sample-level distributions are not available.</p>
            </Panel>
            <Panel className="analytics-quality">
              <div className="panel-heading"><div><h2>Mean SSIM</h2><p>Multimodal U-Net · test split</p></div><Activity size={16} color="#12d9f5" /></div>
              <MetricRing value={multimodal?.ssim_mean ?? null} display={multimodal?.ssim_mean?.toFixed(3) ?? "—"} label="SSIM" sublabel={`${multimodal?.samples ?? "—"} reported samples`} />
              <div className="metric-strip analytics-metric-strip">
                <MiniMetric label="SAM" value="Not recorded" />
                <MiniMetric label="Inference time" value="Not recorded" />
              </div>
            </Panel>
            <Panel className="panel-pad">
              <div className="panel-heading"><div><h2>Training validation trend</h2><p>Saved multimodal U-Net history · PSNR</p></div><span className="chart-legend"><span className="legend-dot" />PSNR · dB</span></div>
              {trend.length ? <><PerformanceChart data={trend} metric="psnr" label="Validation PSNR" /><p className="chart-footnote">{trend.length} recorded validation epochs · see training log for run details</p></> : <div className="empty-state"><h3>No training points</h3><p>No saved validation history is available for this model.</p></div>}
            </Panel>
            <Panel className="panel-pad">
              <div className="panel-heading"><div><h2>Cloud / clear-region PSNR</h2><p>Region means included in the saved comparison report</p></div><StatusBadge tone="neutral">dB</StatusBadge></div>
              {selected.some((row) => Number.isFinite(row.cloud_psnr_mean) || Number.isFinite(row.clear_psnr_mean)) ? <CloudClearChart data={selected} /> : <div className="empty-state"><h3>Region metrics unavailable</h3><p>The selected report rows do not contain cloud / clear-region PSNR values.</p></div>}
              <p className="chart-footnote">These are cloud- and clear-region means, not cloud-coverage buckets or a correlation analysis.</p>
            </Panel>
          </div>
          <Panel className="panel-pad evidence-panel">
            <div className="panel-heading"><div><h2>Evidence table</h2><p>Values below are copied from the saved test comparison, not generated by the UI.</p></div><Timer size={16} color="#12d9f5" /></div>
            <div className="table-wrap"><table className="data-table">
              <thead><tr><th>Model</th><th>PSNR mean ± SD</th><th>SSIM mean ± SD</th><th>MAE mean</th><th>Samples</th></tr></thead>
              <tbody>{selected.map((row) => <tr key={row.model}><td><strong>{row.model.replace(/^[A-Z]_/, "").replace(/_/g, " ")}</strong></td><td>{format(row.psnr_mean)} ± {format(row.psnr_std)}</td><td>{format(row.ssim_mean)} ± {format(row.ssim_std)}</td><td>{format(row.mae_mean)}</td><td>{row.samples ?? "—"}</td></tr>)}</tbody>
            </table></div>
          </Panel>
        </>
      ) : !error ? <div className="empty-state"><h3>No evaluation report</h3><p>Generate a real evaluation report to populate model analytics.</p></div> : null}
      <div className="analytics-footnote"><Timer size={13} /> Checkpoint: {health.data?.model.available ? "loaded by the API" : health.loading ? "checking…" : "unavailable"} · Live analytics update when saved reports or experiment records change.</div>
    </>
  );
}

function MiniMetric({ label, value }: { label: string; value: string }) {
  return <div className="metric-chip"><span>{label}</span><strong className="not-recorded">{value}</strong></div>;
}

function format(value: number | undefined) {
  return value === undefined ? "—" : value.toFixed(3);
}
