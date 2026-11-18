"use client";

import { useState } from "react";
import { ArrowDownToLine, ImageOff } from "lucide-react";
import { ModelBarChart } from "@/components/charts";
import { SceneImageCard } from "@/components/scene-image-card";
import { ErrorAlert, LoadingState, Notice, Panel, SectionTitle, SelectField, StatusBadge } from "@/components/ui";
import { useResource } from "@/hooks/use-resource";
import type { ComparisonReport, ModelRow, SceneMetadata } from "@/lib/api";

type MetricKey = "psnr_mean" | "ssim_mean" | "mae_mean" | "rmse_mean";
const multimodalName = "C_Multimodal_UNet_CNN";
const baselineName = "B_Optical_Only_UNet";

export default function ComparisonPage() {
  const report = useResource<ComparisonReport>("/analytics/model-comparison");
  const scene = useResource<SceneMetadata>("/dataset/sample");
  const [metric, setMetric] = useState<MetricKey>("psnr_mean");
  const multimodal = report.data?.summary.find((row) => row.model === multimodalName);
  const opticalOnly = report.data?.summary.find((row) => row.model === baselineName);
  const models = [multimodal, opticalOnly].filter((row): row is ModelRow => row !== undefined);

  function exportCsv() {
    if (!models.length) return;
    const headers = ["model", "split", "samples", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std", "mae_mean", "rmse_mean"];
    const rows = models.map((row) => headers.map((header) => {
      const value = header === "split" ? report.data?.split : row[header as keyof ModelRow];
      return typeof value === "string" ? `"${value.replace(/"/g, '""')}"` : value ?? "";
    }).join(","));
    const csv = [headers.join(","), ...rows].join("\r\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `model-comparison-${report.data?.split ?? "report"}.csv`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  return (
    <>
      <SectionTitle
        eyebrow="SCIENTIFIC EVALUATION"
        title="Model comparison"
        detail="Compare the stored multimodal U-Net and optical-only baseline evaluation summaries. Original inputs are shown where available; model output imagery was not retained."
        action={<button className="button button-secondary" disabled={!models.length} onClick={exportCsv} type="button"><ArrowDownToLine size={14} /> Export CSV</button>}
      />
      {report.error && <ErrorAlert message={report.error} onRetry={() => void report.refresh()} />}
      <Notice tone="warning">
        The report identifies the test split and sample counts, but does not include per-scene identifiers or preprocessing hashes. It is not possible to verify scene-by-scene pairing from these aggregates alone. Treat metric differences as descriptive, not as a controlled significance claim.
      </Notice>

      <div className="comparison-inputs">
        <SceneImageCard scene={scene.data} kind="cloudy_optical" title="Cloudy Sentinel-2 · input" caption="Held-out sample input" />
        <SceneImageCard scene={scene.data} kind="sar" title="Sentinel-1 · multimodal input" caption="Matched test-manifest preview" />
        <SceneImageCard scene={scene.data} kind="reference" title="Clear reference · target" caption="Reference sample, not a model output" />
      </div>

      {report.loading ? <LoadingState label="Loading saved model results…" /> : report.data ? (
        <>
          <div className="model-cards">
            <ModelSummaryCard row={multimodal} title="Multimodal U-Net" subtitle="Sentinel-1 SAR + cloudy Sentinel-2" accent="cyan" />
            <ModelSummaryCard row={opticalOnly} title="Optical-only baseline" subtitle="Cloudy Sentinel-2 only" accent="muted" />
          </div>
          <Panel className="panel-pad comparison-chart-panel">
            <div className="panel-heading panel-heading-wrap">
              <div><h2>Reported metric comparison</h2><p>Model means from the saved “{report.data.split}” evaluation split</p></div>
              <SelectField label="Metric" value={metric} onChange={(value) => setMetric(value as MetricKey)}>
                <option value="psnr_mean">PSNR</option><option value="ssim_mean">SSIM</option><option value="mae_mean">MAE</option><option value="rmse_mean">RMSE</option>
              </SelectField>
            </div>
            {models.length ? <ModelBarChart data={models} metric={metric} /> : <div className="empty-state"><h3>Comparison rows unavailable</h3><p>The saved report does not contain the expected model pair.</p></div>}
          </Panel>
          <Panel className="panel-pad">
            <div className="panel-heading"><div><h2>Evaluation details</h2><p>Computed values only · no estimates or synthetic metrics</p></div><StatusBadge tone="cyan">TEST · {report.data.total_samples} total samples</StatusBadge></div>
            <div className="table-wrap"><table className="data-table comparison-table">
              <thead><tr><th>Model</th><th>PSNR · dB</th><th>SSIM</th><th>SAM</th><th>MAE</th><th>RMSE</th><th>Inference time</th><th>N</th></tr></thead>
              <tbody>{models.map((row) => <tr key={row.model}>
                <td><strong>{row.model === multimodalName ? "Multimodal U-Net" : "Optical-only baseline"}</strong></td>
                <td>{value(row.psnr_mean)}</td><td>{value(row.ssim_mean)}</td><td>Not recorded</td><td>{value(row.mae_mean)}</td><td>{value(row.rmse_mean)}</td><td>Not recorded</td><td>{row.samples ?? "—"}</td>
              </tr>)}</tbody>
            </table></div>
          </Panel>
          <div className="comparison-output-note"><ImageOff size={16} /><div><strong>Reconstructed outputs are unavailable for side-by-side display.</strong><span>The evaluation JSON stores aggregate metrics only. It does not contain prediction image files, per-scene outputs, SAM scores, or inference timings.</span></div></div>
        </>
      ) : !report.error ? <div className="empty-state"><h3>No model comparison report</h3><p>Run and save an evaluation report to compare model metrics.</p></div> : null}
    </>
  );
}

function ModelSummaryCard({ row, title, subtitle, accent }: { row?: ModelRow; title: string; subtitle: string; accent: "cyan" | "muted" }) {
  return (
    <Panel className={`model-summary-card model-${accent}`}>
      <div className="panel-heading">
        <div><h2>{title}</h2><p>{subtitle}</p></div>
        <StatusBadge tone={row ? "success" : "warning"}>{row ? `${row.samples ?? "—"} samples` : "Not reported"}</StatusBadge>
      </div>
      {row ? (
        <div className="model-summary-metrics">
          <SummaryMetric name="PSNR" value={value(row.psnr_mean)} unit="dB" />
          <SummaryMetric name="SSIM" value={value(row.ssim_mean)} />
          <SummaryMetric name="MAE" value={value(row.mae_mean)} />
          <SummaryMetric name="Inference" value="—" unit="not recorded" />
        </div>
      ) : <div className="empty-state small-empty"><h3>Model missing from report</h3><p>This comparison uses only stored model results.</p></div>}
    </Panel>
  );
}

function SummaryMetric({ name, value: result, unit }: { name: string; value: string; unit?: string }) {
  return <div><span>{name}</span><strong>{result}</strong><small>{unit ?? "mean"}</small></div>;
}

function value(value?: number) {
  return value === undefined ? "—" : value.toFixed(3);
}
