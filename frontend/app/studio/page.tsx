"use client";

import JSZip from "jszip";
import Image from "next/image";
import {
  Check,
  Download,
  FileImage,
  Maximize2,
  Minus,
  Plus,
  RotateCcw,
  UploadCloud,
} from "lucide-react";
import { type FormEvent, type MutableRefObject, useMemo, useRef, useState } from "react";
import { BeforeAfter } from "@/components/before-after";
import {
  Button,
  ErrorAlert,
  LoadingState,
  Notice,
  Panel,
  SectionTitle,
  SelectField,
  StatusBadge,
} from "@/components/ui";
import { useResource } from "@/hooks/use-resource";
import { API_BASE, type Health } from "@/lib/api";
import { alignmentIssue, inspectRaster, renderMaskPreview, renderRasterPreview, type RasterMetadata } from "@/lib/raster";

type InputKind = "optical" | "sar" | "reference" | "mask";
type SelectedRaster = { file: File; metadata: RasterMetadata; preview?: string };
type Evaluation = { psnr: number | null; psnr_is_infinite?: boolean; ssim: number; mae: number; rmse?: number };
type PredictionMetadata = { filename: string; format: string; height: number; width: number; channels: number; model_device: string };
type Phase = "validating" | "uploading" | "processing" | "evaluating" | null;

const displayBands = {
  true: { label: "True color · B4 / B3 / B2", bands: [3, 2, 1] as [number, number, number] },
  false: { label: "False color · B8 / B4 / B3", bands: [7, 3, 2] as [number, number, number] },
  swir: { label: "SWIR · B12 / B8 / B4", bands: [12, 7, 3] as [number, number, number] },
};

export default function StudioPage() {
  const health = useResource<Health>("/health");
  const [inputs, setInputs] = useState<Partial<Record<InputKind, SelectedRaster>>>({});
  const [experimentName, setExperimentName] = useState("");
  const [modelChoice, setModelChoice] = useState("multimodal");
  const [phase, setPhase] = useState<Phase>(null);
  const [progress, setProgress] = useState(0);
  const [phaseText, setPhaseText] = useState("");
  const [validationError, setValidationError] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [evaluationError, setEvaluationError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [metrics, setMetrics] = useState<Evaluation | null>(null);
  const [prediction, setPrediction] = useState<PredictionMetadata | null>(null);
  const [outputPreview, setOutputPreview] = useState<string | null>(null);
  const [outputFile, setOutputFile] = useState<File | null>(null);
  const [archiveUrl, setArchiveUrl] = useState<string | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState<number | null>(null);
  const [bandMode, setBandMode] = useState<keyof typeof displayBands>("true");
  const [viewerZoom, setViewerZoom] = useState(1);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<{ x: number; y: number } | null>(null);

  const opticalChannels = health.data?.model.optical_channels ?? 13;
  const sarChannels = health.data?.model.sar_channels ?? 2;
  const maskRequired = health.data?.model.uses_cloud_mask ?? false;
  const alignmentErrors = useMemo(() => {
    const optical = inputs.optical?.metadata;
    if (!optical) return [];
    return ([
      ["sar", "Sentinel-1 SAR"],
      ["reference", "Ground-truth reference"],
      ["mask", "Cloud mask"],
    ] as const).flatMap(([key, label]) => {
      const metadata = inputs[key]?.metadata;
      const issue = metadata ? alignmentIssue(optical, metadata, label) : null;
      return issue ? [issue] : [];
    });
  }, [inputs]);

  async function selectFile(kind: InputKind, file: File | undefined) {
    if (!file) return;
    setValidationError(null);
    setRunError(null);
    setSuccess(false);
    const previous = inputs[kind];
    if (previous?.preview) URL.revokeObjectURL(previous.preview);
    setInputs((current) => {
      const next = { ...current };
      delete next[kind];
      return next;
    });
    if (outputPreview) URL.revokeObjectURL(outputPreview);
    if (archiveUrl) URL.revokeObjectURL(archiveUrl);
    setOutputPreview(null);
    setOutputFile(null);
    setArchiveUrl(null);
    setPrediction(null);
    setMetrics(null);
    setPhase("validating");
    try {
      const channels = kind === "optical" || kind === "reference" ? opticalChannels : kind === "sar" ? sarChannels : 1;
      const metadata = await inspectRaster(file, channels, kind === "mask");
      const preview = kind === "mask"
        ? await renderMaskPreview(file)
        : kind === "optical" || kind === "reference"
          ? await renderRasterPreview(file, displayBands[bandMode].bands)
          : undefined;
      setInputs((current) => {
        const previous = current[kind]?.preview;
        if (previous) URL.revokeObjectURL(previous);
        return { ...current, [kind]: { file, metadata, preview } };
      });
    } catch (error) {
      setValidationError(error instanceof Error ? error.message : "Input validation failed.");
    } finally {
      setPhase(null);
    }
  }

  async function changeBandMode(value: string) {
    const mode = value as keyof typeof displayBands;
    setBandMode(mode);
    setPhase("validating");
    try {
      const next: Partial<Record<InputKind, SelectedRaster>> = { ...inputs };
      for (const kind of ["optical", "reference"] as const) {
        const selected = inputs[kind];
        if (selected) {
          const preview = await renderRasterPreview(selected.file, displayBands[mode].bands);
          if (selected.preview) URL.revokeObjectURL(selected.preview);
          next[kind] = { ...selected, preview };
        }
      }
      setInputs(next);
      if (outputFile && /\.tiff?$/i.test(outputFile.name)) {
        const preview = await renderRasterPreview(outputFile, displayBands[mode].bands);
        if (outputPreview) URL.revokeObjectURL(outputPreview);
        setOutputPreview(preview);
      }
    } catch (error) {
      setValidationError(error instanceof Error ? error.message : "Could not change the display bands.");
    } finally {
      setPhase(null);
    }
  }

  const isModelReady = Boolean(health.data?.model.available);
  const validationReady = Boolean(
    inputs.optical && inputs.sar && (!maskRequired || inputs.mask) && alignmentErrors.length === 0,
  );
  const canRun = Boolean(
    experimentName.trim() &&
    validationReady &&
    isModelReady &&
    modelChoice === "multimodal" &&
    !phase,
  );

  async function startReconstruction(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setRunError(null);
    setEvaluationError(null);
    setSuccess(false);
    setMetrics(null);
    if (modelChoice !== "multimodal") {
      setRunError("The API only exposes the configured multimodal U-Net for inference. Use Model Comparison for saved optical-only baseline metrics.");
      return;
    }
    if (!isModelReady || !health.data) {
      setRunError(health.error ?? "The backend reports that no compatible checkpoint is loaded.");
      return;
    }
    if (!validationReady || !inputs.optical || !inputs.sar) {
      setRunError("Provide required inputs and resolve alignment or band validation before starting.");
      return;
    }
    if (alignmentErrors.length) {
      setRunError(alignmentErrors.join(" "));
      return;
    }

    const form = new FormData();
    form.append("optical", inputs.optical.file);
    form.append("sar", inputs.sar.file);
    if (health.data.model.uses_cloud_mask && inputs.mask) form.append("cloud_mask", inputs.mask.file);
    setPhase("uploading");
    setProgress(0);
    setPhaseText("Uploading selected GeoTIFF inputs…");
    const startedAt = performance.now();
    try {
      const archive = await postPrediction(form, (loaded, total) => {
        if (total > 0) setProgress(Math.round((loaded / total) * 100));
        if (total > 0 && loaded >= total) {
          setPhase("processing");
          setPhaseText("Upload complete · waiting for the inference API response");
        }
      });
      setPhase("processing");
      setPhaseText("Reconstruction response received · unpacking output");
      const zip = await JSZip.loadAsync(archive);
      const metadataFile = zip.file("metadata.json");
      if (!metadataFile) throw new Error("The API response archive does not contain metadata.json.");
      const responseMetadata = JSON.parse(await metadataFile.async("string")) as PredictionMetadata;
      const outputEntry = Object.values(zip.files).find((entry) => /^reconstruction\.(tif|tiff|png|npz)$/i.test(entry.name) && !entry.dir);
      if (!outputEntry) throw new Error("The API response archive does not contain a reconstruction output.");
      const output = await outputEntry.async("blob");
      const outputFile = new File([output], outputEntry.name, { type: responseMetadata.format });
      let preview: string | null = null;
      if (/\.tiff?$/i.test(outputEntry.name)) {
        preview = await renderRasterPreview(outputFile, displayBands[bandMode].bands);
      } else if (/\.png$/i.test(outputEntry.name)) {
        preview = URL.createObjectURL(output);
      }
      if (outputPreview) URL.revokeObjectURL(outputPreview);
      if (archiveUrl) URL.revokeObjectURL(archiveUrl);
      setOutputPreview(preview);
      setOutputFile(outputFile);
      setArchiveUrl(URL.createObjectURL(archive));
      setPrediction(responseMetadata);
      setElapsedSeconds((performance.now() - startedAt) / 1000);
      setSuccess(true);
      setPhaseText(`Reconstruction estimate ready · ${responseMetadata.width} × ${responseMetadata.height} pixels · ${responseMetadata.channels} output bands`);

      if (inputs.reference && /\.tiff?$/i.test(outputEntry.name)) {
        setPhase("evaluating");
        setPhaseText("Calculating metrics against the uploaded reference…");
        const evaluationForm = new FormData();
        evaluationForm.append("reconstruction", outputFile);
        evaluationForm.append("reference", inputs.reference.file);
        const response = await fetch(`${API_BASE}/evaluate`, { method: "POST", body: evaluationForm, credentials: "include" });
        if (!response.ok) {
          if (response.status === 401) window.dispatchEvent(new Event("orbit:unauthorized"));
          const detail = await responseDetail(response);
          throw new EvaluationError(detail);
        }
        const result = await response.json() as { metrics: Evaluation };
        setMetrics(result.metrics);
        setPhaseText("Reconstruction and reference evaluation completed.");
      }
    } catch (error) {
      if (error instanceof EvaluationError) {
        setEvaluationError(error.message);
        setPhaseText("Reconstruction completed; reference evaluation failed.");
      } else {
        setRunError(error instanceof Error ? error.message : "The reconstruction request failed.");
      }
    } finally {
      setPhase(null);
    }
  }

  const steps = [
    { name: "Cloudy optical", detail: `${opticalChannels} bands`, done: Boolean(inputs.optical) },
    { name: "SAR radar", detail: `${sarChannels} bands`, done: Boolean(inputs.sar) },
    { name: "Reference image", detail: "Optional · metrics", done: Boolean(inputs.reference) },
    { name: "Cloud mask", detail: maskRequired ? "Required by checkpoint" : "Optional", done: Boolean(inputs.mask) },
  ];

  return (
    <>
      <SectionTitle
        eyebrow="RECONSTRUCTION WORKSPACE"
        title="Cloud removal studio"
        detail="Validate co-registered, normalized Sentinel-2 and Sentinel-1 GeoTIFFs before sending them to the configured inference checkpoint."
        action={<StatusBadge tone={health.data?.model.available ? "success" : health.loading ? "neutral" : "warning"}>{health.loading ? "Checking API" : health.data?.model.available ? "Checkpoint loaded" : "Inference unavailable"}</StatusBadge>}
      />
      {health.error && <ErrorAlert message={health.error} onRetry={() => void health.refresh()} />}
      {health.data && (
        <Notice tone={health.data.model.uses_cloud_mask ? "warning" : "info"}>
          Configured checkpoint: {health.data.model.optical_channels} optical + {health.data.model.sar_channels} SAR channels{health.data.model.uses_cloud_mask ? " + 1 cloud-mask channel" : ""} · {health.data.model.device}. The checkpoint configuration uses a mask input; this differs from the stated 15-channel (13 + 2) model contract. Supply a matching mask or update and retrain/configure a checkpoint for the 15-channel contract before scientific use. Inputs are passed through the backend normalized GeoTIFF route; cloud masks are binary (1 = obscured).
        </Notice>
      )}
      <Notice tone="info">
        Inputs must already follow the training pipeline: Sentinel-2 bands in the configured B1–B12 order (13 channels) scaled to [0, 1], Sentinel-1 VV/VH normalized from the configured dB range to [0, 1], matched spatial grids, and at least 16 × 16 pixels. The UI samples pixels for preflight; the API validates full arrays, band counts, dimensions, value ranges, CRS, and affine transforms.
      </Notice>

      <div className="step-list">
        {steps.map((step, index) => <div className={`step-item ${step.done ? "done" : index === 0 ? "active" : ""}`} key={step.name}>
          <span className="step-number">{step.done ? <Check size={13} /> : index + 1}</span>
          <div><strong>{step.name}</strong><span>{step.detail}</span></div>
        </div>)}
      </div>

      <form onSubmit={startReconstruction}>
        <div className="studio-columns">
          <div className="stack">
            <Panel className="panel-pad">
              <div className="panel-heading"><div><h2>1. Required inputs</h2><p>GeoTIFF only · max 50 MB per file · aligned pixel grid</p></div><FileImage size={16} color="#12d9f5" /></div>
              <div className="upload-grid">
                <UploadTile label="Cloudy Sentinel-2 optical" detail={`${opticalChannels} normalized bands · B1–B12 order`} selected={inputs.optical} onFile={(file) => void selectFile("optical", file)} required disabled={phase === "validating"} />
                <UploadTile label="Sentinel-1 SAR" detail={`${sarChannels} normalized bands · VV / VH`} selected={inputs.sar} onFile={(file) => void selectFile("sar", file)} required disabled={phase === "validating"} />
                <UploadTile label="Cloud-free reference" detail={`${opticalChannels} bands · enables computed metrics`} selected={inputs.reference} onFile={(file) => void selectFile("reference", file)} disabled={phase === "validating"} />
                <UploadTile label="Cloud mask" detail={maskRequired ? "Required by current checkpoint · binary 1 = obscured" : "Optional · binary 1 = obscured"} selected={inputs.mask} onFile={(file) => void selectFile("mask", file)} required={maskRequired} disabled={phase === "validating"} />
              </div>
              {phase === "validating" && <div className="preflight-loading"><LoadingState label="Reading GeoTIFF metadata and preview…" /></div>}
              {validationError && <ErrorAlert message={validationError} />}
              {alignmentErrors.map((message) => <ErrorAlert key={message} message={message} />)}
              {inputs.optical && inputs.sar && !alignmentErrors.length && (
                <Notice tone={inputs.optical.metadata.crs && inputs.sar.metadata.crs ? "success" : "warning"}>
                  {inputs.optical.metadata.width} × {inputs.optical.metadata.height} pixels · matching dimensions
                  {inputs.optical.metadata.crs && inputs.sar.metadata.crs ? ` · ${inputs.optical.metadata.crs} georeferencing checked` : " · CRS/geotransform unavailable; full spatial alignment cannot be confirmed"}
                </Notice>
              )}
            </Panel>

            <Panel className="panel-pad">
              <div className="panel-heading"><div><h2>2. Experiment configuration</h2><p>Configuration for this browser session</p></div></div>
              <div className="configuration-fields">
                <label className="field"><span>Experiment name</span><input className="text-input" maxLength={100} onChange={(event) => setExperimentName(event.target.value)} placeholder="e.g. AOI-17 rainy season" value={experimentName} /></label>
                <SelectField label="Inference model" value={modelChoice} onChange={setModelChoice}>
                  <option value="multimodal">Multimodal U-Net · configured checkpoint</option>
                  <option value="optical_only">Optical-only baseline · not available for inference</option>
                </SelectField>
              </div>
              <p className="helper">The API does not accept an experiment-name field; the name is not persisted with the server-side summary. The selected model must match the loaded checkpoint architecture.</p>
              {modelChoice === "optical_only" && <Notice tone="warning">The backend exposes only its configured checkpoint at <code>/predict</code>. Optical-only baseline results are available in Model Comparison, but this endpoint cannot run that baseline.</Notice>}
              {health.data && !health.data.model.available && <Notice tone="warning">{health.data.model.error ?? "No model checkpoint is loaded. Inference is disabled."}</Notice>}
              {runError && <ErrorAlert message={runError} />}
              {evaluationError && <ErrorAlert message={evaluationError} />}
              {success && <Notice tone="success">Reconstruction completed. The output is an estimate, not a direct satellite observation.{elapsedSeconds !== null && ` End-to-end API time: ${elapsedSeconds.toFixed(1)} s (upload + processing + download).`}</Notice>}
              <div className="start-row">
                <Button disabled={!canRun} type="submit"><UploadCloud size={15} /> Start cloud removal</Button>
                {phase && <div className="phase-status"><span className="phase-spinner" />{phaseText}</div>}
              </div>
              {phase === "uploading" && <div className="progress-wrap"><div className="progress-track"><span style={{ width: `${progress}%` }} /></div><span>{progress}% uploaded · inference time is not estimated</span></div>}
              {phase === "processing" && <div className="progress-wrap"><div className="indeterminate-track"><span /></div><span>{phaseText}</span></div>}
              {phase === "evaluating" && <div className="progress-wrap"><div className="indeterminate-track"><span /></div><span>{phaseText}</span></div>}
            </Panel>
          </div>

          <div className="stack">
            <Panel className="panel-pad preview-panel">
              <div className="panel-heading panel-heading-wrap">
                <div><h2>Input preview</h2><p>Rendered from selected GeoTIFF · visualization only</p></div>
                <SelectField label="Display bands" value={bandMode} onChange={(value) => void changeBandMode(value)}>
                  {Object.entries(displayBands).map(([key, band]) => <option key={key} value={key}>{band.label}</option>)}
                </SelectField>
              </div>
              {inputs.optical?.preview ? (
                <RasterViewer
                  src={inputs.optical.preview}
                  zoom={viewerZoom}
                  onZoom={setViewerZoom}
                  dragging={dragging}
                  onDragging={setDragging}
                  pointerRef={inputRef}
                  reset={() => setViewerZoom(1)}
                  label="Cloudy optical input"
                />
              ) : <div className="viewer viewer-empty"><div className="empty-state"><FileImage size={20} /><h3>Preview appears after preflight</h3><p>Upload a valid multispectral GeoTIFF to inspect its RGB rendering.</p></div></div>}
              {inputs.sar && <div className="selected-meta"><span>Matched SAR</span><strong>{inputs.sar.file.name}</strong><span>{inputs.sar.metadata.width} × {inputs.sar.metadata.height} · {inputs.sar.metadata.crs ?? "CRS unavailable"}</span></div>}
              {inputs.mask?.preview && (
                <div className="mask-preview">
                  <div className="panel-heading"><div><h3>Cloud mask preview</h3><p>Orange pixels are marked as obscured · dark pixels are clear</p></div></div>
                  <div className="viewer mask-viewer">
                    <Image alt="Cloud mask preview: orange pixels are obscured and dark pixels are clear" fill sizes="(max-width: 760px) 95vw, 45vw" src={inputs.mask.preview} unoptimized />
                  </div>
                </div>
              )}
            </Panel>

            <Panel className="panel-pad output-panel">
              <div className="panel-heading"><div><h2>3. Reconstructed output</h2><p>Estimated reflectance · not a direct satellite observation</p></div>{prediction && <StatusBadge tone="success">Complete</StatusBadge>}</div>
              {outputPreview && inputs.optical?.preview ? (
                <>
                  <BeforeAfter before={inputs.optical.preview} after={outputPreview} beforeLabel="Cloudy input" afterLabel="Reconstruction estimate" />
                  <div className="output-meta">
                    <span>{prediction?.width} × {prediction?.height}px · {prediction?.channels} bands</span>
                    {archiveUrl && <a className="button button-secondary" download="prediction.zip" href={archiveUrl}><Download size={14} /> Download GeoTIFF bundle</a>}
                  </div>
                </>
              ) : <div className="viewer viewer-empty"><div className="empty-state"><Maximize2 size={20} /><h3>No reconstruction yet</h3><p>Validated inputs and an available model checkpoint are required before inference can run.</p></div></div>}
              {metrics && <div className="computed-metrics"><div className="panel-heading"><div><h2>Reference-based metrics</h2><p>Computed by the backend against your paired reference</p></div><StatusBadge tone="success">Calculated</StatusBadge></div>
                <div className="metric-strip">
                  <Metric label="PSNR" value={metrics.psnr_is_infinite ? "∞" : metrics.psnr === null ? "—" : metrics.psnr.toFixed(2)} unit="dB" />
                  <Metric label="SSIM" value={metrics.ssim.toFixed(4)} />
                  <Metric label="MAE" value={metrics.mae.toFixed(4)} />
                  <Metric label="SAM" value="Not computed" />
                </div>
              </div>}
            </Panel>
          </div>
        </div>
      </form>
      <div className="studio-footer"><span>Expected optical band order: B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12</span><span>Preflight samples 64 × 64 pixels; backend validates complete rasters.</span></div>
    </>
  );
}

function UploadTile({
  label,
  detail,
  selected,
  onFile,
  required = false,
  disabled = false,
}: {
  label: string;
  detail: string;
  selected?: SelectedRaster;
  onFile: (file?: File) => void;
  required?: boolean;
  disabled?: boolean;
}) {
  return (
    <label className={`upload-dropzone ${disabled ? "upload-disabled" : ""}`}>
      <input accept=".tif,.tiff" aria-label={`Upload ${label}`} disabled={disabled} onChange={(event) => onFile(event.target.files?.[0])} type="file" />
      <UploadCloud size={17} />
      <strong>{label}{required && <span className="required-mark"> *</span>}</strong>
      <span className={selected ? "file-selected" : ""}>{selected ? `${selected.file.name} · ${selected.metadata.width} × ${selected.metadata.height} · ${selected.metadata.bands} bands` : detail}</span>
    </label>
  );
}

function RasterViewer({
  src,
  zoom,
  onZoom,
  dragging,
  onDragging,
  pointerRef,
  reset,
  label,
}: {
  src: string;
  zoom: number;
  onZoom: (value: number) => void;
  dragging: boolean;
  onDragging: (value: boolean) => void;
  pointerRef: MutableRefObject<{ x: number; y: number } | null>;
  reset: () => void;
  label: string;
}) {
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  return (
    <div
      className={`viewer raster-viewer ${dragging ? "is-panning" : ""}`}
      onPointerDown={(event) => { if (zoom > 1) { event.currentTarget.setPointerCapture(event.pointerId); pointerRef.current = { x: event.clientX, y: event.clientY }; onDragging(true); } }}
      onPointerMove={(event) => { if (dragging && pointerRef.current) { const dx = event.clientX - pointerRef.current.x; const dy = event.clientY - pointerRef.current.y; setOffset((current) => ({ x: current.x + dx, y: current.y + dy })); pointerRef.current = { x: event.clientX, y: event.clientY }; } }}
      onPointerUp={() => { onDragging(false); pointerRef.current = null; }}
      onPointerCancel={() => { onDragging(false); pointerRef.current = null; }}
    >
      <Image alt={label} className="raster-image" draggable={false} fill sizes="(max-width: 760px) 95vw, 45vw" src={src} style={{ transform: `translate(${offset.x}px, ${offset.y}px) scale(${zoom})` }} unoptimized />
      <div className="viewer-tools">
        <button aria-label="Zoom in" onClick={() => onZoom(Math.min(4, zoom + .25))} type="button"><Plus size={14} /></button>
        <button aria-label="Zoom out" onClick={() => onZoom(Math.max(1, zoom - .25))} type="button"><Minus size={14} /></button>
        <button aria-label="Reset zoom and pan" onClick={() => { reset(); setOffset({ x: 0, y: 0 }); }} type="button"><RotateCcw size={13} /></button>
      </div>
      <span className="zoom-label">{Math.round(zoom * 100)}%</span>
    </div>
  );
}

function Metric({ label, value, unit }: { label: string; value: string; unit?: string }) {
  return <div className="metric-chip"><span>{label}</span><strong>{value}</strong>{unit && <small>{unit}</small>}</div>;
}

async function postPrediction(form: FormData, onProgress: (loaded: number, total: number) => void): Promise<Blob> {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", `${API_BASE}/predict`);
    request.withCredentials = true;
    request.responseType = "blob";
    request.upload.onprogress = (event) => onProgress(event.loaded, event.total);
    request.onload = async () => {
      if (request.status >= 200 && request.status < 300) {
        resolve(request.response as Blob);
      } else {
        if (request.status === 401) window.dispatchEvent(new Event("orbit:unauthorized"));
        reject(new Error(await responseDetailFromBlob(request.response as Blob, request.status)));
      }
    };
    request.onerror = () => reject(new Error(`Network error while contacting ${API_BASE}/predict`));
    request.onabort = () => reject(new Error("The prediction request was cancelled."));
    request.send(form);
  });
}

async function responseDetail(response: Response): Promise<string> {
  try {
    const body = await response.json() as { detail?: string };
    return body.detail ?? `Evaluation failed (${response.status}).`;
  } catch {
    return `Evaluation failed (${response.status}).`;
  }
}

async function responseDetailFromBlob(blob: Blob, status: number): Promise<string> {
  try {
    const body = JSON.parse(await blob.text()) as { detail?: string };
    return body.detail ?? `Prediction failed (${status}).`;
  } catch {
    return `Prediction failed (${status}).`;
  }
}

class EvaluationError extends Error {}
