export const API_BASE = (
  process.env.NEXT_PUBLIC_API_URL ??
  (process.env.NODE_ENV === "production" ? "/api" : "http://localhost:8000")
).replace(/\/+$/, "");

export type Health = {
  status: string;
  model: {
    available: boolean;
    checkpoint: string;
    device: string;
    optical_channels: number;
    sar_channels: number;
    uses_cloud_mask: boolean;
    error: string | null;
  };
};

export type Experiment = {
  id: number;
  kind: string;
  created_at: string;
  metrics: Record<string, unknown>;
};

export type ModelRow = {
  model: string;
  psnr_mean?: number;
  psnr_std?: number;
  ssim_mean?: number;
  ssim_std?: number;
  mae_mean?: number;
  rmse_mean?: number;
  cloud_psnr_mean?: number;
  clear_psnr_mean?: number;
  samples?: number;
};

export type ComparisonReport = {
  summary: ModelRow[];
  split: string;
  total_samples: number;
};

export type SceneMetadata = {
  scene_id: string;
  acquisition_date: string;
  center: [number, number] | null;
  cloudy_optical: { bands: number; width: number; height: number; crs: string | null; bounds: number[] };
  sar: { bands: number; width: number; height: number; crs: string | null; bounds: number[] };
  reference: { bands: number; width: number; height: number; crs: string | null; bounds: number[] };
};

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, { ...init, credentials: "include" });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Network request failed";
    throw new Error(`Could not reach the API at ${API_BASE}: ${message}`);
  }
  if (!response.ok) {
    if (response.status === 401 && typeof window !== "undefined") {
      window.dispatchEvent(new Event("orbit:unauthorized"));
    }
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // Retain the HTTP status when the server did not return JSON.
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "An unexpected error occurred.";
}

export function formatDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

export function numeric(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}
