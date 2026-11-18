import { fromArrayBuffer } from "geotiff";

export type RasterMetadata = {
  bands: number;
  width: number;
  height: number;
  crs: string | null;
  origin: number[] | null;
  resolution: number[] | null;
  sampleRange: [number, number];
};

export async function inspectRaster(file: File, expectedBands: number, binary = false): Promise<RasterMetadata> {
  if (!/\.tiff?$/i.test(file.name)) throw new Error("Use a GeoTIFF (.tif or .tiff) for model inputs.");
  if (file.size > 50 * 1024 * 1024) throw new Error("File exceeds the backend 50 MB upload limit.");
  const image = await (await fromArrayBuffer(await file.arrayBuffer())).getImage();
  const bands = image.getSamplesPerPixel();
  const width = image.getWidth();
  const height = image.getHeight();
  if (bands !== expectedBands) throw new Error(`${file.name} has ${bands} bands; the configured model requires ${expectedBands}.`);
  if (width < 16 || height < 16) throw new Error("Images must be at least 16 × 16 pixels.");

  const sampleWidth = Math.min(64, width);
  const sampleHeight = Math.min(64, height);
  const raster = await image.readRasters({
    samples: Array.from({ length: bands }, (_, index) => index),
    window: [0, 0, sampleWidth, sampleHeight],
    interleave: true,
  });
  const values = raster as ArrayLike<number>;
  let min = Infinity;
  let max = -Infinity;
  for (let index = 0; index < values.length; index++) {
    const value = Number(values[index]);
    if (!Number.isFinite(value)) throw new Error("Input contains non-finite pixel values.");
    if (binary && value !== 0 && value !== 1) throw new Error("Cloud-mask values must be binary: 0 = clear, 1 = obscured.");
    min = Math.min(min, value);
    max = Math.max(max, value);
  }
  if (min < 0 || max > 1) throw new Error(`Sampled pixel range [${min.toPrecision(4)}, ${max.toPrecision(4)}] is outside [0, 1]. Normalize Sentinel-2 reflectance by 10,000 and Sentinel-1 dB to [0, 1] using the project preprocessing before upload.`);

  const keys = image.getGeoKeys();
  const epsg = keys?.GeographicTypeGeoKey ?? keys?.ProjectedCSTypeGeoKey;
  return {
    bands,
    width,
    height,
    crs: typeof epsg === "number" ? `EPSG:${epsg}` : null,
    origin: image.getOrigin() ?? null,
    resolution: image.getResolution() ?? null,
    sampleRange: [min, max],
  };
}

export function alignmentIssue(reference: RasterMetadata, candidate: RasterMetadata, label: string): string | null {
  if (candidate.width !== reference.width || candidate.height !== reference.height) {
    return `${label} dimensions must match the cloudy optical image (${reference.width} × ${reference.height}px).`;
  }
  if (reference.crs && candidate.crs && reference.crs !== candidate.crs) {
    return `${label} CRS (${candidate.crs}) differs from the optical CRS (${reference.crs}).`;
  }
  if (reference.origin && candidate.origin && !close(reference.origin, candidate.origin)) {
    return `${label} georeferencing origin does not match the optical image.`;
  }
  if (reference.resolution && candidate.resolution && !close(reference.resolution, candidate.resolution)) {
    return `${label} pixel resolution does not match the optical image.`;
  }
  return null;
}

function close(left: number[], right: number[]): boolean {
  return left.length === right.length && left.every((value, index) => Math.abs(value - right[index]) <= 1e-9);
}

export async function renderRasterPreview(file: File | Blob, bands: [number, number, number] = [3, 2, 1]): Promise<string> {
  const image = await (await fromArrayBuffer(await file.arrayBuffer())).getImage();
  const count = image.getSamplesPerPixel();
  if (Math.max(...bands) >= count) throw new Error("Selected display bands are not available in this raster.");
  const values = await image.readRasters({ samples: bands, interleave: true });
  const pixels = values as ArrayLike<number>;
  const pixelCount = image.getWidth() * image.getHeight();
  const canvas = document.createElement("canvas");
  canvas.width = image.getWidth();
  canvas.height = image.getHeight();
  const context = canvas.getContext("2d");
  if (!context) throw new Error("This browser could not create an image preview.");
  const output = context.createImageData(canvas.width, canvas.height);

  const limits = bands.map((_, channel) => {
    const sample: number[] = [];
    const stride = Math.max(1, Math.floor(pixelCount / 10000));
    for (let pixel = 0; pixel < pixelCount; pixel += stride) {
      const value = Number(pixels[pixel * bands.length + channel]);
      if (Number.isFinite(value)) sample.push(value);
    }
    sample.sort((a, b) => a - b);
    if (!sample.length) return [0, 1] as [number, number];
    return [percentile(sample, 0.02), percentile(sample, 0.98)] as [number, number];
  });

  for (let pixel = 0; pixel < pixelCount; pixel++) {
    const out = pixel * 4;
    for (let channel = 0; channel < 3; channel++) {
      const [low, high] = limits[channel];
      const value = Number(pixels[pixel * bands.length + channel]);
      output.data[out + channel] = Math.round(255 * (high > low ? Math.max(0, Math.min(1, (value - low) / (high - low))) : 0));
    }
    output.data[out + 3] = 255;
  }
  context.putImageData(output, 0, 0);
  const blob = await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((result) => result ? resolve(result) : reject(new Error("The browser could not encode the image preview.")), "image/png");
  });
  return URL.createObjectURL(blob);
}

export async function renderMaskPreview(file: File | Blob): Promise<string> {
  const image = await (await fromArrayBuffer(await file.arrayBuffer())).getImage();
  if (image.getSamplesPerPixel() !== 1) throw new Error("Cloud mask previews require a single-band GeoTIFF.");

  const values = await image.readRasters({ samples: [0], interleave: true }) as ArrayLike<number>;
  const canvas = document.createElement("canvas");
  canvas.width = image.getWidth();
  canvas.height = image.getHeight();
  const context = canvas.getContext("2d");
  if (!context) throw new Error("This browser could not create an image preview.");
  const output = context.createImageData(canvas.width, canvas.height);

  for (let pixel = 0; pixel < values.length; pixel++) {
    const value = Number(values[pixel]);
    if (value !== 0 && value !== 1) {
      throw new Error("Cloud-mask values must be binary: 0 = clear, 1 = obscured.");
    }
    const offset = pixel * 4;
    output.data[offset] = value === 1 ? 255 : 34;
    output.data[offset + 1] = value === 1 ? 107 : 40;
    output.data[offset + 2] = value === 1 ? 74 : 46;
    output.data[offset + 3] = 255;
  }

  context.putImageData(output, 0, 0);
  const blob = await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((result) => result ? resolve(result) : reject(new Error("The browser could not encode the mask preview.")), "image/png");
  });
  return URL.createObjectURL(blob);
}

function percentile(sorted: number[], fraction: number): number {
  return sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * fraction))];
}
