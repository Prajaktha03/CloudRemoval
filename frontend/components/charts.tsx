"use client";

import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ModelRow } from "@/lib/api";

const axis = { fill: "#747a82", fontSize: 9 };
const tooltipStyle = {
  background: "#111316",
  border: "1px solid #353a41",
  borderRadius: 8,
  color: "#eef0f2",
  fontSize: 10,
};

export type TrendPoint = { epoch: number; psnr?: number; ssim?: number };

export function PerformanceChart({
  data,
  metric = "psnr",
  label,
}: {
  data: TrendPoint[];
  metric?: "psnr" | "ssim";
  label: string;
}) {
  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 11, right: 5, left: -23, bottom: 0 }}>
          <defs>
            <linearGradient id="cyanFade" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#12d9f5" stopOpacity={0.23} />
              <stop offset="100%" stopColor="#12d9f5" stopOpacity={0.005} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="#292c32" strokeDasharray="0" vertical />
          <XAxis dataKey="epoch" axisLine={{ stroke: "#373b41" }} tickLine={false} tick={axis} tickFormatter={(n) => `E${n}`} />
          <YAxis axisLine={false} tickLine={false} tick={axis} domain={["auto", "auto"]} />
          <Tooltip contentStyle={tooltipStyle} labelFormatter={(epoch) => `Epoch ${epoch}`} formatter={(value) => [Number(value).toFixed(metric === "psnr" ? 2 : 3), label]} />
          <Area
            type="monotone"
            dataKey={metric}
            stroke="#12d9f5"
            strokeWidth={2}
            fill="url(#cyanFade)"
            activeDot={{ r: 4, stroke: "#101113", strokeWidth: 2, fill: "#12d9f5" }}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ModelBarChart({ data, metric }: { data: ModelRow[]; metric: "psnr_mean" | "ssim_mean" | "mae_mean" | "rmse_mean" }) {
  const shortLabel = (value: string) =>
    value.replace(/^[A-Z]_/, "").replace(/_UNet_CNN/i, " U-Net").replace(/_/g, " ");
  return (
    <div className="bar-chart-wrap">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 12, left: -19, bottom: 38 }}>
          <CartesianGrid stroke="#292c32" vertical={false} />
          <XAxis
            dataKey="model"
            axisLine={false}
            tickLine={false}
            tick={axis}
            interval={0}
            angle={-22}
            textAnchor="end"
            height={52}
            tickFormatter={shortLabel}
          />
          <YAxis axisLine={false} tickLine={false} tick={axis} />
          <Tooltip contentStyle={tooltipStyle} formatter={(value) => [Number(value).toFixed(3), metric.replace("_mean", "").toUpperCase()]} labelFormatter={shortLabel} />
          <Bar dataKey={metric} fill="#12d9f5" radius={[4, 4, 0, 0]} maxBarSize={38} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function CloudClearChart({ data }: { data: ModelRow[] }) {
  const available = data.filter((row) => Number.isFinite(row.cloud_psnr_mean) && Number.isFinite(row.clear_psnr_mean));
  return (
    <div className="bar-chart-wrap">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={available} margin={{ top: 8, right: 12, left: -19, bottom: 38 }}>
          <CartesianGrid stroke="#292c32" vertical={false} />
          <XAxis dataKey="model" axisLine={false} tickLine={false} tick={axis} interval={0} angle={-22} textAnchor="end" height={52} tickFormatter={(name) => shortModel(name)} />
          <YAxis axisLine={false} tickLine={false} tick={axis} />
          <Tooltip contentStyle={tooltipStyle} formatter={(value) => [`${Number(value).toFixed(2)} dB`, "PSNR"]} labelFormatter={shortModel} />
          <Bar dataKey="cloud_psnr_mean" name="Cloud region" fill="#12d9f5" radius={[3, 3, 0, 0]} maxBarSize={22} />
          <Bar dataKey="clear_psnr_mean" name="Clear region" fill="#597d87" radius={[3, 3, 0, 0]} maxBarSize={22} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function shortModel(value: string) {
  return value.replace(/^[A-Z]_/, "").replace(/_UNet_CNN/i, " U-Net").replace(/_/g, " ");
}

export function MetricRing({
  value,
  display,
  label,
  sublabel,
}: {
  value: number | null;
  display: string;
  label: string;
  sublabel: string;
}) {
  const circumference = 2 * Math.PI * 43;
  const fraction = value === null ? 0 : Math.min(1, Math.max(0, value));
  return (
    <div className="single-ring">
      <div className="ring-wrap" role="img" aria-label={`${label}: ${display}`}>
        <svg className="ring-svg" viewBox="0 0 100 100" aria-hidden="true">
          <circle className="ring-track" cx="50" cy="50" r="43" />
          {value !== null && (
            <circle
              className="ring-value"
              cx="50"
              cy="50"
              r="43"
              strokeDasharray={circumference}
              strokeDashoffset={circumference * (1 - fraction)}
            />
          )}
        </svg>
        <div className="ring-center">
          <strong>{display}</strong>
          <span>{label}</span>
        </div>
      </div>
      <p className="ring-subtitle">{sublabel}</p>
    </div>
  );
}
