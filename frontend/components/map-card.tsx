"use client";

import dynamic from "next/dynamic";
import { MapPin } from "lucide-react";
import type { SceneMetadata } from "@/lib/api";

const LeafletSceneMap = dynamic(() => import("./scene-map"), {
  ssr: false,
  loading: () => <div className="map-empty">Loading geospatial view…</div>,
});

export function MapCard({ scene }: { scene: SceneMetadata | null }) {
  return (
    <section className="panel map-panel">
      <div className="panel-heading">
        <div>
          <h2>Scene footprint</h2>
          <p>Held-out sample · coordinates from GeoTIFF metadata</p>
        </div>
        {scene?.center && <span className="badge badge-cyan">Georeferenced</span>}
      </div>
      <div className="map-frame">
        {scene?.center ? (
          <>
            <LeafletSceneMap center={scene.center} label={scene.scene_id} />
            <span className="map-tag"><MapPin size={11} /> {scene.scene_id}</span>
            <span className="map-location">
              {Math.abs(scene.center[0]).toFixed(4)}° {scene.center[0] < 0 ? "S" : "N"} · {Math.abs(scene.center[1]).toFixed(4)}° {scene.center[1] < 0 ? "W" : "E"} · {scene.cloudy_optical.crs}
            </span>
          </>
        ) : (
          <div className="map-empty">
            <div><MapPin size={18} /><strong>Scene coordinates unavailable</strong>Geographic coordinates will appear here when present in the source imagery.</div>
          </div>
        )}
      </div>
    </section>
  );
}
