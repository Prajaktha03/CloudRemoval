import Image from "next/image";
import { CalendarDays, ScanLine } from "lucide-react";
import { API_BASE, SceneMetadata } from "@/lib/api";

export function SceneImageCard({
  scene,
  kind,
  title,
  caption,
}: {
  scene: SceneMetadata | null;
  kind: "cloudy_optical" | "sar" | "reference";
  title: string;
  caption: string;
}) {
  return (
    <article className="panel image-card">
      {scene ? (
        <Image
          alt={`${title}, ${scene.scene_id}`}
          className="scene-card-image"
          fill
          sizes="(max-width: 760px) 48vw, 25vw"
          src={`${API_BASE}/dataset/sample/${kind}`}
          unoptimized
        />
      ) : (
        <div className="map-empty"><ScanLine size={18} />Connect the API to load real scene previews.</div>
      )}
      {scene && <div className="image-shade" />}
      <div className="image-label"><ScanLine size={12} />{title}</div>
      <div className="image-caption">
        <CalendarDays size={11} /> {scene ? `${scene.scene_id} · ${scene.acquisition_date}` : caption}
      </div>
    </article>
  );
}
