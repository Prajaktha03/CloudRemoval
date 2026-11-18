"use client";

import L from "leaflet";
import { MapContainer, Marker, Popup, TileLayer } from "react-leaflet";
import { useState } from "react";

const markerIcon = L.divIcon({
  className: "scene-marker",
  html: '<span style="display:block;width:14px;height:14px;border:3px solid #07191d;border-radius:50%;background:#12d9f5;box-shadow:0 0 0 5px rgba(18,217,245,.2),0 0 18px rgba(18,217,245,.7)"></span>',
  iconSize: [24, 24],
  iconAnchor: [12, 12],
});

export default function SceneMap({ center, label }: { center: [number, number]; label: string }) {
  const [tilesUnavailable, setTilesUnavailable] = useState(false);

  return (
    <MapContainer center={center} zoom={8} scrollWheelZoom={false} className="leaflet-map">
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>'
        url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
        subdomains="abcd"
        opacity={tilesUnavailable ? 0 : 1}
        eventHandlers={{ tileerror: () => setTilesUnavailable(true) }}
      />
      {tilesUnavailable && <div className="map-fallback"><span>Basemap unavailable · scene coordinates preserved</span></div>}
      <Marker position={center} icon={markerIcon}>
        <Popup>{label}</Popup>
      </Marker>
    </MapContainer>
  );
}
