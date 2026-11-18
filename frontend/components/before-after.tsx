"use client";

import { useState } from "react";
import Image from "next/image";

export function BeforeAfter({
  before,
  after,
  beforeLabel = "Cloudy input",
  afterLabel = "Reconstruction estimate",
}: {
  before: string;
  after: string;
  beforeLabel?: string;
  afterLabel?: string;
}) {
  const [position, setPosition] = useState(50);
  return (
    <div>
      <div className="compare-frame">
        <Image alt={beforeLabel} fill sizes="(max-width: 760px) 95vw, 50vw" src={before} unoptimized />
        <span className="compare-label left">{beforeLabel}</span>
        <div className="compare-after" style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}>
          <Image alt={afterLabel} fill sizes="(max-width: 760px) 95vw, 50vw" src={after} unoptimized />
        </div>
        <span className="compare-label right">{afterLabel}</span>
        <div aria-hidden="true" className="compare-divider" style={{ left: `${position}%` }} />
      </div>
      <label className="compare-control">
        <span>Reveal reconstruction</span>
        <input aria-label="Before and after image comparison" max="100" min="0" onChange={(event) => setPosition(Number(event.target.value))} type="range" value={position} />
      </label>
    </div>
  );
}
