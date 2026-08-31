/**
 * Zeichnet eine `Scene` als SVG.
 *
 * Es gibt bewusst keinen zweiten Zeichenweg für den Bildschirm: Dieselbe Szene,
 * die hier gerendert wird, geht als SVG, PNG oder PDF in den Export. Was man
 * sieht, ist die Abbildung.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import type { Scene } from "../charts/scene.js";
import { rgb } from "../charts/scene.js";
import { DARK, LIGHT, type PlotTheme } from "../charts/plot.js";
import { currentTheme, onThemeChange } from "../core/theme.js";

/** Farbschema der Diagramme, das dem Hell/Dunkel-Zustand der App folgt. */
export function usePlotTheme(): PlotTheme {
  const [dark, setDark] = useState(() => currentTheme() === "dark");
  useEffect(() => onThemeChange((t) => setDark(t === "dark")), []);
  return dark ? DARK : LIGHT;
}

/** Misst die verfügbare Breite eines Elements — die Diagramme wachsen mit dem Fenster. */
export function useWidth<T extends HTMLElement>(fallback = 900): [React.RefObject<T>, number] {
  const ref = useRef<T>(null);
  const [w, setW] = useState(fallback);
  useEffect(() => {
    const el = ref.current; if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(([e]) => {
      const next = Math.max(320, Math.round(e.contentRect.width));
      setW((prev) => (Math.abs(prev - next) > 2 ? next : prev));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

const f = (n: number) => Math.round(n * 100) / 100;

export function ChartCanvas({ scene, title }: { scene: Scene; title?: string }) {
  const paths = useMemo(() => scene.paths.map((p) => {
    let d = "";
    for (let i = 0; i + 1 < p.pts.length; i += 2) d += `${i === 0 ? "M" : "L"}${f(p.pts[i])},${f(p.pts[i + 1])} `;
    if (p.closed) d += "Z";
    return { ...p, d };
  }), [scene]);

  return (
    <svg
      className="chart"
      viewBox={`0 0 ${scene.w} ${scene.h}`}
      width="100%"
      style={{ maxWidth: scene.w, height: "auto", display: "block" }}
      role="img"
      aria-label={title}
      preserveAspectRatio="xMinYMin meet"
    >
      <rect width={scene.w} height={scene.h} fill={rgb(scene.bg)} />
      {scene.rects.map((r, i) => (
        <rect key={"r" + i} x={f(r.x)} y={f(r.y)} width={f(r.w)} height={f(r.h)} fill={rgb(r.c)} />
      ))}
      {paths.map((p, i) => (
        <path
          key={"p" + i} d={p.d}
          fill={p.fill ? rgb(p.fill) : "none"}
          stroke={p.stroke ? rgb(p.stroke) : "none"}
          strokeWidth={p.stroke ? p.width ?? 1 : undefined}
          strokeLinejoin={p.stroke ? "round" : undefined}
          strokeLinecap={p.stroke ? "round" : undefined}
          strokeDasharray={p.dash?.length ? p.dash.join(" ") : undefined}
        />
      ))}
      {scene.texts.map((t, i) => (
        <text
          key={"t" + i} x={f(t.x)} y={f(t.y)} fontSize={t.size} fill={rgb(t.c)}
          textAnchor={t.anchor} fontWeight={t.bold ? 600 : undefined}
          transform={t.rot === -90 ? `rotate(-90 ${f(t.x)} ${f(t.y)})` : undefined}
        >{t.s}</text>
      ))}
    </svg>
  );
}
