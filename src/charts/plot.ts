/**
 * Gemeinsames Gerüst aller Diagramme: Ränder, Skalen, Achsen, Legende.
 *
 * Alle Diagramme von STOCHASI teilen sich dieses Gerüst, damit sie in einer
 * Publikation nebeneinander stehen können, ohne dass Schriftgrößen,
 * Achsenabstände und Farbtöne springen.
 */
import { type RGB, type Scene, type Txt, emptyScene, blend } from "./scene.js";
import { textWidth } from "../export/textMetrics.js";

export interface PlotTheme {
  bg: RGB; text: RGB; label: RGB; dim: RGB; grid: RGB; axis: RGB; accent: RGB;
}

/** Helles Papierschema — die Grundlage jedes Exports. */
export const LIGHT: PlotTheme = {
  bg: [255, 255, 255], text: [28, 27, 26], label: [93, 88, 79], dim: [139, 133, 124],
  grid: [232, 229, 225], axis: [180, 175, 168], accent: [168, 29, 38],
};
/** Dunkles Schema für die Bildschirmansicht. */
export const DARK: PlotTheme = {
  bg: [26, 24, 23], text: [241, 237, 232], label: [201, 195, 187], dim: [148, 141, 132],
  grid: [57, 53, 47], axis: [75, 70, 63], accent: [255, 106, 99],
};

export const FONT = { tick: 9, axis: 10, title: 13, legend: 9, value: 9 };

export interface Margins { top: number; right: number; bottom: number; left: number; }

export interface PlotOptions {
  width?: number;
  height?: number;
  theme?: PlotTheme;
  title?: string;
  subtitle?: string;
  /** Legende zeichnen (Vorgabe: ja, wenn Einträge übergeben werden). */
  legend?: boolean;
}

export interface Scale { (v: number): number; domain: [number, number]; range: [number, number]; }

export function linear(domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain, [r0, r1] = range;
  const span = d1 - d0;
  const s = ((v: number) => (span === 0 ? (r0 + r1) / 2 : r0 + ((v - d0) / span) * (r1 - r0))) as Scale;
  s.domain = domain; s.range = range;
  return s;
}

/**
 * Achsenteilung mit runden Schrittweiten (1, 2, 2.5, 5, 10 × 10^k).
 * Liefert Werte innerhalb des Bereichs, höchstens `count` + 1 Stück.
 */
export function niceTicks(min: number, max: number, count = 6): number[] {
  if (!(max > min)) return [min];
  const raw = (max - min) / Math.max(1, count);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 2.5 ? 2.5 : norm <= 5 ? 5 : 10) * mag;
  const out: number[] = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) {
    out.push(Math.abs(v) < step * 1e-9 ? 0 : Math.round(v * 1e6) / 1e6);
  }
  return out;
}

/** Jahreszahl für die Achse: negative Jahre als „v. Chr." kenntlich. */
export function yearLabel(y: number): string {
  return y < 0 ? `${-y} v.` : String(y);
}

export interface LegendItem { label: string; color: RGB; dash?: number[]; }

export interface Frame {
  scene: Scene;
  m: Margins;
  /** Zeichenfläche. */
  plot: { x: number; y: number; w: number; h: number };
  theme: PlotTheme;
}

/**
 * Baut Titel, Achsenraum und Legende und liefert die freie Zeichenfläche.
 * Die Ränder folgen den tatsächlichen Textbreiten, damit nichts abgeschnitten
 * wird und bei wenigen Kategorien kein leerer Streifen stehen bleibt.
 */
export function frame(opts: PlotOptions & {
  xTicks: number[]; yTicks: number[];
  /** Wertebereich der Achsen. Ohne Angabe spannen die Teilstriche ihn auf.
      Getrennt geführt, weil Teilstriche runde Zahlen sind und deshalb selten
      genau am Datenrand liegen — ohne diese Trennung liefe die letzte Kurve
      über den Rahmen hinaus. */
  xDomain?: [number, number]; yDomain?: [number, number];
  /** Welche Achse Gitterlinien bekommt. Bei liegenden Balken ist das die x-Achse. */
  grid?: "x" | "y" | "none";
  /** Achsenschiene links und unten zeichnen. Bei liegenden Balken stört die linke. */
  rail?: "both" | "bottom" | "none";
  /** Mindestbreite des linken Rands — für Diagramme mit Zeilenbeschriftung. */
  minLeft?: number;
  xFormat?: (v: number) => string; yFormat?: (v: number) => string;
  xLabel?: string; yLabel?: string;
  legendItems?: LegendItem[];
}): Frame & { xs: Scale; ys: Scale } {
  const th = opts.theme ?? LIGHT;
  const W = opts.width ?? 900, H = opts.height ?? 520;
  const xFmt = opts.xFormat ?? String, yFmt = opts.yFormat ?? String;
  const items = opts.legendItems ?? [];
  const showLegend = (opts.legend ?? items.length > 0) && items.length > 0;

  const yTickW = Math.max(...opts.yTicks.map((t) => textWidth(yFmt(t), FONT.tick)), 0);
  const m: Margins = {
    top: 14 + (opts.title ? FONT.title + 6 : 0) + (opts.subtitle ? FONT.tick + 4 : 0),
    right: 16,
    bottom: 14 + FONT.tick + 6 + (opts.xLabel ? FONT.axis + 6 : 0),
    left: Math.max(opts.minLeft ?? 0, 14 + yTickW + 8) + (opts.yLabel ? FONT.axis + 6 : 0),
  };

  // Legende unten, umbrechend
  const SW = 14, GAP = 5, ITEM_GAP = 16, LINE = 15;
  let legendLines: LegendItem[][] = [];
  if (showLegend) {
    const avail = W - m.left - m.right;
    let line: LegendItem[] = [], lw = 0;
    for (const it of items) {
      const w = SW + GAP + textWidth(it.label, FONT.legend);
      if (line.length && lw + ITEM_GAP + w > avail) { legendLines.push(line); line = []; lw = 0; }
      lw += (line.length ? ITEM_GAP : 0) + w;
      line.push(it);
    }
    if (line.length) legendLines.push(line);
    m.bottom += 10 + legendLines.length * LINE;
  }

  const plot = { x: m.left, y: m.top, w: Math.max(10, W - m.left - m.right), h: Math.max(10, H - m.top - m.bottom) };
  const scene = emptyScene(W, H, th.bg);

  if (opts.title) scene.texts.push({ x: m.left, y: 14 + FONT.title, s: opts.title, size: FONT.title, c: th.text, anchor: "start", rot: 0, bold: true });
  if (opts.subtitle) scene.texts.push({ x: m.left, y: 14 + (opts.title ? FONT.title + 6 : 0) + FONT.tick, s: opts.subtitle, size: FONT.tick, c: th.dim, anchor: "start", rot: 0 });

  const xDom = opts.xDomain ?? [opts.xTicks[0] ?? 0, opts.xTicks[opts.xTicks.length - 1] ?? 1];
  const yDom = opts.yDomain ?? [opts.yTicks[0] ?? 0, opts.yTicks[opts.yTicks.length - 1] ?? 1];
  const xs = linear(xDom, [plot.x, plot.x + plot.w]);
  const ys = linear(yDom, [plot.y + plot.h, plot.y]);

  const grid = opts.grid ?? "y";
  const rail = opts.rail ?? "both";

  for (const t of opts.yTicks) {
    if (t < Math.min(yDom[0], yDom[1]) - 1e-9 || t > Math.max(yDom[0], yDom[1]) + 1e-9) continue;
    const y = ys(t);
    if (grid === "y") scene.paths.push({ pts: [plot.x, y, plot.x + plot.w, y], stroke: th.grid, width: 1 });
    const lab = yFmt(t);
    if (lab) scene.texts.push({ x: plot.x - 8, y: y + FONT.tick * 0.35, s: lab, size: FONT.tick, c: th.label, anchor: "end", rot: 0 });
  }
  for (const t of opts.xTicks) {
    if (t < Math.min(xDom[0], xDom[1]) - 1e-9 || t > Math.max(xDom[0], xDom[1]) + 1e-9) continue;
    const x = xs(t);
    if (grid === "x") scene.paths.push({ pts: [x, plot.y, x, plot.y + plot.h], stroke: th.grid, width: 1 });
    scene.paths.push({ pts: [x, plot.y + plot.h, x, plot.y + plot.h + 4], stroke: th.axis, width: 1 });
    const lab = xFmt(t);
    if (lab) scene.texts.push({ x, y: plot.y + plot.h + 4 + FONT.tick + 3, s: lab, size: FONT.tick, c: th.label, anchor: "middle", rot: 0 });
  }
  if (rail === "both") scene.paths.push({ pts: [plot.x, plot.y, plot.x, plot.y + plot.h, plot.x + plot.w, plot.y + plot.h], stroke: th.axis, width: 1 });
  else if (rail === "bottom") scene.paths.push({ pts: [plot.x, plot.y + plot.h, plot.x + plot.w, plot.y + plot.h], stroke: th.axis, width: 1 });

  if (opts.xLabel) scene.texts.push({ x: plot.x + plot.w / 2, y: H - (showLegend ? 10 + legendLines.length * LINE : 0) - 10, s: opts.xLabel, size: FONT.axis, c: th.label, anchor: "middle", rot: 0 });
  if (opts.yLabel) scene.texts.push({ x: 14 + FONT.axis, y: plot.y + plot.h / 2, s: opts.yLabel, size: FONT.axis, c: th.label, anchor: "middle", rot: -90 });

  if (showLegend) {
    const top = H - 10 - legendLines.length * LINE;
    legendLines.forEach((ln, k) => {
      let x = m.left;
      const baseline = top + k * LINE + FONT.legend;
      for (const it of ln) {
        if (it.dash) scene.paths.push({ pts: [x, baseline - FONT.legend * 0.35, x + SW, baseline - FONT.legend * 0.35], stroke: it.color, width: 2, dash: it.dash });
        else scene.rects.push({ x, y: baseline - FONT.legend + 1, w: SW, h: FONT.legend - 1, c: it.color });
        scene.texts.push({ x: x + SW + GAP, y: baseline, s: it.label, size: FONT.legend, c: th.label, anchor: "start", rot: 0 });
        x += SW + GAP + textWidth(it.label, FONT.legend) + ITEM_GAP;
      }
    });
  }

  return { scene, m, plot, theme: th, xs, ys };
}

/** Kurze Notiz unten rechts, z. B. Parameterangabe zur Abbildung. */
export function note(fr: Frame, text: string): void {
  fr.scene.texts.push({
    x: fr.plot.x + fr.plot.w, y: fr.plot.y - 4,
    s: text, size: FONT.tick, c: fr.theme.dim, anchor: "end", rot: 0,
  });
}

/** Relative Leuchtdichte nach WCAG. */
function luminance(c: RGB): number {
  const f = (v: number) => { const x = v / 255; return x <= 0.03928 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4; };
  return 0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2]);
}
function contrast(a: RGB, b: RGB): number {
  const la = luminance(a), lb = luminance(b);
  return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
}

/**
 * Hebt eine Kategorienfarbe so weit vom Hintergrund ab, dass die Kurve sichtbar
 * bleibt.
 *
 * Die Farben der Produktionszentren sind für weißes Papier gewählt; auf dem
 * dunklen Bildschirmhintergrund verschwindet ein tiefes Weinrot fast. Statt
 * einen zweiten Farbsatz zu pflegen, wird die Farbe hier schrittweise zum
 * helleren Ende gezogen, bis der Kontrast reicht. Im hellen Schema greift das
 * praktisch nie — die exportierte Abbildung bleibt also unverändert.
 */
export function readableOn(c: RGB, bg: RGB, min = 2.6): RGB {
  const toward: RGB = luminance(bg) < 0.5 ? [255, 255, 255] : [0, 0, 0];
  let out = c;
  for (let i = 0; i < 14 && contrast(out, bg) < min; i++) out = blend(toward, out, 0.09);
  return out;
}

/** Bandfarbe: Kategorienfarbe schwach über den Hintergrund gelegt. */
export function bandColor(c: RGB, th: PlotTheme, strength = 0.22): RGB {
  return blend(c, th.bg, strength);
}

export function pushText(sc: Scene, t: Txt): void { sc.texts.push(t); }
