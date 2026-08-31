/**
 * Die fünf Diagramme von STOCHASI.
 *
 * Jede Funktion baut eine `Scene` und nichts sonst — dieselbe Szene zeichnet die
 * Oberfläche und geht in den Export. Es gibt keine zweite, „schönere" Fassung
 * für die Abbildung.
 *
 * Eine bewusste Abweichung von Version 1: Das Jahresspektrum ist kein
 * Kreisdiagramm mehr, sondern ein Balkendiagramm mit Unsicherheitsbalken. Bei
 * fünf Kategorien war der Kreis noch lesbar; bei zwanzig Produktionszentren ist
 * er es nicht, und er kann die Simulationsunsicherheit nicht darstellen.
 */
import type { Assemblage, Category, MarketTable, SimParams } from "../core/model.js";
import { interpolateMarket } from "../core/model.js";
import type { EnsembleStats } from "../sim/simulate.js";
import type { DatingResult, ResidualCurve } from "../sim/inverse.js";
import { type RGB, type Scene, blend, hexToRgb } from "./scene.js";
import { FONT, LIGHT, type PlotTheme, bandColor, frame, niceTicks, note, readableOn, yearLabel } from "./plot.js";
import { textWidth } from "../export/textMetrics.js";

export interface ChartOptions {
  width?: number;
  height?: number;
  theme?: PlotTheme;
  title?: string;
  subtitle?: string;
  /** Kurze Parameternotiz oben rechts in der Zeichenfläche. */
  footnote?: string;
}

/* Kategorienfarbe, gegen den Hintergrund des jeweiligen Schemas lesbar gemacht. */
const col = (c: Category, th: PlotTheme = LIGHT): RGB => readableOn(hexToRgb(c.color), th.bg);

/* ── 1. Simulation: Mittelwertkurven mit Unsicherheitsband ── */
export function simulationScene(
  e: EnsembleStats,
  categories: readonly Category[],
  opts: ChartOptions & { showBand?: boolean; lineWidth?: number } = {},
): Scene {
  const th = opts.theme ?? LIGHT;
  const cats = categories.filter((c) => e.ids.includes(c.id));
  const nc = e.ids.length;
  const showBand = opts.showBand ?? true;

  let ymax = 0;
  for (const c of cats) {
    const k = e.ids.indexOf(c.id);
    for (let y = 0; y < e.years.length; y++) ymax = Math.max(ymax, showBand ? e.p90[y * nc + k] : e.mean[y * nc + k]);
  }
  const yTicks = niceTicks(0, Math.min(100, Math.max(10, Math.ceil(ymax / 10) * 10)), 5);
  const fr = frame({
    ...opts, theme: th,
    xTicks: niceTicks(e.years[0], e.years[e.years.length - 1], 7),
    xDomain: [e.years[0], e.years[e.years.length - 1]],
    yTicks, yDomain: [0, yTicks[yTicks.length - 1]],
    xFormat: yearLabel, yFormat: (v) => `${v} %`,
    xLabel: "Jahr n. Chr.", yLabel: "Anteil am Umlaufbestand",
    legendItems: cats.map((c) => ({ label: c.name, color: col(c, th) })),
  });

  // erst alle Bänder, dann alle Linien — sonst verdeckt ein späteres Band eine frühere Linie
  if (showBand) {
    for (const c of cats) {
      const k = e.ids.indexOf(c.id);
      const up: number[] = [], dn: number[] = [];
      for (let y = 0; y < e.years.length; y++) {
        up.push(fr.xs(e.years[y]), fr.ys(e.p90[y * nc + k]));
        dn.push(fr.xs(e.years[y]), fr.ys(e.p10[y * nc + k]));
      }
      const back: number[] = [];
      for (let i = dn.length - 2; i >= 0; i -= 2) back.push(dn[i], dn[i + 1]);
      fr.scene.paths.push({ pts: [...up, ...back], fill: bandColor(col(c, th), th), closed: true });
    }
  }
  for (const c of cats) {
    const k = e.ids.indexOf(c.id);
    const pts: number[] = [];
    for (let y = 0; y < e.years.length; y++) pts.push(fr.xs(e.years[y]), fr.ys(e.mean[y * nc + k]));
    fr.scene.paths.push({ pts, stroke: col(c, th), width: opts.lineWidth ?? 2 });
  }
  if (opts.footnote) note(fr, opts.footnote);
  return fr.scene;
}

/* ── 2. Marktangebot ── */
export function marketScene(
  market: MarketTable,
  categories: readonly Category[],
  opts: ChartOptions & { start?: number; end?: number } = {},
): Scene {
  const th = opts.theme ?? LIGHT;
  const ids = categories.map((c) => c.id);
  const start = opts.start ?? market.years[0] ?? 0;
  const end = opts.end ?? market.years[market.years.length - 1] ?? start + 100;
  const years: number[] = [];
  for (let y = start; y <= end; y++) years.push(y);
  const M = interpolateMarket(market, ids, years);
  const nc = ids.length;

  let ymax = 0;
  for (let i = 0; i < M.length; i++) ymax = Math.max(ymax, M[i]);
  const mTicks = niceTicks(0, Math.min(100, Math.max(10, Math.ceil(ymax / 10) * 10)), 5);
  const fr = frame({
    ...opts, theme: th,
    xTicks: niceTicks(start, end, 7),
    xDomain: [start, end],
    yTicks: mTicks, yDomain: [0, mTicks[mTicks.length - 1]],
    xFormat: yearLabel, yFormat: (v) => `${v} %`,
    xLabel: "Jahr n. Chr.", yLabel: "Anteil am Marktangebot",
    legendItems: categories.map((c) => ({ label: c.name, color: col(c, th) })),
  });
  categories.forEach((c, k) => {
    const pts: number[] = [];
    for (let y = 0; y < years.length; y++) pts.push(fr.xs(years[y]), fr.ys(M[y * nc + k]));
    fr.scene.paths.push({ pts, stroke: col(c, th), width: 2 });
  });
  // Stützjahre als kleine Marken — zeigt, wo Daten stehen und wo interpoliert wird
  for (const sy of market.years) {
    if (sy < start || sy > end) continue;
    const x = fr.xs(sy);
    fr.scene.paths.push({ pts: [x, fr.plot.y + fr.plot.h, x, fr.plot.y + fr.plot.h - 5], stroke: th.dim, width: 1 });
  }
  if (opts.footnote) note(fr, opts.footnote);
  return fr.scene;
}

/* ── 3. Jahresspektrum: Balken mit Unsicherheit, optional gemessene Werte ── */
export function spectrumScene(
  e: EnsembleStats,
  categories: readonly Category[],
  year: number,
  opts: ChartOptions & { observed?: Assemblage } = {},
): Scene {
  const th = opts.theme ?? LIGHT;
  const nc = e.ids.length;
  const yi = Math.max(0, e.years.indexOf(year));
  const cats = categories.filter((c) => e.ids.includes(c.id));

  const obsTotal = opts.observed ? Object.values(opts.observed.counts).reduce((a, b) => a + b, 0) : 0;
  const withObs = obsTotal > 0;
  const rows = cats.map((c) => {
    const k = e.ids.indexOf(c.id);
    return {
      cat: c,
      mean: e.mean[yi * nc + k], p10: e.p10[yi * nc + k], p90: e.p90[yi * nc + k],
      obs: withObs ? ((opts.observed!.counts[c.id] ?? 0) / obsTotal) * 100 : 0,
      count: withObs ? opts.observed!.counts[c.id] ?? 0 : 0,
    };
  });
  const vmax = Math.max(10, ...rows.map((r) => Math.max(r.p90, r.obs)));
  const xTicks = niceTicks(0, Math.ceil(vmax / 10) * 10, 5);

  const labelW = Math.max(...rows.map((r) => textWidth(r.cat.name, FONT.tick)), 30);
  const rowH = withObs ? 26 : 18;
  const W = opts.width ?? 860;
  const H = opts.height ?? 74 + rows.length * rowH
    + (opts.title ? FONT.title + 6 : 0) + (opts.subtitle ? FONT.tick + 4 : 0) + (withObs ? 25 : 0);

  const fr = frame({
    ...opts, width: W, height: H, theme: th,
    xTicks, xDomain: [0, xTicks[xTicks.length - 1]],
    yTicks: [], yDomain: [0, 1],
    xFormat: (v) => `${v} %`, yFormat: () => "",
    grid: "x", rail: "bottom", minLeft: 14 + labelW + 10,
    xLabel: `Anteil im Jahr ${yearLabel(year)}`,
    legendItems: withObs
      ? [
        { label: "Simulation: Mittelwert, Balken 10.–90. Perzentil", color: blend(th.label, th.bg, 0.45) },
        { label: `Fundkomplex ${opts.observed!.name} (n = ${obsTotal})`, color: th.label },
      ]
      : [{ label: "Simulation: Mittelwert, Balken 10.–90. Perzentil", color: blend(th.label, th.bg, 0.45) }],
  });

  const xs = fr.xs, x0 = xs(0);
  const step = fr.plot.h / Math.max(1, rows.length);
  const barH = Math.min(withObs ? 9 : 11, step * (withObs ? 0.34 : 0.55));

  rows.forEach((r, i) => {
    const cy = fr.plot.y + i * step + step / 2;
    fr.scene.texts.push({ x: fr.plot.x - 10, y: cy + FONT.tick * 0.35, s: r.cat.name, size: FONT.tick, c: th.label, anchor: "end", rot: 0 });

    // Simulation — obere Zeile, wenn ein Fundkomplex daruntersteht
    const ySim = withObs ? cy - barH - 1 : cy - barH / 2;
    fr.scene.rects.push({ x: x0, y: ySim, w: Math.max(0, xs(r.mean) - x0), h: barH, c: blend(col(r.cat, th), th.bg, 0.45) });
    const yw = ySim + barH / 2;
    fr.scene.paths.push({ pts: [xs(r.p10), yw, xs(r.p90), yw], stroke: th.label, width: 1 });
    fr.scene.paths.push({ pts: [xs(r.p10), yw - 3, xs(r.p10), yw + 3], stroke: th.label, width: 1 });
    fr.scene.paths.push({ pts: [xs(r.p90), yw - 3, xs(r.p90), yw + 3], stroke: th.label, width: 1 });

    if (withObs) {
      fr.scene.rects.push({ x: x0, y: cy + 1, w: Math.max(0, xs(r.obs) - x0), h: barH, c: col(r.cat, th) });
      // Die Stückzahl steht rechts neben dem längsten Element der Zeile. Reicht
      // der Platz nicht, rückt sie in den Balken — abgeschnitten wäre sie wertlos.
      const label = `${r.count}`;
      const right = fr.plot.x + fr.plot.w;
      const outside = xs(Math.max(r.mean, r.obs, r.p90)) + 7;
      const fits = outside + textWidth(label, FONT.value) <= right;
      fr.scene.texts.push({
        x: fits ? outside : xs(r.obs) - 5, y: cy + barH * 0.9,
        s: label, size: FONT.value,
        c: fits ? th.dim : th.bg, anchor: fits ? "start" : "end", rot: 0,
      });
    }
  });
  if (opts.footnote) note(fr, opts.footnote);
  return fr.scene;
}

/* ── 4. Abweichung Simulation ↔ Fundkomplex ── */
export function deviationScene(
  e: EnsembleStats,
  categories: readonly Category[],
  year: number,
  observed: Assemblage,
  opts: ChartOptions = {},
): Scene {
  const th = opts.theme ?? LIGHT;
  const nc = e.ids.length;
  const yi = Math.max(0, e.years.indexOf(year));
  const total = Object.values(observed.counts).reduce((a, b) => a + b, 0) || 1;
  const cats = categories.filter((c) => e.ids.includes(c.id));
  const rows = cats.map((c) => {
    const k = e.ids.indexOf(c.id);
    return {
      cat: c,
      d: ((observed.counts[c.id] ?? 0) / total) * 100 - e.mean[yi * nc + k],
      /** Liegt der gemessene Anteil innerhalb des simulierten 10.–90.-Perzentils? */
      inBand: ((observed.counts[c.id] ?? 0) / total) * 100 >= e.p10[yi * nc + k] - 1e-9
           && ((observed.counts[c.id] ?? 0) / total) * 100 <= e.p90[yi * nc + k] + 1e-9,
    };
  });
  const amax = Math.max(5, ...rows.map((r) => Math.abs(r.d)));
  const lim = Math.ceil(amax / 5) * 5;

  const rowH = 20;
  const W = opts.width ?? 860;
  const H = opts.height ?? 74 + rows.length * rowH + (opts.title ? FONT.title + 6 : 0) + (opts.subtitle ? FONT.tick + 4 : 0) + 25;
  const fr = frame({
    ...opts, width: W, height: H, theme: th,
    xTicks: niceTicks(-lim, lim, 6), xDomain: [-lim, lim],
    yTicks: [], yDomain: [0, 1],
    xFormat: (v) => `${v > 0 ? "+" : ""}${v}`, yFormat: () => "",
    grid: "x", rail: "bottom",
    xLabel: `Prozentpunkte Fundkomplex minus Simulation (Jahr ${yearLabel(year)})`,
    legendItems: [
      { label: "innerhalb des 10.–90. Perzentils der Simulation", color: th.label },
      { label: "außerhalb — hier weicht der Befund vom Modell ab", color: th.accent },
    ],
  });

  const zero = fr.xs(0);
  fr.scene.paths.push({ pts: [zero, fr.plot.y, zero, fr.plot.y + fr.plot.h], stroke: th.axis, width: 1 });
  const step = fr.plot.h / Math.max(1, rows.length);
  const barH = Math.min(12, step * 0.6);
  rows.forEach((r, i) => {
    const cy = fr.plot.y + i * step + step / 2;
    const x = fr.xs(r.d);
    fr.scene.rects.push({ x: Math.min(zero, x), y: cy - barH / 2, w: Math.abs(x - zero), h: barH, c: col(r.cat, th) });
    // Abweichungen außerhalb des Bands bekommen eine Marke — sonst liest man
    // jeden Balken als Befund, obwohl die meisten im Zufallsbereich liegen.
    if (!r.inBand) {
      const tip = r.d >= 0 ? x + 4 : x - 4;
      fr.scene.paths.push({ pts: [tip, cy - barH / 2 - 3, tip, cy + barH / 2 + 3], stroke: th.accent, width: 2 });
    }
    fr.scene.texts.push({
      x: r.d >= 0 ? zero - 8 : zero + 8, y: cy + FONT.tick * 0.35,
      s: r.cat.name, size: FONT.tick, c: r.inBand ? th.label : th.accent, anchor: r.d >= 0 ? "end" : "start", rot: 0,
    });
  });
  if (opts.footnote) note(fr, opts.footnote);
  return fr.scene;
}

/* ── 5. Inverse Datierung ── */
export interface DatingSeries { label: string; result: DatingResult; color: RGB; dash?: number[]; }

export function datingScene(series: readonly DatingSeries[], opts: ChartOptions & { markHdi?: boolean } = {}): Scene {
  const th = opts.theme ?? LIGHT;
  if (!series.length) return frame({ ...opts, theme: th, xTicks: [0, 1], yTicks: [0, 1] }).scene;
  const years = series[0].result.years;
  let pmax = 0;
  for (const s of series) for (const p of s.result.prob) pmax = Math.max(pmax, p);
  const yTicks = niceTicks(0, pmax * 1.12, 4);

  const fr = frame({
    ...opts, theme: th,
    xTicks: niceTicks(years[0], years[years.length - 1], 7),
    xDomain: [years[0], years[years.length - 1]],
    yTicks, yDomain: [0, Math.max(yTicks[yTicks.length - 1], pmax * 1.05)],
    xFormat: yearLabel,
    yFormat: (v) => (v === 0 ? "0" : `${(v * 100).toFixed(v < 0.01 ? 2 : 1)} %`),
    xLabel: "Ablagerungsjahr n. Chr.", yLabel: "Wahrscheinlichkeit je Jahr",
    legendItems: series.map((s) => ({ label: s.label, color: s.color, dash: s.dash })),
  });

  for (const s of series) {
    if (opts.markHdi !== false && series.length <= 2 && !s.result.empty) {
      const [lo, hi] = s.result.hdi;
      const x0 = fr.xs(lo), x1 = fr.xs(hi);
      fr.scene.rects.push({ x: Math.min(x0, x1), y: fr.plot.y, w: Math.abs(x1 - x0), h: fr.plot.h, c: bandColor(s.color, th, 0.12) });
    }
    const pts: number[] = [];
    for (let i = 0; i < years.length; i++) pts.push(fr.xs(years[i]), fr.ys(s.result.prob[i]));
    fr.scene.paths.push({ pts, stroke: s.color, width: 2, dash: s.dash });
  }
  // Modus der ersten Reihe markieren
  const first = series[0].result;
  if (!first.empty) {
    const x = fr.xs(first.mode);
    fr.scene.paths.push({ pts: [x, fr.plot.y + fr.plot.h, x, fr.plot.y], stroke: series[0].color, width: 1, dash: [3, 3] });
    fr.scene.texts.push({
      x, y: fr.plot.y - 4,
      s: `${yearLabel(first.mode)} (${yearLabel(first.hdi[0])}–${yearLabel(first.hdi[1])}, ${Math.round(first.level * 100)} %)`,
      size: FONT.tick, c: th.text, anchor: "middle", rot: 0,
    });
  }
  if (opts.footnote) note(fr, opts.footnote);
  return fr.scene;
}

/** Bequemer Weg von `dateAcrossResidual` zu den Reihen des Datierungsdiagramms. */
export function residualSeries(curves: readonly ResidualCurve[], base: RGB, th: PlotTheme = LIGHT): DatingSeries[] {
  const dashes: number[][] = [[], [5, 3], [2, 3], [8, 3, 2, 3]];
  return curves.map((c, i) => ({
    label: `Residualität ${Math.round(c.residual * 100)} %`,
    result: c.result,
    color: i === 0 ? base : bandColor(base, th, 1 - i * 0.18),
    dash: dashes[i % dashes.length].length ? dashes[i % dashes.length] : undefined,
  }));
}

/** Kurzfassung eines Parametersatzes für die Fußnote einer Abbildung. */
export function paramNote(p: SimParams): string {
  const parts = [
    `Ersatzrate ${(p.replacementDefault * 100).toFixed(0)} %/Jahr`,
    `σ ${p.noiseSd}`,
    `${p.runs} Läufe`,
    p.residual > 0 ? `Residualität ${(p.residual * 100).toFixed(0)} %` : null,
    p.seed ? `Seed ${p.seed}` : "Seed zufällig",
  ].filter(Boolean);
  return parts.join(" · ");
}
