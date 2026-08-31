/**
 * Datenexport: alles, was am Bildschirm steht, auch als Tabelle.
 *
 * Grundsatz: exportiert werden die Zahlen, aus denen die Abbildung entstanden
 * ist — nicht eine gerundete Zusammenfassung. Wer eine Abbildung in einer
 * Publikation zeigt, soll die zugehörige Tabelle beilegen können, und wer sie
 * nachrechnen will, findet in der Projektdatei dieselben Parameter wieder.
 */
import type { Assemblage, Category, MarketTable, ProjectV2 } from "../core/model.js";
import { interpolateMarket, yearRange } from "../core/model.js";
import type { EnsembleStats } from "../sim/simulate.js";
import type { DatingResult } from "../sim/inverse.js";

export type Cell = string | number;
export type Aoa = Cell[][];

/** UTF-8-Bytereihenfolge-Marke — Excel unter Windows liest UTF-8-CSV nur damit richtig. */
export const UTF8_BOM = "﻿";

export function toCSV(aoa: Aoa, delimiter = ","): string {
  return aoa.map((row) => row.map((v) => csvCell(v, delimiter)).join(delimiter)).join("\r\n");
}
export function toCSVForDownload(aoa: Aoa, delimiter = ","): string { return UTF8_BOM + toCSV(aoa, delimiter); }

function csvCell(v: Cell, delimiter: string): string {
  const s = typeof v === "number" ? (Number.isFinite(v) ? String(v) : "") : String(v);
  return s.includes(delimiter) || /["\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

const r2 = (v: number) => Math.round(v * 100) / 100;
const r4 = (v: number) => Math.round(v * 1e6) / 1e6;

/* ── Simulationsergebnis: je Jahr Mittelwert und Perzentile ── */
export function simulationAoa(e: EnsembleStats, categories: readonly Category[]): Aoa {
  const cats = categories.filter((c) => e.ids.includes(c.id));
  const nc = e.ids.length;
  const head: Cell[] = ["Jahr"];
  for (const c of cats) head.push(`${c.id} Mittel`, `${c.id} P10`, `${c.id} P90`);
  const rows: Aoa = [head];
  for (let y = 0; y < e.years.length; y++) {
    const row: Cell[] = [e.years[y]];
    for (const c of cats) {
      const k = e.ids.indexOf(c.id);
      row.push(r2(e.mean[y * nc + k]), r2(e.p10[y * nc + k]), r2(e.p90[y * nc + k]));
    }
    rows.push(row);
  }
  return rows;
}

/* ── Marktangebot, auf Einzeljahre interpoliert ── */
export function marketAoa(market: MarketTable, categories: readonly Category[], start: number, end: number): Aoa {
  const ids = categories.map((c) => c.id);
  const years = yearRange(start, end);
  const M = interpolateMarket(market, ids, years);
  const nc = ids.length;
  const rows: Aoa = [["Jahr", ...categories.map((c) => c.id), "Stützjahr"]];
  for (let y = 0; y < years.length; y++) {
    rows.push([years[y], ...ids.map((_, k) => r2(M[y * nc + k])), market.years.includes(years[y]) ? "ja" : ""]);
  }
  return rows;
}

/* ── Fundkomplexe ── */
export function assemblagesAoa(assemblages: readonly Assemblage[], categories: readonly Category[]): Aoa {
  const rows: Aoa = [["Fundkomplex", ...categories.map((c) => c.id), "Summe"]];
  for (const a of assemblages) {
    const vals = categories.map((c) => a.counts[c.id] ?? 0);
    rows.push([a.name, ...vals, vals.reduce((x, y) => x + y, 0)]);
  }
  return rows;
}

/* ── Datierung: Wahrscheinlichkeit je Jahr, eine Spalte je Fundkomplex ── */
export function datingCurvesAoa(results: ReadonlyArray<{ label: string; result: DatingResult }>): Aoa {
  if (!results.length) return [["Jahr"]];
  const years = results[0].result.years;
  const rows: Aoa = [["Jahr", ...results.map((r) => r.label)]];
  for (let i = 0; i < years.length; i++) {
    rows.push([years[i], ...results.map((r) => r4(r.result.prob[i] ?? 0))]);
  }
  return rows;
}

/* ── Datierung: Übersicht je Fundkomplex ── */
export function datingSummaryAoa(results: ReadonlyArray<{ label: string; result: DatingResult }>): Aoa {
  const rows: Aoa = [["Fundkomplex", "n", "Modus", "Erwartungswert", "Intervall von", "Intervall bis", "Intervallbreite", "Niveau", "Verfahren", "Kategorien im Komplex"]];
  for (const { label, result: r } of results) {
    rows.push([
      label, r.n,
      r.empty ? "" : r.mode,
      r.empty ? "" : r2(r.expected),
      r.empty ? "" : r.hdi[0],
      r.empty ? "" : r.hdi[1],
      r.empty ? "" : r.hdi[1] - r.hdi[0],
      r.level, r.method, r.usedIds.join(" "),
    ]);
  }
  return rows;
}

/* ── Parameterblatt: macht eine Abbildung nachrechenbar ── */
export function parametersAoa(p: ProjectV2): Aoa {
  const rows: Aoa = [["Größe", "Wert"]];
  rows.push(["Projekt", p.name]);
  rows.push(["STOCHASI-Version", p._meta.version]);
  rows.push(["Zeitraum von", p.params.startYear]);
  rows.push(["Zeitraum bis", p.params.endYear]);
  rows.push(["Ersatzrate (Vorgabe, je Jahr)", p.params.replacementDefault]);
  for (const c of p.categories) {
    const r = p.params.replacement[c.id];
    if (Number.isFinite(r)) rows.push([`Ersatzrate ${c.id} (${c.name})`, r]);
  }
  rows.push(["Streuung sigma (Prozentpunkte)", p.params.noiseSd]);
  rows.push(["Residualanteil", p.params.residual]);
  rows.push(["Läufe", p.params.runs]);
  rows.push(["Seed", p.params.seed === 0 ? "zufällig" : p.params.seed]);
  rows.push(["Neugründungsmodus", p.params.settlementMode ? "ja" : "nein"]);
  rows.push(["Vergleichsjahr", p.comparisonYear]);
  for (const c of p.categories) rows.push([`Startanteil ${c.id} (${c.name})`, r2(p.params.initial[c.id] ?? 0)]);
  return rows;
}

/** Arbeitsmappe mit allen Blättern. `xlsx` wird erst hier geladen (Code-Splitting). */
export async function toXLSX(sheets: ReadonlyArray<{ name: string; aoa: Aoa }>): Promise<Uint8Array> {
  const XLSX = await import("xlsx");
  const wb = XLSX.utils.book_new();
  for (const s of sheets) {
    // Excel erlaubt höchstens 31 Zeichen und keine der Zeichen : \ / ? * [ ]
    const name = s.name.replace(/[:\\/?*[\]]/g, "-").slice(0, 31) || "Blatt";
    XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet(s.aoa as Cell[][]), name);
  }
  return new Uint8Array(XLSX.write(wb, { type: "array", bookType: "xlsx" }) as ArrayBuffer);
}
