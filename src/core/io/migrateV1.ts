/**
 * Konfigurationen von STOCHASI 1 einlesen.
 *
 * Version 1 hat zwei Spielarten geschrieben: eine schlanke, die nur die
 * Parameter enthält, und eine vollständige mit eingebetteten Markt- und
 * Grabungsdaten (`_meta.includes_data`). Beide werden hier auf `ProjectV2`
 * abgebildet.
 *
 * Was dabei entsteht und was nicht:
 *  - Kategorien, Marktkurve, Zeitraum, Startverteilung, Rauschen, Läufe, Seed,
 *    Neugründungsmodus und der Fundkomplex werden übernommen.
 *  - Die eine Ersatzrate von v1 wird zum `replacementDefault`; kategoriespezifische
 *    Raten bleiben leer und damit wirkungsgleich zu v1.
 *  - Der Residualanteil ist in v1 nicht vorhanden und startet bei 0. Damit rechnet
 *    v2 zunächst exakt wie v1.
 *  - Reine Anzeigeeinstellungen (Deckkraft des Bands, Linienstärke, Demodaten)
 *    werden nicht übernommen; sie hängen an der alten Oberfläche.
 *
 * Der umgekehrte Weg wird bewusst nicht angeboten: v1 kennt weder
 * kategoriespezifische Raten noch Residualität und würde eine v2-Datei
 * stillschweigend falsch rechnen.
 */
import type { Assemblage, Category, MarketTable, ProjectV2, SimParams } from "../model.js";
import { DEFAULT_NOISE, DEFAULT_REPLACEMENT, DEFAULT_RUNS, normalizeTo100 } from "../model.js";
import { PRESET_V1, centreById } from "../../data/centres.js";
import { paletteColor } from "../palette.js";

export interface V1Report {
  /** Meldungen über nicht Übernommenes oder Angepasstes. */
  notes: string[];
  hadMarketData: boolean;
  hadExcavationData: boolean;
}

export interface V1Result { project: ProjectV2; report: V1Report; }

export class MigrateError extends Error {}

type Any = Record<string, unknown>;
const obj = (v: unknown): Any => (v && typeof v === "object" && !Array.isArray(v) ? (v as Any) : {});
const nums = (v: unknown): number[] => (Array.isArray(v) ? v.map((x) => Number(x)).filter((x) => Number.isFinite(x)) : []);
const numOr = (v: unknown, d: number): number => { const n = Number(v); return Number.isFinite(n) ? n : d; };

/** Erkennt eine v1-Datei am Kopf. */
export function isV1Config(data: unknown): boolean {
  const d = obj(data), m = obj(d._meta);
  if (String(m.app ?? "").toUpperCase() !== "STOCHASI") return false;
  const v = String(m.version ?? "1");
  return v.startsWith("1");
}

export function migrateV1(data: unknown, name = "STOCHASI-1-Projekt"): V1Result {
  const d = obj(data);
  if (!isV1Config(d)) throw new MigrateError("Das ist keine Konfiguration von STOCHASI 1.");
  const notes: string[] = [];
  const p = obj(d.parameters);
  const catBlock = obj(d.categories);
  const market = obj(d.market_data);
  const exc = obj(d.excavation_data);

  // Kategorien: Liste aus dem Block, sonst aus den Marktdaten, sonst v1-Preset.
  let ids: string[] = Array.isArray(catBlock.list) ? (catBlock.list as unknown[]).map(String) : [];
  if (!ids.length) ids = Object.keys(market).filter((k) => k !== "years");
  if (!ids.length) { ids = PRESET_V1.map((c) => c.id); notes.push("Keine Kategorien in der Datei; die fünf Kategorien von Version 1 wurden eingesetzt."); }

  const names = obj(catBlock.names);
  const categories: Category[] = ids.map((id, i) => {
    const preset = PRESET_V1.find((c) => c.id === id);
    const centre = centreById(id);
    return {
      id,
      name: String(names[id] ?? preset?.name ?? centre?.name ?? id),
      color: preset?.color ?? centre?.color ?? fallbackColor(i),
      group: preset?.group ?? centre?.group,
    };
  });

  // Marktdaten
  const years = nums(market.years);
  const hadMarketData = years.length > 0;
  const shares: Record<string, number[]> = {};
  if (hadMarketData) {
    for (const id of ids) {
      const col = nums(market[id]);
      shares[id] = years.map((_, k) => col[k] ?? 0);
    }
    // v1 hat nicht immer normiert
    let rescaled = false;
    for (let k = 0; k < years.length; k++) {
      const row = ids.map((id) => shares[id][k]);
      const sum = row.reduce((a, b) => a + b, 0);
      if (sum > 0 && Math.abs(sum - 100) > 0.5) rescaled = true;
      const n = normalizeTo100(row);
      ids.forEach((id, c) => (shares[id][k] = n[c]));
    }
    if (rescaled) notes.push("Die Marktdaten summierten nicht auf 100 und wurden zeilenweise auf Prozent umgerechnet.");
  } else {
    notes.push("Die Datei enthielt keine Marktdaten; die Marktkurve muss noch geladen werden.");
    for (const id of ids) shares[id] = [];
  }
  const marketTable: MarketTable = { years, shares };

  // Startverteilung
  const init = obj(p.initial_values);
  const initial: Record<string, number> = {};
  let initSum = 0;
  for (const id of ids) { const v = Math.max(0, numOr(init[id], 0)); initial[id] = v; initSum += v; }
  if (initSum <= 0) { for (const id of ids) initial[id] = 100 / ids.length; notes.push("Keine Startverteilung in der Datei; es wurde gleichverteilt begonnen."); }

  const params: SimParams = {
    startYear: Math.round(numOr(p.start_year, years[0] ?? 0)),
    endYear: Math.round(numOr(p.end_year, years[years.length - 1] ?? 300)),
    replacement: {},
    replacementDefault: numOr(p.replacement_rate, DEFAULT_REPLACEMENT),
    noiseSd: numOr(p.noise_sd, DEFAULT_NOISE),
    runs: Math.round(numOr(p.n_runs, DEFAULT_RUNS)),
    seed: Math.round(numOr(p.seed, 0)),
    settlementMode: Boolean(p.settlement_mode),
    residual: 0,
    initial,
  };
  if (params.endYear <= params.startYear) {
    params.endYear = params.startYear + 100;
    notes.push("Endjahr lag nicht nach dem Startjahr und wurde auf Start + 100 gesetzt.");
  }

  // Grabungsspektrum → ein Fundkomplex
  const assemblages: Assemblage[] = [];
  const abs = obj(exc.absolute);
  const pct = obj(exc.percent);
  const hadExcavationData = Object.keys(abs).length > 0 || Object.keys(pct).length > 0;
  if (Object.keys(abs).length) {
    const counts: Record<string, number> = {};
    let total = 0;
    for (const id of ids) { const v = Math.max(0, Math.round(numOr(abs[id], 0))); counts[id] = v; total += v; }
    if (total > 0) assemblages.push({ id: "A1", name: "Grabungsspektrum", counts, color: paletteColor(0) });
  } else if (Object.keys(pct).length) {
    // Nur Prozente vorhanden: v1 hat den Stichprobenumfang dann nicht gespeichert.
    // Aus 100 „Stück" zu rechnen wäre eine erfundene Genauigkeit, deshalb ein
    // ausdrücklicher Hinweis statt stiller Annahme.
    const counts: Record<string, number> = {};
    for (const id of ids) counts[id] = Math.round(numOr(pct[id], 0));
    assemblages.push({ id: "A1", name: "Grabungsspektrum (nur Prozente)", counts, color: paletteColor(0) });
    notes.push("Das Grabungsspektrum lag nur als Prozentwerte vor. Die inverse Datierung braucht Stückzahlen — bitte die absoluten Zahlen nachtragen, sonst ist das ausgewiesene Intervall zu eng.");
  }

  if (Object.keys(obj(d.parameters)).length === 0) notes.push("Die Datei enthielt keinen Parameterblock; es wurden Vorgabewerte eingesetzt.");
  if (p.auto_normalize !== undefined || p.uncertainty_opacity !== undefined || p.line_width !== undefined)
    notes.push("Anzeigeeinstellungen von Version 1 (Deckkraft, Linienstärke, Auto-Normierung) wurden nicht übernommen.");

  const project: ProjectV2 = {
    _meta: { app: "STOCHASI", version: "2.1", created: new Date().toISOString() },
    name,
    categories,
    market: marketTable,
    params,
    assemblages,
    comparisonYear: Math.round(numOr(exc.comparison_year, Math.round((params.startYear + params.endYear) / 2))),
  };
  if (project.comparisonYear < params.startYear || project.comparisonYear > params.endYear)
    project.comparisonYear = Math.round((params.startYear + params.endYear) / 2);

  return { project, report: { notes, hadMarketData, hadExcavationData } };
}

const FALLBACK = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#008080"];
function fallbackColor(i: number): string { return FALLBACK[i % FALLBACK.length]; }
