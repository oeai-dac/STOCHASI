/**
 * Projektdatei von STOCHASI 2 — lesen, schreiben, prüfen.
 *
 * Die Datei ist ein JSON mit `_meta.app = "STOCHASI"` und `_meta.version = "2.0"`.
 * Sie enthält alles, was zur Wiederholung einer Auswertung nötig ist:
 * Kategorien, Marktkurve, Parameter und Fundkomplexe. Wer eine Zahl aus der
 * Publikation nachrechnen will, braucht nur diese eine Datei.
 *
 * `readProject` nimmt auch v1-Dateien an und wandelt sie um.
 */
import type { Assemblage, Category, MarketTable, ProjectV2, SimParams } from "../model.js";
import { DEFAULT_NOISE, DEFAULT_REPLACEMENT, DEFAULT_RUNS, MAX_NOISE, MAX_RESIDUAL, MAX_RUNS, MAX_REPLACEMENT, MIN_RUNS } from "../model.js";
import { isV1Config, migrateV1 } from "./migrateV1.js";

export class ProjectError extends Error {}

export interface ReadResult { project: ProjectV2; notes: string[]; fromV1: boolean; }

type Any = Record<string, unknown>;
const obj = (v: unknown): Any => (v && typeof v === "object" && !Array.isArray(v) ? (v as Any) : {});
const arr = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);
const clamp = (v: unknown, lo: number, hi: number, d: number) => { const n = Number(v); return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : d; };

export function readProject(text: string, name = "Projekt"): ReadResult {
  let data: unknown;
  try { data = JSON.parse(text); }
  catch { throw new ProjectError("Die Datei ist kein gültiges JSON."); }
  if (isV1Config(data)) {
    const r = migrateV1(data, name);
    return { project: r.project, notes: r.report.notes, fromV1: true };
  }
  const d = obj(data);
  const m = obj(d._meta);
  if (String(m.app ?? "").toUpperCase() !== "STOCHASI")
    throw new ProjectError("Das ist keine STOCHASI-Projektdatei.");
  const notes: string[] = [];
  const version = String(m.version ?? "");
  if (version && !version.startsWith("2"))
    notes.push(`Die Datei nennt Version ${version}; sie wurde als 2.0 gelesen.`);
  return { project: sanitize(d, name, notes), notes, fromV1: false };
}

/** Prüft und normalisiert ein rohes Objekt zu einem gültigen Projekt. */
export function sanitize(d: Any, fallbackName: string, notes: string[] = []): ProjectV2 {
  const rawCats = arr(d.categories).map(obj);
  const seen = new Set<string>();
  const categories: Category[] = [];
  rawCats.forEach((c, i) => {
    const id = String(c.id ?? "").trim();
    if (!id) { notes.push(`Kategorie ${i + 1} ohne Kennung wurde verworfen.`); return; }
    if (seen.has(id)) { notes.push(`Kategorie „${id}" kam mehrfach vor; nur die erste wurde behalten.`); return; }
    seen.add(id);
    const color = String(c.color ?? "");
    categories.push({
      id,
      name: String(c.name ?? id),
      color: /^#[0-9a-f]{6}$/i.test(color) ? color : "#808080",
      group: c.group ? String(c.group) : undefined,
    });
  });
  if (!categories.length) throw new ProjectError("Das Projekt enthält keine Kategorien.");
  const ids = categories.map((c) => c.id);

  const rawMarket = obj(d.market);
  const years = arr(rawMarket.years).map(Number).filter(Number.isFinite).map(Math.round);
  const rawShares = obj(rawMarket.shares);
  const shares: Record<string, number[]> = {};
  for (const id of ids) {
    const col = arr(rawShares[id]).map(Number);
    shares[id] = years.map((_, k) => (Number.isFinite(col[k]) && col[k] >= 0 ? col[k] : 0));
  }
  const market: MarketTable = { years, shares };
  if (!years.length) notes.push("Das Projekt enthält keine Marktkurve.");

  const rp = obj(d.params);
  const rawInit = obj(rp.initial);
  const initial: Record<string, number> = {};
  let sum = 0;
  for (const id of ids) { const v = Math.max(0, Number(rawInit[id]) || 0); initial[id] = v; sum += v; }
  if (sum <= 0) for (const id of ids) initial[id] = 100 / ids.length;

  const replacement: Record<string, number> = {};
  const rawRep = obj(rp.replacement);
  for (const id of ids) {
    const v = Number(rawRep[id]);
    if (Number.isFinite(v)) replacement[id] = Math.min(MAX_REPLACEMENT, Math.max(0, v));
  }

  let startYear = Math.round(clamp(rp.startYear, -3000, 3000, years[0] ?? 0));
  let endYear = Math.round(clamp(rp.endYear, -3000, 3000, years[years.length - 1] ?? 300));
  if (endYear <= startYear) { endYear = startYear + 100; notes.push("Endjahr lag nicht nach dem Startjahr und wurde korrigiert."); }
  if (endYear - startYear > 3000) { endYear = startYear + 3000; notes.push("Der Zeitraum war länger als 3000 Jahre und wurde gekürzt."); }

  const params: SimParams = {
    startYear, endYear,
    replacement,
    replacementDefault: clamp(rp.replacementDefault, 0, MAX_REPLACEMENT, DEFAULT_REPLACEMENT),
    noiseSd: clamp(rp.noiseSd, 0, MAX_NOISE, DEFAULT_NOISE),
    runs: Math.round(clamp(rp.runs, MIN_RUNS, MAX_RUNS, DEFAULT_RUNS)),
    seed: Math.round(clamp(rp.seed, 0, 2 ** 31 - 1, 0)),
    settlementMode: Boolean(rp.settlementMode),
    residual: clamp(rp.residual, 0, MAX_RESIDUAL, 0),
    initial,
  };

  const assemblages: Assemblage[] = [];
  arr(d.assemblages).map(obj).forEach((a, i) => {
    const counts: Record<string, number> = {};
    const rawCounts = obj(a.counts);
    let total = 0;
    for (const id of ids) { const v = Math.max(0, Number(rawCounts[id]) || 0); counts[id] = v; total += v; }
    const name = String(a.name ?? `Fundkomplex ${i + 1}`);
    if (total <= 0) { notes.push(`„${name}" enthält keine Funde und wurde übersprungen.`); return; }
    assemblages.push({ id: String(a.id ?? `A${assemblages.length + 1}`), name, counts });
  });

  let comparisonYear = Math.round(clamp(d.comparisonYear, startYear, endYear, Math.round((startYear + endYear) / 2)));
  if (comparisonYear < startYear || comparisonYear > endYear) comparisonYear = Math.round((startYear + endYear) / 2);

  return {
    _meta: { app: "STOCHASI", version: "2.0", created: String(obj(d._meta).created ?? new Date().toISOString()) },
    name: String(d.name ?? fallbackName),
    categories, market, params, assemblages, comparisonYear,
  };
}

export function writeProject(p: ProjectV2): string {
  return JSON.stringify({ ...p, _meta: { ...p._meta, app: "STOCHASI", version: "2.0" } }, null, 2);
}

/** Leeres, aber lauffähiges Projekt. */
export function emptyProject(categories: Category[], name = "Neues Projekt"): ProjectV2 {
  const ids = categories.map((c) => c.id);
  const initial: Record<string, number> = {};
  for (const id of ids) initial[id] = 100 / Math.max(1, ids.length);
  return {
    _meta: { app: "STOCHASI", version: "2.0", created: new Date().toISOString() },
    name, categories,
    market: { years: [], shares: Object.fromEntries(ids.map((id) => [id, []])) },
    params: {
      startYear: 100, endYear: 300, replacement: {}, replacementDefault: DEFAULT_REPLACEMENT,
      noiseSd: DEFAULT_NOISE, runs: DEFAULT_RUNS, seed: 0, settlementMode: false, residual: 0, initial,
    },
    assemblages: [],
    comparisonYear: 200,
  };
}
