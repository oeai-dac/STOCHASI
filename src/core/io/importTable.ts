/**
 * Tabellen-Import: Zellraster → Marktangebot oder Fundkomplexe.
 *
 * Der Kern arbeitet auf einem bereits geparsten `string[][]`, damit dieselbe
 * Logik CSV und XLSX bedient (siehe `parseDelimited` bzw. `importXLSX`).
 *
 * Erkannt werden:
 *
 *  **Marktangebot** — eine Spalte mit Jahreszahlen (`Year`, `Jahr` oder eine
 *  erste Spalte aus lauter plausiblen Jahren), daneben je Zentrum eine Spalte.
 *  Werte dürfen Prozente oder Rohzahlen sein; jede Zeile wird auf 100 normiert.
 *
 *  **Fundkomplexe** — eine Tabelle ohne Jahresspalte. Die Ausrichtung wird
 *  erkannt: Kategorien können in den Spalten stehen (eine Zeile je Komplex) oder
 *  in den Zeilen (eine Spalte je Komplex, wie die Pivot-Tabellen Provenienz ×
 *  Insula). Beides kommt in der Praxis vor, deshalb wird geraten statt gefragt.
 *
 *  **Langformat** — Spalten `Typ`/`Kategorie`/`Type` und `Anzahl`/`Count`/`n`,
 *  optional zusätzlich eine Spalte mit dem Komplexnamen. Das ist das Format,
 *  das STOCHASI 1 für Grabungsspektren gelesen hat.
 */
import type { Assemblage, MarketTable } from "../model.js";
import { normalizeTo100 } from "../model.js";

export class ImportError extends Error {}

export interface ImportWarnings { warnings: string[]; }

export interface MarketImport extends ImportWarnings {
  market: MarketTable;
  ids: string[];
}

export interface AssemblageImport extends ImportWarnings {
  assemblages: Assemblage[];
  ids: string[];
  /** true, wenn die Kategorien in den Zeilen standen und transponiert wurde. */
  transposed: boolean;
}

const YEAR_HEADERS = ["year", "jahr", "years", "jahre", "anno", "date", "datum"];
const TYPE_HEADERS = ["typ", "type", "kategorie", "category", "provenienz", "provenance", "herkunft", "zentrum", "centre", "center"];
const COUNT_HEADERS = ["anzahl", "count", "menge", "n", "number", "stück", "stueck", "summe", "total"];
const COMPLEX_HEADERS = ["komplex", "fundkomplex", "befund", "context", "kontext", "insula", "einheit", "unit", "assemblage", "fundort", "site"];
/** Spalten, die in Pivot-Tabellen als Randsummen auftauchen und keine Daten sind. */
const IGNORE_HEADERS = ["summe", "total", "gesamt", "sum", "fst?", "σ"];

function norm(s: string): string { return (s ?? "").toString().trim().toLowerCase(); }

/** Zahl aus einer Zelle; akzeptiert Dezimalkomma und Tausenderpunkt/-leerzeichen. */
export function num(cell: string | number | undefined | null): number {
  if (typeof cell === "number") return Number.isFinite(cell) ? cell : NaN;
  let s = (cell ?? "").toString().trim();
  if (!s) return NaN;
  s = s.replace(/[\s' ]/g, "");
  if (s.includes(",") && s.includes(".")) s = s.lastIndexOf(",") > s.lastIndexOf(".") ? s.replace(/\./g, "").replace(",", ".") : s.replace(/,/g, "");
  else if (s.includes(",")) s = s.replace(",", ".");
  // „12 (34,5 %)" aus formatierten Tabellen: die erste Zahl gilt
  const m = s.match(/-?\d+(?:\.\d+)?/);
  return m ? Number(m[0]) : NaN;
}

function isYearLike(v: number): boolean {
  return Number.isFinite(v) && Number.isInteger(v) && v >= -3000 && v <= 3000;
}

function trimGrid(grid: string[][]): string[][] {
  const rows = grid.map((r) => (r ?? []).map((c) => (c ?? "").toString().trim()));
  // vollständig leere Zeilen und Spalten am Rand entfernen
  const keepRow = rows.filter((r) => r.some((c) => c !== ""));
  if (!keepRow.length) return [];
  const width = Math.max(...keepRow.map((r) => r.length));
  const padded = keepRow.map((r) => Array.from({ length: width }, (_, i) => r[i] ?? ""));
  const keepCol: number[] = [];
  for (let i = 0; i < width; i++) if (padded.some((r) => r[i] !== "")) keepCol.push(i);
  return padded.map((r) => keepCol.map((i) => r[i]));
}

/** Entscheidet, ob das Raster ein Marktangebot beschreibt. */
export function detectKind(grid: string[][]): "market" | "assemblages" {
  const g = trimGrid(grid);
  if (g.length < 2) return "assemblages";
  const head = g[0].map(norm);
  if (head.some((h) => YEAR_HEADERS.includes(h))) return "market";
  // erste Spalte aus lauter Jahreszahlen, mindestens drei Zeilen
  const col = g.slice(1).map((r) => num(r[0]));
  if (col.length >= 3 && col.every(isYearLike)) {
    // aufsteigend und verschieden — sonst sind es eher Zählwerte
    const strictly = col.every((v, i) => i === 0 || v > col[i - 1]);
    if (strictly) return "market";
  }
  return "assemblages";
}

/* ── Marktangebot ── */
export function importMarketGrid(grid: string[][]): MarketImport {
  const g = trimGrid(grid);
  if (g.length < 2) throw new ImportError("Die Tabelle enthält keine Datenzeilen.");
  const head = g[0];
  const warnings: string[] = [];

  let yearCol = head.findIndex((h) => YEAR_HEADERS.includes(norm(h)));
  if (yearCol < 0) yearCol = 0;

  const cols: Array<{ idx: number; id: string }> = [];
  head.forEach((h, i) => {
    if (i === yearCol) return;
    const id = h.trim();
    if (!id || IGNORE_HEADERS.includes(norm(id))) return;
    if (cols.some((c) => c.id === id)) { warnings.push(`Spalte „${id}" kommt mehrfach vor; nur die erste wird verwendet.`); return; }
    cols.push({ idx: i, id });
  });
  if (!cols.length) throw new ImportError("Keine Kategoriespalten gefunden. Erwartet wird eine Jahresspalte und je Zentrum eine weitere Spalte.");

  const years: number[] = [];
  const raw: number[][] = [];
  for (let r = 1; r < g.length; r++) {
    const y = num(g[r][yearCol]);
    if (!isYearLike(y)) { warnings.push(`Zeile ${r + 1} übersprungen: „${g[r][yearCol]}" ist keine Jahreszahl.`); continue; }
    if (years.includes(y)) { warnings.push(`Jahr ${y} kommt mehrfach vor; nur das erste wird verwendet.`); continue; }
    const vals = cols.map((c) => { const v = num(g[r][c.idx]); return Number.isFinite(v) && v > 0 ? v : 0; });
    years.push(y); raw.push(vals);
  }
  if (!years.length) throw new ImportError("Keine gültigen Jahreszeilen gefunden.");

  // nach Jahr sortieren und jede Zeile auf 100 normieren
  const order = years.map((_, i) => i).sort((a, b) => years[a] - years[b]);
  const shares: Record<string, number[]> = {};
  cols.forEach((c) => (shares[c.id] = []));
  const sortedYears: number[] = [];
  let rescaled = false;
  for (const i of order) {
    const sum = raw[i].reduce((a, b) => a + b, 0);
    if (sum > 0 && Math.abs(sum - 100) > 0.5) rescaled = true;
    const n = normalizeTo100(raw[i]);
    sortedYears.push(years[i]);
    cols.forEach((c, k) => shares[c.id].push(n[k]));
  }
  if (rescaled) warnings.push("Nicht alle Zeilen summierten auf 100; die Werte wurden zeilenweise auf Prozent umgerechnet.");

  return { market: { years: sortedYears, shares }, ids: cols.map((c) => c.id), warnings };
}

/* ── Fundkomplexe ── */
export interface AssemblageOptions {
  /** Bekannte Kategoriekennungen des Projekts — hilft, die Ausrichtung sicher zu erkennen. */
  known?: readonly string[];
  /** Ausrichtung erzwingen statt raten. */
  orientation?: "categoriesInColumns" | "categoriesInRows";
  /** Name für den einzigen Komplex, wenn die Tabelle keinen enthält. */
  defaultName?: string;
}

export function importAssemblageGrid(grid: string[][], opts: AssemblageOptions = {}): AssemblageImport {
  const g = trimGrid(grid);
  if (!g.length) throw new ImportError("Die Tabelle ist leer.");
  const warnings: string[] = [];
  const head = g[0].map(norm);

  // Langformat? Spalten für Kategorie und Anzahl vorhanden.
  const typeCol = head.findIndex((h) => TYPE_HEADERS.includes(h));
  const countCol = head.findIndex((h) => COUNT_HEADERS.includes(h));
  if (typeCol >= 0 && countCol >= 0 && typeCol !== countCol) {
    const cplxCol = head.findIndex((h, i) => i !== typeCol && i !== countCol && COMPLEX_HEADERS.includes(h));
    const byName = new Map<string, Record<string, number>>();
    const ids: string[] = [];
    for (let r = 1; r < g.length; r++) {
      const id = g[r][typeCol]?.trim();
      const v = num(g[r][countCol]);
      if (!id || !Number.isFinite(v)) continue;
      const name = (cplxCol >= 0 ? g[r][cplxCol]?.trim() : "") || opts.defaultName || "Fundkomplex";
      if (!ids.includes(id)) ids.push(id);
      const rec = byName.get(name) ?? {};
      rec[id] = (rec[id] ?? 0) + v;
      byName.set(name, rec);
    }
    if (!byName.size) throw new ImportError("Im Langformat wurden keine gültigen Zeilen gefunden.");
    const assemblages = [...byName].map(([name, counts], i) => ({ id: `A${i + 1}`, name, counts }));
    return { assemblages, ids, transposed: false, warnings };
  }

  // Breitformat — Ausrichtung bestimmen.
  const known = new Set((opts.known ?? []).map((k) => k.trim()));
  const headerIds = g[0].slice(1).map((s) => s.trim()).filter((s) => s && !IGNORE_HEADERS.includes(norm(s)));
  const firstColIds = g.slice(1).map((r) => (r[0] ?? "").trim()).filter((s) => s && !IGNORE_HEADERS.includes(norm(s)));

  let transposed: boolean;
  if (opts.orientation) transposed = opts.orientation === "categoriesInRows";
  else if (known.size) {
    const hitHead = headerIds.filter((s) => known.has(s)).length;
    const hitCol = firstColIds.filter((s) => known.has(s)).length;
    if (hitHead === 0 && hitCol === 0) {
      transposed = false;
      warnings.push("Keine der Bezeichnungen entspricht den Kategorien des Projekts; es wurde angenommen, dass die Kategorien in den Spalten stehen.");
    } else transposed = hitCol > hitHead;
  } else {
    // Ohne Vorwissen: Kurze, kennungsartige Bezeichnungen sprechen für Kategorien.
    const idLike = (s: string) => s.length <= 4 && /^[A-Za-zÄÖÜäöü]{1,4}$/.test(s);
    const sh = headerIds.filter(idLike).length / Math.max(1, headerIds.length);
    const sc = firstColIds.filter(idLike).length / Math.max(1, firstColIds.length);
    transposed = sc > sh;
  }

  const grid2 = transposed ? transpose(g) : g;
  const h2 = grid2[0];
  const cols: Array<{ idx: number; id: string }> = [];
  h2.forEach((s, i) => {
    if (i === 0) return;
    const id = s.trim();
    if (!id || IGNORE_HEADERS.includes(norm(id))) return;
    if (cols.some((c) => c.id === id)) { warnings.push(`Kategorie „${id}" kommt mehrfach vor; nur die erste wird verwendet.`); return; }
    cols.push({ idx: i, id });
  });
  if (!cols.length) throw new ImportError("Keine Kategoriespalten gefunden.");

  const assemblages: Assemblage[] = [];
  for (let r = 1; r < grid2.length; r++) {
    const name = (grid2[r][0] ?? "").trim();
    if (!name || IGNORE_HEADERS.includes(norm(name))) continue;
    const counts: Record<string, number> = {};
    let total = 0;
    for (const c of cols) {
      const v = num(grid2[r][c.idx]);
      const n = Number.isFinite(v) && v > 0 ? v : 0;
      counts[c.id] = n; total += n;
    }
    if (total <= 0) { warnings.push(`„${name}" enthält keine Funde und wurde übersprungen.`); continue; }
    assemblages.push({ id: `A${assemblages.length + 1}`, name, counts });
  }
  if (!assemblages.length) throw new ImportError("Keine Fundkomplexe mit Zählwerten gefunden.");

  const nonInteger = assemblages.some((a) => Object.values(a.counts).some((v) => !Number.isInteger(v)));
  if (nonInteger) warnings.push("Die Tabelle enthält gebrochene Werte. Für die Datierung werden absolute Stückzahlen gebraucht — Prozentwerte täuschen eine zu große Genauigkeit vor.");

  return { assemblages, ids: cols.map((c) => c.id), transposed, warnings };
}

function transpose(g: string[][]): string[][] {
  const w = Math.max(...g.map((r) => r.length));
  return Array.from({ length: w }, (_, i) => g.map((r) => r[i] ?? ""));
}
