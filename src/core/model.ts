/**
 * Datenmodell von STOCHASI 2.
 *
 * Zwei Datenquellen stehen dem Modell gegenüber:
 *
 *  - Das **Marktangebot** (`MarketTable`): welcher Anteil des in einem Jahr neu
 *    auf den Markt kommenden Materials aus welchem Produktionszentrum stammt.
 *    Angegeben an Stützjahren, dazwischen linear interpoliert.
 *  - Die **Fundkomplexe** (`Assemblage`): gezählte Stückzahlen je Kategorie aus
 *    einem Befund. Absolute Zahlen, nicht Prozente — der Stichprobenumfang geht
 *    in die inverse Datierung ein und darf nicht verlorengehen.
 *
 * Dazwischen steht der Umlaufbestand, den die Simulation Jahr für Jahr fortschreibt.
 */

export interface Category {
  /** Kurzkennung, wie sie in Tabellenspalten steht (z. B. „RZ"). */
  id: string;
  /** Ausgeschriebene Bezeichnung (z. B. „Rheinzabern"). */
  name: string;
  /** Farbe als #rrggbb. */
  color: string;
  /** Optionale Gruppe für die Legende (z. B. „Ostgallisch"). */
  group?: string;
}

/** Marktangebot an Stützjahren. `shares[id][k]` gehört zu `years[k]`. */
export interface MarketTable {
  years: number[];
  shares: Record<string, number[]>;
}

/** Ein gezählter Fundkomplex. */
export interface Assemblage {
  id: string;
  name: string;
  /** Absolute Stückzahlen je Kategorie. Fehlende Kategorien gelten als 0. */
  counts: Record<string, number>;
}

export interface SimParams {
  startYear: number;
  endYear: number;
  /** Jährliche Ersatzrate je Kategorie (0…1). Fehlende Einträge nutzen `replacementDefault`. */
  replacement: Record<string, number>;
  replacementDefault: number;
  /** Streuung des additiven Rauschens in Prozentpunkten. */
  noiseSd: number;
  runs: number;
  /** 0 = bei jedem Lauf neu würfeln, sonst reproduzierbar. */
  seed: number;
  /** Neugründung: im Startjahr entspricht der Bestand dem Marktangebot. */
  settlementMode: boolean;
  /** Residualanteil 0…1 — Anteil umgelagerten Altmaterials im Fundkomplex. */
  residual: number;
  /** Startverteilung des Umlaufbestands in Prozent (Summe 100). */
  initial: Record<string, number>;
}

export interface ProjectV2 {
  _meta: { app: "STOCHASI"; version: "2.0"; created?: string };
  name: string;
  categories: Category[];
  market: MarketTable;
  params: SimParams;
  assemblages: Assemblage[];
  /** Jahr, für das Simulation und Fundkomplex im Vergleich gegenübergestellt werden. */
  comparisonYear: number;
}

export const MIN_RUNS = 10, MAX_RUNS = 500, DEFAULT_RUNS = 100;
export const MAX_REPLACEMENT = 0.5, DEFAULT_REPLACEMENT = 0.1;
export const MAX_NOISE = 10, DEFAULT_NOISE = 2;
export const MAX_RESIDUAL = 0.5;

/** Kategorienliste → Ids, in Reihenfolge. */
export function catIds(cats: readonly Category[]): string[] {
  return cats.map((c) => c.id);
}

/** Auf Summe 100 normieren; eine Nullsumme bleibt unverändert. */
export function normalizeTo100(v: readonly number[]): number[] {
  let s = 0; for (const x of v) s += x;
  if (!(s > 0)) return v.slice();
  return v.map((x) => (x / s) * 100);
}

/** Ersatzrate einer Kategorie mit Rückfall auf den Vorgabewert. */
export function rateFor(p: SimParams, id: string): number {
  const r = p.replacement[id];
  return Number.isFinite(r) ? Math.min(MAX_REPLACEMENT, Math.max(0, r)) : p.replacementDefault;
}

/** Jahresvektor start…end einschließlich. */
export function yearRange(start: number, end: number): number[] {
  const out: number[] = [];
  for (let y = start; y <= end; y++) out.push(y);
  return out;
}

/**
 * Marktangebot auf jedes einzelne Jahr interpolieren.
 *
 * Zwischen zwei Stützjahren wird linear interpoliert, außerhalb wird der
 * jeweilige Randwert fortgeschrieben (nicht extrapoliert — eine Extrapolation
 * der Lieferanteile über das belegte Ende hinaus wäre eine Behauptung, die die
 * Daten nicht tragen). Jede Jahreszeile wird auf 100 normiert.
 */
export function interpolateMarket(m: MarketTable, ids: readonly string[], years: readonly number[]): Float64Array {
  const nc = ids.length, ny = years.length;
  const out = new Float64Array(ny * nc);
  const ys = m.years;
  if (ys.length === 0) return out;
  for (let k = 0; k < ny; k++) {
    const y = years[k];
    let i = 0;
    while (i < ys.length - 1 && ys[i + 1] < y) i++;
    // i zeigt auf das letzte Stützjahr < y bzw. auf 0
    let lo = i, hi = Math.min(ys.length - 1, i + 1);
    if (y <= ys[0]) { lo = hi = 0; }
    else if (y >= ys[ys.length - 1]) { lo = hi = ys.length - 1; }
    const span = ys[hi] - ys[lo];
    const t = span > 0 ? (y - ys[lo]) / span : 0;
    const row: number[] = [];
    for (let c = 0; c < nc; c++) {
      const s = m.shares[ids[c]] ?? [];
      const a = Number(s[lo] ?? 0), b = Number(s[hi] ?? 0);
      row.push(a + (b - a) * t);
    }
    const norm = normalizeTo100(row);
    for (let c = 0; c < nc; c++) out[k * nc + c] = norm[c];
  }
  return out;
}
