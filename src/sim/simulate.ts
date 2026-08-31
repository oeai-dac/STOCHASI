/**
 * Der Vorwärtskern: stochastische Fortschreibung des Umlaufbestands.
 *
 * Modell (verallgemeinert aus STOCHASI 1, dort mit einer gemeinsamen Rate):
 *
 *     bestand_c(t) = bestand_c(t-1)·(1 - r_c) + markt_c(t)·r_c + ε,  ε ~ N(0, σ)
 *
 * anschließend bei 0 abgeschnitten und die Jahreszeile auf 100 % normiert.
 * `r_c` ist die jährliche Ersatzrate der Kategorie c: der Anteil des Bestands,
 * der binnen eines Jahres ausscheidet und durch Neuware vom Markt ersetzt wird.
 * Sind alle `r_c` gleich, ergibt sich exakt das Modell von Version 1.
 *
 * Das Rauschen ε ist additiv in Prozentpunkten und fasst alles zusammen, was das
 * Modell nicht abbildet — Lieferschwankungen, ungleiche Bruchraten, Zufall der
 * Deponierung. Es ist bewusst kein Messfehler des Fundmaterials; der steckt in
 * der Stichprobengröße und wird erst bei der inversen Datierung wirksam.
 */
import type { SimParams } from "../core/model.js";
import { rateFor } from "../core/model.js";
import { seededGaussian, percentileSorted } from "./stats.js";

/**
 * Was die Diagramme brauchen: Mittelwert und Perzentile, aber nicht die
 * einzelnen Läufe. Getrennt geführt, damit der Worker nur diesen kleinen Teil
 * an den Haupt-Thread schickt — die Läufe sind bei großen Projekten mehrere
 * zehn Megabyte und werden dort nie gebraucht.
 */
export interface EnsembleStats {
  years: number[];
  ids: string[];
  runs: number;
  /** Mittelwert je (Jahr, Kategorie), flach `[y·nc + c]`. */
  mean: Float64Array;
  p10: Float64Array;
  p90: Float64Array;
}

export interface Ensemble extends EnsembleStats {
  /** Alle Läufe, flach: `values[(run·ny + y)·nc + c]`, in Prozent. */
  values: Float32Array;
}

/** Zugriff auf eine Jahreszeile des Mittelwerts. */
export function meanRow(e: EnsembleStats, yearIdx: number): number[] {
  const nc = e.ids.length, out: number[] = [];
  for (let c = 0; c < nc; c++) out.push(e.mean[yearIdx * nc + c]);
  return out;
}

export function rowAt(v: Float64Array, yearIdx: number, nc: number): number[] {
  const out: number[] = [];
  for (let c = 0; c < nc; c++) out.push(v[yearIdx * nc + c]);
  return out;
}

/**
 * Führt `runs` Läufe aus und liefert Ensemble samt Mittelwert und 10-/90-Perzentil.
 *
 * `market` ist bereits auf Einzeljahre interpoliert (`interpolateMarket`), flach
 * abgelegt als `[y·nc + c]`.
 */
export function simulate(
  market: Float64Array,
  years: readonly number[],
  ids: readonly string[],
  params: SimParams,
): Ensemble {
  const ny = years.length, nc = ids.length;
  const runs = Math.max(1, Math.round(params.runs));
  const values = new Float32Array(runs * ny * nc);

  const rates = new Float64Array(nc);
  for (let c = 0; c < nc; c++) rates[c] = rateFor(params, ids[c]);

  const initial = new Float64Array(nc);
  {
    let s = 0;
    for (let c = 0; c < nc; c++) { const v = Math.max(0, Number(params.initial[ids[c]] ?? 0)); initial[c] = v; s += v; }
    if (s > 0) for (let c = 0; c < nc; c++) initial[c] = (initial[c] / s) * 100;
    else for (let c = 0; c < nc; c++) initial[c] = 100 / Math.max(1, nc);
  }

  const cur = new Float64Array(nc);
  for (let run = 0; run < runs; run++) {
    // Seed 0 heißt „jeder Lauf würfelt neu"; sonst ist jeder Lauf eindeutig
    // durch base+run bestimmt und die gesamte Rechnung wiederholbar.
    const g = seededGaussian(params.seed === 0 ? 0 : params.seed + run);
    cur.set(initial);
    for (let y = 0; y < ny; y++) {
      const off = (run * ny + y) * nc;
      if (params.settlementMode && y === 0) {
        // Neugründung: im Startjahr ist ausschließlich Neuware im Umlauf.
        for (let c = 0; c < nc; c++) cur[c] = market[c];
      } else {
        let sum = 0;
        for (let c = 0; c < nc; c++) {
          let v = cur[c] * (1 - rates[c]) + market[y * nc + c] * rates[c];
          if (params.noiseSd > 0) v += g() * params.noiseSd;
          if (v < 0) v = 0;
          cur[c] = v; sum += v;
        }
        if (sum > 0) for (let c = 0; c < nc; c++) cur[c] = (cur[c] / sum) * 100;
      }
      for (let c = 0; c < nc; c++) values[off + c] = cur[c];
    }
  }

  const mean = new Float64Array(ny * nc);
  const p10 = new Float64Array(ny * nc);
  const p90 = new Float64Array(ny * nc);
  const col = new Float64Array(runs);
  for (let y = 0; y < ny; y++) {
    for (let c = 0; c < nc; c++) {
      let s = 0;
      for (let r = 0; r < runs; r++) { const v = values[(r * ny + y) * nc + c]; col[r] = v; s += v; }
      mean[y * nc + c] = s / runs;
      col.sort();
      p10[y * nc + c] = percentileSorted(col, 10);
      p90[y * nc + c] = percentileSorted(col, 90);
    }
  }
  return { years: years.slice(), ids: ids.slice(), runs, values, mean, p10, p90 };
}
