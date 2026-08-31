/**
 * Inverse Datierung — vom gezählten Fundkomplex zurück auf das Ablagerungsjahr.
 *
 * Gegeben sind die Stückzahlen n_k je Kategorie (N = Σ n_k) und das Ensemble der
 * Simulation. Für jedes Kandidatenjahr t liefert die Simulation M Läufe mit je
 * einer Zusammensetzung p^(m)(t). Die Wahrscheinlichkeit, bei gegebener
 * Zusammensetzung genau diese Zählung zu ziehen, ist multinomial:
 *
 *     log L^(m)(t) = Σ_k n_k · log p_k^(m)(t)      (+ Konstante, über t gleich)
 *
 * Die Unsicherheit der Simulation muss dabei ausintegriert werden:
 * L(t) = ∫ L(t|p) dP(p|t). Dafür gibt es zwei Wege, beide sind eingebaut.
 *
 * **`"dirichlet"` (Vorgabe).** An die M simulierten Zusammensetzungen eines
 * Jahres wird über die Momente eine Dirichlet-Verteilung angepasst; das Integral
 * hat dann die geschlossene Form der Dirichlet-Multinomial-Verteilung:
 *
 *     log L(t) = logΓ(α₀) − logΓ(α₀+N) + Σ_k [logΓ(n_k+α_k) − logΓ(α_k)]
 *
 * **`"ensemble"`.** Die rohe Monte-Carlo-Näherung L(t) = (1/M) Σ_m exp(log L^(m)).
 *
 * Warum die Vorgabe nicht die naheliegendere Monte-Carlo-Summe ist: Bei
 * größeren Fundzahlen ist die Likelihood scharf, und die Summe wird von dem
 * einen Lauf beherrscht, der zufällig am besten passt. Die Kurve wird dann
 * zackig, und die Zacken sehen aus wie Struktur, sind aber Rauschen des
 * Schätzers — man liest ein Datierungsmaximum ab, das beim nächsten Seed drei
 * Jahre weiter liegt. Die Dirichlet-Anpassung integriert dieselbe Unsicherheit
 * analytisch und liefert dieselbe Lage bei glatter Kurve. Der
 * Monte-Carlo-Weg bleibt zum Vergleich erhalten.
 *
 * Bei flachem Prior über die Jahre ist die normierte Kurve die A-posteriori-
 * Verteilung des Ablagerungsjahres.
 *
 * Warum multinomial und nicht ein Abstandsmaß: Der Stichprobenumfang geht so
 * richtig ein. 500 Scherben liefern eine schmale Kurve, 12 Scherben eine breite.
 * Ein Chi-Quadrat-Abstand auf Prozentwerten kann das nicht unterscheiden und
 * täuscht bei kleinen Komplexen eine Genauigkeit vor, die nicht da ist.
 *
 * **Was die Kurve nicht ist:** eine absolute Datierung. Sie ist bedingt auf die
 * eingestellte Marktkurve, die Ersatzraten und den Residualanteil. Ändert sich
 * eine dieser Annahmen, ändert sich das Ergebnis. Deshalb sollte man die Kurve
 * immer für mehrere Residualanteile ansehen (`dateAcrossResidual`).
 */
import type { Ensemble } from "./simulate.js";
import { applyResidual } from "./residual.js";
import { logGamma, logSumExp } from "./stats.js";

/**
 * Untergrenze für einen simulierten Anteil. Ohne sie schließt eine einzige
 * Scherbe einer Kategorie, die das Modell für ein Jahr auf exakt 0 setzt, dieses
 * Jahr vollständig aus. Das wäre eine Härte, die das Modell nicht rechtfertigt:
 * Fehlbestimmungen, Altstücke und Lieferungen außerhalb des Modellierten
 * kommen vor. 1e-4 entspricht 0,01 % und kostet eine solche Scherbe rund
 * 9 Log-Einheiten — spürbar, aber nicht ausschließend.
 */
export const MIN_SHARE = 1e-4;

export type DatingMethod = "dirichlet" | "ensemble";

export interface DatingOptions {
  /** Masse des ausgewiesenen Intervalls. Vorgabe 0.95. */
  level?: number;
  /** Verfahren zur Ausintegration der Simulationsunsicherheit. */
  method?: DatingMethod;
}

export interface DatingResult {
  years: number[];
  /** A-posteriori-Wahrscheinlichkeit je Jahr, summiert zu 1. */
  prob: number[];
  /** Unnormierte Log-Likelihood je Jahr (für Diagnosen). */
  logL: number[];
  /** Jahr höchster Wahrscheinlichkeit. */
  mode: number;
  /** Erwartungswert des Jahres unter der A-posteriori-Verteilung. */
  expected: number;
  /** Intervall höchster Dichte, enthält mindestens `level` der Masse. */
  hdi: [number, number];
  level: number;
  /** Stichprobenumfang N. */
  n: number;
  /** Kategorien, die im Komplex vorkommen (n_k > 0). */
  usedIds: string[];
  /** true, wenn N = 0 — dann ist die Kurve flach und ohne Aussage. */
  empty: boolean;
  method: DatingMethod;
}

/** Zählungen in Ensemble-Reihenfolge, fehlende Kategorien als 0. */
function countVector(ids: readonly string[], counts: Record<string, number>): number[] {
  return ids.map((id) => {
    const v = Number(counts[id]);
    return Number.isFinite(v) && v > 0 ? v : 0;
  });
}

/**
 * Konzentration α₀ einer Dirichlet-Verteilung, die zu Mittelwert und Varianz
 * der simulierten Anteile eines Jahres passt.
 *
 * Für eine Dirichlet gilt Var_k = m_k(1−m_k)/(α₀+1). Über alle Kategorien
 * gemittelt ergibt das eine robuste Schätzung. Kategorien ohne Streuung
 * (etwa solche, die im Modell für dieses Jahr konstant 0 sind) tragen nichts
 * bei, weil sie α₀ gegen unendlich treiben würden.
 */
function fitConcentration(mean: Float64Array, varr: Float64Array, off: number, nc: number): number {
  let sum = 0, cnt = 0;
  for (let c = 0; c < nc; c++) {
    const m = mean[off + c], v = varr[off + c];
    if (!(m > 1e-6) || !(m < 1 - 1e-6) || !(v > 1e-12)) continue;
    sum += (m * (1 - m)) / v - 1;
    cnt++;
  }
  if (!cnt) return 1e4;
  // Die Grenzen fangen entartete Fälle ab: ohne Rauschen streut nichts (α₀ → ∞),
  // bei sehr wenigen Läufen kann die Schätzung unter 1 rutschen.
  return Math.min(1e6, Math.max(1, sum / cnt));
}

/**
 * Datiert einen Fundkomplex gegen das Ensemble.
 */
export function dateAssemblage(
  e: Ensemble,
  counts: Record<string, number>,
  opts: DatingOptions | number = {},
): DatingResult {
  const o: DatingOptions = typeof opts === "number" ? { level: opts } : opts;
  const level = o.level ?? 0.95;
  const method: DatingMethod = o.method ?? "dirichlet";
  const ny = e.years.length, nc = e.ids.length, runs = e.runs;
  const n = countVector(e.ids, counts);
  const used: number[] = [];
  let N = 0;
  for (let c = 0; c < nc; c++) { N += n[c]; if (n[c] > 0) used.push(c); }
  const usedIds = used.map((c) => e.ids[c]);

  if (N <= 0 || ny === 0) {
    const flat = ny ? new Array<number>(ny).fill(1 / ny) : [];
    return {
      years: e.years.slice(), prob: flat, logL: new Array<number>(ny).fill(0),
      mode: e.years[0] ?? NaN, expected: NaN, hdi: [e.years[0] ?? NaN, e.years[ny - 1] ?? NaN],
      level, n: 0, usedIds, empty: true, method,
    };
  }

  const logL: number[] = new Array(ny);
  const logMin = Math.log(MIN_SHARE);

  if (method === "ensemble") {
    const perRun = new Float64Array(runs);
    for (let y = 0; y < ny; y++) {
      for (let run = 0; run < runs; run++) {
        const off = (run * ny + y) * nc;
        let s = 0;
        // Nur belegte Kategorien tragen bei: n_k = 0 liefert 0·log p = 0.
        for (let u = 0; u < used.length; u++) {
          const c = used[u];
          const p = e.values[off + c] / 100;
          s += n[c] * (p > MIN_SHARE ? Math.log(p) : logMin);
        }
        perRun[run] = s;
      }
      logL[y] = logSumExp(perRun) - Math.log(runs);
    }
  } else {
    // Anteile und ihre Streuung je Jahr, aus dem Ensemble.
    const m = new Float64Array(ny * nc), v = new Float64Array(ny * nc);
    for (let y = 0; y < ny; y++) for (let c = 0; c < nc; c++) {
      let s = 0, s2 = 0;
      for (let run = 0; run < runs; run++) { const x = e.values[(run * ny + y) * nc + c] / 100; s += x; s2 += x * x; }
      const mm = s / runs;
      m[y * nc + c] = mm;
      v[y * nc + c] = Math.max(0, s2 / runs - mm * mm);
    }
    for (let y = 0; y < ny; y++) {
      const off = y * nc;
      const a0 = fitConcentration(m, v, off, nc);
      let s = logGamma(a0) - logGamma(a0 + N);
      for (let u = 0; u < used.length; u++) {
        const c = used[u];
        const ak = Math.max(MIN_SHARE, m[off + c]) * a0;
        s += logGamma(n[c] + ak) - logGamma(ak);
      }
      logL[y] = s;
    }
    void logMin;
  }

  // Normieren über den maximalen Log-Wert, damit nichts unterläuft.
  let mx = -Infinity;
  for (let y = 0; y < ny; y++) if (logL[y] > mx) mx = logL[y];
  const raw = logL.map((l) => Math.exp(l - mx));
  let s = 0; for (const v of raw) s += v;
  const prob = s > 0 ? raw.map((v) => v / s) : raw.map(() => 1 / ny);

  let mode = 0, expected = 0;
  for (let y = 0; y < ny; y++) { if (prob[y] > prob[mode]) mode = y; expected += prob[y] * e.years[y]; }

  return {
    years: e.years.slice(), prob, logL,
    mode: e.years[mode], expected,
    hdi: hdiSpan(e.years, prob, level), level,
    n: N, usedIds, empty: false, method,
  };
}

/**
 * Spannweite des Intervalls höchster Dichte.
 *
 * Die Jahre werden nach Wahrscheinlichkeit absteigend aufgenommen, bis `level`
 * der Masse erreicht ist; zurückgegeben werden erstes und letztes Jahr dieser
 * Menge. Ist die Verteilung zweigipfelig, liegen dazwischen auch Jahre geringer
 * Wahrscheinlichkeit — das ausgewiesene Intervall ist dann konservativ,
 * also eher zu weit als zu eng. Die Kurve selbst zeigt den Unterschied.
 */
export function hdiSpan(years: readonly number[], prob: readonly number[], level: number): [number, number] {
  const idx = years.map((_, i) => i).sort((a, b) => prob[b] - prob[a]);
  let acc = 0, lo = Infinity, hi = -Infinity;
  for (const i of idx) {
    acc += prob[i];
    if (years[i] < lo) lo = years[i];
    if (years[i] > hi) hi = years[i];
    if (acc >= level) break;
  }
  return [lo, hi];
}

export interface ResidualCurve { residual: number; result: DatingResult; }

/**
 * Dieselbe Datierung für mehrere Residualanteile.
 *
 * Der eigentliche Ertrag: Liegen die Kurven übereinander, hängt das Ergebnis
 * nicht von der unbekannten Residualität ab und ist belastbar. Wandern sie
 * auseinander, ist die Datierung eine Funktion der Annahme und sollte als
 * Spanne über alle Kurven berichtet werden.
 */
export function dateAcrossResidual(
  base: Ensemble,
  counts: Record<string, number>,
  residuals: readonly number[] = [0, 0.1, 0.2, 0.3],
  opts: DatingOptions = {},
): ResidualCurve[] {
  // Für die Datierung werden keine Perzentile gebraucht — das spart bei vielen
  // Kategorien den größten Einzelposten der Rechnung.
  return residuals.map((r) => ({ residual: r, result: dateAssemblage(applyResidual(base, r, false), counts, opts) }));
}
