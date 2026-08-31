/**
 * Kleine statistische Helfer für Simulation und inverse Datierung.
 * Framework-frei, damit sie in Worker, Tests und App gleichermaßen laufen.
 */
import { mulberry32 } from "../core/rng.js";

/** Standardnormalverteilte Zufallszahlen (Box-Muller) auf Basis von mulberry32. */
export function gaussian(rand: () => number): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) { const s = spare; spare = null; return s; }
    let u = 0, v = 0, s = 0;
    do { u = rand() * 2 - 1; v = rand() * 2 - 1; s = u * u + v * v; } while (s === 0 || s >= 1);
    const f = Math.sqrt((-2 * Math.log(s)) / s);
    spare = v * f;
    return u * f;
  };
}

/** Gaußgenerator zu einem Seed. Seed 0 heißt „nicht reproduzierbar". */
export function seededGaussian(seed: number): () => number {
  const s = seed === 0 ? (Math.random() * 0x7fffffff) >>> 0 : seed >>> 0;
  return gaussian(mulberry32(s));
}

/**
 * Perzentil nach linearer Interpolation zwischen den Rangplätzen
 * (dieselbe Definition wie `numpy.percentile` mit Voreinstellung, damit
 * Ergebnisse mit STOCHASI 1 vergleichbar bleiben).
 * `sorted` muss aufsteigend sortiert sein.
 */
export function percentileSorted(sorted: ArrayLike<number>, q: number): number {
  const n = sorted.length;
  if (n === 0) return NaN;
  if (n === 1) return sorted[0];
  const pos = (q / 100) * (n - 1);
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

export function mean(v: ArrayLike<number>): number {
  let s = 0; for (let i = 0; i < v.length; i++) s += v[i];
  return v.length ? s / v.length : NaN;
}

/**
 * log(Σ exp(x_i)) ohne Überlauf. Wird gebraucht, weil die Likelihoods der
 * inversen Datierung bei größeren Fundzahlen weit unter den kleinsten
 * darstellbaren Gleitkommawert fallen.
 */
export function logSumExp(xs: ArrayLike<number>): number {
  let m = -Infinity;
  for (let i = 0; i < xs.length; i++) if (xs[i] > m) m = xs[i];
  if (!Number.isFinite(m)) return m;
  let s = 0;
  for (let i = 0; i < xs.length; i++) s += Math.exp(xs[i] - m);
  return m + Math.log(s);
}

/** Auf Summe 1 normieren. Eine Nullsumme ergibt eine Gleichverteilung. */
export function normalizeProb(v: readonly number[]): number[] {
  let s = 0; for (const x of v) s += x;
  if (!(s > 0)) return v.map(() => 1 / Math.max(1, v.length));
  return v.map((x) => x / s);
}

/**
 * log Γ(x) nach Lanczos (g = 7, 9 Koeffizienten). Genau auf etwa 15 Stellen für
 * x > 0. Wird für die Dirichlet-Multinomial-Likelihood gebraucht.
 */
const LANCZOS = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028,
  771.32342877765313, -176.61502916214059, 12.507343278686905,
  -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
];
export function logGamma(x: number): number {
  if (!(x > 0)) return NaN;
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  const z = x - 1;
  let a = LANCZOS[0];
  const t = z + 7.5;
  for (let i = 1; i < 9; i++) a += LANCZOS[i] / (z + i);
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(a);
}
