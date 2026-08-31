import { percentileSorted, logSumExp, normalizeProb, seededGaussian, mean } from "./stats.js";

let pass = 0, fail = 0; const F: string[] = [];
function c(n: string, ok: boolean, d = "") { ok ? pass++ : (fail++, F.push(n)); console.log((ok ? "  \x1b[32m✓\x1b[0m " : "  \x1b[31m✗\x1b[0m ") + n + (d ? " — " + d : "")); }
const near = (a: number, b: number, e = 1e-9) => Math.abs(a - b) < e;

console.log("\n\x1b[1mStatistische Helfer\x1b[0m\n");

// Perzentil — Referenzwerte aus numpy.percentile (lineare Interpolation)
const s = [1, 2, 3, 4, 5];
c("Median von [1..5] = 3", near(percentileSorted(s, 50), 3));
c("10-%-Perzentil von [1..5] = 1.4", near(percentileSorted(s, 10), 1.4));
c("90-%-Perzentil von [1..5] = 4.6", near(percentileSorted(s, 90), 4.6));
c("0 % = Minimum", near(percentileSorted(s, 0), 1));
c("100 % = Maximum", near(percentileSorted(s, 100), 5));
c("Einzelwert", near(percentileSorted([7], 90), 7));
c("leer → NaN", Number.isNaN(percentileSorted([], 50)));

// logSumExp
c("logSumExp([0,0]) = ln2", near(logSumExp([0, 0]), Math.LN2, 1e-12));
c("logSumExp überlebt sehr kleine Werte", near(logSumExp([-1000, -1000]), -1000 + Math.LN2, 1e-9));
c("logSumExp([-Inf,-Inf]) = -Inf", logSumExp([-Infinity, -Infinity]) === -Infinity);
c("logSumExp ignoriert -Inf-Summanden", near(logSumExp([-Infinity, 0]), 0, 1e-12));

// normalizeProb
const p = normalizeProb([1, 3]);
c("normalizeProb summiert auf 1", near(p[0] + p[1], 1));
c("normalizeProb erhält Verhältnis", near(p[1] / p[0], 3, 1e-12));
c("Nullsumme → Gleichverteilung", near(normalizeProb([0, 0, 0])[0], 1 / 3));

// Gauß: Reproduzierbarkeit und grobe Momente
const g1 = seededGaussian(42), g2 = seededGaussian(42);
c("gleicher Seed → gleiche Folge", [0, 1, 2, 3, 4].every(() => near(g1(), g2())));
const g = seededGaussian(7); const xs: number[] = [];
for (let i = 0; i < 20000; i++) xs.push(g());
const m = mean(xs); const sd = Math.sqrt(mean(xs.map((x) => (x - m) ** 2)));
c(`Mittelwert ≈ 0 (${m.toFixed(4)})`, Math.abs(m) < 0.03);
c(`Standardabweichung ≈ 1 (${sd.toFixed(4)})`, Math.abs(sd - 1) < 0.03);
c("Seed 0 ist nicht reproduzierbar", seededGaussian(0)() !== seededGaussian(0)());

console.log(`\n\x1b[1mErgebnis:\x1b[0m ${pass} bestanden, ${fail} fehlgeschlagen`);
if (fail) { console.log("Fehlgeschlagen: " + F.join(", ")); process.exit(1); }
