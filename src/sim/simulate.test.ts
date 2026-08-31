import { simulate, rowAt } from "./simulate.js";
import { interpolateMarket, yearRange, type SimParams, type MarketTable } from "../core/model.js";

let pass = 0, fail = 0; const F: string[] = [];
function c(n: string, ok: boolean, d = "") { ok ? pass++ : (fail++, F.push(n)); console.log((ok ? "  \x1b[32m✓\x1b[0m " : "  \x1b[31m✗\x1b[0m ") + n + (d ? " — " + d : "")); }
const near = (a: number, b: number, e = 1e-6) => Math.abs(a - b) < e;

console.log("\n\x1b[1mSimulationskern\x1b[0m\n");

const ids = ["A", "B", "C"];
const market: MarketTable = { years: [100, 150, 200], shares: { A: [100, 0, 0], B: [0, 100, 0], C: [0, 0, 100] } };
const years = yearRange(100, 200);
const M = interpolateMarket(market, ids, years);

function params(over: Partial<SimParams> = {}): SimParams {
  return {
    startYear: 100, endYear: 200,
    replacement: {}, replacementDefault: 0.1,
    noiseSd: 0, runs: 20, seed: 7,
    settlementMode: false, residual: 0,
    initial: { A: 100, B: 0, C: 0 },
    ...over,
  };
}

// Interpolation
c("Marktzeile am Stützjahr trifft den Stützwert", near(M[0 * 3 + 0], 100) && near(M[0 * 3 + 1], 0));
c("Marktzeile in der Mitte interpoliert linear (A=50, B=50 bei Jahr 125)", near(M[25 * 3 + 0], 50) && near(M[25 * 3 + 1], 50));
c("jede Marktzeile summiert auf 100", years.every((_, y) => near(M[y * 3] + M[y * 3 + 1] + M[y * 3 + 2], 100, 1e-9)));
{
  const short = interpolateMarket(market, ids, yearRange(80, 220));
  c("vor dem ersten Stützjahr wird der Randwert gehalten, nicht extrapoliert", near(short[0 * 3 + 0], 100));
  c("nach dem letzten Stützjahr wird der Randwert gehalten", near(short[140 * 3 + 2], 100));
}

// Determinismus
{
  const a = simulate(M, years, ids, params({ noiseSd: 2 }));
  const b = simulate(M, years, ids, params({ noiseSd: 2 }));
  c("gleicher Seed → identisches Ensemble", a.values.every((v, i) => v === b.values[i]));
  const d = simulate(M, years, ids, params({ noiseSd: 2, seed: 8 }));
  c("anderer Seed → anderes Ensemble", !a.values.every((v, i) => v === d.values[i]));
}

// Grundeigenschaften
{
  const e = simulate(M, years, ids, params({ noiseSd: 2, runs: 30 }));
  let ok = true;
  for (let r = 0; r < e.runs; r++) for (let y = 0; y < years.length; y++) {
    let s = 0; for (let k = 0; k < 3; k++) s += e.values[(r * years.length + y) * 3 + k];
    if (Math.abs(s - 100) > 0.02) ok = false;
  }
  c("jede Jahreszeile jedes Laufs summiert auf 100", ok);
  c("keine negativen Anteile", e.values.every((v) => v >= 0));
  let band = true;
  for (let i = 0; i < e.mean.length; i++) if (!(e.p10[i] <= e.mean[i] + 1e-9 && e.mean[i] <= e.p90[i] + 1e-9)) band = false;
  c("p10 ≤ Mittelwert ≤ p90", band);
}

// Grenzfälle des Modells
{
  const e = simulate(M, years, ids, params({ replacementDefault: 0, noiseSd: 0 }));
  const last = rowAt(e.mean, years.length - 1, 3);
  c("Ersatzrate 0 → Bestand bleibt die Startverteilung", near(last[0], 100, 1e-4));
}
{
  const e = simulate(M, years, ids, params({ replacementDefault: 1, noiseSd: 0 }));
  const y = 25; // Jahr 125, Markt A=50 B=50
  const row = rowAt(e.mean, y, 3);
  c("Ersatzrate 1 → Bestand ist das Marktangebot", near(row[0], 50, 1e-3) && near(row[1], 50, 1e-3));
}
{
  const e = simulate(M, years, ids, params({ settlementMode: true, noiseSd: 0, initial: { A: 0, B: 0, C: 100 } }));
  const row = rowAt(e.mean, 0, 3);
  c("Neugründung: Startjahr entspricht dem Markt, nicht der Startverteilung", near(row[0], 100, 1e-3));
}
{
  // Konstanter Markt: der Bestand muss gegen das Marktspektrum laufen.
  const flat: MarketTable = { years: [100, 200], shares: { A: [20, 20], B: [30, 30], C: [50, 50] } };
  const Mf = interpolateMarket(flat, ids, years);
  const e = simulate(Mf, years, ids, params({ replacementDefault: 0.1, noiseSd: 0, initial: { A: 100, B: 0, C: 0 } }));
  const row = rowAt(e.mean, years.length - 1, 3);
  c(`konstanter Markt → Bestand konvergiert dorthin (${row.map((x) => x.toFixed(1)).join("/")})`,
    near(row[0], 20, 0.5) && near(row[1], 30, 0.5) && near(row[2], 50, 0.5));
}
{
  const e = simulate(M, years, ids, params({ initial: { A: 0, B: 0, C: 0 } }));
  const row = rowAt(e.mean, 0, 3);
  c("Startverteilung aus lauter Nullen → Gleichverteilung statt NaN", row.every((v) => Number.isFinite(v)));
}
{
  const e = simulate(M, years, ids, params({ replacement: { A: 0.4, B: 0.05 }, replacementDefault: 0.2, noiseSd: 0 }));
  c("kategoriespezifische Raten greifen (Ensemble endlich)", e.mean.every((v) => Number.isFinite(v)));
}

console.log(`\n\x1b[1mErgebnis:\x1b[0m ${pass} bestanden, ${fail} fehlgeschlagen`);
if (fail) { console.log("Fehlgeschlagen: " + F.join(", ")); process.exit(1); }
