import { simulate, rowAt } from "./simulate.js";
import { applyResidual } from "./residual.js";
import { interpolateMarket, yearRange, type SimParams, type MarketTable } from "../core/model.js";

let pass = 0, fail = 0; const F: string[] = [];
function c(n: string, ok: boolean, d = "") { ok ? pass++ : (fail++, F.push(n)); console.log((ok ? "  \x1b[32m✓\x1b[0m " : "  \x1b[31m✗\x1b[0m ") + n + (d ? " — " + d : "")); }
const near = (a: number, b: number, e = 1e-4) => Math.abs(a - b) < e;

console.log("\n\x1b[1mResidualität\x1b[0m\n");

const ids = ["A", "B", "C"];
const years = yearRange(100, 200);
const ramp: MarketTable = { years: [100, 150, 200], shares: { A: [100, 0, 0], B: [0, 100, 0], C: [0, 0, 100] } };
const M = interpolateMarket(ramp, ids, years);
const P: SimParams = {
  startYear: 100, endYear: 200, replacement: {}, replacementDefault: 0.15,
  noiseSd: 0, runs: 12, seed: 3, settlementMode: false, residual: 0,
  initial: { A: 100, B: 0, C: 0 },
};
const base = simulate(M, years, ids, P);

c("r = 0 liefert dasselbe Ensemble-Objekt", applyResidual(base, 0) === base);

const mixed = applyResidual(base, 0.3);
c("Startjahr bleibt unverändert", rowAt(mixed.mean, 0, 3).every((v, k) => near(v, rowAt(base.mean, 0, 3)[k])));
{
  let ok = true;
  for (let r = 0; r < mixed.runs; r++) for (let y = 0; y < years.length; y++) {
    let s = 0; for (let k = 0; k < 3; k++) s += mixed.values[(r * years.length + y) * 3 + k];
    if (Math.abs(s - 100) > 0.02) ok = false;
  }
  c("jede Jahreszeile summiert weiterhin auf 100", ok);
}
{
  // Altbestand ist bei diesem Markt älter, also A-lastiger: der A-Anteil im
  // Endjahr muss durch Residualität steigen.
  const a0 = rowAt(base.mean, years.length - 1, 3)[0];
  const a3 = rowAt(mixed.mean, years.length - 1, 3)[0];
  c(`Altmaterial hebt den Anteil der frühen Kategorie (${a0.toFixed(2)} → ${a3.toFixed(2)})`, a3 > a0 + 0.5);
}
{
  const c0 = rowAt(base.mean, years.length - 1, 3)[2];
  const c3 = rowAt(mixed.mean, years.length - 1, 3)[2];
  c(`und senkt den Anteil der späten Kategorie (${c0.toFixed(2)} → ${c3.toFixed(2)})`, c3 < c0 - 0.5);
}
{
  // Stationärer Markt: Bestand ist konstant, Altbestand gleicht ihm — Residualität
  // darf dann nichts ändern. Guter Beleg, dass nicht versehentlich verschoben wird.
  const flat: MarketTable = { years: [100, 200], shares: { A: [20, 20], B: [30, 30], C: [50, 50] } };
  const Ef = simulate(interpolateMarket(flat, ids, years), years, ids, { ...P, initial: { A: 20, B: 30, C: 50 } });
  const Rf = applyResidual(Ef, 0.4);
  const a = rowAt(Ef.mean, years.length - 1, 3), b = rowAt(Rf.mean, years.length - 1, 3);
  c("im Gleichgewicht ändert Residualität nichts", a.every((v, k) => near(v, b[k], 1e-3)));
}
{
  const m5 = applyResidual(base, 0.5), m1 = applyResidual(base, 0.1);
  const a1 = rowAt(m1.mean, years.length - 1, 3)[0], a5 = rowAt(m5.mean, years.length - 1, 3)[0];
  c("stärkere Residualität wirkt stärker", a5 > a1);
}
{
  const noisy = simulate(M, years, ids, { ...P, noiseSd: 3, runs: 40 });
  const nm = applyResidual(noisy, 0.3);
  const spanBase = rowAt(noisy.p90, 60, 3)[1] - rowAt(noisy.p10, 60, 3)[1];
  const spanMix = rowAt(nm.p90, 60, 3)[1] - rowAt(nm.p10, 60, 3)[1];
  c(`Mittelung über Vorjahre verschmälert das Band (${spanBase.toFixed(2)} → ${spanMix.toFixed(2)})`, spanMix < spanBase);
}


{
  // Ohne Perzentile müssen die Läufe selbst identisch bleiben — nur die
  // abgeleiteten Kennzahlen fehlen dann.
  const full = applyResidual(base, 0.25);
  const lean = applyResidual(base, 0.25, false);
  c("ohne Perzentile bleiben die Läufe identisch", lean.values.every((v, i) => v === full.values[i]));
  c("ohne Perzentile werden Mittelwert und Perzentile nicht neu gerechnet", lean.mean === base.mean && lean.p10 === base.p10);
}

console.log(`\n\x1b[1mErgebnis:\x1b[0m ${pass} bestanden, ${fail} fehlgeschlagen`);
if (fail) { console.log("Fehlgeschlagen: " + F.join(", ")); process.exit(1); }
