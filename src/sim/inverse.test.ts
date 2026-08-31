import { simulate, rowAt } from "./simulate.js";
import { dateAssemblage, dateAcrossResidual, hdiSpan, MIN_SHARE } from "./inverse.js";
import { interpolateMarket, yearRange, type SimParams, type MarketTable } from "../core/model.js";

let pass = 0, fail = 0; const F: string[] = [];
function c(n: string, ok: boolean, d = "") { ok ? pass++ : (fail++, F.push(n)); console.log((ok ? "  \x1b[32m✓\x1b[0m " : "  \x1b[31m✗\x1b[0m ") + n + (d ? " — " + d : "")); }

console.log("\n\x1b[1mInverse Datierung\x1b[0m\n");

const ids = ["A", "B", "C"];
const years = yearRange(100, 250);
const ramp: MarketTable = { years: [100, 160, 220], shares: { A: [100, 0, 0], B: [0, 100, 0], C: [0, 0, 100] } };
const M = interpolateMarket(ramp, ids, years);
const P: SimParams = {
  startYear: 100, endYear: 250, replacement: {}, replacementDefault: 0.15,
  noiseSd: 1.5, runs: 150, seed: 11, settlementMode: false, residual: 0,
  initial: { A: 100, B: 0, C: 0 },
};
const E = simulate(M, years, ids, P);

/** Baut Zählungen aus dem simulierten Mittelwert eines bekannten Jahres. */
function countsAtYear(year: number, N: number): Record<string, number> {
  const y = years.indexOf(year);
  const row = rowAt(E.mean, y, 3);
  const raw = row.map((p) => (p / 100) * N);
  // größte Restanteile zuerst runden, damit die Summe exakt N ergibt
  const fl = raw.map(Math.floor);
  let rest = N - fl.reduce((a, b) => a + b, 0);
  const order = raw.map((v, i) => [v - fl[i], i] as const).sort((a, b) => b[0] - a[0]);
  for (const [, i] of order) { if (rest <= 0) break; fl[i]++; rest--; }
  return { A: fl[0], B: fl[1], C: fl[2] };
}

// Kernvalidierung: das eingespeiste Jahr muss zurückkommen.
for (const y of [130, 170, 210]) {
  const r = dateAssemblage(E, countsAtYear(y, 400));
  c(`Rückgewinnung Jahr ${y} aus 400 Stück: Modus ${r.mode}, Intervall ${r.hdi[0]}–${r.hdi[1]}`,
    Math.abs(r.mode - y) <= 8 && r.hdi[0] <= y && y <= r.hdi[1]);
}

// Stichprobenumfang
{
  const small = dateAssemblage(E, countsAtYear(170, 20));
  const big = dateAssemblage(E, countsAtYear(170, 2000));
  const wS = small.hdi[1] - small.hdi[0], wB = big.hdi[1] - big.hdi[0];
  c(`größere Stichprobe → engeres Intervall (N=20: ${wS} Jahre, N=2000: ${wB} Jahre)`, wB < wS);
  c("kleine Stichprobe liefert trotzdem ein gültiges Intervall", Number.isFinite(wS) && wS > 0);
}

// Normierung und Randfälle
{
  const r = dateAssemblage(E, countsAtYear(170, 300));
  const s = r.prob.reduce((a, b) => a + b, 0);
  c("A-posteriori summiert auf 1", Math.abs(s - 1) < 1e-9);
  c("alle Wahrscheinlichkeiten endlich und ≥ 0", r.prob.every((p) => Number.isFinite(p) && p >= 0));
  c("Erwartungswert liegt im Zeitraum", r.expected >= 100 && r.expected <= 250);
  c("n wird durchgereicht", r.n === 300);
  c("usedIds nennt nur belegte Kategorien", r.usedIds.every((id) => (countsAtYear(170, 300) as Record<string, number>)[id] > 0));
}
{
  const r = dateAssemblage(E, {});
  c("leerer Komplex → empty-Kennzeichen", r.empty && r.n === 0);
  c("leerer Komplex → flache Kurve", r.prob.every((p) => Math.abs(p - 1 / years.length) < 1e-12));
}
{
  const r = dateAssemblage(E, { A: 5, Z: 99 });
  c("unbekannte Kategorien werden ignoriert", r.n === 5);
}
{
  // Eine einzelne Scherbe einer im Modell unmöglichen Kategorie darf ein Jahr
  // dämpfen, aber nicht die gesamte Kurve zerstören.
  const good = countsAtYear(210, 200);
  const withOutlier = { ...good, A: (good.A ?? 0) + 1 };
  const r = dateAssemblage(E, withOutlier);
  c("einzelne unerwartete Scherbe zerstört die Kurve nicht (Untergrenze wirkt)",
    r.prob.some((p) => p > 0.001) && Number.isFinite(r.expected));
}

// HDI
{
  const yy = [1, 2, 3, 4, 5];
  c("HDI bei Punktmasse ist ein einzelnes Jahr",
    JSON.stringify(hdiSpan(yy, [0, 0, 1, 0, 0], 0.95)) === JSON.stringify([3, 3]));
  c("HDI bei Gleichverteilung umfasst alles",
    JSON.stringify(hdiSpan(yy, [0.2, 0.2, 0.2, 0.2, 0.2], 0.95)) === JSON.stringify([1, 5]));
  c("niedrigeres Niveau → engeres Intervall",
    hdiSpan(yy, [0.05, 0.15, 0.6, 0.15, 0.05], 0.5)[1] - hdiSpan(yy, [0.05, 0.15, 0.6, 0.15, 0.05], 0.5)[0]
    <= hdiSpan(yy, [0.05, 0.15, 0.6, 0.15, 0.05], 0.95)[1] - hdiSpan(yy, [0.05, 0.15, 0.6, 0.15, 0.05], 0.95)[0]);
}

// Residual-Schar
{
  const curves = dateAcrossResidual(E, countsAtYear(200, 300), [0, 0.1, 0.2, 0.3]);
  c("dateAcrossResidual liefert eine Kurve je Anteil", curves.length === 4 && curves[2].residual === 0.2);
  const modes = curves.map((x) => x.result.mode);
  c(`Residualität verschiebt die Datierung nach hinten (Modi ${modes.join(", ")})`,
    modes[3] >= modes[0]);
  c("alle Kurven normiert", curves.every((x) => Math.abs(x.result.prob.reduce((a, b) => a + b, 0) - 1) < 1e-9));
}


/* ── Verfahrensvergleich: Dirichlet gegen rohe Monte-Carlo-Summe ── */
{
  const counts = countsAtYear(180, 300);
  const d = dateAssemblage(E, counts, { method: "dirichlet" });
  const m = dateAssemblage(E, counts, { method: "ensemble" });
  c("Vorgabe ist das Dirichlet-Verfahren", dateAssemblage(E, counts).method === "dirichlet");
  c(`beide Verfahren datieren gleich (Dirichlet ${d.mode}, Monte Carlo ${m.mode})`, Math.abs(d.mode - m.mode) <= 6);

  /**
   * Richtungswechsel im tragenden Teil der Kurve — ein Maß für die Zackigkeit.
   * Der flache Schwanz bleibt außen vor: dort liegen beide Verfahren bei nahezu
   * null, und Gleitkommarauschen würde die Zählung beherrschen.
   */
  const wiggles = (p: number[]) => {
    const cut = Math.max(...p) * 0.05;
    let n = 0;
    for (let i = 1; i + 1 < p.length; i++) {
      if (p[i] < cut) continue;
      if ((p[i] - p[i - 1]) * (p[i + 1] - p[i]) < 0) n++;
    }
    return n;
  };
  const wd = wiggles(d.prob), wm = wiggles(m.prob);
  c(`Dirichlet liefert die glattere Kurve (${wd} gegen ${wm} Richtungswechsel im Hauptteil)`, wd < wm / 2);

  // Reproduzierbarkeit über verschiedene Seeds: die eigentliche Schwäche der
  // Monte-Carlo-Summe ist, dass der Modus mit dem Seed springt.
  const spread = (method: "dirichlet" | "ensemble") => {
    const modes = [11, 12, 13, 14].map((seed) =>
      dateAssemblage(simulate(M, years, ids, { ...P, seed }), counts, { method }).mode);
    return Math.max(...modes) - Math.min(...modes);
  };
  const sd = spread("dirichlet"), sm = spread("ensemble");
  c(`Dirichlet ist über Seeds hinweg stabiler (Streuung ${sd} gegen ${sm} Jahre)`, sd <= sm);
}
{
  // Ohne Rauschen streut das Ensemble nicht; die Konzentrationsschätzung darf
  // dann nicht entgleisen.
  const flat = simulate(M, years, ids, { ...P, noiseSd: 0, runs: 20 });
  const r = dateAssemblage(flat, countsAtYear(170, 100));
  c("Ensemble ohne Streuung liefert trotzdem eine gültige Kurve",
    r.prob.every((p) => Number.isFinite(p) && p >= 0) && Math.abs(r.prob.reduce((a, b) => a + b, 0) - 1) < 1e-9);
}
{
  const curves = dateAcrossResidual(E, countsAtYear(200, 200), [0, 0.2], { method: "ensemble" });
  c("dateAcrossResidual reicht das Verfahren durch", curves.every((x) => x.result.method === "ensemble"));
  c("dateAssemblage nimmt weiterhin eine Zahl als Niveau", dateAssemblage(E, countsAtYear(170, 100), 0.5).level === 0.5);
}

c(`Untergrenze für Anteile ist gesetzt (${MIN_SHARE})`, MIN_SHARE > 0 && MIN_SHARE < 1e-3);

console.log(`\n\x1b[1mErgebnis:\x1b[0m ${pass} bestanden, ${fail} fehlgeschlagen`);
if (fail) { console.log("Fehlgeschlagen: " + F.join(", ")); process.exit(1); }
