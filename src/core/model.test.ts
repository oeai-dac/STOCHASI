import { catIds, normalizeTo100, rateFor, yearRange, interpolateMarket, assemblageColor, appendAssemblages, MAX_REPLACEMENT, type MarketTable, type SimParams } from "./model.js";
import { OKABE_ITO } from "./palette.js";

let pass = 0, fail = 0; const F: string[] = [];
function c(n: string, ok: boolean, d = "") { ok ? pass++ : (fail++, F.push(n)); console.log((ok ? "  \x1b[32m✓\x1b[0m " : "  \x1b[31m✗\x1b[0m ") + n + (d ? " — " + d : "")); }
const near = (a: number, b: number, e = 1e-9) => Math.abs(a - b) < e;

console.log("\n\x1b[1mDatenmodell\x1b[0m\n");

c("catIds hält die Reihenfolge", catIds([{ id: "B", name: "b", color: "#000" }, { id: "A", name: "a", color: "#000" }]).join(",") === "B,A");

c("normalizeTo100 summiert auf 100", near(normalizeTo100([1, 3]).reduce((a, b) => a + b, 0), 100));
c("normalizeTo100 erhält Verhältnisse", near(normalizeTo100([1, 3])[1] / normalizeTo100([1, 3])[0], 3));
c("normalizeTo100 lässt eine Nullsumme unverändert", normalizeTo100([0, 0]).join(",") === "0,0");
c("normalizeTo100 verändert das Original nicht", (() => { const v = [1, 1]; normalizeTo100(v); return v.join(",") === "1,1"; })());

const P: SimParams = {
  startYear: 100, endYear: 200, replacement: { A: 0.3, B: -1, C: 9, D: NaN },
  replacementDefault: 0.12, noiseSd: 1, runs: 10, seed: 1,
  settlementMode: false, residual: 0, initial: {},
};
c("rateFor nimmt den eigenen Wert", near(rateFor(P, "A"), 0.3));
c("rateFor fällt auf die Vorgabe zurück", near(rateFor(P, "Z"), 0.12));
c("rateFor begrenzt nach unten", near(rateFor(P, "B"), 0));
c("rateFor begrenzt nach oben", near(rateFor(P, "C"), MAX_REPLACEMENT));
c("rateFor behandelt NaN als nicht gesetzt", near(rateFor(P, "D"), 0.12));

c("yearRange ist einschließlich", yearRange(10, 13).join(",") === "10,11,12,13");
c("yearRange bei einem einzigen Jahr", yearRange(5, 5).join(",") === "5");
c("yearRange bei umgekehrten Grenzen ist leer", yearRange(10, 5).length === 0);
c("yearRange verträgt Jahre vor Christus", yearRange(-2, 1).join(",") === "-2,-1,0,1");

/* ── Interpolation ── */
const ids = ["A", "B"];
{
  const m: MarketTable = { years: [100, 200], shares: { A: [100, 0], B: [0, 100] } };
  const M = interpolateMarket(m, ids, [100, 150, 200]);
  c("Stützjahre werden exakt getroffen", near(M[0], 100) && near(M[5], 100));
  c("Mitte wird linear interpoliert", near(M[2], 50) && near(M[3], 50));
}
{
  const m: MarketTable = { years: [100], shares: { A: [40], B: [60] } };
  const M = interpolateMarket(m, ids, [50, 100, 150]);
  c("ein einzelnes Stützjahr wird nach beiden Seiten gehalten", near(M[0], 40) && near(M[4], 40));
}
{
  const m: MarketTable = { years: [], shares: {} };
  c("leere Markttabelle ergibt Nullen statt NaN", interpolateMarket(m, ids, [100, 110]).every((v) => v === 0));
}
{
  const m: MarketTable = { years: [100, 200], shares: { A: [1, 1] } };
  const M = interpolateMarket(m, ids, [150]);
  c("fehlende Kategoriespalte zählt als 0", near(M[0], 100) && near(M[1], 0));
}
{
  const m: MarketTable = { years: [100, 200], shares: { A: [0, 0], B: [0, 0] } };
  c("Jahreszeile aus lauter Nullen bleibt 0 statt NaN", interpolateMarket(m, ids, [150]).every((v) => v === 0));
}
{
  const m: MarketTable = { years: [100, 200], shares: { A: [3, 3], B: [1, 1] } };
  const M = interpolateMarket(m, ids, [100, 150, 200]);
  c("unnormierte Eingabe wird zeilenweise auf 100 gebracht", near(M[0], 75) && near(M[1], 25) && near(M[2], 75));
}
{
  // Nicht monoton sortierte Stützjahre kommen bei handgepflegten Tabellen vor.
  const m: MarketTable = { years: [200, 100], shares: { A: [0, 100], B: [100, 0] } };
  const M = interpolateMarket(m, ids, [100, 200]);
  c("unsortierte Stützjahre liefern endliche Werte", M.every((v) => Number.isFinite(v)));
}
{
  const m: MarketTable = { years: [100, 200], shares: { A: [100, 0], B: [0, 100] } };
  const M = interpolateMarket(m, ids, yearRange(100, 200));
  let ok = true;
  for (let y = 0; y < 101; y++) if (Math.abs(M[y * 2] + M[y * 2 + 1] - 100) > 1e-9) ok = false;
  c("jede interpolierte Jahreszeile summiert auf 100", ok);
  c("die Interpolation ist monoton", (() => { for (let y = 1; y < 101; y++) if (M[y * 2] > M[(y - 1) * 2] + 1e-9) return false; return true; })());
}

/* ── Farbe je Fundkomplex und Anhängen ── */
{
  const a = { id: "A1", name: "eins", counts: { A: 1 } };
  c("ohne eigene Farbe greift die Palette", assemblageColor(a, 0) === OKABE_ITO[0]);
  c("die Palette folgt der Reihenfolge", assemblageColor(a, 1) === OKABE_ITO[1]);
  c("eine gesetzte Farbe gilt", assemblageColor({ ...a, color: "#123456" }, 0) === "#123456");
  c("eine unsinnige Farbe fällt auf die Palette zurück", assemblageColor({ ...a, color: "rot" }, 2) === OKABE_ITO[2]);
}
{
  const existing = [
    { id: "A1", name: "Insula I", counts: { A: 1 }, color: "#123456" },
    { id: "A2", name: "Insula II", counts: { A: 2 }, color: "#654321" },
  ];
  const incoming = [
    { id: "A1", name: "Insula I", counts: { A: 3 } },
    { id: "A9", name: "Insula IX", counts: { A: 4 } },
  ];
  const merged = appendAssemblages(existing, incoming);
  c("Anhängen behält die vorhandenen Komplexe", merged.length === 4 && merged[0].id === "A1" && merged[0].counts.A === 1);
  c("kollidierende Kennungen werden neu vergeben", new Set(merged.map((x) => x.id)).size === 4);
  c("gleiche Namen werden unterschieden", merged[2].name === "Insula I (2)" && merged[3].name === "Insula IX");
  c("die Farben setzen die Reihe fort", merged[2].color === OKABE_ITO[2] && merged[3].color === OKABE_ITO[3]);
  c("gesetzte Farben bleiben unangetastet", merged[0].color === "#123456");
  c("Ersetzen ist derselbe Aufruf mit leerer Grundmenge",
    appendAssemblages([], incoming).map((x) => x.name).join(",") === "Insula I,Insula IX");
}

console.log(`\n\x1b[1mErgebnis:\x1b[0m ${pass} bestanden, ${fail} fehlgeschlagen`);
if (fail) { console.log("Fehlgeschlagen: " + F.join(", ")); process.exit(1); }
