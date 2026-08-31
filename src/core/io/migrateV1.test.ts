import { readFileSync } from "node:fs";
import { isV1Config, migrateV1, MigrateError } from "./migrateV1.js";
import { readProject, writeProject, sanitize, emptyProject, ProjectError } from "./project.js";
import { PRESET_V1 } from "../../data/centres.js";

let pass = 0, fail = 0; const F: string[] = [];
function c(n: string, ok: boolean, d = "") { ok ? pass++ : (fail++, F.push(n)); console.log((ok ? "  \x1b[32m✓\x1b[0m " : "  \x1b[31m✗\x1b[0m ") + n + (d ? " — " + d : "")); }
const near = (a: number, b: number, e = 1e-6) => Math.abs(a - b) < e;

console.log("\n\x1b[1mProjektdateien und v1-Übernahme\x1b[0m\n");

/**
 * Die echte Konfiguration aus STOCHASI 1 (Flavia Solva, Insula XLI). Sie liegt
 * als Beispieldatei im Repository und dient hier zugleich als Prüfstück: Die
 * Übernahme lässt sich nur an einer echten v1-Datei belegen, nicht an einer
 * nachgebauten.
 */
const V1_PATHS = [
  "examples/stochasi-1_flavia-solva_insula-xli.json",
  "../STOCHASI_version-1-0/config/config_f-solva_insula-xli.json",
];
let v1raw: string | null = null;
for (const path of V1_PATHS) {
  try { v1raw = readFileSync(path, "utf8"); break; } catch { /* nächster Pfad */ }
}

c("isV1Config erkennt v1", isV1Config({ _meta: { app: "STOCHASI", version: "1.0" } }));
c("isV1Config lehnt v2 ab", !isV1Config({ _meta: { app: "STOCHASI", version: "2.0" } }));
c("isV1Config lehnt Fremdes ab", !isV1Config({ _meta: { app: "CombiTab", version: "1.0" } }));
c("isV1Config verträgt Unsinn", !isV1Config(null) && !isV1Config("x") && !isV1Config([]));

{
  let threw = false;
  try { migrateV1({ _meta: { app: "X" } }); } catch (e) { threw = e instanceof MigrateError; }
  c("migrateV1 wirft bei fremder Datei", threw);
}

if (v1raw) {
  const { project, report } = migrateV1(JSON.parse(v1raw), "Insula XLI");
  c("echte v1-Datei: fünf Kategorien", project.categories.length === 5);
  c("echte v1-Datei: Kennungen erhalten", project.categories.map((x) => x.id).join(",") === "IT,LG,BA,MG,RZ");
  c("echte v1-Datei: Namen erhalten", project.categories[3].name === "Central Gaulish");
  c("echte v1-Datei: Farben von v1 erhalten", project.categories[0].color === PRESET_V1[0].color);
  c("echte v1-Datei: Zeitraum 125–290", project.params.startYear === 125 && project.params.endYear === 290);
  c("echte v1-Datei: Ersatzrate 0,1", near(project.params.replacementDefault, 0.1));
  c("echte v1-Datei: Rauschen 2", near(project.params.noiseSd, 2));
  c("echte v1-Datei: 100 Läufe", project.params.runs === 100);
  c("echte v1-Datei: 18 Marktjahre", project.market.years.length === 18);
  c("echte v1-Datei: Marktzeilen auf 100 normiert",
    project.market.years.every((_, k) => near(project.categories.reduce((a, cc) => a + project.market.shares[cc.id][k], 0), 100, 1e-6)));
  c("echte v1-Datei: Startverteilung übernommen", near(project.params.initial.IT, 60) && near(project.params.initial.BA, 30));
  c("echte v1-Datei: Fundkomplex mit 77 Stück",
    project.assemblages.length === 1 && Object.entries(project.assemblages[0].counts).reduce((a, [, v]) => a + v, 0) === 77);
  c("echte v1-Datei: MG = 56", project.assemblages[0].counts.MG === 56);
  c("echte v1-Datei: Vergleichsjahr 170", project.comparisonYear === 170);
  c("echte v1-Datei: Residualität startet bei 0 (rechnet wie v1)", project.params.residual === 0);
  c("echte v1-Datei: keine kategoriespezifischen Raten (rechnet wie v1)", Object.keys(project.params.replacement).length === 0);
  c("echte v1-Datei: als v2 markiert", project._meta.version === "2.0");
  c("Marktdaten als vorhanden gemeldet", report.hadMarketData && report.hadExcavationData);
  // Rundlauf
  const back = readProject(writeProject(project), "x");
  c("Rundlauf schreiben/lesen erhält den Fundkomplex", back.project.assemblages[0].counts.MG === 56);
  c("Rundlauf erhält die Marktkurve", back.project.market.years.length === 18);
  c("Rundlauf wird nicht mehr als v1 gelesen", !back.fromV1);
} else {
  c("v1-Beispieldatei ist im Repository vorhanden", false, V1_PATHS[0] + " fehlt");
}

// v1 ohne Daten
{
  const { project, report } = migrateV1({
    _meta: { app: "STOCHASI", version: "1.0", includes_data: false },
    parameters: { start_year: 100, end_year: 200, replacement_rate: 0.2, n_runs: 50 },
    categories: { list: ["A", "B"], names: { A: "Alpha" } },
  });
  c("v1 ohne Daten: Parameter übernommen", project.params.startYear === 100 && project.params.runs === 50);
  c("v1 ohne Daten: Hinweis auf fehlende Marktkurve", !report.hadMarketData && report.notes.some((n) => n.includes("Marktdaten")));
  c("v1 ohne Daten: gleichverteilter Start", near(project.params.initial.A, 50));
  c("v1 ohne Daten: Name aus der Liste, Rest als Kennung", project.categories[0].name === "Alpha" && project.categories[1].name === "B");
}
// v1 nur mit Prozenten
{
  const { project, report } = migrateV1({
    _meta: { app: "STOCHASI", version: "1.0" },
    parameters: { start_year: 100, end_year: 200 },
    categories: { list: ["A", "B"] },
    excavation_data: { percent: { A: 40, B: 60 } },
  });
  c("nur Prozente: Warnung zur Stichprobengröße", report.notes.some((n) => n.includes("Stückzahlen")));
  c("nur Prozente: Komplex trotzdem angelegt", project.assemblages.length === 1);
}
// unnormierte Marktdaten
{
  const { report } = migrateV1({
    _meta: { app: "STOCHASI", version: "1.0" },
    parameters: {}, categories: { list: ["A", "B"] },
    market_data: { years: [100, 200], A: [1, 2], B: [1, 2] },
  });
  c("unnormierte Marktdaten werden umgerechnet und gemeldet", report.notes.some((n) => n.includes("Prozent")));
}
// kaputtes Endjahr
{
  const { project, report } = migrateV1({
    _meta: { app: "STOCHASI", version: "1.0" },
    parameters: { start_year: 200, end_year: 100 }, categories: { list: ["A"] },
  });
  c("Endjahr vor Startjahr wird korrigiert und gemeldet", project.params.endYear === 300 && report.notes.some((n) => n.includes("Endjahr")));
}

/* ── v2-Projektdatei ── */
{
  let threw = false;
  try { readProject("kein json"); } catch (e) { threw = e instanceof ProjectError; }
  c("ungültiges JSON wirft ProjectError", threw);
  threw = false;
  try { readProject(JSON.stringify({ _meta: { app: "CombiTab" } })); } catch (e) { threw = e instanceof ProjectError; }
  c("fremde App wirft ProjectError", threw);
  threw = false;
  try { sanitize({ categories: [] }, "x"); } catch (e) { threw = e instanceof ProjectError; }
  c("Projekt ohne Kategorien wirft ProjectError", threw);
}
{
  const notes: string[] = [];
  const p = sanitize({
    name: "Test",
    categories: [{ id: "A", name: "Alpha", color: "nichtHex" }, { id: "A", name: "Doppelt" }, { id: "", name: "Leer" }],
    market: { years: [100, 200], shares: { A: [50, -3] } },
    params: { startYear: 200, endYear: 100, runs: 9999, noiseSd: 99, residual: 0.9, replacementDefault: 5, seed: -4, replacement: { A: 9 } },
    assemblages: [{ name: "leer", counts: { A: 0 } }, { name: "gut", counts: { A: 5, Z: 3 } }],
    comparisonYear: 9999,
  }, "x", notes);
  c("doppelte und leere Kategorien werden verworfen", p.categories.length === 1);
  c("ungültige Farbe wird durch Grau ersetzt", p.categories[0].color === "#808080");
  c("negativer Marktanteil wird auf 0 gesetzt", p.market.shares.A[1] === 0);
  c("Endjahr wird korrigiert", p.params.endYear === 300);
  c("Läufe werden begrenzt", p.params.runs === 500);
  c("Rauschen wird begrenzt", p.params.noiseSd === 10);
  c("Residualanteil wird begrenzt", near(p.params.residual, 0.5));
  c("Ersatzraten werden begrenzt", near(p.params.replacementDefault, 0.5) && near(p.params.replacement.A, 0.5));
  c("negativer Seed wird auf 0 gesetzt", p.params.seed === 0);
  c("leerer Fundkomplex wird verworfen", p.assemblages.length === 1 && p.assemblages[0].name === "gut");
  c("unbekannte Kategorie im Komplex wird verworfen", p.assemblages[0].counts.Z === undefined);
  c("Vergleichsjahr wird in den Zeitraum gezogen", p.comparisonYear >= p.params.startYear && p.comparisonYear <= p.params.endYear);
  c("Korrekturen werden gemeldet", notes.length >= 4);
}
{
  const p = emptyProject(PRESET_V1, "Neu");
  c("emptyProject ist gültig", readProject(writeProject(p), "x").project.categories.length === 5);
  c("emptyProject verteilt den Start gleich", near(p.params.initial.IT, 20));
}

console.log(`\n\x1b[1mErgebnis:\x1b[0m ${pass} bestanden, ${fail} fehlgeschlagen`);
if (fail) { console.log("Fehlgeschlagen: " + F.join(", ")); process.exit(1); }
