import * as XLSX from "xlsx";
import { toCSV, toCSVForDownload, UTF8_BOM, simulationAoa, marketAoa, assemblagesAoa, datingCurvesAoa, datingSummaryAoa, parametersAoa, toXLSX } from "./exportTable.js";
import { safeFilename, canDownload } from "./download.js";
import { textWidth, truncateToWidth, maxTextWidth } from "./textMetrics.js";
import { simulate } from "../sim/simulate.js";
import { dateAssemblage } from "../sim/inverse.js";
import { emptyProject } from "../core/io/project.js";
import { PRESET_V1 } from "../data/centres.js";
import { interpolateMarket, yearRange, type MarketTable } from "../core/model.js";

let pass = 0, fail = 0; const F: string[] = [];
function c(n: string, ok: boolean, d = "") { ok ? pass++ : (fail++, F.push(n)); console.log((ok ? "  \x1b[32m✓\x1b[0m " : "  \x1b[31m✗\x1b[0m ") + n + (d ? " — " + d : "")); }

console.log("\n\x1b[1mExport\x1b[0m\n");

const p = emptyProject(PRESET_V1, "Flavia Solva");
p.params.startYear = 120; p.params.endYear = 200; p.params.runs = 30; p.params.seed = 3;
p.params.replacement = { RZ: 0.2 };
p.market = { years: [120, 200], shares: { IT: [60, 0], LG: [40, 0], BA: [0, 10], MG: [0, 50], RZ: [0, 40] } };
p.assemblages = [{ id: "A1", name: 'Insula "XLI"', counts: { IT: 10, LG: 0, BA: 1, MG: 56, RZ: 10 } }];
const ids = p.categories.map((x) => x.id);
const years = yearRange(p.params.startYear, p.params.endYear);
const E = simulate(interpolateMarket(p.market, ids, years), years, ids, p.params);
const dr = [{ label: p.assemblages[0].name, result: dateAssemblage(E, p.assemblages[0].counts) }];

/* ── CSV ── */
c("CSV nutzt CRLF", toCSV([["a", "b"], [1, 2]]) === "a,b\r\n1,2");
c("CSV schützt Trennzeichen", toCSV([["x,y"]]) === '"x,y"');
c("CSV verdoppelt Anführungszeichen", toCSV([['sagt "ja"']]) === '"sagt ""ja"""');
c("CSV schützt Zeilenumbrüche", toCSV([["a\nb"]]).includes('"a\nb"'));
c("CSV mit Semikolon schützt nur das Semikolon", toCSV([["a,b"]], ";") === "a,b");
c("nicht endliche Zahlen werden leer", toCSV([[NaN, Infinity]]) === ",");
c("Download-Fassung trägt das BOM", toCSVForDownload([["a"]]).startsWith(UTF8_BOM));

/* ── Tabellen ── */
{
  const a = simulationAoa(E, p.categories);
  c("Simulation: Kopfzeile plus je Jahr eine Zeile", a.length === years.length + 1);
  c("Simulation: drei Spalten je Kategorie", a[0].length === 1 + 3 * 5);
  c("Simulation: Kopf nennt Mittel und Perzentile", String(a[0][1]).includes("Mittel") && String(a[0][3]).includes("P90"));
  c("Simulation: erstes Jahr stimmt", a[1][0] === 120);
  c("Simulation: Werte gerundet, nicht abgeschnitten", a.slice(1).every((r) => r.slice(1).every((v) => typeof v === "number" && Number.isFinite(v))));
  c("Simulation: Zeilen summieren auf 100",
    a.slice(1).every((r) => { let s = 0; for (let i = 1; i < r.length; i += 3) s += r[i] as number; return Math.abs(s - 100) < 0.2; }));
}
{
  const a = marketAoa(p.market, p.categories, 120, 200);
  c("Markt: je Jahr eine Zeile", a.length === 82);
  c("Markt: Stützjahre sind gekennzeichnet", a[1][6] === "ja" && a[2][6] === "");
  c("Markt: interpolierte Zeile summiert auf 100",
    Math.abs((a[41].slice(1, 6) as number[]).reduce((x, y) => x + y, 0) - 100) < 0.2);
}
{
  const a = assemblagesAoa(p.assemblages, p.categories);
  c("Fundkomplexe: Summe wird mitgeführt", a[1][6] === 77);
  c("Fundkomplexe: Name in der ersten Spalte", a[1][0] === 'Insula "XLI"');
}
{
  const a = datingCurvesAoa(dr);
  c("Datierungskurve: je Jahr eine Zeile", a.length === years.length + 1);
  c("Datierungskurve: summiert auf 1", Math.abs(a.slice(1).reduce((s, r) => s + (r[1] as number), 0) - 1) < 1e-3);
  c("Datierungskurve ohne Ergebnisse liefert nur den Kopf", datingCurvesAoa([]).length === 1);
}
{
  const a = datingSummaryAoa(dr);
  c("Übersicht: eine Zeile je Komplex", a.length === 2);
  c("Übersicht: n, Modus und Intervall gesetzt", a[1][1] === 77 && typeof a[1][2] === "number" && (a[1][6] as number) >= 0);
  c("Übersicht: Verfahren wird genannt", a[1][8] === "dirichlet");
  const leer = datingSummaryAoa([{ label: "leer", result: dateAssemblage(E, {}) }]);
  c("Übersicht: leerer Komplex lässt die Datierungsspalten frei", leer[1][2] === "" && leer[1][1] === 0);
}
{
  const a = parametersAoa(p);
  const find = (k: string) => a.find((r) => String(r[0]).startsWith(k))?.[1];
  c("Parameter: Zeitraum", find("Zeitraum von") === 120 && find("Zeitraum bis") === 200);
  c("Parameter: kategoriespezifische Rate erscheint", find("Ersatzrate RZ") === 0.2);
  c("Parameter: nicht gesetzte Raten erscheinen nicht", a.every((r) => !String(r[0]).startsWith("Ersatzrate IT")));
  c("Parameter: gesetzter Seed erscheint als Zahl", find("Seed") === 3);
  const a0 = parametersAoa({ ...p, params: { ...p.params, seed: 0 } });
  c("Parameter: zufälliger Seed wird als solcher benannt", a0.find((r) => String(r[0]).startsWith("Seed"))?.[1] === "zufällig");
  c("Parameter: Startanteile vollständig", p.categories.every((cat) => a.some((r) => String(r[0]).includes(`Startanteil ${cat.id}`))));
}

/* ── XLSX ── */
{
  const run = async () => {
    const bytes = await toXLSX([
      { name: "Simulation", aoa: simulationAoa(E, p.categories) },
      { name: "Datierung", aoa: datingSummaryAoa(dr) },
      { name: "Ein sehr langer Blattname mit [Klammern]", aoa: parametersAoa(p) },
    ]);
    c("XLSX ist nicht leer", bytes.length > 1000);
    const wb = XLSX.read(bytes, { type: "array" });
    c("XLSX hat drei Blätter", wb.SheetNames.length === 3);
    c("XLSX kürzt zu lange Blattnamen auf 31 Zeichen", wb.SheetNames[2].length <= 31);
    c("XLSX ersetzt unzulässige Zeichen im Blattnamen", !/[:\\/?*[\]]/.test(wb.SheetNames[2]));
    const sim = XLSX.utils.sheet_to_json<string[]>(wb.Sheets["Simulation"], { header: 1 });
    c("XLSX: Simulationsblatt vollständig", sim.length === years.length + 1);
    c("XLSX: Zahlen bleiben Zahlen", typeof (sim[1] as unknown as number[])[1] === "number");

    console.log(`\n\x1b[1mErgebnis:\x1b[0m ${pass} bestanden, ${fail} fehlgeschlagen`);
    if (fail) { console.log("Fehlgeschlagen: " + F.join(", ")); process.exit(1); }
  };

  /* ── Dateinamen und Textmaße ── */
  c("safeFilename transliteriert Umlaute und ß", safeFilename("Flavia Solva Größe") === "Flavia_Solva_Grosse");
  c("safeFilename verträgt Insula-Namen", safeFilename("405 [XLI]") === "405_XLI");
  c("safeFilename entfernt Pfadtrenner", !safeFilename("a/b\\c").includes("/"));
  c("safeFilename fällt auf den Vorgabenamen zurück", safeFilename("") === "stochasi");
  c("safeFilename begrenzt die Länge", safeFilename("x".repeat(300)).length <= 120);
  c("canDownload ohne DOM meldet false", canDownload() === false);
  c("textWidth wächst mit der Länge", textWidth("mm", 10) > textWidth("m", 10));
  c("truncateToWidth kürzt sichtbar", truncateToWidth("Chémery-Faulquemont", 9, 40).endsWith("…"));
  c("truncateToWidth lässt Kurzes unberührt", truncateToWidth("RZ", 9, 40) === "RZ");
  c("maxTextWidth nimmt das Längste", maxTextWidth(["a", "mmmm"], 10) === textWidth("mmmm", 10));

  void run();
}
