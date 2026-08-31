import { detectKind, importMarketGrid, importAssemblageGrid, num, ImportError } from "./importTable.js";
import { parseDelimited } from "./parseDelimited.js";

let pass = 0, fail = 0; const F: string[] = [];
function c(n: string, ok: boolean, d = "") { ok ? pass++ : (fail++, F.push(n)); console.log((ok ? "  \x1b[32m✓\x1b[0m " : "  \x1b[31m✗\x1b[0m ") + n + (d ? " — " + d : "")); }
const near = (a: number, b: number, e = 1e-6) => Math.abs(a - b) < e;
function threw(f: () => unknown): boolean { try { f(); return false; } catch (e) { return e instanceof ImportError; } }

console.log("\n\x1b[1mTabellen-Import\x1b[0m\n");

// Zahlenparser
c("num: einfache Zahl", num("42") === 42);
c("num: Dezimalkomma", near(num("12,5"), 12.5));
c("num: Tausenderpunkt und Dezimalkomma", near(num("1.234,5"), 1234.5));
c("num: englisches Format", near(num("1,234.5"), 1234.5));
c('num: formatierte Pivot-Zelle „56 (72,7 %)"', num("56 (72,7 %)") === 56);
c("num: leer → NaN", Number.isNaN(num("")));
c("num: Text → NaN", Number.isNaN(num("keine Zahl")));
c("num: echte Zahl bleibt", num(7) === 7);

// Art erkennen
const marketCSV = "Year;IT;LG;RZ\n120;30;20;0\n160;10;0;40\n200;0;0;100";
c("Jahresspalte erkannt", detectKind(parseDelimited(marketCSV)) === "market");
c("erste Spalte aus Jahren erkannt", detectKind(parseDelimited("x;A;B\n100;1;2\n150;3;4\n200;5;6")) === "market");
c("Fundkomplex-Tabelle erkannt", detectKind(parseDelimited("Insula;IT;RZ\n405;10;56\n403;3;20")) === "assemblages");
c("Zählwerte in Spalte 1 gelten nicht als Jahre", detectKind(parseDelimited("Name;A\nx;1\ny;2\nz;3")) === "assemblages");

// Markt
{
  const r = importMarketGrid(parseDelimited(marketCSV));
  c("Marktjahre gelesen", r.market.years.join(",") === "120,160,200");
  c("Kategorien gelesen", r.ids.join(",") === "IT,LG,RZ");
  c("Zeile auf 100 normiert", near(r.market.shares.IT[0] + r.market.shares.LG[0] + r.market.shares.RZ[0], 100));
  c("Verhältnis erhalten (30:20 → 60:40)", near(r.market.shares.IT[0], 60) && near(r.market.shares.LG[0], 40));
  c("Hinweis auf Umrechnung", r.warnings.some((w) => w.includes("Prozent")));
}
{
  const r = importMarketGrid(parseDelimited("Jahr;A;B\n200;50;50\n100;100;0"));
  c("unsortierte Jahre werden sortiert", r.market.years.join(",") === "100,200");
  c("Werte folgen der Sortierung", near(r.market.shares.A[0], 100));
}
{
  const r = importMarketGrid(parseDelimited("Year;A;B;Summe\n100;50;50;100"));
  c("Randsummenspalte wird ignoriert", r.ids.join(",") === "A,B");
}
{
  const r = importMarketGrid(parseDelimited("Year;A;B\n100;50;50\nkaputt;1;2\n200;20;80"));
  c("ungültige Jahreszeile wird übersprungen und gemeldet", r.market.years.length === 2 && r.warnings.some((w) => w.includes("kaputt")));
}
c("Markttabelle ohne Kategoriespalten wirft ImportError", threw(() => importMarketGrid(parseDelimited("Year\n100\n200"))));
c("Markttabelle ohne Datenzeilen wirft ImportError", threw(() => importMarketGrid([["Year", "A"]])));

// Fundkomplexe, Kategorien in Spalten
{
  const r = importAssemblageGrid(parseDelimited("Insula;IT;LG;BA;MG;RZ\n405;10;0;1;56;10\n403;3;0;0;20;7"), { known: ["IT", "LG", "BA", "MG", "RZ"] });
  c("zwei Komplexe gelesen", r.assemblages.length === 2 && !r.transposed);
  c("Namen übernommen", r.assemblages.map((a) => a.name).join(",") === "405,403");
  c("Zählwerte stimmen", r.assemblages[0].counts.MG === 56 && r.assemblages[0].counts.LG === 0);
  c("Kategorien gemeldet", r.ids.join(",") === "IT,LG,BA,MG,RZ");
}
// Kategorien in Zeilen (Pivot Provenienz × Insula, wie in Flavia Solva)
{
  const r = importAssemblageGrid(parseDelimited("Provenienz;405;403;802\nIT;10;3;1\nMG;56;20;5\nRZ;10;7;2"), { known: ["IT", "MG", "RZ"] });
  c("Ausrichtung erkannt und transponiert", r.transposed && r.assemblages.length === 3);
  c("Komplexnamen aus der Kopfzeile", r.assemblages.map((a) => a.name).join(",") === "405,403,802");
  c("Werte richtig zugeordnet", r.assemblages[0].counts.MG === 56 && r.assemblages[2].counts.RZ === 2);
}
{
  const r = importAssemblageGrid(parseDelimited("Provenienz;405\nIT;10\nMG;56"));
  c("ohne Vorwissen wird die Ausrichtung geraten", r.assemblages.length >= 1);
}
{
  const r = importAssemblageGrid(parseDelimited("Insula;IT;RZ;Summe\n405;10;5;15\nSumme;10;5;15"), { known: ["IT", "RZ"] });
  c("Randsummen-Zeile und -Spalte werden ignoriert", r.assemblages.length === 1 && r.ids.join(",") === "IT,RZ");
}
{
  const r = importAssemblageGrid(parseDelimited("Insula;IT;RZ\n405;10;5\n999;0;0"), { known: ["IT", "RZ"] });
  c("leerer Komplex wird übersprungen und gemeldet", r.assemblages.length === 1 && r.warnings.some((w) => w.includes("999")));
}
{
  const r = importAssemblageGrid(parseDelimited("Insula;IT;RZ\n405;12,5;87,5"), { known: ["IT", "RZ"] });
  c("Prozentwerte werden gelesen, aber angemahnt", r.warnings.some((w) => w.includes("Stückzahlen")));
}
// Langformat (das Format von STOCHASI 1)
{
  const r = importAssemblageGrid(parseDelimited("Typ;Anzahl\nIT;10\nMG;56\nRZ;10"));
  c("Langformat ohne Komplexspalte → ein Komplex", r.assemblages.length === 1 && r.assemblages[0].counts.MG === 56);
}
{
  const r = importAssemblageGrid(parseDelimited("Insula;Typ;Anzahl\n405;IT;10\n405;MG;56\n403;IT;3"), { defaultName: "x" });
  c("Langformat mit Komplexspalte → mehrere Komplexe", r.assemblages.length === 2);
  c("gleiche Kategorie wird aufsummiert", importAssemblageGrid(parseDelimited("Typ;Anzahl\nIT;3\nIT;4")).assemblages[0].counts.IT === 7);
}
c("leere Tabelle wirft ImportError", threw(() => importAssemblageGrid([])));
c("Tabelle ohne Zählwerte wirft ImportError", threw(() => importAssemblageGrid(parseDelimited("Insula;IT\n405;0"), { known: ["IT"] })));

console.log(`\n\x1b[1mErgebnis:\x1b[0m ${pass} bestanden, ${fail} fehlgeschlagen`);
if (fail) { console.log("Fehlgeschlagen: " + F.join(", ")); process.exit(1); }
