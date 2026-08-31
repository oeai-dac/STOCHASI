import { CENTRES, GROUP_ORDER, PRESET_V1, PRESET_FLAVIA_SOLVA_IDS, centreById, centresByGroup, centresToCategories, suggestedPeriod } from "./centres.js";

let pass = 0, fail = 0; const F: string[] = [];
function c(n: string, ok: boolean, d = "") { ok ? pass++ : (fail++, F.push(n)); console.log((ok ? "  \x1b[32m✓\x1b[0m " : "  \x1b[31m✗\x1b[0m ") + n + (d ? " — " + d : "")); }

console.log("\n\x1b[1mReferenzliste der Produktionszentren\x1b[0m\n");

c(`${CENTRES.length} Zentren erfasst`, CENTRES.length >= 30);
{
  const ids = CENTRES.map((x) => x.id);
  c("Kennungen sind eindeutig", new Set(ids).size === ids.length);
  c("Kennungen sind zwei Großbuchstaben", ids.every((i) => /^[A-Z]{2}$/.test(i)));
}
{
  const names = CENTRES.map((x) => x.name);
  c("Bezeichnungen sind eindeutig", new Set(names).size === names.length);
}
c("jedes Zentrum hat eine Gruppe aus GROUP_ORDER", CENTRES.every((x) => (GROUP_ORDER as readonly string[]).includes(x.group)));
c("jede Gruppe ist belegt", GROUP_ORDER.every((g) => CENTRES.some((x) => x.group === g)));
c("Farben sind gültige Hex-Werte", CENTRES.every((x) => /^#[0-9a-f]{6}$/i.test(x.color)));
c("Farben sind eindeutig", new Set(CENTRES.map((x) => x.color.toLowerCase())).size === CENTRES.length);
c("Produktionsbeginn liegt vor dem Ende", CENTRES.every((x) => x.from < x.to));
c("Zeiträume liegen im plausiblen Rahmen (−200 bis 700)", CENTRES.every((x) => x.from >= -200 && x.to <= 700));
c("certainty ist gesetzt", CENTRES.every((x) => x.certainty === "established" || x.certainty === "approximate"));
c("es gibt sowohl gesicherte als auch grobe Angaben", CENTRES.some((x) => x.certainty === "established") && CENTRES.some((x) => x.certainty === "approximate"));

// Die im Flavia-Solva-Skript fehlenden Zentren müssen enthalten sein
for (const n of ["Montans", "Trier", "Sinzig", "La Madeleine", "Chémery-Faulquemont"]) {
  c(`neu gegenüber der Flavia-Solva-Liste: ${n}`, CENTRES.some((x) => x.name === n));
}
// und die 19 der Flavia-Solva-Liste müssen auflösbar sein
c("alle 19 Provenienzen von Flavia Solva sind auflösbar",
  PRESET_FLAVIA_SOLVA_IDS.every((id) => centreById(id) !== undefined));
c("Flavia-Solva-Auswahl ergibt 19 Kategorien", centresToCategories(PRESET_FLAVIA_SOLVA_IDS).length === 19);

c("PRESET_V1 hat die fünf Kategorien von Version 1",
  PRESET_V1.length === 5 && PRESET_V1.map((x) => x.id).join(",") === "IT,LG,BA,MG,RZ");
c("PRESET_V1 behält die Farben von Version 1", PRESET_V1[0].color === "#8B0000" && PRESET_V1[4].color === "#FF8C00");

c("centreById findet Rheinzabern", centreById("RZ")?.name === "Rheinzabern");
c("centreById liefert undefined für Unbekanntes", centreById("ZZ") === undefined);
c("centresByGroup gruppiert vollständig", centresByGroup().reduce((a, g) => a + g.centres.length, 0) === CENTRES.length);
c("centresByGroup hält die Gruppenreihenfolge", centresByGroup()[0].group === "Italisch");
c("centresToCategories hält die Listenreihenfolge", centresToCategories(["RZ", "LG"]).map((x) => x.id).join(",") === "LG,RZ");
c("centresToCategories ignoriert Unbekanntes", centresToCategories(["RZ", "ZZ"]).length === 1);

{
  const p = suggestedPeriod(["LG", "RZ"]);
  c(`Zeitvorschlag für La Graufesenque + Rheinzabern: ${p?.start}–${p?.end}`, p !== null && p.start === 20 && p.end === 260);
  c("Zeitvorschlag ohne Auswahl ist null", suggestedPeriod([]) === null);
}

console.log(`\n\x1b[1mErgebnis:\x1b[0m ${pass} bestanden, ${fail} fehlgeschlagen`);
if (fail) { console.log("Fehlgeschlagen: " + F.join(", ")); process.exit(1); }
