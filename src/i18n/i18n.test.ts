import { dictionaries, translate, LANGS, type Lang } from "./i18n.js";

let pass = 0, fail = 0; const F: string[] = [];
function c(n: string, ok: boolean, d = "") { ok ? pass++ : (fail++, F.push(n)); console.log((ok ? "  \x1b[32m✓\x1b[0m " : "  \x1b[31m✗\x1b[0m ") + n + (d ? " — " + d : "")); }

console.log("\n\x1b[1mZweisprachigkeit\x1b[0m\n");

const de = dictionaries.de, en = dictionaries.en;
const kde = Object.keys(de), ken = Object.keys(en);

c(`Deutsch hat ${kde.length} Einträge`, kde.length > 150);
c("beide Sprachen haben gleich viele Einträge", kde.length === ken.length);
{
  const missingEn = kde.filter((k) => !(k in en));
  const missingDe = ken.filter((k) => !(k in de));
  c("kein Schlüssel fehlt im Englischen", missingEn.length === 0, missingEn.join(", "));
  c("kein Schlüssel fehlt im Deutschen", missingDe.length === 0, missingDe.join(", "));
}
c("kein Eintrag ist leer", [...Object.values(de), ...Object.values(en)].every((v) => v.trim().length > 0));

// Platzhalter müssen in beiden Sprachen dieselben sein, sonst bleibt eine Variable stehen
{
  const vars = (s: string) => (s.match(/\{(\w+)\}/g) ?? []).sort().join(",");
  const bad = kde.filter((k) => vars(de[k]) !== vars(en[k]));
  c("Platzhalter stimmen in beiden Sprachen überein", bad.length === 0, bad.join(", "));
}
// Die deutschen Texte dürfen nicht bloß aus dem Englischen kopiert sein
{
  const identical = kde.filter((k) => de[k] === en[k] && de[k].length > 12);
  c(`höchstens wenige längere Einträge sind in beiden Sprachen gleich (${identical.length})`, identical.length <= 4, identical.join(", "));
}
// Schlüssel sollen ein Präfix haben, damit sie auffindbar bleiben
c("alle Schlüssel sind bereichsweise benannt", kde.every((k) => /^[a-z0-9]+\.[A-Za-z0-9.]+$/.test(k)));
c("die erwarteten Bereiche kommen vor",
  ["app", "tab", "side", "cat", "market", "sim", "year", "compare", "dating", "data", "import", "export", "share", "common"]
    .every((pfx) => kde.some((k) => k.startsWith(pfx + "."))));

/* ── translate ── */
c("übersetzt ins Deutsche", translate("de", "tab.dating") === "Datierung");
c("übersetzt ins Englische", translate("en", "tab.dating") === "Dating");
c("interpoliert Variablen", translate("de", "cat.count", { n: 19 }) === "19 Kategorien");
c("interpoliert mehrfach vorkommende Variablen",
  translate("de", "dating.subtitle", { name: "Insula XLI", n: 77 }) === "Insula XLI, n = 77");
c("unbekannter Schlüssel fällt auf sich selbst zurück", translate("de", "gibt.es.nicht") === "gibt.es.nicht");
c("fehlende englische Fassung fiele auf Deutsch zurück", translate("en", "gibt.es.nicht") === "gibt.es.nicht");
c("fehlende Variable bleibt sichtbar statt undefined",
  translate("de", "cat.count").includes("{n}"));
c("LANGS nennt beide Sprachen", (LANGS as Lang[]).join(",") === "de,en");

/* ── Inhaltliche Stichproben ── */
c("Fachbegriff: assemblage statt find complex", en["data.assemblage"] === "Assemblage");
c("Fachbegriff: replacement rate", en["side.replacement"].includes("Replacement rate"));
c("Fachbegriff: stock in circulation", en["sim.yAxis"].includes("stock in circulation"));
c("Vorbehalt zur Datierung ist in beiden Sprachen da",
  de["dating.caveat"].includes("keine absolute") && en["dating.caveat"].includes("not an absolute"));
c("Hinweis, dass die Referenzliste keine Marktanteile enthält",
  de["cat.referenceNote"].includes("Marktanteile") && en["cat.referenceNote"].includes("market shares"));

console.log(`\n\x1b[1mErgebnis:\x1b[0m ${pass} bestanden, ${fail} fehlgeschlagen`);
if (fail) { console.log("Fehlgeschlagen: " + F.join(", ")); process.exit(1); }
