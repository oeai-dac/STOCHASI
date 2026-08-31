import { encodeShare, decodeShare, buildShareUrl, readShareFromHash, LINK_MAX } from "./shareLink.js";
import { emptyProject } from "./io/project.js";
import { PRESET_V1 } from "../data/centres.js";
import type { ProjectV2 } from "./model.js";

let pass = 0, fail = 0; const F: string[] = [];
function c(n: string, ok: boolean, d = "") { ok ? pass++ : (fail++, F.push(n)); console.log((ok ? "  \x1b[32m✓\x1b[0m " : "  \x1b[31m✗\x1b[0m ") + n + (d ? " — " + d : "")); }

console.log("\n\x1b[1mTeilbarer Link\x1b[0m\n");

const p: ProjectV2 = emptyProject(PRESET_V1, "Testprojekt");
p.market = { years: [100, 200], shares: { IT: [80, 0], LG: [20, 0], BA: [0, 30], MG: [0, 40], RZ: [0, 30] } };
p.assemblages = [{ id: "A1", name: "Insula XLI", counts: { IT: 10, LG: 0, BA: 1, MG: 56, RZ: 10 } }];

const run = async () => {
  const frag = await encodeShare({ project: p, ui: { tab: "dating" } });
  c("Fragment ist URL-sicher", /^[gr][A-Za-z0-9_-]+$/.test(frag));
  c("gzip wird genutzt, wenn verfügbar", typeof CompressionStream === "undefined" ? frag[0] === "r" : frag[0] === "g");

  const back = await decodeShare(frag);
  c("Rundlauf erhält den Projektnamen", back?.project.name === "Testprojekt");
  c("Rundlauf erhält die Fundzahlen", back?.project.assemblages[0].counts.MG === 56);
  c("Rundlauf erhält die Marktkurve", back?.project.market.years.join(",") === "100,200");
  c("Rundlauf erhält den Oberflächenzustand", back?.ui.tab === "dating");

  c("kaputtes Fragment ergibt null", (await decodeShare("gNICHTBASE64!!")) === null);
  c("unbekanntes Kennzeichen ergibt null", (await decodeShare("xAAAA")) === null);
  c("leeres Fragment ergibt null", (await decodeShare("")) === null);
  c("gültiges JSON ohne Kategorien ergibt null",
    (await decodeShare("r" + btoa(JSON.stringify({ project: { categories: [] }, ui: {} })).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, ""))) === null);

  const { url, tooLong } = await buildShareUrl({ project: p, ui: {} }, "https://oeai-dac.github.io", "/STOCHASI/");
  c("URL enthält das Fragment", url.includes("#s="));
  c("kleines Projekt passt in einen Link", !tooLong && url.length < LINK_MAX);
  c("aus dem Hash wieder lesbar", (await readShareFromHash(new URL(url).hash))?.project.name === "Testprojekt");
  c("Hash ohne s-Parameter ergibt null", (await readShareFromHash("#tab=sim")) === null);

  // Großes Projekt: die Grenze muss greifen, statt einen unbrauchbaren Link zu liefern.
  const big: ProjectV2 = { ...p, market: { years: [], shares: {} }, assemblages: [] };
  big.market.years = Array.from({ length: 400 }, (_, i) => i);
  for (const cat of PRESET_V1) big.market.shares[cat.id] = big.market.years.map(() => Math.random() * 20);
  big.assemblages = Array.from({ length: 300 }, (_, i) => ({
    id: `A${i}`, name: `Fundkomplex mit einem recht ausführlichen Namen Nr. ${i}`,
    counts: Object.fromEntries(PRESET_V1.map((cat) => [cat.id, Math.floor(Math.random() * 500)])),
  }));
  const r2 = await buildShareUrl({ project: big, ui: {} }, "https://example.org", "/");
  c(`großes Projekt wird als zu lang erkannt (${r2.url.length} Zeichen)`, r2.tooLong);

  console.log(`\n\x1b[1mErgebnis:\x1b[0m ${pass} bestanden, ${fail} fehlgeschlagen`);
  if (fail) { console.log("Fehlgeschlagen: " + F.join(", ")); process.exit(1); }
};
void run();
