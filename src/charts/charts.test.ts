import { simulationScene, marketScene, spectrumScene, deviationScene, datingScene, rankScene, residualSeries, paramNote, type RankRow } from "./charts.js";
import { sceneToSVG, sceneToPDF, emptyScene, hexToRgb, blend } from "./scene.js";
import { niceTicks, linear, yearLabel, LIGHT, DARK } from "./plot.js";
import { simulate } from "../sim/simulate.js";
import { dateAssemblage, dateAcrossResidual } from "../sim/inverse.js";
import { interpolateMarket, yearRange, type Category, type MarketTable, type SimParams } from "../core/model.js";

let pass = 0, fail = 0; const F: string[] = [];
function c(n: string, ok: boolean, d = "") { ok ? pass++ : (fail++, F.push(n)); console.log((ok ? "  \x1b[32m✓\x1b[0m " : "  \x1b[31m✗\x1b[0m ") + n + (d ? " — " + d : "")); }

console.log("\n\x1b[1mDiagramme\x1b[0m\n");

const cats: Category[] = [
  { id: "IT", name: "Italian", color: "#8B0000" },
  { id: "MG", name: "Central Gaulish", color: "#228B22" },
  { id: "RZ", name: "Rheinzabern", color: "#FF8C00" },
];
const ids = cats.map((x) => x.id);
const market: MarketTable = { years: [100, 170, 240], shares: { IT: [100, 0, 0], MG: [0, 100, 0], RZ: [0, 0, 100] } };
const years = yearRange(100, 240);
const params: SimParams = {
  startYear: 100, endYear: 240, replacement: {}, replacementDefault: 0.12,
  noiseSd: 2, runs: 60, seed: 5, settlementMode: false, residual: 0,
  initial: { IT: 100, MG: 0, RZ: 0 },
};
const E = simulate(interpolateMarket(market, ids, years), years, ids, params);
const observed = { id: "A1", name: "Insula XLI", counts: { IT: 10, MG: 56, RZ: 10 } };

/* ── Hilfsfunktionen ── */
c("niceTicks liefert runde Schritte", niceTicks(0, 100, 5).join(",") === "0,20,40,60,80,100");
c("niceTicks bei kleinem Bereich", niceTicks(0, 1, 4).every((v) => Number.isFinite(v)));
c("niceTicks bei entartetem Bereich", niceTicks(5, 5).length === 1);
c("niceTicks trifft die Null exakt", niceTicks(-10, 10, 4).includes(0));
c("linear bildet die Ränder ab", linear([0, 10], [0, 100])(5) === 50);
c("linear bei entartetem Bereich stürzt nicht ab", Number.isFinite(linear([3, 3], [0, 10])(3)));
c("yearLabel kennzeichnet Jahre vor Christus", yearLabel(-40) === "40 v." && yearLabel(120) === "120");
c("hexToRgb", hexToRgb("#ff8c00").join(",") === "255,140,0");
c("hexToRgb Kurzform", hexToRgb("#fff").join(",") === "255,255,255");
c("blend mischt zum Hintergrund", blend([0, 0, 0], [255, 255, 255], 0)[0] === 255 && blend([0, 0, 0], [255, 255, 255], 1)[0] === 0);

/* ── Szenen ── */
const scenes: Array<[string, ReturnType<typeof simulationScene>]> = [
  ["Simulation", simulationScene(E, cats, { title: "Simulation", footnote: paramNote(params) })],
  ["Simulation ohne Band", simulationScene(E, cats, { showBand: false })],
  ["Markt", marketScene(market, cats, { title: "Marktangebot" })],
  ["Jahresspektrum", spectrumScene(E, cats, 170, { title: "Jahr 170", observed })],
  ["Jahresspektrum ohne Fundkomplex", spectrumScene(E, cats, 170)],
  ["Abweichung", deviationScene(E, cats, 170, observed, { title: "Abweichung" })],
  ["Datierung", datingScene(residualSeries(dateAcrossResidual(E, observed.counts, [0, 0.1, 0.2]), [168, 29, 38]), { title: "Datierung" })],
  ["Datierung, eine Reihe", datingScene([{ label: "x", result: dateAssemblage(E, observed.counts), color: [168, 29, 38] }])],
  ["dunkles Schema", simulationScene(E, cats, { theme: DARK })],
];

for (const [name, sc] of scenes) {
  c(`${name}: Maße positiv`, sc.w > 0 && sc.h > 0);
  c(`${name}: alle Koordinaten endlich`,
    sc.paths.every((p) => p.pts.every(Number.isFinite)) &&
    sc.rects.every((r) => [r.x, r.y, r.w, r.h].every(Number.isFinite)) &&
    sc.texts.every((t) => Number.isFinite(t.x) && Number.isFinite(t.y)));
  c(`${name}: keine negativen Rechteckmaße`, sc.rects.every((r) => r.w >= 0 && r.h >= 0));
  c(`${name}: nichts ragt aus der Bildfläche`,
    sc.paths.every((p) => { for (let i = 0; i + 1 < p.pts.length; i += 2) if (p.pts[i] < -2 || p.pts[i] > sc.w + 2 || p.pts[i + 1] < -2 || p.pts[i + 1] > sc.h + 2) return false; return true; }));
  const svg = sceneToSVG(sc);
  c(`${name}: SVG wohlgeformt`, svg.startsWith("<?xml") && svg.trimEnd().endsWith("</svg>") && !svg.includes("NaN") && !svg.includes("undefined"));
  const pdf = sceneToPDF(sc);
  const txt = new TextDecoder("latin1").decode(pdf);
  c(`${name}: PDF hat Kopf, xref und EOF`, txt.startsWith("%PDF-1.4") && txt.includes("xref") && txt.trimEnd().endsWith("%%EOF") && !txt.includes("NaN"));
}

/* ── Inhaltliche Eigenschaften ── */
{
  const sc = simulationScene(E, cats, {});
  c("Simulation: je Kategorie ein Band und eine Linie", sc.paths.filter((p) => p.closed && p.fill).length === 3 && sc.paths.filter((p) => p.stroke && !p.closed && p.pts.length > 100).length === 3);
  const noBand = simulationScene(E, cats, { showBand: false });
  c("ohne Band entfallen die Flächen", noBand.paths.filter((p) => p.closed && p.fill).length === 0);
  c("Legende nennt alle Kategorien", cats.every((x) => sc.texts.some((t) => t.s === x.name)));
}
{
  const sc = simulationScene(E, [], {});
  c("Simulation ohne Kategorien liefert eine gültige Szene", sc.w > 0 && sceneToSVG(sc).includes("</svg>"));
}
{
  const sc = spectrumScene(E, cats, 170, { observed });
  c("Jahresspektrum zeigt die Stückzahlen", sc.texts.some((t) => t.s === "56"));
  c("Jahresspektrum nennt den Stichprobenumfang", sc.texts.some((t) => t.s.includes("n = 76")));
  const nums = sc.texts.filter((t) => /^\d+$/.test(t.s));
  c("alle Stückzahlen stehen im Bild", nums.length === 3 && nums.every((t) => t.x >= 0 && t.x <= sc.w));
}
{
  const sc = deviationScene(E, cats, 170, observed, {});
  // drei Balken plus die beiden Legenden-Farbfelder
  c("Abweichung: ein Balken je Kategorie", sc.rects.length === 3 + 2);
  c("Abweichung: Kategorien beschriftet", cats.every((x) => sc.texts.some((t) => t.s === x.name)));
  c("Abweichung: Legende erklärt die Markierung", sc.texts.some((t) => t.s.includes("Perzentil")));
}
{
  // Ein Fundkomplex, der genau dem Simulationsmittel entspricht, darf keine
  // Kategorie als auffällig markieren.
  const nc = E.ids.length, yi = E.years.indexOf(170);
  const exact = { id: "X", name: "exakt", counts: Object.fromEntries(E.ids.map((id, k) => [id, Math.round(E.mean[yi * nc + k] * 10)])) };
  const sc = deviationScene(E, cats, 170, exact, {});
  const marks = sc.paths.filter((p) => p.width === 2 && p.pts.length === 4);
  c("passender Fundkomplex wird nicht als auffällig markiert", marks.length === 0);
}
{
  const sc = spectrumScene(E, cats, 170, { observed });
  const minTextX = Math.min(...sc.texts.filter((t) => cats.some((x) => x.name === t.s)).map((t) => t.x));
  c("Zeilenbeschriftung bleibt im Bild", minTextX > 0);
}
{
  const r = dateAssemblage(E, observed.counts);
  const sc = datingScene([{ label: "x", result: r, color: [168, 29, 38] }]);
  c("Datierung beschriftet Modus und Intervall", sc.texts.some((t) => t.s.includes(String(r.mode)) && t.s.includes("95 %")));
  const empty = datingScene([{ label: "leer", result: dateAssemblage(E, {}), color: [0, 0, 0] }]);
  c("leerer Komplex ergibt trotzdem eine gültige Szene", sceneToSVG(empty).includes("</svg>"));
  c("Datierung ohne Reihen stürzt nicht ab", sceneToSVG(datingScene([])).includes("</svg>"));
}
{
  const s = residualSeries(dateAcrossResidual(E, observed.counts, [0, 0.2]), [168, 29, 38]);
  c("residualSeries beschriftet die Anteile", s[0].label.includes("0 %") && s[1].label.includes("20 %"));
  c("residualSeries unterscheidet die Linien", s[0].dash === undefined && Array.isArray(s[1].dash));
}
c("paramNote nennt Rate, Streuung und Läufe",
  paramNote(params).includes("12 %") && paramNote(params).includes("60 Läufe") && paramNote({ ...params, residual: 0.2 }).includes("Residualität"));
c("paramNote kennzeichnet den zufälligen Seed", paramNote({ ...params, seed: 0 }).includes("zufällig"));

/* ── Ausgabeformate ── */
{
  const sc = emptyScene(100, 50, [255, 255, 255]);
  sc.texts.push({ x: 10, y: 20, s: 'Größe „test" (a–b) 50 %', size: 10, c: [0, 0, 0], anchor: "start", rot: 0 });
  const txt = new TextDecoder("latin1").decode(sceneToPDF(sc));
  c("PDF: Umlaut wird oktal kodiert", txt.includes("\\366"));
  c("PDF: Klammern werden geschützt", txt.includes("\\(a-b\\)"));
  c("PDF: typografische Anführungszeichen werden ersetzt", !txt.includes("„"));
  const svg = sceneToSVG(sc);
  c("SVG: Sonderzeichen bleiben erhalten", svg.includes("Größe"));
}
{
  const sc = emptyScene(80, 40);
  sc.texts.push({ x: 40, y: 20, s: "Mitte", size: 10, c: [0, 0, 0], anchor: "middle", rot: 0 });
  c("SVG kennt die Mittelausrichtung", sceneToSVG(sc).includes('text-anchor="middle"'));
  c("PDF verschiebt mittig ausgerichteten Text", new TextDecoder("latin1").decode(sceneToPDF(sc)).includes("Tm"));
}
{
  const sc = emptyScene(60, 30);
  sc.paths.push({ pts: [0, 0, 10, 10], stroke: [0, 0, 0], width: 1, dash: [4, 2] });
  c("SVG setzt das Strichmuster", sceneToSVG(sc).includes('stroke-dasharray="4 2"'));
  c("PDF setzt das Strichmuster", new TextDecoder("latin1").decode(sceneToPDF(sc)).includes("[4 2] 0 d"));
}
c("helles und dunkles Schema unterscheiden sich", LIGHT.bg[0] !== DARK.bg[0]);


/* ── Achsenbereich: die Daten müssen im Rahmen bleiben, nicht nur im Bild ── */
{
  // 125–290 ist der Fall aus der v1-Konfiguration: die runden Teilstriche enden
  // bei 275, die Daten laufen bis 290. Ohne getrennten Wertebereich ragten die
  // Kurven über den gezeichneten Rahmen hinaus.
  const yrs = yearRange(125, 290);
  const mk: MarketTable = { years: [125, 200, 290], shares: { IT: [100, 0, 0], MG: [0, 100, 0], RZ: [0, 0, 100] } };
  const e2 = simulate(interpolateMarket(mk, ids, yrs), yrs, ids, { ...params, startYear: 125, endYear: 290, initial: { IT: 100, MG: 0, RZ: 0 } });
  const sc = simulationScene(e2, cats, { width: 900, height: 500 });
  // rechter Rahmenrand = größte x-Koordinate der Achsenlinie
  const axis = sc.paths.find((p) => p.pts.length === 6 && !p.fill)!;
  const right = Math.max(axis.pts[0], axis.pts[2], axis.pts[4]);
  const maxX = Math.max(...sc.paths.flatMap((p) => p.pts.filter((_, i) => i % 2 === 0)));
  c(`Kurven enden am Rahmen, nicht dahinter (Rahmen ${right.toFixed(0)}, Daten ${maxX.toFixed(0)})`, maxX <= right + 0.5);
  const maxY = Math.max(...sc.paths.flatMap((p) => p.pts.filter((_, i) => i % 2 === 1)));
  const minY = Math.min(...sc.paths.flatMap((p) => p.pts.filter((_, i) => i % 2 === 1)));
  c("Kurven bleiben senkrecht im Rahmen", minY >= 0 && maxY <= sc.h);
  c("Teilstriche außerhalb des Bereichs werden weggelassen", !sc.texts.some((t) => t.s === "300"));
}


/* ── Rangfolge ── */
{
  /** Baut Fundkomplexe, deren Spektren aus verschiedenen Jahren stammen. */
  const at = (year: number, N: number, name: string) => {
    const yi = E.years.indexOf(year), nc = E.ids.length;
    const raw = E.ids.map((_, k) => (E.mean[yi * nc + k] / 100) * N);
    const fl = raw.map(Math.floor);
    let rest = N - fl.reduce((a, b) => a + b, 0);
    raw.map((v, i) => [v - fl[i], i] as const).sort((a, b) => b[0] - a[0]).forEach(([, i]) => { if (rest > 0) { fl[i]++; rest--; } });
    return { label: name, result: dateAssemblage(E, Object.fromEntries(E.ids.map((id, i) => [id, fl[i]]))) };
  };
  const rows: RankRow[] = [at(210, 600, "spät"), at(140, 600, "früh"), at(175, 600, "mitte")];
  const sc = rankScene(rows, { title: "Rangfolge", width: 900 });
  c("Rangfolge: Szene gültig", sc.w > 0 && sceneToSVG(sc).includes("</svg>") && !sceneToSVG(sc).includes("NaN"));
  c("Rangfolge: ein Balken je Komplex plus zwei Legendenfelder", sc.rects.length === 3 + 2);
  c("Rangfolge: alle Komplexe beschriftet", rows.every((r) => sc.texts.some((t) => t.s.startsWith(r.label))));
  c("Rangfolge: Stückzahl steht in der Beschriftung", sc.texts.some((t) => t.s.includes("n = 600")));
  // Sortierung: der früheste Komplex steht oben
  const ys = rows.map((r) => sc.texts.find((t) => t.s.startsWith(r.label))!.y);
  c(`Rangfolge: nach Datum sortiert (früh oben)`, ys[1] < ys[2] && ys[2] < ys[0]);
  // Balken liegen an der richtigen Stelle
  const bars = sc.rects.slice(0, 3).map((r) => r.x);
  c("Rangfolge: früher Komplex liegt links", Math.min(...bars) === sc.rects[0].x);
  c("Rangfolge: PDF gültig", new TextDecoder("latin1").decode(sceneToPDF(sc)).trimEnd().endsWith("%%EOF"));
}
{
  // Gut getrennte Komplexe: kein Jahr passt zu allen, also kein Schnittband.
  const at = (year: number, N: number, name: string) => {
    const yi = E.years.indexOf(year), nc = E.ids.length;
    const cnt = Object.fromEntries(E.ids.map((id, k) => [id, Math.round((E.mean[yi * nc + k] / 100) * N)]));
    return { label: name, result: dateAssemblage(E, cnt) };
  };
  const wide = rankScene([at(130, 2000, "a"), at(215, 2000, "b")], { overlapLabel: "ÜBERALL" });
  c("kein gemeinsames Datum → kein Schnittband und kein Hinweis", !wide.texts.some((t) => t.s === "ÜBERALL"));

  // Winzige Stichproben: die Intervalle überlappen, das Band muss erscheinen.
  const tiny = rankScene([at(150, 6, "a"), at(190, 6, "b")], { overlapLabel: "ÜBERALL" });
  c("gemeinsames Datum → Schnittband mit Hinweis", tiny.texts.some((t) => t.s === "ÜBERALL"));

  // Gezählt statt indiziert: die Legende belegt beide Farben ohnehin je einmal,
  // die Balken kommen obendrauf. Bei großen Stichproben also dreimal die kräftige
  // Farbe, bei winzigen dreimal die blasse.
  const count = (sc: ReturnType<typeof rankScene>, col: string) => sc.rects.filter((r) => r.c.join(",") === col).length;
  const strong = LIGHT.accent.join(",");
  const pale = blend(LIGHT.accent, LIGHT.bg, 0.45).join(",");
  c(`große Stichproben werden kräftig gezeichnet (${count(wide, strong)}× kräftig, ${count(wide, pale)}× blass)`,
    count(wide, strong) === 3 && count(wide, pale) === 1);
  c(`kleine Stichproben werden blass gezeichnet (${count(tiny, strong)}× kräftig, ${count(tiny, pale)}× blass)`,
    count(tiny, strong) === 1 && count(tiny, pale) === 3);
}
{
  c("Rangfolge ohne Komplexe stürzt nicht ab", sceneToSVG(rankScene([])).includes("</svg>"));
  c("Rangfolge mit nur leeren Komplexen stürzt nicht ab",
    sceneToSVG(rankScene([{ label: "leer", result: dateAssemblage(E, {}) }])).includes("</svg>"));
  const one = rankScene([{ label: "x", result: dateAssemblage(E, observed.counts) }], { overlapLabel: "ÜBERALL" });
  c("bei einem einzigen Komplex gibt es keinen Schnitt zu zeigen", !one.texts.some((t) => t.s === "ÜBERALL"));
}
{
  // Ein weiteres Intervall (Residualschar) muss den Balken verbreitern.
  const r = dateAssemblage(E, observed.counts);
  const narrow = rankScene([{ label: "x", result: r }]);
  const broad = rankScene([{ label: "x", result: r, span: [r.hdi[0] - 20, r.hdi[1] + 20] }]);
  const widest = (sc: ReturnType<typeof rankScene>) => Math.max(...sc.rects.map((x) => x.w));
  c("span verbreitert den Balken", widest(broad) > widest(narrow));
}

console.log(`\n\x1b[1mErgebnis:\x1b[0m ${pass} bestanden, ${fail} fehlgeschlagen`);
if (fail) { console.log("Fehlgeschlagen: " + F.join(", ")); process.exit(1); }
