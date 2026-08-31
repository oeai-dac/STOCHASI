import { OKABE_ITO, contrastRatio, relativeLuminance, simulateCVD, deltaE, assignCvdSafePalette, indistinctPairs } from "./palette.js";

let pass = 0, fail = 0; const F: string[] = [];
function c(n: string, ok: boolean, d = "") { ok ? pass++ : (fail++, F.push(n)); console.log((ok ? "  \x1b[32m✓\x1b[0m " : "  \x1b[31m✗\x1b[0m ") + n + (d ? " — " + d : "")); }
const approx = (a: number, b: number, e = 0.01) => Math.abs(a - b) < e;

console.log("\n\x1b[1mFarbenblind-sichere Paletten & Kontrast (§9.5)\x1b[0m\n");

// ── WCAG-Kontrast ──
c("Kontrast Schwarz/Weiß = 21", approx(contrastRatio("#000000", "#ffffff"), 21, 0.1));
c("Kontrast gleiche Farbe = 1", approx(contrastRatio("#5d584f", "#5d584f"), 1));
c("Kontrast ist symmetrisch", approx(contrastRatio("#0072B2", "#f6f4f2"), contrastRatio("#f6f4f2", "#0072B2")));
c("Weiß heller als Schwarz (Luminanz)", relativeLuminance("#ffffff") > relativeLuminance("#000000"));

// ── CVD-Simulation: Richtung/Plausibilität ──
{
  // Grau bleibt (nahezu) grau — achromatisch ist von CVD unbetroffen.
  const g = simulateCVD("#808080", "deuteranopia");
  c("Grau bleibt unter Deuteranopie nahezu grau", deltaE("#808080", g) < 8, `ΔE=${deltaE("#808080", g).toFixed(1)}`);
}
{
  // Rot und Grün sind normal gut unterscheidbar, unter Deuteranopie deutlich weniger.
  const normal = deltaE("#D55E00", "#009E73");
  const deut = deltaE(simulateCVD("#D55E00", "deuteranopia"), simulateCVD("#009E73", "deuteranopia"));
  c("Rot/Grün rücken unter Deuteranopie zusammen", deut < normal, `normal=${normal.toFixed(0)} deut=${deut.toFixed(0)}`);
}

// ── Okabe-Ito: unter Normalsicht paarweise unterscheidbar ──
{
  const pal: Record<string, string> = {}; OKABE_ITO.forEach((h, i) => (pal["c" + i] = h));
  const bad = indistinctPairs(pal, null, 12);
  c("Okabe-Ito: alle 9 unter Normalsicht unterscheidbar", bad.length === 0, bad.map((p) => `${p.a}/${p.b}`).join(", "));
}
// ── Okabe-Ito robuster als eine naive Rot/Grün/Braun-Palette unter Deuteranopie ──
{
  const oi: Record<string, string> = {}; OKABE_ITO.slice(0, 5).forEach((h, i) => (oi["c" + i] = h));
  const naive = { keramik: "#CD853F", metall: "#B22222", glas: "#8B4513", bein: "#A0522D", stein: "#556B2F" };
  const oiBad = indistinctPairs(oi, "deuteranopia", 12).length;
  const naiveBad = indistinctPairs(naive, "deuteranopia", 12).length;
  c("Okabe-Ito unter Deuteranopie robuster als Braun/Rot-Palette", oiBad < naiveBad, `okabe=${oiBad} naiv=${naiveBad}`);
}

// ── Zuweisung ──
{
  const groups = ["Keramik", "Metall", "Glas", "Bein", "Stein"];
  const m = assignCvdSafePalette(groups);
  c("Zuweisung deckt alle Gruppen ab", groups.every((g) => /^#[0-9A-Fa-f]{6}$/.test(m[g])));
  c("Zuweisung vergibt distinkte Farben", new Set(Object.values(m)).size === groups.length);
  c("erste Gruppe erhält erste Palettenfarbe", m["Keramik"] === OKABE_ITO[0]);
}
{
  // Über 9 Gruppen hinaus: weiter distinkt (Helligkeitsvariante)
  const many = Array.from({ length: 12 }, (_, i) => "G" + i);
  const m = assignCvdSafePalette(many);
  c("mehr als 9 Gruppen bekommen weiterhin distinkte Farben", new Set(Object.values(m)).size >= 11, `${new Set(Object.values(m)).size}`);
}

// ── indistinctPairs erkennt ein bewusst schlechtes Paar ──
{
  const pair = { a: "#D55E00", b: "#E06010" };
  c("nahe Farben werden als ununterscheidbar gemeldet", indistinctPairs(pair, null, 12).length === 1);
}

console.log(`\n\x1b[1mErgebnis:\x1b[0m ${pass} bestanden, ${fail} fehlgeschlagen`);
if (fail) { console.log("\x1b[31mFehlgeschlagen:\x1b[0m " + F.join(", ")); process.exit(1); }
