/**
 * Farbenblind-sichere Paletten & Kontrast (§9.5) — framework-frei, ohne Fremdbibliothek.
 *
 * Enthält:
 *  - die Okabe-Ito-Qualitativpalette (für alle gängigen Farbsehschwächen entworfen),
 *  - WCAG-Kontrast (relative Leuchtdichte, Kontrastverhältnis),
 *  - Dichromasie-Simulation (Protanopie/Deuteranopie/Tritanopie, Viénot 1999,
 *    korrekt im linearen Licht gerechnet),
 *  - CIELAB-ΔE76 und eine Prüfung, welche Materialfarben unter einer Farbsehschwäche
 *    kaum noch unterscheidbar sind.
 */

export type CVD = "protanopia" | "deuteranopia" | "tritanopia";

/**
 * Okabe-Ito, umsortiert so, dass die ersten Einträge maximal unterscheidbar sind.
 * Pures Schwarz/Grau ans Ende, damit kleine Gruppenzahlen kräftige Farben bekommen.
 */
export const OKABE_ITO: string[] = [
  "#0072B2", // Blau
  "#E69F00", // Orange
  "#009E73", // Blaugrün
  "#CC79A7", // Rotviolett
  "#56B4E9", // Himmelblau
  "#D55E00", // Zinnoberrot
  "#F0E442", // Gelb
  "#000000", // Schwarz
  "#999999", // Grau
];

/* ── Konvertierung ── */
export function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  const s = h.length === 3 ? h.split("").map((c) => c + c).join("") : h;
  return [parseInt(s.slice(0, 2), 16), parseInt(s.slice(2, 4), 16), parseInt(s.slice(4, 6), 16)];
}
export function rgbToHex(r: number, g: number, b: number): string {
  const c = (v: number) => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, "0");
  return "#" + c(r) + c(g) + c(b);
}
const srgbToLinear = (c: number) => { const x = c / 255; return x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4; };
const linearToSrgb = (x: number) => { const c = x <= 0.0031308 ? 12.92 * x : 1.055 * x ** (1 / 2.4) - 0.055; return c * 255; };

/* ── WCAG-Kontrast ── */
export function relativeLuminance(hex: string): number {
  const [r, g, b] = hexToRgb(hex);
  return 0.2126 * srgbToLinear(r) + 0.7152 * srgbToLinear(g) + 0.0722 * srgbToLinear(b);
}
/** WCAG-Kontrastverhältnis (1..21). */
export function contrastRatio(a: string, b: string): number {
  const la = relativeLuminance(a), lb = relativeLuminance(b);
  const hi = Math.max(la, lb), lo = Math.min(la, lb);
  return (hi + 0.05) / (lo + 0.05);
}

/* ── Dichromasie-Simulation (Viénot 1999, linearer RGB-Raum) ── */
const CVD_MAT: Record<CVD, number[]> = {
  protanopia: [0.152286, 1.052583, -0.204868, 0.114503, 0.786281, 0.099216, -0.003882, -0.048116, 1.051998],
  deuteranopia: [0.367322, 0.860646, -0.227968, 0.280085, 0.672501, 0.047413, -0.011820, 0.042940, 0.968910],
  tritanopia: [1.255528, -0.076749, -0.178779, -0.078411, 0.930809, 0.147602, 0.004733, 0.691367, 0.303900],
};
export function simulateCVD(hex: string, type: CVD): string {
  const [r, g, b] = hexToRgb(hex).map(srgbToLinear);
  const m = CVD_MAT[type];
  const clamp = (x: number) => Math.max(0, Math.min(1, x));
  const R = clamp(m[0] * r + m[1] * g + m[2] * b);
  const G = clamp(m[3] * r + m[4] * g + m[5] * b);
  const B = clamp(m[6] * r + m[7] * g + m[8] * b);
  return rgbToHex(linearToSrgb(R), linearToSrgb(G), linearToSrgb(B));
}

/* ── CIELAB & ΔE76 ── */
function toLab(hex: string): [number, number, number] {
  const [r, g, b] = hexToRgb(hex).map(srgbToLinear);
  // linear sRGB → XYZ (D65)
  let X = r * 0.4124 + g * 0.3576 + b * 0.1805;
  let Y = r * 0.2126 + g * 0.7152 + b * 0.0722;
  let Z = r * 0.0193 + g * 0.1192 + b * 0.9505;
  X /= 0.95047; Y /= 1.0; Z /= 1.08883;
  const f = (t: number) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
  const fx = f(X), fy = f(Y), fz = f(Z);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}
/** Perzeptueller Farbabstand (CIE76). Grob: <~12 = schwer unterscheidbar. */
export function deltaE(a: string, b: string): number {
  const [l1, a1, b1] = toLab(a), [l2, a2, b2] = toLab(b);
  return Math.hypot(l1 - l2, a1 - a2, b1 - b2);
}

/* ── Palette zuweisen & Ununterscheidbarkeit prüfen ── */
/**
 * Die i-te Farbe der Palette, zyklisch.
 *
 * Über die neun Grundfarben hinaus wird die Basisfarbe je Umlauf abgedunkelt.
 * Das ist keine gute Lösung für zwanzig Reihen in einem Diagramm — die gibt es
 * nicht —, aber es liefert verschiedene Farben statt derselben neun noch einmal.
 */
export function paletteColor(i: number): string {
  const k = ((i % OKABE_ITO.length) + OKABE_ITO.length) % OKABE_ITO.length;
  const round = Math.max(0, Math.floor(i / OKABE_ITO.length));
  if (round === 0) return OKABE_ITO[k];
  const f = 1 - 0.22 * round;
  const [r, g, b] = hexToRgb(OKABE_ITO[k]);
  return rgbToHex(r * f, g * f, b * f);
}

/** Weist jeder Gruppe eine CVD-sichere Farbe zu (zyklisch, mit Helligkeits-Variation ab 9). */
export function assignCvdSafePalette(groups: string[]): Record<string, string> {
  const out: Record<string, string> = {};
  groups.forEach((g, i) => { out[g] = paletteColor(i); });
  return out;
}

export interface IndistinctPair { a: string; b: string; deltaE: number; }
/**
 * Findet Farbpaare, die unter der gegebenen Farbsehschwäche kaum unterscheidbar sind
 * (simulierter ΔE unter der Schwelle). `null` als Typ = Normalsicht.
 */
export function indistinctPairs(colors: Record<string, string>, type: CVD | null, threshold = 12): IndistinctPair[] {
  const keys = Object.keys(colors);
  const sim = (h: string) => (type ? simulateCVD(h, type) : h);
  const out: IndistinctPair[] = [];
  for (let i = 0; i < keys.length; i++) for (let j = i + 1; j < keys.length; j++) {
    const d = deltaE(sim(colors[keys[i]]), sim(colors[keys[j]]));
    if (d < threshold) out.push({ a: keys[i], b: keys[j], deltaE: d });
  }
  return out.sort((x, y) => x.deltaE - y.deltaE);
}
