/**
 * Textbreiten-Metrik für den Bild-Export.
 *
 * Der PDF-Export setzt die Base-14-Schrift Helvetica; SVG und PNG verwenden
 * dieselbe metrische Familie (Outfit/Helvetica/Arial). Statt einer pauschalen
 * Mittelbreite („jedes Zeichen ist 0,55 em breit") werden hier die echten
 * Vorschubbreiten aus der Helvetica-AFM verwendet. Das ist die Voraussetzung
 * dafür, dass die reservierten Ränder tatsächlich zur Beschriftung passen —
 * eine Pauschale unterschätzt breite Namen („WWM…") und überschätzt schmale
 * („Illi…"), was zu Überlappungen bzw. zu viel Weißraum führt.
 *
 * Einheit der Tabelle: 1/1000 em (AFM-Konvention).
 */

/* eslint-disable */
const W: Record<string, number> = {
  " ": 278, "!": 278, '"': 355, "#": 556, $: 556, "%": 889, "&": 667, "'": 191,
  "(": 333, ")": 333, "*": 389, "+": 584, ",": 278, "-": 333, ".": 278, "/": 278,
  "0": 556, "1": 556, "2": 556, "3": 556, "4": 556, "5": 556, "6": 556, "7": 556,
  "8": 556, "9": 556, ":": 278, ";": 278, "<": 584, "=": 584, ">": 584, "?": 556,
  "@": 1015,
  A: 667, B: 667, C: 722, D: 722, E: 667, F: 611, G: 778, H: 722, I: 278, J: 500,
  K: 667, L: 556, M: 833, N: 722, O: 778, P: 667, Q: 778, R: 722, S: 667, T: 611,
  U: 722, V: 667, W: 944, X: 667, Y: 667, Z: 611,
  "[": 278, "\\": 278, "]": 278, "^": 469, _: 556, "`": 333,
  a: 556, b: 556, c: 500, d: 556, e: 556, f: 278, g: 556, h: 556, i: 222, j: 222,
  k: 500, l: 222, m: 833, n: 556, o: 556, p: 556, q: 556, r: 333, s: 500, t: 278,
  u: 556, v: 500, w: 722, x: 500, y: 500, z: 500,
  "{": 334, "|": 260, "}": 334, "~": 584,
  // häufige nicht-ASCII-Zeichen in Fundtypen/Kontextnamen
  "\u00e4": 556, "\u00f6": 556, "\u00fc": 556, "\u00df": 556,
  "\u00c4": 667, "\u00d6": 778, "\u00dc": 722,
  "\u2013": 556, "\u2014": 1000, "\u2026": 1000, "\u00b7": 278,
};
/* eslint-enable */

/** Fallbackbreite für unbekannte Zeichen (mittlere Kleinbuchstabenbreite). */
const FALLBACK = 556;

/** Vorschubbreite eines einzelnen Zeichens bei gegebener Schriftgröße. */
export function charWidth(ch: string, size: number): number {
  return ((W[ch] ?? FALLBACK) / 1000) * size;
}

/** Gesamtbreite einer Zeichenkette bei gegebener Schriftgröße (in Punkt/px). */
export function textWidth(s: string, size: number): number {
  let w = 0;
  for (const ch of s) w += (W[ch] ?? FALLBACK) / 1000;
  return w * size;
}

/**
 * Kürzt einen Text auf die verfügbare Breite und hängt ein Auslassungszeichen an.
 * Wird verwendet, damit sehr lange Bezeichnungen den reservierten Rand nicht
 * sprengen und in die Matrix hineinlaufen — lieber sichtbar gekürzt als
 * überlappend.
 */
export function truncateToWidth(s: string, size: number, maxWidth: number): string {
  if (textWidth(s, size) <= maxWidth) return s;
  const ell = "\u2026";
  const ellW = textWidth(ell, size);
  if (ellW > maxWidth) return "";
  let out = "";
  let w = 0;
  for (const ch of s) {
    const cw = charWidth(ch, size);
    if (w + cw + ellW > maxWidth) break;
    out += ch;
    w += cw;
  }
  return out + ell;
}

/** Größte Textbreite einer Liste (0 bei leerer Liste). */
export function maxTextWidth(list: readonly string[], size: number): number {
  let m = 0;
  for (const s of list) { const w = textWidth(s, size); if (w > m) m = w; }
  return m;
}
