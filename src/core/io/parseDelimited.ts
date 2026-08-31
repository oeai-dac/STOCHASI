/**
 * Robuster Parser für getrennte Textformate (CSV/TSV/…), framework-frei.
 *
 * Beherrscht: Anführungszeichen mit eingebetteten Trennzeichen und Zeilenumbrüchen,
 * verdoppelte Anführungszeichen ("") als Escape, CRLF/CR/LF, sowie automatische
 * Erkennung des Trennzeichens.
 */

export type Delimiter = "," | ";" | "\t" | "|";
const CANDIDATES: Delimiter[] = [",", ";", "\t", "|"];

/** Errät das Trennzeichen aus der ersten nicht-leeren Zeile (außerhalb von Quotes). */
export function detectDelimiter(text: string): Delimiter {
  const line = firstDataLine(text);
  let best: Delimiter = ",", bestN = -1;
  for (const d of CANDIDATES) {
    const n = countOutsideQuotes(line, d);
    if (n > bestN) { bestN = n; best = d; }
  }
  return best;
}

/** Zerlegt getrennten Text in ein Zellraster `string[][]`. */
export function parseDelimited(text: string, delimiter?: Delimiter): string[][] {
  const delim = delimiter ?? detectDelimiter(text);
  const rows: string[][] = [];
  let field = "", row: string[] = [], inQuotes = false;
  const s = text.charCodeAt(0) === 0xfeff ? text.slice(1) : text; // BOM entfernen
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (inQuotes) {
      if (ch === '"') {
        if (s[i + 1] === '"') { field += '"'; i++; }   // "" → "
        else inQuotes = false;
      } else field += ch;
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === delim) {
      row.push(field); field = "";
    } else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && s[i + 1] === "\n") i++;
      row.push(field); field = ""; rows.push(row); row = [];
    } else field += ch;
  }
  if (field.length > 0 || row.length > 0) { row.push(field); rows.push(row); }
  // vollständig leere Zeilen (nur ein leeres Feld) am Rand entfernen
  return rows.filter((r) => !(r.length === 1 && r[0].trim() === ""));
}

/* ── intern ── */
function firstDataLine(text: string): string {
  const s = text.charCodeAt(0) === 0xfeff ? text.slice(1) : text;
  let line = "", inQ = false;
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (ch === '"') { inQ = !inQ; line += ch; }
    else if ((ch === "\n" || ch === "\r") && !inQ) { if (line.trim() !== "") break; line = ""; }
    else line += ch;
  }
  return line;
}
function countOutsideQuotes(line: string, d: Delimiter): number {
  let n = 0, inQ = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') inQ = !inQ;
    else if (ch === d && !inQ) n++;
  }
  return n;
}
