/**
 * Eine auflösungsunabhängige Szene, drei Ausgaben.
 *
 * Jedes Diagramm wird als Liste aus Rechtecken, Pfaden und Beschriftungen
 * gebaut. Daraus entstehen SVG (Vektor), PNG (rasterisiert) und PDF (Vektor,
 * ohne Fremdbibliothek) — und dieselbe Szene rendert die Oberfläche. Die
 * Bildschirmansicht und die exportierte Abbildung sind deshalb nicht nur
 * ähnlich, sondern identisch.
 *
 * Bewusst ohne Transparenz: Halbdurchsichtige Flächen bräuchten im PDF einen
 * ExtGState und sähen je nach Betrachter verschieden aus. Unsicherheitsbänder
 * werden stattdessen vorab gegen den Hintergrund verrechnet (`blend`). Das
 * Ergebnis ist in allen drei Formaten pixelgleich.
 */
import { textWidth } from "../export/textMetrics.js";

export type RGB = [number, number, number];

export interface Rect { x: number; y: number; w: number; h: number; c: RGB; }
export interface Txt {
  x: number; y: number; s: string; size: number; c: RGB;
  anchor: "start" | "middle" | "end"; rot: 0 | -90;
  /** Fett setzen (Helvetica-Bold im PDF). */
  bold?: boolean;
}
export interface Path {
  /** Flach: x0, y0, x1, y1, … */
  pts: number[];
  fill?: RGB;
  stroke?: RGB;
  width?: number;
  closed?: boolean;
  /** Strichmuster in Punkt, z. B. [4, 3]. */
  dash?: number[];
}
export interface Scene { w: number; h: number; bg: RGB; rects: Rect[]; paths: Path[]; texts: Txt[]; }

export function emptyScene(w: number, h: number, bg: RGB = [255, 255, 255]): Scene {
  return { w, h, bg, rects: [], paths: [], texts: [] };
}

export function hexToRgb(hex: string): RGB {
  const h = (hex || "").replace("#", "");
  const n = h.length === 3 ? h.split("").map((c) => c + c).join("") : h;
  return [parseInt(n.slice(0, 2), 16) || 0, parseInt(n.slice(2, 4), 16) || 0, parseInt(n.slice(4, 6), 16) || 0];
}

/** Farbe `a` mit Anteil `t` über `b` legen — der Ersatz für Transparenz. */
export function blend(a: RGB, b: RGB, t: number): RGB {
  const k = Math.min(1, Math.max(0, t));
  return [
    Math.round(b[0] + (a[0] - b[0]) * k),
    Math.round(b[1] + (a[1] - b[1]) * k),
    Math.round(b[2] + (a[2] - b[2]) * k),
  ];
}

/* ── SVG ── */
export function sceneToSVG(sc: Scene): string {
  const out: string[] = [
    `<?xml version="1.0" encoding="UTF-8" standalone="no"?>`,
    `<svg xmlns="http://www.w3.org/2000/svg" width="${sc.w}" height="${sc.h}" viewBox="0 0 ${sc.w} ${sc.h}" font-family="Outfit, Helvetica, Arial, sans-serif">`,
    `<rect width="${sc.w}" height="${sc.h}" fill="${rgb(sc.bg)}"/>`,
  ];
  for (const r of sc.rects) out.push(`<rect x="${f(r.x)}" y="${f(r.y)}" width="${f(r.w)}" height="${f(r.h)}" fill="${rgb(r.c)}"/>`);
  for (const p of sc.paths) {
    if (p.pts.length < 4) continue;
    const d = pathD(p);
    const attrs = [
      `fill="${p.fill ? rgb(p.fill) : "none"}"`,
      `stroke="${p.stroke ? rgb(p.stroke) : "none"}"`,
      p.stroke ? `stroke-width="${f(p.width ?? 1)}"` : "",
      p.stroke ? `stroke-linejoin="round" stroke-linecap="round"` : "",
      p.dash?.length ? `stroke-dasharray="${p.dash.map(f).join(" ")}"` : "",
    ].filter(Boolean).join(" ");
    out.push(`<path d="${d}" ${attrs}/>`);
  }
  for (const t of sc.texts) {
    const tr = t.rot === -90 ? ` transform="rotate(-90 ${f(t.x)} ${f(t.y)})"` : "";
    const w = t.bold ? ` font-weight="600"` : "";
    out.push(`<text x="${f(t.x)}" y="${f(t.y)}" font-size="${t.size}" text-anchor="${t.anchor}" fill="${rgb(t.c)}"${w}${tr}>${esc(t.s)}</text>`);
  }
  out.push("</svg>");
  return out.join("\n");
}

function pathD(p: Path): string {
  const parts: string[] = [];
  for (let i = 0; i + 1 < p.pts.length; i += 2) parts.push(`${i === 0 ? "M" : "L"}${f(p.pts[i])},${f(p.pts[i + 1])}`);
  if (p.closed) parts.push("Z");
  return parts.join(" ");
}

/* ── PNG ── */
export async function sceneToPNG(sc: Scene, scale = 2): Promise<Blob> {
  const cv = document.createElement("canvas");
  cv.width = Math.ceil(sc.w * scale); cv.height = Math.ceil(sc.h * scale);
  const g = cv.getContext("2d")!;
  g.scale(scale, scale);
  g.fillStyle = rgb(sc.bg); g.fillRect(0, 0, sc.w, sc.h);
  for (const r of sc.rects) { g.fillStyle = rgb(r.c); g.fillRect(r.x, r.y, r.w, r.h); }
  for (const p of sc.paths) {
    if (p.pts.length < 4) continue;
    g.beginPath();
    g.moveTo(p.pts[0], p.pts[1]);
    for (let i = 2; i + 1 < p.pts.length; i += 2) g.lineTo(p.pts[i], p.pts[i + 1]);
    if (p.closed) g.closePath();
    if (p.fill) { g.fillStyle = rgb(p.fill); g.fill(); }
    if (p.stroke) {
      g.strokeStyle = rgb(p.stroke); g.lineWidth = p.width ?? 1;
      g.lineJoin = "round"; g.lineCap = "round";
      g.setLineDash(p.dash ?? []);
      g.stroke();
      g.setLineDash([]);
    }
  }
  g.textBaseline = "alphabetic";
  for (const t of sc.texts) {
    g.save();
    g.font = `${t.bold ? "600 " : ""}${t.size}px Outfit, Helvetica, Arial, sans-serif`;
    g.fillStyle = rgb(t.c);
    g.textAlign = t.anchor === "end" ? "right" : t.anchor === "middle" ? "center" : "left";
    g.translate(t.x, t.y);
    if (t.rot === -90) g.rotate(-Math.PI / 2);
    g.fillText(t.s, 0, 0);
    g.restore();
  }
  return await new Promise<Blob>((res, rej) => cv.toBlob((b) => (b ? res(b) : rej(new Error("toBlob"))), "image/png"));
}

/* ── PDF (Vektor, Base-14-Helvetica) ── */
export function sceneToPDF(sc: Scene): Uint8Array {
  const H = sc.h;
  const body: string[] = [];
  let fillCol = "", strokeCol = "", lineW = -1, dashState = "";
  const setFill = (c: RGB) => { const s = `${dec(c[0])} ${dec(c[1])} ${dec(c[2])}`; if (s !== fillCol) { body.push(`${s} rg`); fillCol = s; } };
  const setStroke = (c: RGB) => { const s = `${dec(c[0])} ${dec(c[1])} ${dec(c[2])}`; if (s !== strokeCol) { body.push(`${s} RG`); strokeCol = s; } };
  const setWidth = (w: number) => { if (w !== lineW) { body.push(`${f(w)} w`); lineW = w; } };
  const setDash = (d?: number[]) => { const s = d?.length ? `[${d.map(f).join(" ")}] 0 d` : "[] 0 d"; if (s !== dashState) { body.push(s); dashState = s; } };

  setFill(sc.bg);
  body.push(`0 0 ${f(sc.w)} ${f(sc.h)} re f`);
  for (const r of sc.rects) { setFill(r.c); body.push(`${f(r.x)} ${f(H - r.y - r.h)} ${f(r.w)} ${f(r.h)} re f`); }
  for (const p of sc.paths) {
    if (p.pts.length < 4) continue;
    if (p.fill) setFill(p.fill);
    if (p.stroke) { setStroke(p.stroke); setWidth(p.width ?? 1); setDash(p.dash); }
    body.push(`${f(p.pts[0])} ${f(H - p.pts[1])} m`);
    for (let i = 2; i + 1 < p.pts.length; i += 2) body.push(`${f(p.pts[i])} ${f(H - p.pts[i + 1])} l`);
    if (p.closed) body.push("h");
    body.push(p.fill && p.stroke ? "B" : p.fill ? "f" : "S");
  }
  for (const t of sc.texts) {
    setFill(t.c);
    const w = textWidth(t.s, t.size);
    const x = t.x, y = H - t.y;
    body.push("BT", `/${t.bold ? "F2" : "F1"} ${t.size} Tf`);
    const shift = t.anchor === "end" ? w : t.anchor === "middle" ? w / 2 : 0;
    if (t.rot === -90) body.push(`0 1 -1 0 ${f(x)} ${f(y - shift)} Tm`);
    else body.push(`1 0 0 1 ${f(x - shift)} ${f(y)} Tm`);
    body.push(`(${pdfEsc(t.s)}) Tj`, "ET");
  }

  const content = body.join("\n");
  const objs: string[] = [];
  objs[1] = "<< /Type /Catalog /Pages 2 0 R >>";
  objs[2] = "<< /Type /Pages /Kids [3 0 R] /Count 1 >>";
  objs[3] = `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${f(sc.w)} ${f(sc.h)}] /Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> /Contents 4 0 R >>`;
  objs[4] = `<< /Length ${byteLen(content)} >>\nstream\n${content}\nendstream`;
  objs[5] = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>";
  objs[6] = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>";

  let pdf = "%PDF-1.4\n"; const offsets: number[] = [];
  const N = 6;
  for (let i = 1; i <= N; i++) { offsets[i] = byteLen(pdf); pdf += `${i} 0 obj\n${objs[i]}\nendobj\n`; }
  const xrefPos = byteLen(pdf);
  pdf += `xref\n0 ${N + 1}\n0000000000 65535 f \n`;
  for (let i = 1; i <= N; i++) pdf += `${String(offsets[i]).padStart(10, "0")} 00000 n \n`;
  pdf += `trailer\n<< /Size ${N + 1} /Root 1 0 R >>\nstartxref\n${xrefPos}\n%%EOF`;
  return new TextEncoder().encode(pdf);
}

/* ── Helfer ── */
export function f(n: number): string { return (Math.round(n * 100) / 100).toString(); }
function dec(v: number): string { return (v / 255).toFixed(3); }
export function rgb(c: RGB): string { return `rgb(${c[0]},${c[1]},${c[2]})`; }
function esc(s: string): string { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;"); }
/** Zeichenketten für den PDF-Inhaltsstrom (WinAnsi; Sonderzeichen auf ASCII). */
function pdfEsc(s: string): string {
  return s
    .replace(/\\/g, "\\\\").replace(/\(/g, "\\(").replace(/\)/g, "\\)")
    .replace(/\u2026/g, "...").replace(/[\u2013\u2014\u2212]/g, "-")
    .replace(/[\u2018\u2019]/g, "'").replace(/[\u201c\u201d\u201e]/g, '"')
    .replace(/\u00b1/g, "+/-")
    .replace(/[\u00a0-\u00ff]/g, (ch) => "\\" + ch.charCodeAt(0).toString(8).padStart(3, "0"))
    .replace(/[^\x20-\x7e]/g, "?");
}
function byteLen(s: string): number { return new TextEncoder().encode(s).length; }
