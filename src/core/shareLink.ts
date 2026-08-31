/**
 * Teilbarer Projekt-Link / URL-State (§9.8) — framework-frei, ohne Fremdbibliothek.
 *
 * Kodiert das vollständige Projekt (Kategorien, Marktkurve, Parameter,
 * Fundkomplexe) plus etwas Oberflächenzustand in das URL-Fragment. Komprimiert per nativem `CompressionStream`
 * (gzip) und base64url; fehlt die API, wird unkomprimiert kodiert (durch ein Flag am
 * Anfang gekennzeichnet, damit der Empfänger es unabhängig davon dekodieren kann).
 *
 * Große Projekte passen nicht in eine URL — der Aufrufer prüft die Länge und weicht
 * dann auf „Projektdatei teilen" aus. Das Fragment (`#`) wird nicht an Server gesendet.
 */
import type { ProjectV2 } from "./model.js";

export interface ShareState { project: ProjectV2; ui: { tab?: string }; }

/** Obergrenze für die Gesamt-URL-Länge, ab der ein Link nicht mehr zuverlässig teilbar ist. */
export const LINK_MAX = 16000;

const hasCS = typeof CompressionStream !== "undefined";
const hasDS = typeof DecompressionStream !== "undefined";

function b64urlEncode(bytes: Uint8Array): string {
  let bin = ""; const CH = 0x8000;
  for (let i = 0; i < bytes.length; i += CH) bin += String.fromCharCode(...bytes.subarray(i, i + CH));
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}
function b64urlDecode(s: string): Uint8Array {
  const b = atob(s.replace(/-/g, "+").replace(/_/g, "/"));
  const out = new Uint8Array(b.length);
  for (let i = 0; i < b.length; i++) out[i] = b.charCodeAt(i);
  return out;
}

async function gzip(data: Uint8Array): Promise<Uint8Array> {
  const cs = new CompressionStream("gzip");
  const w = cs.writable.getWriter(); void w.write(data as BufferSource); void w.close();
  return new Uint8Array(await new Response(cs.readable).arrayBuffer());
}
async function gunzip(data: Uint8Array): Promise<Uint8Array> {
  const ds = new DecompressionStream("gzip");
  const w = ds.writable.getWriter(); void w.write(data as BufferSource); void w.close();
  return new Uint8Array(await new Response(ds.readable).arrayBuffer());
}

/** Kodiert den Zustand zu einem Fragmentwert (`g…` = gzip, `r…` = roh). */
export async function encodeShare(state: ShareState): Promise<string> {
  const json = JSON.stringify(state);
  const raw = new TextEncoder().encode(json);
  if (hasCS) return "g" + b64urlEncode(await gzip(raw));
  return "r" + b64urlEncode(raw);
}

/** Dekodiert einen Fragmentwert; `null` bei Fehler/Unlesbarkeit. */
export async function decodeShare(fragment: string): Promise<ShareState | null> {
  try {
    const flag = fragment[0], body = fragment.slice(1);
    const bytes = b64urlDecode(body);
    let json: string;
    if (flag === "g") { if (!hasDS) return null; json = new TextDecoder().decode(await gunzip(bytes)); }
    else if (flag === "r") json = new TextDecoder().decode(bytes);
    else return null;
    const s = JSON.parse(json) as ShareState;
    return s && s.project && Array.isArray(s.project.categories) && s.project.categories.length > 0 ? s : null;
  } catch { return null; }
}

/** Baut die vollständige teilbare URL (aktueller Ursprung + Pfad + Fragment). */
export async function buildShareUrl(state: ShareState, origin: string, pathname: string): Promise<{ url: string; tooLong: boolean }> {
  const frag = await encodeShare(state);
  const url = `${origin}${pathname}#s=${frag}`;
  return { url, tooLong: url.length > LINK_MAX };
}

/** Liest einen Share-State aus einem `#s=…`-Hash, sonst `null`. */
export async function readShareFromHash(hash: string): Promise<ShareState | null> {
  const m = /[#&]s=([^&]+)/.exec(hash || "");
  if (!m) return null;
  return decodeShare(decodeURIComponent(m[1]));
}
