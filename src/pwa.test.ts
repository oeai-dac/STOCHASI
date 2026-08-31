/**
 * PWA-Validierung (§13) — statisch, ohne Browser.
 * Prüft Manifest-Pflichtfelder, Icon-Dateien (PNG-Signatur + Maße), die
 * Service-Worker-Logik und die Verweise im <head>.
 */
import { readFileSync } from "node:fs";

let pass = 0, fail = 0; const F: string[] = [];
function c(n: string, ok: boolean, d = "") { ok ? pass++ : (fail++, F.push(n)); console.log((ok ? "  \x1b[32m✓\x1b[0m " : "  \x1b[31m✗\x1b[0m ") + n + (d ? " — " + d : "")); }

console.log("\n\x1b[1mPWA-Validierung (§13)\x1b[0m\n");

// ── Manifest ──
const mani = JSON.parse(readFileSync("public/manifest.webmanifest", "utf8"));
c("Manifest: Name + Kurzname", typeof mani.name === "string" && mani.name.length > 0 && typeof mani.short_name === "string");
// Relativ, damit die App auch unter einem Unterpfad (GitHub Pages: /STOCHASI/)
// korrekt installiert. Absolute Pfade zeigten dort auf die Domainwurzel.
c("Manifest: start_url + scope relativ", mani.start_url === "./" && mani.scope === "./");
c("Manifest: kein absoluter id-Eintrag", mani.id === undefined || !String(mani.id).startsWith("/"));
c("Manifest: display standalone", mani.display === "standalone");
c("Manifest: theme_color/background_color als Hex", /^#[0-9a-f]{6}$/i.test(mani.theme_color) && /^#[0-9a-f]{6}$/i.test(mani.background_color));
c("Manifest: lang gesetzt", typeof mani.lang === "string" && mani.lang.length >= 2);
{
  const icons: Array<{ src: string; sizes: string; type: string; purpose?: string }> = mani.icons || [];
  const has = (sz: string, purpose?: string) => icons.some((i) => i.sizes === sz && i.type === "image/png" && (!purpose || (i.purpose || "").includes(purpose)));
  c("Manifest: 192er-PNG-Icon", has("192x192"));
  c("Manifest: 512er-PNG-Icon", has("512x512"));
  c("Manifest: maskable-Icon vorhanden", has("512x512", "maskable"));
  c("Manifest: Icon-Pfade relativ", icons.every((i) => i.src.startsWith("./")));
  // referenzierte Dateien existieren + sind gültige PNGs
  for (const ic of icons) {
    const p = "public/" + ic.src.replace(/^\.\//, "");
    const d = readFileSync(p);
    const sigOK = d[0] === 0x89 && d[1] === 0x50 && d[2] === 0x4e && d[3] === 0x47;
    const w = d.readUInt32BE(16), h = d.readUInt32BE(20);
    c(`Icon ${ic.src}: PNG ${ic.sizes}`, sigOK && `${w}x${h}` === ic.sizes, `${w}x${h}`);
  }
}

// ── Service Worker ──
{
  const sw = readFileSync("public/sw.js", "utf8");
  c("SW: benannter Cache", /const CACHE\s*=/.test(sw));
  c("SW: install cached App-Shell", sw.includes("install") && sw.includes("addAll"));
  c("SW: activate räumt alte Caches", sw.includes("activate") && sw.includes("caches.delete"));
  c("SW: fetch-Handler mit Navigationsfallback", sw.includes('addEventListener("fetch"') && sw.includes("navigate"));
  c("SW: nur GET wird behandelt", sw.includes('req.method !== "GET"'));
  // Der Worker darf keine absoluten Pfade mehr verdrahten, sonst scheitert der
  // Install unter einem Unterpfad und die App ist dort nicht offline-fähig.
  c("SW: Shell-Pfade relativ aufgelöst", sw.includes('new URL("./", self.location)') && !/["']\/(index\.html|manifest\.webmanifest|icon-)/.test(sw));
}

// ── index.html head ──
{
  const html = readFileSync("index.html", "utf8");
  c("HTML: Manifest verlinkt", /<link[^>]+rel="manifest"[^>]+manifest\.webmanifest/.test(html));
  c("HTML: theme-color", /<meta[^>]+name="theme-color"/.test(html));
  c("HTML: apple-touch-icon", /apple-touch-icon/.test(html));
}

console.log(`\n\x1b[1mErgebnis:\x1b[0m ${pass} bestanden, ${fail} fehlgeschlagen`);
if (fail) { console.log("\x1b[31mFehlgeschlagen:\x1b[0m " + F.join(", ")); process.exit(1); }
