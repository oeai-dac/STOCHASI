// PWA-Icon-Generator — rastert das STOCHASI-Gitterlogo als PNG.
// Dependency-frei: nur Node-Builtins (zlib). Aufruf: node scripts/gen-icons.mjs
import { deflateSync } from "node:zlib";
import { writeFileSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const OUT = join(ROOT, "public");
const BUILD_OUT = join(ROOT, "build");
mkdirSync(OUT, { recursive: true });

// ── CRC32 ──
const CRC = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
function crc32(buf) { let c = 0xffffffff; for (let i = 0; i < buf.length; i++) c = CRC[(c ^ buf[i]) & 0xff] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; }

function chunk(type, data) {
  const len = Buffer.alloc(4); len.writeUInt32BE(data.length, 0);
  const td = Buffer.concat([Buffer.from(type, "latin1"), data]);
  const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(td), 0);
  return Buffer.concat([len, td, crc]);
}

function encodePNG(w, h, rgba) {
  const sig = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4); ihdr[8] = 8; ihdr[9] = 6; // 8-bit RGBA
  const raw = Buffer.alloc(h * (1 + w * 4));
  for (let y = 0; y < h; y++) { raw[y * (1 + w * 4)] = 0; rgba.copy(raw, y * (1 + w * 4) + 1, y * w * 4, (y + 1) * w * 4); }
  const idat = deflateSync(raw, { level: 9 });
  return Buffer.concat([sig, chunk("IHDR", ihdr), chunk("IDAT", idat), chunk("IEND", Buffer.alloc(0))]);
}

// ── Palette (aus theme.css / Shell-Logo) ──
const ACCENT = [210, 38, 48], BGL = [246, 244, 242], LIGHT = [226, 223, 218];
const mix = (a, b, t) => a.map((v, i) => Math.round(v * (1 - t) + b[i] * t));
const BAND = mix(ACCENT, BGL, 0.62);   // Unsicherheitsband
const BAND2 = mix(ACCENT, BGL, 0.80);  // äußeres, schwächeres Band
const AXIS = mix(LIGHT, BGL, 0.25);

// Das Zeichen ist der Signaturplot von STOCHASI: eine Mittelwertkurve mit
// Unsicherheitsband, wie sie die Simulation ausgibt. Bewusst anders als das
// 3x3-Gitter von CombiTab, damit sich die beiden ÖAI-Werkzeuge im Panel
// unterscheiden lassen.
const bell = (x, mu, sigma) => Math.exp(-((x - mu) ** 2) / (2 * sigma * sigma));
/** Mittelwertkurve, normiert auf 0..1. */
const meanAt = (x) => 0.12 + 0.56 * bell(x, 0.44, 0.26);
/** Halbe Bandbreite an der Stelle x: dort breit, wo viel Material umläuft. */
const halfAt = (x) => 0.045 + 0.115 * bell(x, 0.50, 0.30);

function drawIcon(size, { maskable = false } = {}) {
  const SS = 3;                        // Supersampling gegen Treppenkanten
  const acc = new Float64Array(size * size * 3);
  const cnt = new Float64Array(size * size);
  const pad = size * (maskable ? 0.215 : 0.135);
  const area = size - 2 * pad;
  const lw = Math.max(area * 0.055, 1.0);   // Strichstärke der Mittelwertkurve

  const toX = (px) => (px - pad) / area;                 // Pixel -> 0..1
  const toV = (py) => 1 - (py - pad) / area;             // Pixel -> Wert 0..1

  for (let sy = 0; sy < size * SS; sy++) {
    const py = (sy + 0.5) / SS;
    for (let sx = 0; sx < size * SS; sx++) {
      const px = (sx + 0.5) / SS;
      let col = BGL;
      const x = toX(px), v = toV(py);
      if (x >= 0 && x <= 1) {
        const m = meanAt(x), h = halfAt(x);
        // Grundlinie
        if (v >= -0.03 && v <= 0.012) col = AXIS;
        if (v >= 0 && v <= m + 1.55 * h && v >= m - 1.55 * h) col = BAND2;
        if (v >= 0 && v <= m + h && v >= m - h) col = BAND;
        // Mittelwertkurve: senkrechter Abstand reicht, die Kurve ist flach genug
        if (Math.abs(v - m) * area <= lw / 2) col = ACCENT;
      }
      const i = (Math.floor(py) * size + Math.floor(px));
      if (i < 0 || i >= size * size) continue;
      acc[i * 3] += col[0]; acc[i * 3 + 1] += col[1]; acc[i * 3 + 2] += col[2];
      cnt[i] += 1;
    }
  }
  const out = Buffer.alloc(size * size * 4);
  for (let i = 0; i < size * size; i++) {
    const n = cnt[i] || 1;
    out[i * 4] = Math.round(acc[i * 3] / n);
    out[i * 4 + 1] = Math.round(acc[i * 3 + 1] / n);
    out[i * 4 + 2] = Math.round(acc[i * 3 + 2] / n);
    out[i * 4 + 3] = 255;
  }
  return encodePNG(size, size, out);
}

writeFileSync(join(OUT, "icon-192.png"), drawIcon(192));
writeFileSync(join(OUT, "icon-512.png"), drawIcon(512));
writeFileSync(join(OUT, "icon-maskable-512.png"), drawIcon(512, { maskable: true }));

// Quellbild für die Desktop-Pakete. electron-builder leitet daraus selbst das
// Windows-.ico, das macOS-.icns und die Linux-Icongrößen ab — deshalb genügt
// eine einzige, ausreichend große PNG-Datei und es braucht kein ImageMagick
// auf dem Build-Rechner.
mkdirSync(BUILD_OUT, { recursive: true });
writeFileSync(join(BUILD_OUT, "icon.png"), drawIcon(1024));

// Icon-Satz für Linux. electron-builder legt jede Größe unter
// /usr/share/icons/hicolor/<größe>/apps/ ab. Ein einzelnes 1024er-Icon reicht
// nicht: Panels und Anwendungsmenüs suchen die gängigen Stufen und zeigen sonst
// ein Platzhaltersymbol.
const LINUX_SIZES = [16, 24, 32, 48, 64, 128, 256, 512];
const ICONS_OUT = join(BUILD_OUT, "icons");
mkdirSync(ICONS_OUT, { recursive: true });
for (const size of LINUX_SIZES) {
  writeFileSync(join(ICONS_OUT, `${size}x${size}.png`), drawIcon(size));
}

// Vektor-Favicon (scharf in jeder Größe), gleiche Geometrie in 40x40-Einheiten.
const rgbHex = (c) => "#" + c.map((v) => v.toString(16).padStart(2, "0")).join("");
const VP = 40, VPAD = 5.4, VA = VP - 2 * VPAD;
const sx = (x) => (VPAD + x * VA).toFixed(2);
const sy = (v) => (VPAD + (1 - v) * VA).toFixed(2);
const N = 48;
const xs = Array.from({ length: N + 1 }, (_, i) => i / N);
const bandPath = (k) => {
  const up = xs.map((x) => `${sx(x)},${sy(Math.min(1, meanAt(x) + k * halfAt(x)))}`);
  const dn = xs.slice().reverse().map((x) => `${sx(x)},${sy(Math.max(0, meanAt(x) - k * halfAt(x)))}`);
  return `M${up.join(" L")} L${dn.join(" L")} Z`;
};
const meanPath = `M${xs.map((x) => `${sx(x)},${sy(meanAt(x))}`).join(" L")}`;
writeFileSync(join(OUT, "favicon.svg"),
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${VP} ${VP}">` +
  `<rect width="${VP}" height="${VP}" rx="6" fill="${rgbHex(BGL)}"/>` +
  `<path d="${bandPath(1.55)}" fill="${rgbHex(BAND2)}"/>` +
  `<path d="${bandPath(1)}" fill="${rgbHex(BAND)}"/>` +
  `<path d="${meanPath}" fill="none" stroke="${rgbHex(ACCENT)}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>` +
  `</svg>\n`);

console.log("PWA-Icons in public/: icon-192.png, icon-512.png, icon-maskable-512.png, favicon.svg");
console.log("Paket-Icon in build/: icon.png (1024x1024)");
console.log(`Linux-Icon-Satz in build/icons/: ${LINUX_SIZES.map((s) => `${s}px`).join(", ")}`);
