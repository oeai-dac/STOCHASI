/**
 * Lädt die drei verwendeten Google-Schriften einmalig als WOFF2 nach
 * `public/fonts/` und erzeugt `src/fonts.css` mit lokalen @font-face-Regeln.
 *
 * Grund: Die Desktop-Fassung muss offline vollständig funktionieren, und die
 * App soll beim Start keinen Fremdserver kontaktieren ("alles bleibt lokal").
 *
 * Nur nötig, wenn Schnitte oder Schriftstärken geändert werden:
 *     npm run vendor-fonts
 *
 * Lizenzen: Cormorant Garamond (OFL 1.1), JetBrains Mono (OFL 1.1),
 * Outfit (OFL 1.1) — Weitergabe im Bundle ausdrücklich erlaubt.
 */
import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";

// Genau die Schnitte, die index.html bisher vom CDN geholt hat — damit sich
// am Schriftbild nichts ändert.
const FAMILIES = [
  { name: "Cormorant Garamond", slug: "cormorant-garamond", weights: [400, 500, 600, 700] },
  { name: "JetBrains Mono",     slug: "jetbrains-mono",     weights: [400, 500] },
  { name: "Outfit",             slug: "outfit",             weights: [300, 400, 500, 600] },
];

// "latin" deckt Deutsch/Englisch ab, "latin-ext" zusätzlich die mittel- und
// osteuropäischen Diakritika, die in archäologischen Fundortnamen vorkommen.
const WANTED_SUBSETS = new Set(["latin", "latin-ext"]);

// Ohne moderne UA liefert Google TTF statt WOFF2.
const UA =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 " +
  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36";

const OUT_DIR = new URL("../src/fonts/", import.meta.url);
const CSS_OUT = new URL("../src/fonts.css", import.meta.url);

/** Zerlegt die Google-CSS in @font-face-Blöcke samt vorangestelltem Subset-Kommentar. */
function parseFaces(css) {
  const faces = [];
  // Google stellt jedem Block ein `/* latin */`-Kommentar voran.
  const re = /\/\*\s*([a-z0-9-]+)\s*\*\/\s*@font-face\s*\{([^}]*)\}/gi;
  let m;
  while ((m = re.exec(css)) !== null) {
    const subset = m[1];
    const body = m[2];
    const url = body.match(/url\((https:[^)]+\.woff2)\)/)?.[1];
    const weight = body.match(/font-weight:\s*(\d+)/)?.[1];
    const style = body.match(/font-style:\s*(\w+)/)?.[1] ?? "normal";
    const unicodeRange = body.match(/unicode-range:\s*([^;]+);/)?.[1]?.trim();
    if (url && weight) faces.push({ subset, url, weight: Number(weight), style, unicodeRange });
  }
  return faces;
}

async function main() {
  await mkdir(OUT_DIR, { recursive: true });
  const cssRules = [];
  let downloaded = 0;
  let bytes = 0;

  for (const fam of FAMILIES) {
    const spec = `${fam.name.replace(/ /g, "+")}:wght@${fam.weights.join(";")}`;
    const cssUrl = `https://fonts.googleapis.com/css2?family=${spec}&display=swap`;

    const res = await fetch(cssUrl, { headers: { "User-Agent": UA } });
    if (!res.ok) throw new Error(`CSS für ${fam.name} nicht erhalten: HTTP ${res.status}`);
    const css = await res.text();

    const faces = parseFaces(css).filter(
      (f) => WANTED_SUBSETS.has(f.subset) && fam.weights.includes(f.weight),
    );
    if (faces.length === 0) throw new Error(`Keine passenden Schnitte für ${fam.name} gefunden`);

    // Google liefert Variable Fonts: alle Schriftstärken einer Familie teilen
    // sich pro Subset *eine* Datei. Deshalb je URL nur einmal herunterladen und
    // die Datei aus allen Gewichts-Regeln referenzieren — jede @font-face-Regel
    // pinnt die Variable Font auf ihre Stärke.
    const fileByUrl = new Map();
    const urlsPerKey = new Map();
    for (const f of faces) {
      const key = `${fam.slug}-${f.subset}`;
      if (!urlsPerKey.has(key)) urlsPerKey.set(key, new Set());
      urlsPerKey.get(key).add(f.url);
    }

    for (const f of faces) {
      const key = `${fam.slug}-${f.subset}`;
      // Nur wenn ein Subset ausnahmsweise mehrere echte Dateien hat (statische
      // Schnitte), wandert die Stärke in den Dateinamen.
      const needsWeight = urlsPerKey.get(key).size > 1;
      const file = needsWeight ? `${fam.slug}-${f.weight}-${f.subset}.woff2` : `${key}.woff2`;

      if (!fileByUrl.has(f.url)) {
        const bin = await fetch(f.url, { headers: { "User-Agent": UA } });
        if (!bin.ok) throw new Error(`Download fehlgeschlagen: ${f.url} (HTTP ${bin.status})`);
        const buf = Buffer.from(await bin.arrayBuffer());
        await writeFile(join(OUT_DIR.pathname, file), buf);
        fileByUrl.set(f.url, file);
        downloaded++;
        bytes += buf.length;
        console.log(`  ${file}  ${(buf.length / 1024).toFixed(1)} KB`);
      }
      const named = fileByUrl.get(f.url);

      cssRules.push(
        [
          "@font-face{",
          `font-family:'${fam.name}';`,
          `font-style:${f.style};`,
          `font-weight:${f.weight};`,
          "font-display:swap;",
          // Liegt unter src/, läuft also durch Vites Asset-Pipeline: der Bundler
          // hasht die Datei und schreibt die URL relativ zum CSS um — trägt so
          // unter file:// (Electron) wie unter einem Unterpfad (GitHub Pages).
          `src:url(./fonts/${named}) format('woff2');`,
          f.unicodeRange ? `unicode-range:${f.unicodeRange};` : "",
          "}",
        ].join(""),
      );
    }
  }

  const header =
    "/* Erzeugt von scripts/vendor-fonts.mjs — nicht von Hand bearbeiten.\n" +
    "   Schriftdateien liegen in src/fonts/ und werden mitgeliefert,\n" +
    "   damit die App offline und ohne Fremdserver-Aufruf läuft.\n" +
    "   Cormorant Garamond, JetBrains Mono, Outfit: SIL Open Font License 1.1. */\n";
  await writeFile(CSS_OUT, header + cssRules.join("\n") + "\n");

  console.log(`\n${downloaded} Schriftdateien, ${(bytes / 1024).toFixed(0)} KB gesamt.`);
  console.log("src/fonts.css erzeugt.");
}

main().catch((e) => {
  console.error("Fehler:", e.message);
  process.exit(1);
});
