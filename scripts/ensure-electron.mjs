/**
 * Stellt sicher, dass die Electron-Binärdatei vorliegt.
 *
 * Hintergrund: Ab npm 12 werden Installations-Skripte von Abhängigkeiten
 * standardmäßig blockiert. Bei `npm ci` greift die `allowScripts`-Freigabe aus
 * der package.json nicht zuverlässig, sodass Electron zwar als Paket, aber ohne
 * seine Programmdatei ankommt — `npm run electron` und `npm run dist` scheitern
 * dann mit einer schwer deutbaren Meldung.
 *
 * Dieses Skript läuft als `postinstall` des Projekts selbst; solche Skripte
 * werden von npm nie blockiert. Es holt den Download nur nach, wenn er fehlt.
 *
 * Bricht die Installation nie ab: Wer nur die Web-Fassung bauen will
 * (`npm run build`), braucht Electron nicht.
 */
import { existsSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";

const require = createRequire(import.meta.url);

let electronDir;
try {
  electronDir = dirname(require.resolve("electron/package.json"));
} catch {
  // Electron ist nicht installiert (z. B. `npm ci --omit=dev`) — nichts zu tun.
  process.exit(0);
}

// `path.txt` schreibt Electrons Installer am Ende; sein Vorhandensein zusammen
// mit der tatsächlichen Datei ist das verlässliche Zeichen für „fertig".
const pathTxt = join(electronDir, "path.txt");
const installer = join(electronDir, "install.js");

if (existsSync(pathTxt)) {
  process.exit(0);
}

if (!existsSync(installer)) {
  console.warn("[ensure-electron] install.js nicht gefunden — übersprungen.");
  process.exit(0);
}

console.log("[ensure-electron] Electron-Binärdatei fehlt, wird nachgeholt …");
const res = spawnSync(process.execPath, [installer], {
  cwd: electronDir,
  stdio: "inherit",
});

if (res.status !== 0) {
  console.warn(
    "\n[ensure-electron] Der Download ist fehlgeschlagen.\n" +
      "Die Web-Fassung (npm run build) funktioniert trotzdem.\n" +
      "Für die Desktop-Fassung bitte nachholen mit:  npm run ensure-electron\n",
  );
}
