/**
 * Preload für STOCHASI v2.
 *
 * Muss CommonJS sein (.cjs): Electron lädt Preload-Skripte in der Sandbox
 * ausschließlich als CommonJS, auch wenn das Projekt sonst ESM ist.
 *
 * Die Oberfläche braucht keinerlei Systemzugriff — sie liest und schreibt
 * ausschließlich über die Browser-APIs. Hier wird deshalb bewusst KEINE
 * Dateisystem- oder Node-Funktion durchgereicht, sondern nur eine Kennung,
 * an der die App merkt, dass sie im Desktop-Gehäuse läuft (siehe
 * `src/main.tsx`: dort unterbleibt daraufhin die Service-Worker-Registrierung,
 * weil die App bereits lokal liegt).
 */
const { contextBridge } = require("electron");

contextBridge.exposeInMainWorld("stochasiDesktop", {
  platform: process.platform,
  electron: process.versions.electron,
  chrome: process.versions.chrome,
});
