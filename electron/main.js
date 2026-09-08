/**
 * STOCHASI v2 — Electron-Hauptprozess.
 *
 * Die Weboberfläche wird NICHT über file:// geladen, sondern über ein eigenes,
 * als sicher registriertes Schema `app://`. Grund: Die App erzeugt ihre Rechen-
 * Worker als Modul-Worker (`new Worker(url, { type: "module" })`). Chromium
 * blockiert die unter file:// über die CORS-Regeln, womit Simulation und
 * inverse Datierung im Haupt-Thread hängen blieben. Ein eigenes Schema liefert einen
 * echten, stabilen Origin und damit zusätzlich verlässliches IndexedDB
 * (Autosave), localStorage (Theme/Sprache) und einen sicheren Kontext für
 * CompressionStream (Teilen-Link).
 */
import { app, BrowserWindow, Menu, dialog, shell, protocol, net, nativeTheme, session } from "electron";
import { fileURLToPath, pathToFileURL } from "node:url";
import { join, dirname, normalize, sep } from "node:path";
import { readFile, writeFile, mkdir } from "node:fs/promises";

const __dirname = dirname(fileURLToPath(import.meta.url));

/** Wurzel der gebauten Weboberfläche (dist/ wird von electron-builder mitgepackt). */
const WEB_ROOT = join(__dirname, "..", "dist");

/** Selbsttest-Betrieb: Fenster bleibt unsichtbar, Prüfbericht auf die Konsole. */
const SMOKE = process.env.STOCHASI_SMOKE === "1";

const APP_SCHEME = "app";
const APP_ORIGIN = `${APP_SCHEME}://stochasi`;

/**
 * Alles bleibt lokal: Die Richtlinie verbietet jede Verbindung nach außen.
 * `unsafe-inline` bei Skripten ist nötig für den Theme-Vorabsetzer in
 * index.html, der das Aufblitzen der hellen Fläche verhindert; da `default-src`
 * auf 'self' steht, ist von außen ohnehin nichts ladbar.
 */
const CSP = [
  "default-src 'self'",
  "script-src 'self' 'unsafe-inline'",
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data: blob:",
  "font-src 'self'",
  "connect-src 'self' data: blob:",
  "worker-src 'self' blob:",
  "object-src 'none'",
  "base-uri 'self'",
  "form-action 'none'",
  "frame-ancestors 'none'",
].join("; ");

// Muss vor `app.whenReady()` stehen, sonst gilt das Schema nicht als sicher.
protocol.registerSchemesAsPrivileged([
  {
    scheme: APP_SCHEME,
    privileges: { standard: true, secure: true, supportFetchAPI: true, corsEnabled: true, stream: true },
  },
]);

/** Bedient app://stochasi/… aus dem dist-Verzeichnis, mit Schutz vor Pfadausbruch. */
function serveWebRoot() {
  protocol.handle(APP_SCHEME, async (request) => {
    const { pathname } = new URL(request.url);
    const decoded = decodeURIComponent(pathname);
    const relative = normalize(decoded).replace(/^([/\\])+/, "");
    const target = join(WEB_ROOT, relative === "" ? "index.html" : relative);

    // Nichts außerhalb von dist/ ausliefern.
    if (target !== WEB_ROOT && !target.startsWith(WEB_ROOT + sep)) {
      return new Response("Forbidden", { status: 403 });
    }

    const res = await net.fetch(pathToFileURL(target).toString());
    // Antwort mit CSP nachreichen; net.fetch liefert die Kopfzeilen der Datei.
    const headers = new Headers(res.headers);
    headers.set("Content-Security-Policy", CSP);
    return new Response(res.body, { status: res.status, statusText: res.statusText, headers });
  });
}

/* ---------------------------------------------------------------- Fenstermaße */

const stateFile = () => join(app.getPath("userData"), "window-state.json");

async function loadWindowState() {
  try {
    const raw = await readFile(stateFile(), "utf8");
    const s = JSON.parse(raw);
    if (Number.isFinite(s.width) && Number.isFinite(s.height)) return s;
  } catch { /* erster Start oder unlesbar — Vorgabewerte */ }
  return { width: 1440, height: 900 };
}

async function saveWindowState(win) {
  if (!win || win.isDestroyed()) return;
  try {
    const b = win.isMaximized() || win.isFullScreen() ? win.getNormalBounds() : win.getBounds();
    await mkdir(app.getPath("userData"), { recursive: true });
    await writeFile(
      stateFile(),
      JSON.stringify({ ...b, maximized: win.isMaximized() }, null, 2),
      "utf8",
    );
  } catch { /* Fenstermaße zu sichern ist entbehrlich — nie den Start blockieren */ }
}

/* -------------------------------------------------------------------- Fenster */

async function createWindow() {
  const state = await loadWindowState();

  const win = new BrowserWindow({
    width: state.width,
    height: state.height,
    x: state.x,
    y: state.y,
    minWidth: 1024,
    minHeight: 640,
    // Verhindert das Aufblitzen einer weißen Fläche vor dem ersten Paint.
    backgroundColor: nativeTheme.shouldUseDarkColors ? "#1a1817" : "#ffffff",
    show: false,
    title: "STOCHASI",
    webPreferences: {
      preload: join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      webSecurity: true,
    },
  });

  if (state.maximized) win.maximize();
  // Im Selbsttest bleibt das Fenster verborgen — es soll auf einem Build-Rechner
  // nichts aufpoppen.
  if (!SMOKE) win.once("ready-to-show", () => win.show());

  // Externe Verweise gehören in den Standardbrowser, nicht in ein App-Fenster
  // ohne Adressleiste.
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith("http://") || url.startsWith("https://")) shell.openExternal(url);
    return { action: "deny" };
  });

  // Wegnavigieren aus der App unterbinden (der Teilen-Link ändert nur das Fragment).
  win.webContents.on("will-navigate", (event, url) => {
    if (!url.startsWith(APP_ORIGIN)) {
      event.preventDefault();
      if (url.startsWith("http://") || url.startsWith("https://")) shell.openExternal(url);
    }
  });

  let saveTimer;
  const scheduleSave = () => {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => saveWindowState(win), 400);
  };
  if (!SMOKE) {
    win.on("resize", scheduleSave);
    win.on("move", scheduleSave);
    win.on("close", () => { clearTimeout(saveTimer); saveWindowState(win); });
  }

  await win.loadURL(`${APP_ORIGIN}/index.html`);
  return win;
}

/* ------------------------------------------------------------------ Downloads */

/**
 * Die Exporte der App laufen über `<a download>`. Ohne Zutun legte Electron sie
 * wortlos im Download-Ordner ab; der Nutzer soll stattdessen sehen, wohin die
 * Datei geht.
 */
function handleDownloads(sess) {
  sess.on("will-download", (_event, item) => {
    const name = item.getFilename();
    const ext = name.includes(".") ? name.slice(name.lastIndexOf(".") + 1).toLowerCase() : "";
    const filters = [];
    const known = {
      pdf: "PDF-Dokument", svg: "SVG-Grafik", png: "PNG-Grafik",
      csv: "CSV-Tabelle", xlsx: "Excel-Arbeitsmappe", json: "STOCHASI-Projekt",
    };
    if (ext && known[ext]) filters.push({ name: known[ext], extensions: [ext] });
    filters.push({ name: "Alle Dateien", extensions: ["*"] });

    item.setSaveDialogOptions({
      title: "Export speichern",
      defaultPath: join(app.getPath("documents"), name),
      filters,
    });
  });
}

/* ---------------------------------------------------------------------- Menü */

function buildMenu(win) {
  const de = app.getLocale().toLowerCase().startsWith("de");
  const t = (d, e) => (de ? d : e);
  const isMac = process.platform === "darwin";

  const about = () => {
    dialog.showMessageBox(win, {
      type: "info",
      title: t("Über STOCHASI", "About STOCHASI"),
      message: `STOCHASI ${app.getVersion()}`,
      detail: t(
        "Stochastische Simulation archäologischer Fundspektren und\n" +
          "inverse Datierung von Fundkomplexen.\n\n" +
          "© Christian Gugl / Österreichisches Archäologisches Institut (ÖAW)\n" +
          "MIT-Lizenz\n\n" +
          `Electron ${process.versions.electron} · Chromium ${process.versions.chrome}\n\n` +
          "Alle Daten bleiben auf diesem Rechner. Die Anwendung stellt keine\n" +
          "Verbindung ins Internet her.",
        "Stochastic simulation of archaeological find spectra and\n" +
          "inverse dating of assemblages.\n\n" +
          "© Christian Gugl / Austrian Archaeological Institute (ÖAW)\n" +
          "MIT License\n\n" +
          `Electron ${process.versions.electron} · Chromium ${process.versions.chrome}\n\n` +
          "All data stays on this computer. The application makes no\n" +
          "network connections.",
      ),
      buttons: ["OK"],
    });
  };

  const template = [
    ...(isMac
      ? [{
          label: app.name,
          submenu: [
            { label: t("Über STOCHASI", "About STOCHASI"), click: about },
            { type: "separator" },
            { role: "services" },
            { type: "separator" },
            { role: "hide", label: t("STOCHASI ausblenden", "Hide STOCHASI") },
            { role: "hideOthers", label: t("Andere ausblenden", "Hide Others") },
            { role: "unhide", label: t("Alle einblenden", "Show All") },
            { type: "separator" },
            { role: "quit", label: t("STOCHASI beenden", "Quit STOCHASI") },
          ],
        }]
      : []),
    {
      label: t("Datei", "File"),
      submenu: [
        // Laden und Exportieren geschieht in der Oberfläche selbst (Drag & Drop,
        // "Datei laden…", Export-Menü). Hier steht nur, was das Fenster betrifft.
        isMac
          ? { role: "close", label: t("Fenster schließen", "Close Window") }
          : { role: "quit", label: t("Beenden", "Quit") },
      ],
    },
    {
      label: t("Bearbeiten", "Edit"),
      submenu: [
        { role: "undo", label: t("Rückgängig", "Undo") },
        { role: "redo", label: t("Wiederherstellen", "Redo") },
        { type: "separator" },
        { role: "cut", label: t("Ausschneiden", "Cut") },
        { role: "copy", label: t("Kopieren", "Copy") },
        { role: "paste", label: t("Einfügen", "Paste") },
        { role: "selectAll", label: t("Alles auswählen", "Select All") },
      ],
    },
    {
      label: t("Ansicht", "View"),
      submenu: [
        { role: "reload", label: t("Neu laden", "Reload") },
        { role: "forceReload", label: t("Neu laden (Zwischenspeicher leeren)", "Force Reload") },
        { type: "separator" },
        { role: "resetZoom", label: t("Normale Größe", "Actual Size") },
        { role: "zoomIn", label: t("Vergrößern", "Zoom In") },
        { role: "zoomOut", label: t("Verkleinern", "Zoom Out") },
        { type: "separator" },
        { role: "togglefullscreen", label: t("Vollbild", "Toggle Full Screen") },
        { type: "separator" },
        { role: "toggleDevTools", label: t("Entwicklerwerkzeuge", "Developer Tools") },
      ],
    },
    {
      role: "window",
      label: t("Fenster", "Window"),
      submenu: [
        { role: "minimize", label: t("Minimieren", "Minimize") },
        ...(isMac ? [{ role: "zoom", label: t("Zoomen", "Zoom") }, { type: "separator" }, { role: "front", label: t("Alle nach vorne", "Bring All to Front") }] : []),
      ],
    },
    {
      role: "help",
      label: t("Hilfe", "Help"),
      submenu: [
        {
          label: t("Kurzanleitung", "Quick start guide"),
          click: () => shell.openExternal(t(
            "https://github.com/oeai-dac/STOCHASI/blob/main/docs/QUICKSTART.de.md",
            "https://github.com/oeai-dac/STOCHASI/blob/main/docs/QUICKSTART.md",
          )),
        },
        {
          label: t("Handbuch (Website)", "Documentation (website)"),
          click: () => shell.openExternal("https://github.com/oeai-dac/STOCHASI"),
        },
        {
          label: t("Fehler melden", "Report an issue"),
          click: () => shell.openExternal("https://github.com/oeai-dac/STOCHASI/issues"),
        },
        ...(isMac ? [] : [{ type: "separator" }, { label: t("Über STOCHASI", "About STOCHASI"), click: about }]),
      ],
    },
  ];

  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

/* ------------------------------------------------------------------ Lebenslauf */

// Zweite Instanz: vorhandenes Fenster nach vorn holen, statt neu zu starten.
if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  let mainWindow = null;

  app.on("second-instance", () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });

  app.whenReady().then(async () => {
    serveWebRoot();
    handleDownloads(session.defaultSession);
    mainWindow = await createWindow();
    buildMenu(mainWindow);

    if (SMOKE) {
      const { runSmoke } = await import("./smoke.js");
      await runSmoke(app, mainWindow, WEB_ROOT);
      return;
    }

    app.on("activate", async () => {
      if (BrowserWindow.getAllWindows().length === 0) {
        mainWindow = await createWindow();
        buildMenu(mainWindow);
      }
    });
  });

  app.on("window-all-closed", () => {
    if (process.platform !== "darwin") app.quit();
  });
}
