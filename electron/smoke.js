/**
 * Selbsttest der Desktop-Fassung.
 *
 * Prüft im laufenden Fenster genau die Eigenschaften, die beim Wechsel vom
 * Browser ins Electron-Gehäuse brechen können — allen voran der Modul-Worker,
 * der unter file:// nicht laden würde und ohne den Simulation und inverse
 * Datierung im Haupt-Thread hängen blieben.
 *
 * Aufruf:  npm run smoke
 * Beendet sich mit Code 0 (alles bestanden) oder 1 (mindestens ein Fehlschlag).
 *
 * Ist STOCHASI_SMOKE_REPORT gesetzt, wandert der Bericht zusätzlich als
 * schmuckloser Text in diese Datei. Nötig beim Prüfen der fertigen Pakete:
 * Unter Windows hängt die installierte .exe am GUI-Subsystem und schreibt
 * nicht in die Konsole des Aufrufers — ohne Datei bliebe dort nur der
 * Exit-Code, und man wüsste nicht, welche Prüfung gescheitert ist.
 */
import { readdir, writeFile } from "node:fs/promises";
import { join } from "node:path";

const GREEN = "\x1b[32m", RED = "\x1b[31m", OFF = "\x1b[0m";

/** Findet den gehashten Dateinamen des Rechen-Workers im gebauten Bundle. */
async function findSimWorker(webRoot) {
  const files = await readdir(join(webRoot, "assets"));
  const hit = files.find((f) => /^sim\.worker-.*\.js$/.test(f));
  if (!hit) throw new Error("sim.worker-*.js nicht in dist/assets/ gefunden");
  return `assets/${hit}`;
}

/** Im Renderer ausgeführt. Liefert eine Liste { name, ok, detail }. */
async function rendererProbe(simWorkerPath) {
  const checks = [];
  {
    const add = (name, ok, detail = "") =>
      checks.push({ name, ok: Boolean(ok), detail: String(detail) });

    add("Oberfläche gemountet", document.querySelector("#root")?.childElementCount > 0,
        `${document.querySelector("#root")?.childElementCount ?? 0} Kindknoten`);
    add("Desktop-Kennung im Fenster", Boolean(window.stochasiDesktop),
        window.stochasiDesktop ? `Electron ${window.stochasiDesktop.electron}` : "fehlt");
    add("Origin ist app://", location.protocol === "app:", location.origin);
    add("Sicherer Kontext", window.isSecureContext === true, String(window.isSecureContext));
    add("CompressionStream (Teilen-Link)", typeof CompressionStream === "function");
    add("IndexedDB vorhanden (Autosave)", typeof indexedDB === "object" && indexedDB !== null);

    // Die Diagramme sind SVG; ein Canvas wird nur für den PNG-Export gebraucht.
    let ctx = null;
    try { ctx = document.createElement("canvas").getContext("2d"); } catch { /* egal */ }
    add("Canvas-2D (PNG-Export)", Boolean(ctx));

    // Das Startbeispiel muss durchgerechnet werden. Die Rechnung läuft im Worker
    // und ist bewusst kurz aufgeschoben, damit das Ziehen an einem Regler nicht
    // jeden Zwischenwert rechnet — deshalb wird hier gewartet, nicht sofort geprüft.
    let svg = null;
    for (let i = 0; i < 60 && !svg; i++) {
      svg = document.querySelector("svg.chart");
      if (svg && svg.querySelectorAll("path").length > 3) break;
      svg = null;
      await new Promise((r) => setTimeout(r, 100));
    }
    add("Startbeispiel durchgerechnet und gezeichnet", Boolean(svg),
        svg ? `${svg.querySelectorAll("path").length} Pfade` : "nach 6 s kein svg.chart");

    // Eingebettete Schriften: müssen sich ohne Netz laden lassen. `check()`
    // allein taugt nicht — es meldet false, solange eine Schrift noch nicht
    // angefordert wurde. `load()` erzwingt den Abruf und beweist damit, dass
    // die gebündelten woff2-Dateien unter app:// tatsächlich erreichbar sind.
    const families = ["Outfit", "Cormorant Garamond", "JetBrains Mono"];
    const loaded = [];
    for (const f of families) {
      try {
        const faces = await document.fonts.load(`16px "${f}"`, "Aa");
        if (faces.length > 0) loaded.push(f);
      } catch { /* zählt als nicht geladen */ }
    }
    add("Lokale Schriften geladen", loaded.length === families.length,
        `${loaded.length}/${families.length}: ${loaded.join(", ") || "keine"}`);

    // Kein Verweis mehr auf einen Fremdserver irgendwo im Dokument.
    const external = [...document.querySelectorAll("link[href], script[src]")]
      .map((n) => n.getAttribute("href") || n.getAttribute("src"))
      .filter((u) => u && /^https?:/i.test(u));
    add("Keine externen Ressourcen", external.length === 0, external.join(", ") || "keine");

  }

  // Der eigentliche Prüfstein: ein Modul-Worker muss laden UND antworten.
  await new Promise((resolve) => {
    const add = (name, ok, detail = "") =>
      checks.push({ name, ok: Boolean(ok), detail: String(detail) });
    const finish = () => resolve();
    let settled = false;
    try {
      const w = new Worker(new URL(simWorkerPath, document.baseURI), { type: "module" });
      const timer = setTimeout(() => {
        if (settled) return;
        settled = true;
        add("Modul-Worker antwortet", false, "Zeitüberschreitung nach 8 s");
        w.terminate(); finish();
      }, 8000);
      w.onerror = (e) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        add("Modul-Worker antwortet", false, `Ladefehler: ${e.message || "unbekannt"}`);
        w.terminate(); finish();
      };
      w.onmessage = (ev) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        const d = ev.data;
        add("Modul-Worker antwortet", d && d.id === 1 && d.type === "simulated" && d.ensemble?.years?.length === 51,
            d ? `type=${d.type}, ${d.ensemble?.years?.length ?? 0} Jahre` : "leere Antwort");
        w.terminate(); finish();
      };
      // Kleine, echte Anfrage nach dem SimRequest-Protokoll.
      w.postMessage({
        id: 1, type: "simulate",
        market: { years: [100, 150], shares: { A: [100, 0], B: [0, 100] } },
        categories: [{ id: "A", name: "A", color: "#8B0000" }, { id: "B", name: "B", color: "#1E90FF" }],
        params: {
          startYear: 100, endYear: 150, replacement: {}, replacementDefault: 0.1,
          noiseSd: 1, runs: 20, seed: 1, settlementMode: false, residual: 0,
          initial: { A: 100, B: 0 },
        },
      });
    } catch (e) {
      settled = true;
      add("Modul-Worker antwortet", false, `Konstruktor warf: ${e.message}`);
      finish();
    }
  });

  return checks;
}

/**
 * Schreibt den Bericht nach STOCHASI_SMOKE_REPORT, falls gesetzt. Ein
 * Fehlschlag beim Schreiben darf den Selbsttest nicht verfälschen — das
 * Urteil steht da bereits fest und geht über den Exit-Code hinaus.
 */
async function writeReport(text) {
  const target = process.env.STOCHASI_SMOKE_REPORT;
  if (!target) return;
  try {
    await writeFile(target, text, "utf8");
  } catch (e) {
    console.error("Bericht konnte nicht geschrieben werden:", e?.message ?? e);
  }
}

/** Führt den Selbsttest im gegebenen Fenster aus und beendet den Prozess. */
export async function runSmoke(app, win, webRoot) {
  let code = 1;
  try {
    const simWorkerPath = await findSimWorker(webRoot);
    const checks = await win.webContents.executeJavaScript(
      `(${rendererProbe.toString()})(${JSON.stringify(simWorkerPath)})`,
      true,
    );

    let failed = 0;
    const plain = ["Selbsttest der Desktop-Fassung", ""];
    console.log("\nSelbsttest der Desktop-Fassung\n");
    for (const c of checks) {
      if (!c.ok) failed++;
      const mark = c.ok ? `${GREEN}OK  ${OFF}` : `${RED}FEHL${OFF}`;
      console.log(`  ${mark} ${c.name}${c.detail ? ` — ${c.detail}` : ""}`);
      plain.push(`  ${c.ok ? "OK  " : "FEHL"} ${c.name}${c.detail ? ` — ${c.detail}` : ""}`);
    }
    const fazit = failed === 0
      ? `Alle ${checks.length} Prüfungen bestanden.`
      : `${failed} von ${checks.length} Prüfungen fehlgeschlagen.`;
    console.log(failed === 0 ? `\n${GREEN}${fazit}${OFF}\n` : `\n${RED}${fazit}${OFF}\n`);
    plain.push("", fazit);
    await writeReport(plain.join("\n") + "\n");
    code = failed === 0 ? 0 : 1;
  } catch (e) {
    console.error("Selbsttest abgebrochen:", e?.message ?? e);
    await writeReport(`Selbsttest abgebrochen: ${e?.message ?? e}\n`);
  }
  app.exit(code);
}
