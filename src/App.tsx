import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./theme.css";
import type { Assemblage, Category, ProjectV2, SimParams } from "./core/model.js";
import { appendAssemblages } from "./core/model.js";
import type { EnsembleStats } from "./sim/simulate.js";
import { emptyProject, readProject, ProjectError } from "./core/io/project.js";
import { detectKind, importMarketGrid, importAssemblageGrid, ImportError } from "./core/io/importTable.js";
import { parseDelimited } from "./core/io/parseDelimited.js";
import { xlsxToGrid } from "./core/io/importXLSX.js";
import { PRESET_V1, centreById } from "./data/centres.js";
import { DEMO } from "./data/demo.js";
import { Shell, type TabId } from "./components/Shell.js";
import { Sidebar } from "./components/Sidebar.js";
import { MarketView, SimView, YearView, CompareView, DatingView, datingModel, datingNote, type DatingMode } from "./components/views.js";
import { DataView } from "./components/DataView.js";
import { ExportMenu, type ExportPayload } from "./components/ExportMenu.js";
import { simClient, isAbort } from "./workers/simClient.js";
import type { DatingRow } from "./workers/sim.worker.js";
import { saveAutosave, loadAutosave, clearAutosave, type AutosaveRecord } from "./core/autosave.js";
import { buildShareUrl, readShareFromHash } from "./core/shareLink.js";
import { useI18n } from "./i18n/I18nContext.js";
import { marketScene, simulationScene, spectrumScene, deviationScene, datingScene, rankScene, paramNote } from "./charts/charts.js";
import { LIGHT } from "./charts/plot.js";
import { simulationAoa, marketAoa, assemblagesAoa, datingCurvesAoa, datingSummaryAoa, parametersAoa } from "./export/exportTable.js";

/** Residualanteile der Vergleichsschar. */
const RESIDUAL_SCAN = [0, 0.1, 0.2, 0.3];
const DEMO_NAME = "Flavia-Solva-Beispiel";

export default function App() {
  const { t, lang } = useI18n();
  const [project, setProject] = useState<ProjectV2 | null>(null);
  const [tab, setTab] = useState<TabId>("sim");
  const [ensemble, setEnsemble] = useState<EnsembleStats | null>(null);
  const [rows, setRows] = useState<DatingRow[]>([]);
  const [scan, setScan] = useState(false);
  const [datingMode, setDatingMode] = useState<DatingMode>("curves");
  const [datingSel, setDatingSel] = useState("");
  /* Ausgeblendete Kurven. Nur für die Sitzung — in einer weitergegebenen
     Projektdatei stünde sonst eine unsichtbare Auswahl. */
  const [hidden, setHidden] = useState<Record<string, boolean>>({});
  const [markMode, setMarkMode] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notes, setNotes] = useState<string[]>([]);
  const [toast, setToast] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [restore, setRestore] = useState<AutosaveRecord | null>(null);
  const [imported, setImported] = useState<PendingImport | null>(null);
  const [installEvt, setInstallEvt] = useState<{ prompt: () => Promise<unknown> } | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const projectRef = useRef<ProjectV2 | null>(null); projectRef.current = project;
  const dirtyRef = useRef(false);
  const pristineRef = useRef<string | null>(null);

  useEffect(() => { if (import.meta.env.DEV) (globalThis as unknown as { __project?: unknown }).__project = project; }, [project]);

  /* Start: ein geteilter Link hat Vorrang vor Autosave und Beispiel. */
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const shared = await readShareFromHash(location.hash);
      if (cancelled) return;
      if (shared) {
        dirtyRef.current = true;
        setProject(shared.project);
        if (shared.ui?.tab) setTab(shared.ui.tab as TabId);
        return;
      }
      const demo = DEMO();
      pristineRef.current = JSON.stringify(demo);
      setProject(demo);
      const rec = await loadAutosave();
      if (!cancelled && rec) setRestore(rec);
    })();
    return () => { cancelled = true; };
  }, []);

  /* Sitzung beim Verlassen sichern; das unberührte Beispiel wird übersprungen. */
  useEffect(() => {
    const save = () => {
      const p = projectRef.current; if (!p) return;
      if (p.name === DEMO_NAME && !dirtyRef.current && JSON.stringify(p) === pristineRef.current) return;
      void saveAutosave(p);
    };
    const onVis = () => { if (document.visibilityState === "hidden") save(); };
    document.addEventListener("visibilitychange", onVis);
    window.addEventListener("pagehide", save);
    return () => { document.removeEventListener("visibilitychange", onVis); window.removeEventListener("pagehide", save); };
  }, []);

  useEffect(() => {
    const onPrompt = (e: Event) => { e.preventDefault(); setInstallEvt(e as unknown as { prompt: () => Promise<unknown> }); };
    const onInstalled = () => setInstallEvt(null);
    window.addEventListener("beforeinstallprompt", onPrompt);
    window.addEventListener("appinstalled", onInstalled);
    return () => { window.removeEventListener("beforeinstallprompt", onPrompt); window.removeEventListener("appinstalled", onInstalled); };
  }, []);

  useEffect(() => { if (!toast) return; const id = window.setTimeout(() => setToast(null), 3500); return () => window.clearTimeout(id); }, [toast]);

  /* Rechnen. Ein kurzer Aufschub bündelt das Ziehen an einem Regler zu einem Lauf. */
  const canRun = Boolean(project && project.market.years.length && project.params.endYear > project.params.startYear);
  const simKey = project ? JSON.stringify([project.categories.map((c) => c.id), project.market, project.params]) : "";
  const datingKey = project ? simKey + JSON.stringify(project.assemblages) + String(scan) : "";

  useEffect(() => {
    if (!project || !canRun) { setEnsemble(null); return; }
    let alive = true;
    setBusy(true);
    const id = window.setTimeout(() => {
      simClient.simulate({ market: project.market, categories: project.categories, params: project.params })
        .then((e) => { if (alive) { setEnsemble(e); setError(null); } })
        .catch((err) => { if (alive && !isAbort(err)) setError(String(err instanceof Error ? err.message : err)); })
        .finally(() => { if (alive) setBusy(false); });
    }, 140);
    return () => { alive = false; window.clearTimeout(id); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [simKey, canRun]);

  useEffect(() => {
    if (!project || !canRun || !project.assemblages.length) { setRows([]); return; }
    let alive = true;
    const id = window.setTimeout(() => {
      simClient.date({
        market: project.market, categories: project.categories, params: project.params,
        assemblages: project.assemblages, residuals: scan ? RESIDUAL_SCAN : [],
        method: "dirichlet", level: 0.95,
      })
        .then((r) => { if (alive) setRows(r); })
        .catch((err) => { if (alive && !isAbort(err)) setError(String(err instanceof Error ? err.message : err)); });
    }, 220);
    return () => { alive = false; window.clearTimeout(id); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datingKey, canRun]);

  const setParams = useCallback((patch: Partial<SimParams>) => {
    dirtyRef.current = true;
    setProject((p) => (p ? { ...p, params: { ...p.params, ...patch } } : p));
  }, []);
  const updateProject = useCallback((p: ProjectV2) => { dirtyRef.current = true; setProject(p); }, []);

  async function share() {
    const p = projectRef.current; if (!p) return;
    const { url, tooLong } = await buildShareUrl({ project: p, ui: { tab } }, location.origin, location.pathname);
    if (tooLong) { setToast(t("share.tooLarge")); return; }
    try { await navigator.clipboard.writeText(url); setToast(t("share.copied")); }
    catch { location.hash = url.slice(url.indexOf("#") + 1); setToast(t("share.copyManual")); }
  }

  /**
   * Fundkomplexe aus einem Import einsetzen — ersetzend oder anhängend.
   *
   * Die geladenen Komplexe werden ohne Farbe vorgehalten; erst hier bekommen sie
   * eine, denn beim Anhängen muss die Palette die vorhandene Reihe fortsetzen
   * statt bei Blau von vorne zu beginnen.
   */
  const applyImport = useCallback((p: PendingImport, mode: PendingImport["mode"]) => {
    setImported({ ...p, mode });
    dirtyRef.current = true;
    setProject((prev) => (prev ? {
      ...prev,
      assemblages: appendAssemblages(mode === "append" ? p.previous : [], p.incoming),
    } : prev));
  }, []);

  async function loadFile(file: File) {
    setError(null); setNotes([]); setImported(null);
    const name = file.name.replace(/\.[^.]+$/, "");
    try {
      if (/\.json$/i.test(file.name)) {
        const r = readProject(await file.text(), name);
        setProject(r.project); setNotes(r.fromV1 ? [t("import.fromV1"), ...r.notes] : r.notes);
        dirtyRef.current = true; setTab("sim");
        return;
      }
      const grid = /\.(xlsx|xls)$/i.test(file.name)
        ? await xlsxToGrid(await file.arrayBuffer())
        : parseDelimited(await file.text());
      const base = projectRef.current ?? emptyProject(PRESET_V1, name);
      if (detectKind(grid) === "market") {
        const r = importMarketGrid(grid);
        // Kategorien aus der Datei ergänzen, damit die Spalten nicht ins Leere laufen
        const { categories, added } = withNewCategories(base.categories, r.ids);
        const initial = { ...base.params.initial };
        for (const id of added) initial[id] = 0;
        setProject({
          ...base, name: base.name === DEMO_NAME ? name : base.name,
          categories, market: r.market, params: { ...base.params, initial },
          assemblages: base.assemblages.map((a) => ({ ...a, counts: { ...Object.fromEntries(added.map((id) => [id, 0])), ...a.counts } })),
        });
        setNotes([...r.warnings, ...(added.length ? [t("import.newCategories", { n: added.length })] : [])]);
        setTab("market");
      } else {
        const r = importAssemblageGrid(grid, { known: base.categories.map((c) => c.id), defaultName: name });
        // Spalten, die keiner Kategorie des Projekts entsprechen, würden sonst
        // stillschweigend wegfallen — und die Datierung liefe mit einem zu
        // kleinen N, ohne dass es jemand merkt. Also anlegen und melden.
        const { categories, added } = withNewCategories(base.categories, r.ids);
        const ids = categories.map((c) => c.id);
        const initial = { ...base.params.initial };
        const shares = { ...base.market.shares };
        for (const id of added) { initial[id] = 0; shares[id] = base.market.years.map(() => 0); }
        const incoming: Assemblage[] = r.assemblages.map((a) => ({
          ...a, color: undefined, counts: Object.fromEntries(ids.map((id) => [id, a.counts[id] ?? 0])),
        }));
        const previous = base.assemblages.map((a) => ({
          ...a, counts: Object.fromEntries(ids.map((id) => [id, a.counts[id] ?? 0])),
        }));
        setProject({
          ...base, name: base.name === DEMO_NAME ? name : base.name,
          categories, market: { ...base.market, shares }, params: { ...base.params, initial },
          assemblages: appendAssemblages([], incoming),
        });
        setNotes([...r.warnings, ...(r.transposed ? [t("import.transposed")] : []),
          ...(added.length ? [t("import.newCategories", { n: added.length })] : [])]);
        // Waren schon Komplexe da, wird ersetzt — aber sichtbar, mit der Wahl,
        // stattdessen anzuhängen.
        if (previous.length) setImported({ incoming, previous, mode: "replace" });
        setTab("dating");
      }
      dirtyRef.current = true;
    } catch (e) {
      const msg = e instanceof ImportError || e instanceof ProjectError ? e.message : e instanceof Error ? e.message : String(e);
      setError(t("import.failed", { msg }));
    }
  }

  /* Szene und Tabellen der aktuellen Ansicht — im hellen Schema für den Druck. */
  const payload: ExportPayload = useMemo(() => {
    if (!project) return { scene: null, viewName: "stochasi", sheets: [] };
    const W = 900, opt = { theme: LIGHT, width: W, footnote: paramNote(project.params) };
    const params = { name: "Parameter", aoa: parametersAoa(project) };
    switch (tab) {
      case "market":
        return {
          scene: project.market.years.length ? marketScene(project.market, project.categories, { ...opt, height: 460, title: t("market.title"), start: project.params.startYear, end: project.params.endYear }) : null,
          viewName: "markt",
          sheets: [{ name: "Marktangebot", aoa: marketAoa(project.market, project.categories, project.params.startYear, project.params.endYear) }, params],
        };
      case "sim":
        return {
          scene: ensemble ? simulationScene(ensemble, project.categories, { ...opt, height: 520, title: t("sim.title") }) : null,
          viewName: "simulation",
          sheets: ensemble ? [{ name: "Simulation", aoa: simulationAoa(ensemble, project.categories) }, params] : [params],
        };
      case "year":
        return {
          scene: ensemble ? spectrumScene(ensemble, project.categories, project.comparisonYear, { ...opt, title: t("year.title", { year: project.comparisonYear }), observed: project.assemblages[0] }) : null,
          viewName: `spektrum_${project.comparisonYear}`,
          sheets: ensemble ? [{ name: "Simulation", aoa: simulationAoa(ensemble, project.categories) }, { name: "Fundkomplexe", aoa: assemblagesAoa(project.assemblages, project.categories) }, params] : [params],
        };
      case "compare":
        return {
          scene: ensemble && project.assemblages[0] ? deviationScene(ensemble, project.categories, project.comparisonYear, project.assemblages[0], { ...opt, title: t("compare.title") }) : null,
          viewName: "vergleich",
          sheets: [{ name: "Fundkomplexe", aoa: assemblagesAoa(project.assemblages, project.categories) }, params],
        };
      case "dating": {
        const ranking = datingMode === "ranking" && rows.filter((r) => !r.curves[0].result.empty).length >= 2;
        const model = datingModel(project, rows, scan, datingSel, hidden, LIGHT);
        const note = datingNote(project, model, (n, total) => t("dating.hiddenNote", { n, total }));
        // Exportiert wird die Abbildung, die auf dem Bildschirm steht — samt der
        // Fußnote, die ausgeblendete Kurven ausweist. Die Tabellen daneben
        // bleiben vollständig; sie sind der Anhang zum Nachrechnen.
        const scene = ranking
          ? (model.rank.length ? rankScene(model.rank, { ...opt, footnote: note, title: t("dating.ranking.title"), overlapLabel: t("dating.ranking.overlapAll") }) : null)
          : model.visible.length ? datingScene(model.visible, { ...opt, footnote: note, height: 480, title: t("dating.title"), markMode }) : null;
        const flat = rows.map((r) => ({
          label: r.label, result: r.curves[0].result,
          shown: scan ? true : !hidden[r.assemblageId],
        }));
        return {
          scene,
          viewName: ranking ? "rangfolge" : "datierung",
          sheets: rows.length ? [{ name: "Datierung", aoa: datingSummaryAoa(flat) }, { name: "Kurven", aoa: datingCurvesAoa(flat) }, params] : [params],
        };
      }
      default:
        return {
          scene: null, viewName: "daten",
          sheets: [
            { name: "Marktangebot", aoa: marketAoa(project.market, project.categories, project.params.startYear, project.params.endYear) },
            { name: "Fundkomplexe", aoa: assemblagesAoa(project.assemblages, project.categories) },
            params,
          ],
        };
    }
  }, [project, tab, ensemble, rows, scan, datingMode, datingSel, hidden, markMode, t]);

  return (
    <>
      <a className="skip-link" href="#main">{t("a11y.skip")}</a>
      <div className="app"
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); const f = e.dataTransfer.files[0]; if (f) void loadFile(f); }}>
        <Shell project={project} tab={tab} onTab={setTab} onPickFile={() => fileInput.current?.click()} onShare={share}
          exportMenu={project ? <ExportMenu project={project} payload={payload} onToast={setToast} /> : undefined} />
        <input ref={fileInput} type="file" accept=".json,.csv,.tsv,.txt,.xlsx,.xls" style={{ display: "none" }}
          onChange={(e) => { const f = e.target.files?.[0]; if (f) void loadFile(f); e.target.value = ""; }} />

        <div className="body">
          {project && tab !== "data" && <Sidebar project={project} onParams={setParams} onProject={updateProject} />}
          <main id="main" className="content" role="tabpanel" aria-labelledby={`tab-${tab}`}>
            {restore && (
              <div className="restore-bar" role="region" aria-label={t("autosave.title")}>
                <span>{t("autosave.restoreQ", { name: restore.name, time: new Date(restore.savedAt).toLocaleString(lang === "de" ? "de-AT" : "en-GB") })}</span>
                <div className="restore-actions">
                  <button className="btn" onClick={() => { if (restore) { dirtyRef.current = true; setProject(restore.project); setRestore(null); } }}>{t("autosave.restore")}</button>
                  <button className="btn btn-ghost" onClick={() => { setRestore(null); void clearAutosave(); }}>{t("autosave.dismiss")}</button>
                </div>
              </div>
            )}
            {imported && (
              <div className="restore-bar" role="region" aria-label={t("import.title")}>
                <span>{t("import.loadedAssemblages", { n: imported.incoming.length })}</span>
                <div className="restore-actions">
                  <div className="seg" role="group" aria-label={t("import.title")}>
                    <button className={"seg-b" + (imported.mode === "replace" ? " on" : "")} aria-pressed={imported.mode === "replace"}
                      onClick={() => applyImport(imported, "replace")}>{t("import.replaceExisting")}</button>
                    <button className={"seg-b" + (imported.mode === "append" ? " on" : "")} aria-pressed={imported.mode === "append"}
                      onClick={() => applyImport(imported, "append")}>{t("import.appendExisting")}</button>
                  </div>
                  <button className="btn btn-ghost" onClick={() => setImported(null)}>{t("common.close")}</button>
                </div>
              </div>
            )}
            {error && <div className="msg err" role="alert">{error}</div>}
            {notes.length > 0 && (
              <div className="msg note" role="status">
                <button className="x" onClick={() => setNotes([])} aria-label={t("common.close")}>×</button>
                <ul>{notes.slice(0, 8).map((n, i) => <li key={i}>{n}</li>)}</ul>
                {notes.length > 8 && <p>{t("import.warnings", { n: notes.length - 8 })}</p>}
              </div>
            )}
            {!project ? <div className="placeholder">{t("app.loading")}</div> : (
              tab === "market" ? <MarketView project={project} ensemble={ensemble} busy={busy} />
                : tab === "sim" ? <SimView project={project} ensemble={ensemble} busy={busy} />
                  : tab === "year" ? <YearView project={project} ensemble={ensemble} busy={busy} />
                    : tab === "compare" ? <CompareView project={project} ensemble={ensemble} busy={busy} />
                      : tab === "dating" ? <DatingView project={project} rows={rows} busy={busy} scan={scan} onScan={setScan}
                        mode={datingMode} onMode={setDatingMode} sel={datingSel} onSel={setDatingSel}
                        hidden={hidden} onHidden={setHidden} markMode={markMode} onMarkMode={setMarkMode} />
                        : <DataView project={project} onProject={updateProject} />
            )}
          </main>
        </div>

        {installEvt && (
          <button className="install-chip" onClick={() => { void installEvt.prompt(); setInstallEvt(null); }}>
            <span aria-hidden="true">⬇</span> {t("pwa.install")}
          </button>
        )}
        {dragging && <div className="drop-overlay">{t("app.drop")}</div>}
        {toast && <div className="toast" role="status" aria-live="polite">{toast}</div>}
      </div>
    </>
  );
}

const FALLBACK = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#008080"];

/** Ein Fundkomplex-Import, solange die Wahl zwischen Ersetzen und Anhängen offensteht. */
interface PendingImport {
  /** Die geladenen Komplexe, noch ohne Farbe. */
  incoming: Assemblage[];
  /** Die Komplexe, die vor dem Import im Projekt standen. */
  previous: Assemblage[];
  mode: "replace" | "append";
}

/**
 * Kategorien der Datei ergänzen, die das Projekt noch nicht kennt.
 * Name und Farbe kommen aus der Referenzliste, sonst aus der Ersatzpalette.
 */
function withNewCategories(existing: readonly Category[], ids: readonly string[]): { categories: Category[]; added: string[] } {
  const known = new Set(existing.map((c) => c.id));
  const added = ids.filter((id) => !known.has(id));
  const categories = [...existing, ...added.map((id, i) => {
    const c = centreById(id);
    return { id, name: c?.name ?? id, color: c?.color ?? FALLBACK[(existing.length + i) % FALLBACK.length], group: c?.group };
  })];
  return { categories, added };
}
