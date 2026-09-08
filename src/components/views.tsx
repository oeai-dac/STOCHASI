/**
 * Die fünf Auswertungsansichten.
 *
 * Alle bekommen das Ensemble fertig gerechnet und bauen daraus nur noch eine
 * Szene. Die Rechnung selbst liegt im Worker (`workers/simClient`).
 */
import { useMemo, useState } from "react";
import type { Assemblage, ProjectV2 } from "../core/model.js";
import { assemblageColor } from "../core/model.js";
import type { EnsembleStats } from "../sim/simulate.js";
import type { DatingRow } from "../workers/sim.worker.js";
import { marketScene, simulationScene, spectrumScene, deviationScene, datingScene, rankScene, residualSeries, paramNote, type DatingSeries, type RankRow } from "../charts/charts.js";
import { ChartCanvas, usePlotTheme, useWidth } from "./ChartCanvas.js";
import { useI18n, useT } from "../i18n/I18nContext.js";
import { yearLabel, type PlotTheme } from "../charts/plot.js";
import { hexToRgb, type RGB } from "../charts/scene.js";

export interface ViewProps {
  project: ProjectV2;
  ensemble: EnsembleStats | null;
  busy: boolean;
}

function Empty({ text }: { text: string }) {
  return <div className="placeholder">{text}</div>;
}

/* ── Marktangebot ── */
export function MarketView({ project }: ViewProps) {
  const t = useT();
  const th = usePlotTheme();
  const [ref, w] = useWidth<HTMLDivElement>();
  const scene = useMemo(() => marketScene(project.market, project.categories, {
    theme: th, width: w, height: Math.max(340, Math.round(w * 0.5)),
    title: t("market.title"), start: project.params.startYear, end: project.params.endYear,
  }), [project.market, project.categories, project.params.startYear, project.params.endYear, th, w, t]);

  return (
    <div className="view" ref={ref}>
      {!project.market.years.length
        ? <Empty text={t("market.empty")} />
        : <>
          <ChartCanvas scene={scene} title={t("market.title")} />
          <p className="hint">{t("market.interpolated")} {t("market.normalized")}</p>
        </>}
    </div>
  );
}

/* ── Simulation ── */
export function SimView({ project, ensemble, busy }: ViewProps) {
  const t = useT();
  const th = usePlotTheme();
  const [ref, w] = useWidth<HTMLDivElement>();
  const [band, setBand] = useState(true);
  const scene = useMemo(() => ensemble && simulationScene(ensemble, project.categories, {
    theme: th, width: w, height: Math.max(360, Math.round(w * 0.55)),
    title: t("sim.title"), showBand: band, footnote: paramNote(project.params),
  }), [ensemble, project.categories, project.params, th, w, band, t]);

  if (!project.market.years.length) return <div className="view"><Empty text={t("sim.needMarket")} /></div>;
  return (
    <div className="view" ref={ref}>
      <div className="view-bar">
        <label className="chk">
          <input type="checkbox" checked={band} onChange={(e) => setBand(e.target.checked)} />
          <span>{t("sim.showBand")}</span>
        </label>
        {busy && <span className="busy">{t("sim.computing", { runs: project.params.runs })}</span>}
      </div>
      {scene ? <ChartCanvas scene={scene} title={t("sim.title")} /> : <Empty text={t("app.loading")} />}
      {band && <p className="hint">{t("sim.bandLegend")}</p>}
    </div>
  );
}

/* ── Jahresspektrum ── */
export function YearView({ project, ensemble }: ViewProps & { onYear?: (y: number) => void }) {
  const t = useT();
  const th = usePlotTheme();
  const [ref, w] = useWidth<HTMLDivElement>();
  const [year, setYear] = useState(project.comparisonYear);
  const [asmId, setAsmId] = useState<string>(project.assemblages[0]?.id ?? "");
  const y = Math.min(project.params.endYear, Math.max(project.params.startYear, year));
  const asm = project.assemblages.find((a) => a.id === asmId);

  const scene = useMemo(() => ensemble && spectrumScene(ensemble, project.categories, y, {
    theme: th, width: w, title: t("year.title", { year: yearLabel(y) }),
    observed: asm, footnote: paramNote(project.params),
  }), [ensemble, project.categories, project.params, y, asm, th, w, t]);

  if (!ensemble) return <div className="view"><Empty text={t("sim.needMarket")} /></div>;
  return (
    <div className="view" ref={ref}>
      <div className="view-bar">
        <label className="fld-inline">
          <span>{t("year.year")}</span>
          <input type="range" min={project.params.startYear} max={project.params.endYear} step={1}
            value={y} onChange={(e) => setYear(Number(e.target.value))} style={{ width: "min(340px, 40vw)" }} />
          <output className="fld-val">{yearLabel(y)}</output>
        </label>
        <label className="fld-inline">
          <span>{t("year.compareWith")}</span>
          <select value={asmId} onChange={(e) => setAsmId(e.target.value)}>
            <option value="">{t("year.none")}</option>
            {project.assemblages.map((a) => <option key={a.id} value={a.id}>{a.name}</option>)}
          </select>
        </label>
      </div>
      {scene && <ChartCanvas scene={scene} title={t("year.title", { year: yearLabel(y) })} />}
    </div>
  );
}

/* ── Vergleich ── */
export function CompareView({ project, ensemble }: ViewProps) {
  const t = useT();
  const th = usePlotTheme();
  const [ref, w] = useWidth<HTMLDivElement>();
  const [year, setYear] = useState(project.comparisonYear);
  const [asmId, setAsmId] = useState<string>(project.assemblages[0]?.id ?? "");
  const y = Math.min(project.params.endYear, Math.max(project.params.startYear, year));
  const asm: Assemblage | undefined = project.assemblages.find((a) => a.id === asmId) ?? project.assemblages[0];

  const scene = useMemo(() => ensemble && asm && deviationScene(ensemble, project.categories, y, asm, {
    theme: th, width: w, title: t("compare.title"), footnote: paramNote(project.params),
  }), [ensemble, project.categories, project.params, y, asm, th, w, t]);

  if (!project.assemblages.length) return <div className="view"><Empty text={t("compare.needAssemblage")} /></div>;
  if (!ensemble) return <div className="view"><Empty text={t("sim.needMarket")} /></div>;
  return (
    <div className="view" ref={ref}>
      <div className="view-bar">
        <label className="fld-inline">
          <span>{t("data.assemblage")}</span>
          <select value={asm?.id ?? ""} onChange={(e) => setAsmId(e.target.value)}>
            {project.assemblages.map((a) => <option key={a.id} value={a.id}>{a.name}</option>)}
          </select>
        </label>
        <label className="fld-inline">
          <span>{t("year.year")}</span>
          <input type="range" min={project.params.startYear} max={project.params.endYear} step={1}
            value={y} onChange={(e) => setYear(Number(e.target.value))} style={{ width: "min(340px, 40vw)" }} />
          <output className="fld-val">{yearLabel(y)}</output>
        </label>
      </div>
      {scene && <ChartCanvas scene={scene} title={t("compare.title")} />}
    </div>
  );
}

/* ── Inverse Datierung ── */
export type DatingMode = "curves" | "ranking";

/**
 * Fasst die Kurven eines Fundkomplexes zu einer Zeile der Rangfolge zusammen.
 *
 * Wird über mehrere Residualanteile gerechnet, ist die belastbare Aussage die
 * Spanne über alle Kurven — nicht das Intervall zu einer einzelnen Annahme.
 */
export function toRankRow(r: DatingRow, color?: RGB): RankRow {
  return {
    label: r.label,
    result: r.curves[0].result,
    span: r.curves.length > 1
      ? [Math.min(...r.curves.map((cu) => cu.result.hdi[0])), Math.max(...r.curves.map((cu) => cu.result.hdi[1]))]
      : undefined,
    color,
  };
}

/** Eine Kurve samt Schlüssel, unter dem sie sich ein- und ausblenden lässt. */
export interface KeyedSeries extends DatingSeries { key: string; }

export interface DatingModel {
  /** Alle Kurven, auch die ausgeblendeten — die Chip-Reihe zeigt sie vollständig. */
  all: KeyedSeries[];
  /** Die sichtbaren, in der Zeichenreihenfolge. */
  visible: KeyedSeries[];
  /** Zeilen des Rangfolge-Diagramms, nur die sichtbaren Fundkomplexe. */
  rank: RankRow[];
  hiddenCount: number;
}

/**
 * Was im Reiter Datierung gezeichnet wird — an einer Stelle, weil Bildschirm
 * und Export dieselbe Abbildung zeigen müssen.
 *
 * Im Normalfall ist jede Kurve ein Fundkomplex und trägt dessen Farbe. Ist die
 * Residualschar eingeschaltet, sind die Kurven die Residualanteile eines
 * einzigen Komplexes; sie leiten sich dann aus dessen Farbe ab.
 */
export function datingModel(
  project: ProjectV2, rows: readonly DatingRow[], scan: boolean, selectedId: string,
  hidden: Readonly<Record<string, boolean>>, th: PlotTheme,
): DatingModel {
  const colorOf = (id: string): RGB => {
    const i = project.assemblages.findIndex((a) => a.id === id);
    return hexToRgb(i >= 0 ? assemblageColor(project.assemblages[i], i) : "#a81d26");
  };
  const row = rows.find((r) => r.assemblageId === selectedId) ?? rows[0];
  const all: KeyedSeries[] = scan
    ? (row ? residualSeries(row.curves, colorOf(row.assemblageId), th).map((s, i) => ({ ...s, key: `r${i}` })) : [])
    : rows.map((r) => ({
      key: r.assemblageId,
      label: `${r.label} (n = ${r.curves[0]?.result.n ?? 0})`,
      result: r.curves[0].result,
      color: colorOf(r.assemblageId),
    }));
  const visible = all.filter((s) => !hidden[s.key]);
  const rank = rows.filter((r) => !hidden[r.assemblageId]).map((r) => toRankRow(r, colorOf(r.assemblageId)));
  return { all, visible, rank, hiddenCount: all.length - visible.length };
}

/**
 * Fußnote der Abbildung, um die Zahl der ausgeblendeten Kurven ergänzt.
 *
 * Ohne diesen Zusatz könnte eine exportierte Abbildung eine Abfolge behaupten,
 * die nur dadurch entsteht, dass die widersprechenden Komplexe weggeklickt
 * wurden. So trägt die Abbildung ihre Einschränkung selbst mit sich.
 */
export function datingNote(project: ProjectV2, model: DatingModel, hiddenLabel: (n: number, total: number) => string): string {
  const base = paramNote(project.params);
  return model.hiddenCount > 0 ? `${base} · ${hiddenLabel(model.hiddenCount, model.all.length)}` : base;
}

export function DatingView({ project, rows, busy, scan, onScan, mode, onMode, sel, onSel, hidden, onHidden, markMode, onMarkMode }: {
  project: ProjectV2; rows: DatingRow[]; busy: boolean;
  scan: boolean; onScan: (v: boolean) => void;
  mode: DatingMode; onMode: (m: DatingMode) => void;
  sel: string; onSel: (id: string) => void;
  hidden: Record<string, boolean>; onHidden: (h: Record<string, boolean>) => void;
  markMode: boolean; onMarkMode: (v: boolean) => void;
}) {
  const { t, lang } = useI18n();
  const th = usePlotTheme();
  const [ref, w] = useWidth<HTMLDivElement>();
  const locale = lang === "de" ? "de-AT" : "en-GB";
  const row = rows.find((r) => r.assemblageId === sel) ?? rows[0];

  const model = useMemo(() => datingModel(project, rows, scan, sel, hidden, th),
    [project, rows, scan, sel, hidden, th]);

  const canRank = rows.filter((r) => !r.curves[0].result.empty).length >= 2;
  const showRanking = mode === "ranking" && canRank;
  const note = datingNote(project, model, (n, total) => t("dating.hiddenNote", { n, total }));

  const scene = useMemo(() => {
    if (showRanking) {
      return model.rank.length ? rankScene(model.rank, {
        theme: th, width: w, title: t("dating.ranking.title"),
        overlapLabel: t("dating.ranking.overlapAll"), footnote: note,
      }) : null;
    }
    return model.visible.length ? datingScene(model.visible, {
      theme: th, width: w, height: Math.max(340, Math.round(w * 0.48)),
      title: t("dating.title"),
      subtitle: scan && row ? t("dating.subtitle", { name: row.label, n: row.curves[0]?.result.n ?? 0 }) : undefined,
      markMode, footnote: note,
    }) : null;
  }, [showRanking, model, th, w, t, scan, row, markMode, note]);

  const setAll = (on: boolean) => onHidden(on ? {} : Object.fromEntries(model.all.map((sr) => [sr.key, true])));

  if (!project.assemblages.length) return <div className="view"><Empty text={t("dating.needAssemblage")} /></div>;
  return (
    <div className="view" ref={ref}>
      <div className="view-bar">
        <div className="seg" role="group" aria-label={t("dating.title")}>
          <button className={"seg-b" + (!showRanking ? " on" : "")} onClick={() => onMode("curves")} aria-pressed={!showRanking}>
            {t("dating.view.curves")}
          </button>
          <button className={"seg-b" + (showRanking ? " on" : "")} onClick={() => onMode("ranking")}
            aria-pressed={showRanking} disabled={!canRank} title={canRank ? undefined : t("dating.ranking.needTwo")}>
            {t("dating.view.ranking")}
          </button>
        </div>
        {!showRanking && (
          <label className="chk">
            <input type="checkbox" checked={scan} onChange={(e) => onScan(e.target.checked)} />
            <span>{t("dating.residualScan")}</span>
          </label>
        )}
        {!showRanking && (
          <label className="chk">
            <input type="checkbox" checked={markMode} onChange={(e) => onMarkMode(e.target.checked)} />
            <span>{t("dating.markMode")}</span>
          </label>
        )}
        {!showRanking && scan && (
          <label className="fld-inline">
            <span>{t("data.assemblage")}</span>
            <select value={row?.assemblageId ?? ""} onChange={(e) => onSel(e.target.value)}>
              {rows.map((r) => <option key={r.assemblageId} value={r.assemblageId}>{r.label}</option>)}
            </select>
          </label>
        )}
        {busy && <span className="busy">{t("app.loading")}</span>}
      </div>

      {/* Ein- und Ausblenden. Die Chips schalten das, was gerade gezeichnet wird:
          die Fundkomplexe, oder im Residualmodus die Residualanteile des einen
          ausgewählten Komplexes. */}
      {model.all.length > 1 && (
        <div className="chips" role="group" aria-label={t("dating.visibleCurves")}>
          {model.all.map((sr) => {
            const on = !hidden[sr.key];
            return (
              <button key={sr.key} className={"chip" + (on ? " on" : "")} aria-pressed={on}
                onClick={() => onHidden({ ...hidden, [sr.key]: on })}>
                <span className="sw" style={{ background: `rgb(${sr.color.join(",")})` }} aria-hidden="true" />
                {sr.label}
              </button>
            );
          })}
          <button className="lnk" onClick={() => setAll(true)}>{t("dating.showAll")}</button>
          <button className="lnk" onClick={() => setAll(false)}>{t("dating.showNone")}</button>
        </div>
      )}

      {scene
        ? <ChartCanvas scene={scene} title={showRanking ? t("dating.ranking.title") : t("dating.title")} />
        : <Empty text={model.all.length ? t("dating.noneVisible") : t("app.loading")} />}
      <p className="hint">
        {showRanking ? t("dating.ranking.help") : scan ? t("dating.residualScanHelp") : t("dating.caveat")}
      </p>

      <h3 className="tbl-h">{t("dating.overview")}</h3>
      <div className="tbl-wrap">
        <table className="tbl">
          <thead>
            <tr>
              <th>{t("data.assemblage")}</th>
              <th className="num">{t("dating.n")}</th>
              <th className="num" title={t("dating.modeHelp")}>{t("dating.mode")}</th>
              <th className="num" title={t("dating.expectedHelp")}>{t("dating.expected")}</th>
              <th className="num">{t("dating.interval", { level: 95 })}</th>
              <th className="num">{t("dating.width")}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const res = r.curves[0].result;
              // Bei der Residualschar ist die belastbare Aussage die Spanne über
              // alle Anteile, nicht das Intervall zu einer einzelnen Annahme.
              const lo = Math.min(...r.curves.map((cu) => cu.result.hdi[0]));
              const hi = Math.max(...r.curves.map((cu) => cu.result.hdi[1]));
              return (
                <tr key={r.assemblageId}
                  className={(r.assemblageId === row?.assemblageId ? "on" : "") + (hidden[r.assemblageId] ? " off" : "")}>
                  <td>
                    <span className="sw" style={{ background: assemblageColor(asmOf(project, r.assemblageId), idxOf(project, r.assemblageId)) }} aria-hidden="true" />
                    {" "}{r.label}
                  </td>
                  <td className="num">{res.n.toLocaleString(locale)}</td>
                  <td className="num">{res.empty ? "—" : yearLabel(res.mode)}</td>
                  <td className="num">{res.empty ? "—" : yearLabel(Math.round(res.expected))}</td>
                  <td className="num">{res.empty ? "—" : `${yearLabel(lo)}–${yearLabel(hi)}`}</td>
                  <td className="num">{res.empty ? "—" : t("dating.years", { n: hi - lo })}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {rows.some((r) => r.curves[0].result.n > 0 && r.curves[0].result.n < 30) && (
        <p className="hint warn">{t("dating.smallSample", { n: Math.min(...rows.filter((r) => r.curves[0].result.n > 0).map((r) => r.curves[0].result.n)) })}</p>
      )}
    </div>
  );
}

/** Fundkomplex zu einer Kennung; für die Farbe in der Übersichtstabelle. */
function idxOf(project: ProjectV2, id: string): number {
  return project.assemblages.findIndex((a) => a.id === id);
}
function asmOf(project: ProjectV2, id: string): Assemblage {
  return project.assemblages.find((a) => a.id === id) ?? { id, name: id, counts: {} };
}
