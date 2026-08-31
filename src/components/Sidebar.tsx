/**
 * Parameterleiste. Alles, was die Simulation steuert, steht hier an einer
 * Stelle — die Ansichten selbst zeigen nur Ergebnisse.
 *
 * Jeder Regler trägt eine kurze Erklärung, was er im Modell bedeutet. Das ist
 * kein Beiwerk: Wer „Ersatzrate" auf 0,3 stellt, ohne zu wissen, dass damit ein
 * Drittel des Umlaufbestands pro Jahr ausscheidet, liest das Ergebnis falsch.
 */
import { useState } from "react";
import type { Category, ProjectV2, SimParams } from "../core/model.js";
import { MAX_NOISE, MAX_REPLACEMENT, MAX_RESIDUAL, MAX_RUNS, MIN_RUNS, normalizeTo100 } from "../core/model.js";
import { interpolateMarket, yearRange } from "../core/model.js";
import { useT } from "../i18n/I18nContext.js";

function Help({ text }: { text: string }) {
  return <p className="fld-help">{text}</p>;
}

function Slider({ label, help, value, min, max, step, format, onChange }: {
  label: string; help?: string; value: number; min: number; max: number; step: number;
  format: (v: number) => string; onChange: (v: number) => void;
}) {
  return (
    <div className="fld">
      <label className="fld-lbl">
        <span>{label}</span>
        <output className="fld-val">{format(value)}</output>
      </label>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))} />
      {help && <Help text={help} />}
    </div>
  );
}

export function Sidebar({ project, onParams, onProject }: {
  project: ProjectV2;
  onParams: (p: Partial<SimParams>) => void;
  onProject: (p: ProjectV2) => void;
}) {
  const t = useT();
  const p = project.params;
  const [showPerCat, setShowPerCat] = useState(Object.keys(p.replacement).length > 0);

  const setInitial = (id: string, v: number) => onParams({ initial: { ...p.initial, [id]: Math.max(0, v) } });
  const initSum = project.categories.reduce((a, c) => a + (p.initial[c.id] ?? 0), 0);

  function initialFromMarket() {
    const ids = project.categories.map((c) => c.id);
    const years = yearRange(p.startYear, p.startYear);
    const M = interpolateMarket(project.market, ids, years);
    const initial: Record<string, number> = {};
    ids.forEach((id, k) => (initial[id] = Math.round(M[k] * 10) / 10));
    onParams({ initial });
  }

  function normalizeInitial() {
    const ids = project.categories.map((c) => c.id);
    const n = normalizeTo100(ids.map((id) => p.initial[id] ?? 0));
    const initial: Record<string, number> = {};
    ids.forEach((id, k) => (initial[id] = Math.round(n[k] * 10) / 10));
    onParams({ initial });
  }

  return (
    <aside className="side" aria-label={t("side.simulation")}>
      {/* Zeitraum */}
      <section className="side-sec">
        <h2>{t("side.period")}</h2>
        <div className="fld-row">
          <label className="fld-num">
            <span>{t("side.startYear")}</span>
            <input type="number" value={p.startYear} step={5}
              onChange={(e) => onParams({ startYear: Math.round(Number(e.target.value) || 0) })} />
          </label>
          <label className="fld-num">
            <span>{t("side.endYear")}</span>
            <input type="number" value={p.endYear} step={5}
              onChange={(e) => onParams({ endYear: Math.round(Number(e.target.value) || 0) })} />
          </label>
        </div>
        {p.endYear <= p.startYear && <p className="fld-err" role="alert">{t("side.periodError")}</p>}
      </section>

      {/* Simulation */}
      <section className="side-sec">
        <h2>{t("side.simulation")}</h2>
        <Slider label={t("side.replacement")} help={t("side.replacementHelp")}
          value={p.replacementDefault} min={0} max={MAX_REPLACEMENT} step={0.01}
          format={(v) => `${Math.round(v * 100)} %`}
          onChange={(v) => onParams({ replacementDefault: v })} />

        <button className="lnk" aria-expanded={showPerCat} onClick={() => setShowPerCat((s) => !s)}>
          {showPerCat ? "−" : "+"} {t("side.perCategory")}
        </button>
        {showPerCat && (
          <div className="percat">
            <Help text={t("side.perCategoryHelp")} />
            {project.categories.map((c) => {
              const own = p.replacement[c.id];
              const has = Number.isFinite(own);
              return (
                <div className="percat-row" key={c.id}>
                  <span className="sw" style={{ background: c.color }} aria-hidden="true" />
                  <span className="percat-name" title={c.name}>{c.id}</span>
                  <input type="range" min={0} max={MAX_REPLACEMENT} step={0.01}
                    value={has ? own : p.replacementDefault}
                    onChange={(e) => onParams({ replacement: { ...p.replacement, [c.id]: Number(e.target.value) } })} />
                  <span className="percat-val">{Math.round((has ? own : p.replacementDefault) * 100)} %</span>
                  <button className="x" disabled={!has} title={t("common.reset")} aria-label={`${t("common.reset")} ${c.id}`}
                    onClick={() => { const r = { ...p.replacement }; delete r[c.id]; onParams({ replacement: r }); }}>↺</button>
                </div>
              );
            })}
          </div>
        )}

        <Slider label={t("side.noise")} help={t("side.noiseHelp")}
          value={p.noiseSd} min={0} max={MAX_NOISE} step={0.5}
          format={(v) => v.toFixed(1)} onChange={(v) => onParams({ noiseSd: v })} />

        <Slider label={t("side.residual")} help={t("side.residualHelp")}
          value={p.residual} min={0} max={MAX_RESIDUAL} step={0.01}
          format={(v) => `${Math.round(v * 100)} %`} onChange={(v) => onParams({ residual: v })} />

        <Slider label={t("side.runs")} help={t("side.runsHelp")}
          value={p.runs} min={MIN_RUNS} max={MAX_RUNS} step={10}
          format={(v) => String(v)} onChange={(v) => onParams({ runs: Math.round(v) })} />

        <label className="fld-num">
          <span>{t("side.seed")}</span>
          <input type="number" min={0} value={p.seed}
            onChange={(e) => onParams({ seed: Math.max(0, Math.round(Number(e.target.value) || 0)) })} />
        </label>
        <Help text={t("side.seedHelp")} />

        <label className="chk">
          <input type="checkbox" checked={p.settlementMode}
            onChange={(e) => onParams({ settlementMode: e.target.checked })} />
          <span>{t("side.settlement")}</span>
        </label>
        <Help text={t("side.settlementHelp")} />
      </section>

      {/* Startverteilung */}
      <section className="side-sec">
        <h2>{t("side.initial")}</h2>
        <Help text={t("side.initialHelp")} />
        <div className="side-btns">
          <button className="btn btn-ghost" onClick={initialFromMarket} disabled={!project.market.years.length}>
            {t("side.initialFromMarket")}
          </button>
          <button className="btn btn-ghost" onClick={normalizeInitial}>{t("side.normalize")}</button>
        </div>
        {project.categories.map((c: Category) => (
          <div className="init-row" key={c.id}>
            <span className="sw" style={{ background: c.color }} aria-hidden="true" />
            <span className="init-name" title={c.name}>{c.id}</span>
            <input type="number" min={0} max={100} step={1} value={Math.round((p.initial[c.id] ?? 0) * 10) / 10}
              onChange={(e) => setInitial(c.id, Number(e.target.value))}
              aria-label={`${t("side.initial")} ${c.name}`} />
          </div>
        ))}
        <p className={"fld-help" + (Math.abs(initSum - 100) > 0.5 ? " warn" : "")}>
          {t("data.total")}: {initSum.toFixed(1)} %
        </p>
      </section>
    </aside>
  );
}
