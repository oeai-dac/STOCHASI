/**
 * Datenansicht: Kategorien, Marktkurve und Fundkomplexe bearbeiten.
 *
 * Die Marktkurve wird an Stützjahren gepflegt, nicht Jahr für Jahr — so steht in
 * der Tabelle, was man aus der Literatur wirklich weiß, und dazwischen sagt das
 * Programm sichtbar, dass es interpoliert.
 */
import { useMemo, useState } from "react";
import type { Assemblage, Category, ProjectV2 } from "../core/model.js";
import { normalizeTo100 } from "../core/model.js";
import { CENTRES, centresByGroup, type Centre } from "../data/centres.js";
import { useI18n, useT } from "../i18n/I18nContext.js";
import { yearLabel } from "../charts/plot.js";

export function DataView({ project, onProject }: { project: ProjectV2; onProject: (p: ProjectV2) => void }) {
  const t = useT();
  return (
    <div className="view data-view">
      <CategoryPanel project={project} onProject={onProject} />
      <MarketPanel project={project} onProject={onProject} />
      <AssemblagePanel project={project} onProject={onProject} />
      <p className="hint">{t("data.countsHelp")}</p>
    </div>
  );
}

/* ── Kategorien ── */
function CategoryPanel({ project, onProject }: { project: ProjectV2; onProject: (p: ProjectV2) => void }) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const chosen = new Set(project.categories.map((c) => c.id));

  function setCats(categories: Category[]) {
    const ids = categories.map((c) => c.id);
    // Marktkurve und Fundkomplexe auf die neue Kategorienmenge ziehen: neue
    // Kategorien starten bei 0, entfernte verschwinden. Ohne das blieben
    // verwaiste Spalten in den Daten stehen.
    const shares: Record<string, number[]> = {};
    for (const id of ids) shares[id] = project.market.shares[id] ?? project.market.years.map(() => 0);
    const initial: Record<string, number> = {};
    for (const id of ids) initial[id] = project.params.initial[id] ?? 0;
    if (!Object.values(initial).some((v) => v > 0)) for (const id of ids) initial[id] = 100 / Math.max(1, ids.length);
    const replacement: Record<string, number> = {};
    for (const id of ids) if (Number.isFinite(project.params.replacement[id])) replacement[id] = project.params.replacement[id];
    const assemblages = project.assemblages.map((a) => ({
      ...a, counts: Object.fromEntries(ids.map((id) => [id, a.counts[id] ?? 0])),
    }));
    onProject({ ...project, categories, market: { ...project.market, shares }, assemblages, params: { ...project.params, initial, replacement } });
  }

  function toggleCentre(c: Centre) {
    setErr(null);
    if (chosen.has(c.id)) {
      if (project.categories.length <= 1) { setErr(t("cat.lastOne")); return; }
      setCats(project.categories.filter((x) => x.id !== c.id));
    } else {
      const next = [...project.categories, { id: c.id, name: c.name, color: c.color, group: c.group }];
      // Reihenfolge der Referenzliste beibehalten, damit die Legende chronologisch bleibt
      const order = new Map(CENTRES.map((x, i) => [x.id, i] as const));
      next.sort((a, b) => (order.get(a.id) ?? 999) - (order.get(b.id) ?? 999));
      setCats(next);
    }
  }

  function addCustom() {
    setErr(null);
    let n = 1; while (chosen.has(`X${n}`)) n++;
    setCats([...project.categories, { id: `X${n}`, name: `Kategorie ${n}`, color: "#808080" }]);
  }

  function edit(id: string, patch: Partial<Category>) {
    setErr(null);
    if (patch.id !== undefined) {
      const v = patch.id.trim();
      if (!v) { setErr(t("cat.emptyId")); return; }
      if (v !== id && chosen.has(v)) { setErr(t("cat.duplicateId", { id: v })); return; }
      // Kennung ändern heißt: in Markt, Startwerten, Raten und Komplexen umbenennen
      const ren = <T,>(o: Record<string, T>): Record<string, T> =>
        Object.fromEntries(Object.entries(o).map(([k, val]) => [k === id ? v : k, val]));
      onProject({
        ...project,
        categories: project.categories.map((c) => (c.id === id ? { ...c, ...patch, id: v } : c)),
        market: { ...project.market, shares: ren(project.market.shares) },
        params: { ...project.params, initial: ren(project.params.initial), replacement: ren(project.params.replacement) },
        assemblages: project.assemblages.map((a) => ({ ...a, counts: ren(a.counts) })),
      });
      return;
    }
    onProject({ ...project, categories: project.categories.map((c) => (c.id === id ? { ...c, ...patch } : c)) });
  }

  return (
    <section className="panel">
      <h2>{t("cat.title")} <span className="cnt">{project.categories.length}</span></h2>
      {err && <p className="fld-err" role="alert">{err}</p>}
      <div className="tbl-wrap">
        <table className="tbl">
          <thead><tr><th>{t("cat.color")}</th><th>{t("cat.id")}</th><th>{t("cat.name")}</th><th /></tr></thead>
          <tbody>
            {project.categories.map((c) => (
              <tr key={c.id}>
                <td><input type="color" value={c.color} onChange={(e) => edit(c.id, { color: e.target.value })} aria-label={`${t("cat.color")} ${c.name}`} /></td>
                <td><input className="in-sm" value={c.id} onChange={(e) => edit(c.id, { id: e.target.value })} aria-label={`${t("cat.id")} ${c.name}`} /></td>
                <td><input className="in-md" value={c.name} onChange={(e) => edit(c.id, { name: e.target.value })} aria-label={`${t("cat.name")} ${c.id}`} /></td>
                <td><button className="x" title={t("cat.remove")} aria-label={`${t("cat.remove")} ${c.name}`}
                  onClick={() => { if (project.categories.length <= 1) { setErr(t("cat.lastOne")); return; } setCats(project.categories.filter((x) => x.id !== c.id)); }}>×</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="side-btns">
        <button className="btn btn-ghost" aria-expanded={open} onClick={() => setOpen((o) => !o)}>{open ? "−" : "+"} {t("cat.add")}</button>
        <button className="btn btn-ghost" onClick={addCustom}>+ {t("cat.addCustom")}</button>
      </div>
      {open && (
        <div className="centres">
          <p className="fld-help">{t("cat.referenceNote")}</p>
          {centresByGroup().map(({ group, centres }) => (
            <div key={group} className="centre-grp">
              <h3>{group}</h3>
              <div className="centre-list">
                {centres.map((c) => (
                  <button key={c.id} className={"centre" + (chosen.has(c.id) ? " on" : "")} onClick={() => toggleCentre(c)}
                    title={`${t("cat.period", { from: c.from < 0 ? `${-c.from} v. Chr.` : c.from, to: c.to })} · ${t(`cat.certain.${c.certainty}`)}`}>
                    {/* Bereits gewählte Zentren zeigen die Farbe, die im Projekt gilt — sonst
                        widerspräche der Chip der Legende, sobald jemand eine Farbe geändert hat
                        oder ein v1-Projekt mit eigenem Farbsatz geladen wurde. */}
                    <span className="sw" style={{ background: project.categories.find((x) => x.id === c.id)?.color ?? c.color }} aria-hidden="true" />
                    <span className="centre-nm">{c.name}</span>
                    <span className={"centre-yr" + (c.certainty === "approximate" ? " approx" : "")}>
                      {yearLabel(c.from)}–{yearLabel(c.to)}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

/* ── Marktkurve ── */
function MarketPanel({ project, onProject }: { project: ProjectV2; onProject: (p: ProjectV2) => void }) {
  const t = useT();
  const [newYear, setNewYear] = useState<number>(project.params.startYear);
  const ids = project.categories.map((c) => c.id);
  const sums = useMemo(() => project.market.years.map((_, k) => ids.reduce((a, id) => a + (project.market.shares[id]?.[k] ?? 0), 0)), [project.market, ids]);

  function setCell(k: number, id: string, v: number) {
    const shares = { ...project.market.shares };
    const col = (shares[id] ?? project.market.years.map(() => 0)).slice();
    col[k] = Math.max(0, v);
    shares[id] = col;
    onProject({ ...project, market: { ...project.market, shares } });
  }

  function addYear() {
    const y = Math.round(newYear);
    if (project.market.years.includes(y)) return;
    const years = [...project.market.years, y].sort((a, b) => a - b);
    const at = years.indexOf(y);
    const shares: Record<string, number[]> = {};
    for (const id of ids) {
      const col = (project.market.shares[id] ?? project.market.years.map(() => 0)).slice();
      col.splice(at, 0, 0);
      shares[id] = col;
    }
    onProject({ ...project, market: { years, shares } });
  }

  function removeYear(k: number) {
    const years = project.market.years.filter((_, i) => i !== k);
    const shares: Record<string, number[]> = {};
    for (const id of ids) shares[id] = (project.market.shares[id] ?? []).filter((_, i) => i !== k);
    onProject({ ...project, market: { years, shares } });
  }

  function normalizeRow(k: number) {
    const n = normalizeTo100(ids.map((id) => project.market.shares[id]?.[k] ?? 0));
    const shares = { ...project.market.shares };
    ids.forEach((id, i) => {
      const col = (shares[id] ?? project.market.years.map(() => 0)).slice();
      col[k] = Math.round(n[i] * 10) / 10;
      shares[id] = col;
    });
    onProject({ ...project, market: { ...project.market, shares } });
  }

  return (
    <section className="panel">
      <h2>{t("data.market")} <span className="cnt">{project.market.years.length}</span></h2>
      {!project.market.years.length && <p className="fld-help">{t("market.empty")}</p>}
      {project.market.years.length > 0 && (
        <div className="tbl-wrap">
          <table className="tbl mkt">
            <thead>
              <tr>
                <th>{t("market.year")}</th>
                {project.categories.map((c) => (
                  <th key={c.id} className="num" title={c.name}>
                    <span className="sw" style={{ background: c.color }} aria-hidden="true" /> {c.id}
                  </th>
                ))}
                <th className="num">{t("market.sum")}</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {project.market.years.map((y, k) => (
                <tr key={y}>
                  <td className="num mono">{yearLabel(y)}</td>
                  {project.categories.map((c) => (
                    <td key={c.id} className="num">
                      <input className="in-xs num" type="number" min={0} step={1}
                        value={Math.round((project.market.shares[c.id]?.[k] ?? 0) * 10) / 10}
                        onChange={(e) => setCell(k, c.id, Number(e.target.value))}
                        aria-label={`${c.name} ${y}`} />
                    </td>
                  ))}
                  <td className="num mono">
                    {/* Die Summe ist zugleich der Schalter zum Normieren. Rot wird sie nur,
                        wenn sie tatsächlich von 100 abweicht — sonst läse man jede Zeile
                        als Fehler. */}
                    <button className={"sum-btn" + (Math.abs(sums[k] - 100) > 0.5 ? " warn" : "")}
                      onClick={() => normalizeRow(k)} title={t("side.normalize")}>{sums[k].toFixed(0)}</button>
                  </td>
                  <td><button className="x" onClick={() => removeYear(k)} aria-label={`${t("market.removeYear")} ${y}`}>×</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <div className="side-btns">
        <input className="in-sm num" type="number" step={5} value={newYear} onChange={(e) => setNewYear(Number(e.target.value))} aria-label={t("market.addYear")} />
        <button className="btn btn-ghost" onClick={addYear}>+ {t("market.addYear")}</button>
      </div>
      <p className="fld-help">{t("market.interpolated")}</p>
    </section>
  );
}

/* ── Fundkomplexe ── */
function AssemblagePanel({ project, onProject }: { project: ProjectV2; onProject: (p: ProjectV2) => void }) {
  const { t, lang } = useI18n();
  const locale = lang === "de" ? "de-AT" : "en-GB";
  const ids = project.categories.map((c) => c.id);

  function setAsm(list: Assemblage[]) { onProject({ ...project, assemblages: list }); }
  function setCount(aid: string, cid: string, v: number) {
    setAsm(project.assemblages.map((a) => (a.id === aid ? { ...a, counts: { ...a.counts, [cid]: Math.max(0, Math.round(v)) } } : a)));
  }
  function add() {
    let n = project.assemblages.length + 1;
    const names = new Set(project.assemblages.map((a) => a.name));
    while (names.has(`${t("data.assemblage")} ${n}`)) n++;
    setAsm([...project.assemblages, {
      id: `A${Date.now().toString(36)}`, name: `${t("data.assemblage")} ${n}`,
      counts: Object.fromEntries(ids.map((id) => [id, 0])),
    }]);
  }

  return (
    <section className="panel">
      <h2>{t("data.assemblages")} <span className="cnt">{project.assemblages.length}</span></h2>
      {!project.assemblages.length && <p className="fld-help">{t("data.noAssemblages")}</p>}
      {project.assemblages.length > 0 && (
        <div className="tbl-wrap">
          <table className="tbl asm">
            <thead>
              <tr>
                <th>{t("data.assemblage")}</th>
                {project.categories.map((c) => (
                  <th key={c.id} className="num" title={c.name}>
                    <span className="sw" style={{ background: c.color }} aria-hidden="true" /> {c.id}
                  </th>
                ))}
                <th className="num">{t("data.total")}</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {project.assemblages.map((a) => {
                const total = ids.reduce((s, id) => s + (a.counts[id] ?? 0), 0);
                return (
                  <tr key={a.id}>
                    <td><input className="in-md" value={a.name}
                      onChange={(e) => setAsm(project.assemblages.map((x) => (x.id === a.id ? { ...x, name: e.target.value } : x)))}
                      aria-label={t("data.assemblage")} /></td>
                    {project.categories.map((c) => (
                      <td key={c.id} className="num">
                        <input className="in-xs num" type="number" min={0} step={1} value={a.counts[c.id] ?? 0}
                          onChange={(e) => setCount(a.id, c.id, Number(e.target.value))} aria-label={`${a.name} ${c.name}`} />
                      </td>
                    ))}
                    <td className="num mono">{total.toLocaleString(locale)}</td>
                    <td><button className="x" onClick={() => setAsm(project.assemblages.filter((x) => x.id !== a.id))}
                      aria-label={`${t("data.removeAssemblage")} ${a.name}`}>×</button></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      <div className="side-btns">
        <button className="btn btn-ghost" onClick={add}>+ {t("data.addAssemblage")}</button>
      </div>
    </section>
  );
}
