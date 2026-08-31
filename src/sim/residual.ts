/**
 * Residualität — umgelagertes Altmaterial im Fundkomplex.
 *
 * Bei Siedlungsbefunden ist ein Teil der Funde älter als die Schicht, in der sie
 * liegen: aufgearbeitetes Material aus Planierungen, Grubenverfüllungen,
 * Altbeständen. STOCHASI 1 hat das nicht modelliert, was die Datierung
 * systematisch zu alt oder zu jung ziehen kann.
 *
 * Modelliert wird das mit **einem** Parameter r ∈ [0, 0.5]:
 *
 *     beobachtet(t) = (1 − r)·bestand(t) + r·altbestand(t)
 *     altbestand(t) = Mittel der Bestände aller Jahre vor t
 *
 * Der Altbestand ist also das gleichgewichtete Mittel des bisherigen
 * Umlaufmaterials — die einfachste Annahme, die keine weitere unbelegbare
 * Größe einführt. Für das Startjahr gibt es keine Vorjahre; dort bleibt der
 * Bestand unverändert.
 *
 * Die Mischung wird auf **jeden einzelnen Lauf** angewandt, nicht auf den
 * Mittelwert. Nur so bleibt das Unsicherheitsband richtig: Residualität
 * verschmiert die Spektren und macht das Band schmaler, nicht breiter.
 */
import type { Ensemble } from "./simulate.js";
import { percentileSorted } from "./stats.js";

/**
 * Liefert ein neues Ensemble mit eingemischtem Altbestand. r = 0 gibt das
 * Original zurück.
 *
 * `withPercentiles` steuert, ob Mittelwert und Perzentile neu berechnet werden.
 * Das kostet bei vielen Kategorien mehr als die Mischung selbst (je Zelle ein
 * Sortiervorgang über alle Läufe) und wird nur für die Anzeige gebraucht. Die
 * inverse Datierung liest ausschließlich die Läufe und kommt ohne aus.
 */
export function applyResidual(e: Ensemble, r: number, withPercentiles = true): Ensemble {
  const rr = Math.min(1, Math.max(0, r));
  if (rr === 0) return e;
  const ny = e.years.length, nc = e.ids.length, runs = e.runs;
  const out = new Float32Array(runs * ny * nc);
  const cum = new Float64Array(nc);

  for (let run = 0; run < runs; run++) {
    cum.fill(0);
    for (let y = 0; y < ny; y++) {
      const off = (run * ny + y) * nc;
      if (y === 0) {
        for (let c = 0; c < nc; c++) { const v = e.values[off + c]; out[off + c] = v; cum[c] = v; }
        continue;
      }
      let sum = 0;
      for (let c = 0; c < nc; c++) {
        const old = cum[c] / y;                 // Mittel der Jahre 0…y-1
        const v = (1 - rr) * e.values[off + c] + rr * old;
        out[off + c] = v; sum += v;
      }
      if (sum > 0) for (let c = 0; c < nc; c++) out[off + c] = (out[off + c] / sum) * 100;
      for (let c = 0; c < nc; c++) cum[c] += e.values[off + c];
    }
  }
  return withPercentiles ? withStats({ ...e, values: out }) : { ...e, values: out };
}

/** Rechnet Mittelwert und 10-/90-Perzentil aus den Läufen neu. */
export function withStats(e: Ensemble): Ensemble {
  const ny = e.years.length, nc = e.ids.length, runs = e.runs;
  const mean = new Float64Array(ny * nc), p10 = new Float64Array(ny * nc), p90 = new Float64Array(ny * nc);
  const col = new Float64Array(runs);
  for (let y = 0; y < ny; y++) for (let c = 0; c < nc; c++) {
    let s = 0;
    for (let run = 0; run < runs; run++) { const v = e.values[(run * ny + y) * nc + c]; col[run] = v; s += v; }
    mean[y * nc + c] = s / runs;
    col.sort();
    p10[y * nc + c] = percentileSorted(col, 10);
    p90[y * nc + c] = percentileSorted(col, 90);
  }
  return { ...e, mean, p10, p90 };
}
