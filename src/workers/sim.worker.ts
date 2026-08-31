/**
 * Rechen-Worker: Simulation und inverse Datierung.
 *
 * Der Vorwärtslauf kostet bei großen Projekten (Dutzende Zentren, mehrere
 * hundert Jahre, 500 Läufe) Sekunden. Im Haupt-Thread würde die Oberfläche
 * dabei einfrieren, und zwar genau in dem Moment, in dem jemand an einem
 * Schieberegler zieht. Deshalb rechnet ein Worker.
 *
 * Das Ensemble wird als übertragbarer Puffer zurückgegeben, nicht kopiert.
 */
import type { Assemblage, Category, MarketTable, SimParams } from "../core/model.js";
import { interpolateMarket, yearRange } from "../core/model.js";
import { simulate, type Ensemble, type EnsembleStats } from "../sim/simulate.js";
import { applyResidual } from "../sim/residual.js";
import { dateAcrossResidual, dateAssemblage, type DatingMethod, type DatingResult } from "../sim/inverse.js";

export interface SimRequest {
  id: number;
  type: "simulate";
  market: MarketTable;
  categories: Category[];
  params: SimParams;
}

export interface DateRequest {
  id: number;
  type: "date";
  market: MarketTable;
  categories: Category[];
  params: SimParams;
  assemblages: Assemblage[];
  /** Residualanteile, über die gerechnet wird. Leer = nur der eingestellte Wert. */
  residuals: number[];
  method: DatingMethod;
  level: number;
}

export type WorkerRequest = SimRequest | DateRequest;

/** Ensemble ohne die Läufe — die braucht die Anzeige nicht und sie sind groß. */
export type EnsembleSummary = EnsembleStats;

export interface DatingRow { assemblageId: string; label: string; curves: Array<{ residual: number; result: DatingResult }>; }

export type WorkerResponse =
  | { id: number; type: "simulated"; ensemble: EnsembleSummary }
  | { id: number; type: "dated"; rows: DatingRow[] }
  | { id: number; type: "error"; message: string };

export function runSimulation(req: SimRequest): Ensemble {
  const ids = req.categories.map((c) => c.id);
  const years = yearRange(req.params.startYear, req.params.endYear);
  const M = interpolateMarket(req.market, ids, years);
  const base = simulate(M, years, ids, req.params);
  return req.params.residual > 0 ? applyResidual(base, req.params.residual) : base;
}

export function runDating(req: DateRequest): DatingRow[] {
  const ids = req.categories.map((c) => c.id);
  const years = yearRange(req.params.startYear, req.params.endYear);
  const M = interpolateMarket(req.market, ids, years);
  const base = simulate(M, years, ids, req.params);
  const opts = { method: req.method, level: req.level };
  return req.assemblages.map((a) => ({
    assemblageId: a.id,
    label: a.name,
    curves: req.residuals.length
      ? dateAcrossResidual(base, a.counts, req.residuals, opts)
      : [{
        residual: req.params.residual,
        result: dateAssemblage(req.params.residual > 0 ? applyResidual(base, req.params.residual, false) : base, a.counts, opts),
      }],
  }));
}

export function summarize(e: Ensemble): EnsembleSummary {
  return { years: e.years, ids: e.ids, runs: e.runs, mean: e.mean, p10: e.p10, p90: e.p90 };
}

/* Nur im Worker-Kontext an den Nachrichtenkanal hängen; in Tests und im
   Haupt-Thread bleiben die Funktionen oben einfach aufrufbar. Bewusst über eine
   schmale Schnittstelle statt über die DOM-Typen — das Projekt bindet die
   WebWorker-Bibliothek nicht ein, weil sie sich mit den DOM-Typen der App beißt. */
interface WorkerScope {
  postMessage(msg: WorkerResponse, transfer?: Transferable[]): void;
  onmessage: ((ev: MessageEvent<WorkerRequest>) => void) | null;
}
const scope = (globalThis as unknown as { self?: WorkerScope; document?: unknown }).self;
if (scope && typeof scope.postMessage === "function" && typeof (globalThis as { document?: unknown }).document === "undefined") {
  scope.onmessage = (ev: MessageEvent<WorkerRequest>) => {
    const req = ev.data;
    try {
      if (req.type === "simulate") {
        const s = summarize(runSimulation(req));
        scope.postMessage({ id: req.id, type: "simulated", ensemble: s }, [s.mean.buffer, s.p10.buffer, s.p90.buffer]);
      } else {
        scope.postMessage({ id: req.id, type: "dated", rows: runDating(req) });
      }
    } catch (e) {
      scope.postMessage({ id: req.id, type: "error", message: e instanceof Error ? e.message : String(e) });
    }
  };
}
