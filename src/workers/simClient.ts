/**
 * Client um den Rechen-Worker.
 *
 * Es läuft immer nur eine Anfrage je Art; eine neue bricht die vorherige ab.
 * Das ist genau das gewünschte Verhalten an einem Schieberegler: Zwischenstände
 * werden verworfen, gerechnet wird der zuletzt eingestellte Wert.
 *
 * Fehlt die Worker-Unterstützung (ältere Umgebungen, Testkontexte), wird
 * synchron im Haupt-Thread gerechnet — dasselbe Ergebnis, nur mit Stocken.
 */
import type { DatingRow, EnsembleSummary, SimRequest, DateRequest, WorkerRequest, WorkerResponse } from "./sim.worker.js";
import { runSimulation, runDating, summarize } from "./sim.worker.js";

type Pending = { id: number; resolve: (v: never) => void; reject: (e: Error) => void };

class SimClient {
  private worker: Worker | null = null;
  private nextId = 1;
  private pending = new Map<number, Pending>();
  private supported: boolean | null = null;

  private ensure(): Worker | null {
    if (this.supported === false) return null;
    if (typeof Worker === "undefined") { this.supported = false; return null; }
    if (!this.worker) {
      try {
        this.worker = new Worker(new URL("./sim.worker.ts", import.meta.url), { type: "module" });
      } catch { this.supported = false; return null; }
      this.worker.onmessage = (ev: MessageEvent<WorkerResponse>) => {
        const msg = ev.data;
        const p = this.pending.get(msg.id);
        if (!p) return; // abgebrochen
        this.pending.delete(msg.id);
        if (msg.type === "error") p.reject(new Error(msg.message));
        else p.resolve((msg.type === "simulated" ? msg.ensemble : msg.rows) as never);
      };
      this.worker.onerror = () => {
        // Der Worker ließ sich nicht laden — ab jetzt synchron weiterrechnen,
        // statt die Anwendung stehenzulassen.
        this.supported = false;
        for (const [, p] of this.pending) p.reject(new DOMException("Worker nicht verfügbar", "AbortError"));
        this.pending.clear();
        this.worker?.terminate(); this.worker = null;
      };
      this.supported = true;
    }
    return this.worker;
  }

  /** Bricht alle laufenden Anfragen ab. */
  cancel(): void {
    for (const [, p] of this.pending) p.reject(new DOMException("Abgebrochen", "AbortError"));
    this.pending.clear();
    if (this.worker) { this.worker.terminate(); this.worker = null; }
  }

  private post<T>(build: (id: number) => WorkerRequest, sync: () => T): Promise<T> {
    const w = this.ensure();
    if (!w) { try { return Promise.resolve(sync()); } catch (e) { return Promise.reject(e); } }
    const id = this.nextId++;
    const req = build(id);
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { id, resolve: resolve as (v: never) => void, reject });
      w.postMessage(req);
    });
  }

  simulate(req: Omit<SimRequest, "id" | "type">): Promise<EnsembleSummary> {
    this.cancelPendingOfType("simulate");
    return this.post<EnsembleSummary>(
      (id) => ({ ...req, id, type: "simulate" }),
      () => summarize(runSimulation({ ...req, id: 0, type: "simulate" })),
    );
  }

  date(req: Omit<DateRequest, "id" | "type">): Promise<DatingRow[]> {
    this.cancelPendingOfType("date");
    return this.post<DatingRow[]>(
      (id) => ({ ...req, id, type: "date" }),
      () => runDating({ ...req, id: 0, type: "date" }),
    );
  }

  /* Eine neue Anfrage macht die vorherige gleicher Art gegenstandslos. Der Worker
     rechnet sie zwar zu Ende, das Ergebnis wird aber verworfen. */
  private types = new Map<number, string>();
  private cancelPendingOfType(kind: string): void {
    for (const [id, p] of [...this.pending]) {
      if (this.types.get(id) === kind) {
        this.pending.delete(id); this.types.delete(id);
        p.reject(new DOMException("Durch eine neuere Anfrage ersetzt", "AbortError"));
      }
    }
    this.types.set(this.nextId, kind);
  }
}

export const simClient = new SimClient();
/** true, wenn ein Abbruch die Ursache war — solche Fehler gehören nicht in die Oberfläche. */
export function isAbort(e: unknown): boolean {
  return e instanceof DOMException && e.name === "AbortError";
}
