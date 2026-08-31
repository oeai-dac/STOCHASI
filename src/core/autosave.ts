/**
 * Autosave in IndexedDB (§9.8/§13) — framework-frei, ohne Fremdbibliothek.
 *
 * Sichert das zuletzt bearbeitete Projekt lokal im Browser, damit eine Sitzung
 * nach versehentlichem Schließen oder Absturz wiederhergestellt werden kann. Alle
 * Operationen sind fehlertolerant: fehlt IndexedDB (Privatmodus/alte Umgebung),
 * liefern sie still `null`/`void`, statt die Oberfläche zu stören.
 */
import type { ProjectV2 } from "./model.js";

const DB_NAME = "stochasi";
const STORE = "kv";
const KEY = "autosave";
const DB_VERSION = 1;

export interface AutosaveRecord { project: ProjectV2; savedAt: number; name: string; }

export function autosaveAvailable(): boolean {
  return typeof indexedDB !== "undefined";
}

function openDB(): Promise<IDBDatabase | null> {
  return new Promise((resolve) => {
    if (typeof indexedDB === "undefined") { resolve(null); return; }
    let req: IDBOpenDBRequest;
    try { req = indexedDB.open(DB_NAME, DB_VERSION); }
    catch { resolve(null); return; }
    req.onupgradeneeded = () => { const db = req.result; if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE); };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => resolve(null);
    req.onblocked = () => resolve(null);
  });
}

function tx(db: IDBDatabase, mode: IDBTransactionMode): IDBObjectStore {
  return db.transaction(STORE, mode).objectStore(STORE);
}

/** Speichert das Projekt als aktuellen Autosave-Stand. Wirft nie. */
export async function saveAutosave(p: ProjectV2): Promise<boolean> {
  try {
    const db = await openDB(); if (!db) return false;
    const rec: AutosaveRecord = { project: p, savedAt: Date.now(), name: p.name };
    await new Promise<void>((resolve, reject) => {
      const r = tx(db, "readwrite").put(rec, KEY);
      r.onsuccess = () => resolve(); r.onerror = () => reject(r.error);
    });
    db.close();
    return true;
  } catch { return false; }
}

/** Liest den letzten Autosave-Stand, sonst `null`. Wirft nie. */
export async function loadAutosave(): Promise<AutosaveRecord | null> {
  try {
    const db = await openDB(); if (!db) return null;
    const rec = await new Promise<AutosaveRecord | null>((resolve, reject) => {
      const r = tx(db, "readonly").get(KEY);
      r.onsuccess = () => resolve((r.result as AutosaveRecord) ?? null); r.onerror = () => reject(r.error);
    });
    db.close();
    return rec && rec.project ? rec : null;
  } catch { return null; }
}

/** Löscht den Autosave-Stand. Wirft nie. */
export async function clearAutosave(): Promise<void> {
  try {
    const db = await openDB(); if (!db) return;
    await new Promise<void>((resolve) => { const r = tx(db, "readwrite").delete(KEY); r.onsuccess = () => resolve(); r.onerror = () => resolve(); });
    db.close();
  } catch { /* still */ }
}
