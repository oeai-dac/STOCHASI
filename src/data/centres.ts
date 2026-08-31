/**
 * Referenzliste der Terra-Sigillata-Produktionszentren.
 *
 * WAS DIESE LISTE IST: eine Auswahlhilfe. Sie liefert Kennung, ausgeschriebene
 * Bezeichnung, Gruppe, Farbe und einen groben Produktionszeitraum, damit man
 * Kategorien nicht von Hand anlegen muss und die Farbgebung über Projekte hinweg
 * gleich bleibt.
 *
 * WAS SIE NICHT IST: eine Datengrundlage für die Simulation. STOCHASI rechnet mit
 * *Marktanteilen pro Jahr* — also damit, welchen Anteil ein Zentrum am jeweils neu
 * gelieferten Material eines konkreten Fundortes hatte. Solche Kurven sind
 * regional verschieden und hier bewusst NICHT mitgeliefert; sie wären erfunden.
 * Die Marktkurve muss aus der Literatur zum jeweiligen Fundort oder aus eigenen
 * Zählungen kommen.
 *
 * Die Zeiträume `from`/`to` sind gerundete Orientierungswerte der gängigen
 * Handbuchliteratur und dienen nur dazu, in der Oberfläche eine plausible
 * Vorbelegung des Zeitfensters und eine Sortierung anzubieten. `certainty`
 * unterscheidet gut abgesicherte Zentren von solchen, deren Laufzeit in der
 * regionalen Forschung strittig oder nur grob umrissen ist. Vor der Verwendung
 * in einer Publikation ist beides an der Literatur zum eigenen Arbeitsgebiet zu
 * prüfen.
 */
import type { Category } from "../core/model.js";

export type Certainty = "established" | "approximate";

export interface Centre extends Category {
  group: string;
  /** Produktionsbeginn, gerundet, n. Chr. (negative Werte = v. Chr.). */
  from: number;
  /** Produktionsende, gerundet, n. Chr. */
  to: number;
  certainty: Certainty;
}

/* Farbfamilien je Gruppe — innerhalb einer Gruppe abgestuft, zwischen den
   Gruppen deutlich getrennt, damit sich benachbarte Zentren im Diagramm
   unterscheiden lassen. */
export const GROUP_ORDER = [
  "Italisch", "Südgallisch", "Mittelgallisch", "Ostgallisch",
  "Rätisch", "Pannonisch/lokal", "Nordafrikanisch", "Östlich",
] as const;

export const CENTRES: Centre[] = [
  // ── Italisch ──
  { id: "AR", name: "Arezzo",            group: "Italisch", color: "#7f1d1d", from: -40, to: 40,  certainty: "established" },
  { id: "PI", name: "Pisa",              group: "Italisch", color: "#991b1b", from: -20, to: 60,  certainty: "established" },
  { id: "MI", name: "Mittelitalien",     group: "Italisch", color: "#b91c1c", from: -20, to: 50,  certainty: "approximate" },
  { id: "PA", name: "Padana",            group: "Italisch", color: "#c2410c", from: 1,   to: 150, certainty: "established" },
  { id: "TP", name: "Tardopadana",       group: "Italisch", color: "#9a3412", from: 150, to: 400, certainty: "approximate" },

  // ── Südgallisch ──
  { id: "LG", name: "La Graufesenque",   group: "Südgallisch", color: "#f97316", from: 20,  to: 120, certainty: "established" },
  { id: "MT", name: "Montans",           group: "Südgallisch", color: "#fb923c", from: 20,  to: 150, certainty: "established" },
  { id: "BA", name: "Banassac",          group: "Südgallisch", color: "#fdba74", from: 80,  to: 150, certainty: "established" },
  { id: "ES", name: "Espalion",          group: "Südgallisch", color: "#fed7aa", from: 40,  to: 70,  certainty: "approximate" },

  // ── Mittelgallisch ──
  { id: "MV", name: "Les Martres-de-Veyre", group: "Mittelgallisch", color: "#166534", from: 100, to: 165, certainty: "established" },
  { id: "LZ", name: "Lezoux",               group: "Mittelgallisch", color: "#16a34a", from: 120, to: 200, certainty: "established" },
  { id: "TF", name: "Terre-Franche",        group: "Mittelgallisch", color: "#4ade80", from: 120, to: 200, certainty: "approximate" },

  // ── Ostgallisch ──
  { id: "CH", name: "Chémery-Faulquemont", group: "Ostgallisch", color: "#0c4a6e", from: 50,  to: 120, certainty: "approximate" },
  { id: "LM", name: "La Madeleine",        group: "Ostgallisch", color: "#075985", from: 100, to: 160, certainty: "established" },
  { id: "BW", name: "Blickweiler",         group: "Ostgallisch", color: "#0369a1", from: 100, to: 180, certainty: "established" },
  { id: "HB", name: "Heiligenberg",        group: "Ostgallisch", color: "#0284c7", from: 120, to: 180, certainty: "established" },
  { id: "LV", name: "Lavoye",              group: "Ostgallisch", color: "#0ea5e9", from: 130, to: 260, certainty: "established" },
  { id: "TR", name: "Trier",               group: "Ostgallisch", color: "#38bdf8", from: 130, to: 300, certainty: "established" },
  { id: "RZ", name: "Rheinzabern",         group: "Ostgallisch", color: "#1e90ff", from: 140, to: 260, certainty: "established" },
  { id: "IW", name: "Ittenweiler",         group: "Ostgallisch", color: "#7dd3fc", from: 150, to: 200, certainty: "approximate" },
  { id: "SZ", name: "Sinzig",              group: "Ostgallisch", color: "#a5c9ea", from: 150, to: 250, certainty: "approximate" },
  { id: "WB", name: "Waiblingen",          group: "Ostgallisch", color: "#bae6fd", from: 150, to: 200, certainty: "approximate" },
  { id: "AG", name: "Argonnen",            group: "Ostgallisch", color: "#1e3a8a", from: 300, to: 450, certainty: "established" },

  // ── Rätisch ──
  { id: "WD", name: "Westerndorf",   group: "Rätisch", color: "#6b21a8", from: 180, to: 240, certainty: "established" },
  { id: "PF", name: "Pfaffenhofen",  group: "Rätisch", color: "#a855f7", from: 220, to: 270, certainty: "established" },

  // ── Pannonisch / lokal ──
  { id: "AQ", name: "Aquincum",                 group: "Pannonisch/lokal", color: "#a16207", from: 100, to: 250, certainty: "approximate" },
  { id: "SI", name: "Siscia",                   group: "Pannonisch/lokal", color: "#ca8a04", from: 150, to: 250, certainty: "approximate" },
  { id: "NP", name: "Norisch-pannonische Ware", group: "Pannonisch/lokal", color: "#eab308", from: 50,  to: 250, certainty: "approximate" },

  // ── Nordafrikanisch ──
  { id: "AA", name: "African Red Slip A", group: "Nordafrikanisch", color: "#78350f", from: 80,  to: 200, certainty: "established" },
  { id: "AC", name: "African Red Slip C", group: "Nordafrikanisch", color: "#92400e", from: 200, to: 320, certainty: "established" },
  { id: "AD", name: "African Red Slip D", group: "Nordafrikanisch", color: "#b45309", from: 320, to: 600, certainty: "established" },

  // ── Östlich ──
  { id: "EA", name: "Eastern Sigillata A", group: "Östlich", color: "#134e4a", from: -150, to: 150, certainty: "established" },
  { id: "EB", name: "Eastern Sigillata B", group: "Östlich", color: "#0f766e", from: -25,  to: 150, certainty: "established" },
  { id: "EC", name: "Çandarlı (ESC)",      group: "Östlich", color: "#14b8a6", from: 50,   to: 200, certainty: "established" },
  { id: "PS", name: "Pontische Sigillata", group: "Östlich", color: "#5eead4", from: -50,  to: 150, certainty: "approximate" },
  { id: "PH", name: "Phokäische Ware",     group: "Östlich", color: "#2dd4bf", from: 400,  to: 700, certainty: "approximate" },
];

/**
 * Die fünf Kategorien von STOCHASI 1, mit den dortigen Kennungen und Farben.
 * Wird gebraucht, damit v1-Projekte unverändert aussehen. „IT" und „MG" sind
 * dabei Sammelbegriffe, keine einzelnen Werkstätten.
 */
export const PRESET_V1: Category[] = [
  { id: "IT", name: "Italian",         color: "#8B0000", group: "Italisch" },
  { id: "LG", name: "La Graufesenque", color: "#FF6666", group: "Südgallisch" },
  { id: "BA", name: "Banassac",        color: "#1E90FF", group: "Südgallisch" },
  { id: "MG", name: "Central Gaulish", color: "#228B22", group: "Mittelgallisch" },
  { id: "RZ", name: "Rheinzabern",     color: "#FF8C00", group: "Ostgallisch" },
];

/**
 * Die 19 Provenienzen der Auswertung Flavia Solva, in der dort verwendeten
 * Reihenfolge. Dient als Startpunkt für vergleichbare Auswertungen im
 * norisch-pannonischen Raum.
 */
export const PRESET_FLAVIA_SOLVA_IDS = [
  "AR", "MI", "PI", "PA", "TP", "LG", "BA", "MV", "LZ", "LV",
  "BW", "HB", "IW", "RZ", "WB", "WD", "PF", "AG", "AA",
] as const;

export function centreById(id: string): Centre | undefined {
  return CENTRES.find((c) => c.id === id);
}

export function centresByGroup(): Array<{ group: string; centres: Centre[] }> {
  return GROUP_ORDER.map((g) => ({ group: g, centres: CENTRES.filter((c) => c.group === g) }))
    .filter((x) => x.centres.length > 0);
}

/** Auswahl von Zentren zu Kategorien, in der Reihenfolge der Referenzliste. */
export function centresToCategories(ids: readonly string[]): Category[] {
  const want = new Set(ids);
  return CENTRES.filter((c) => want.has(c.id)).map((c) => ({ id: c.id, name: c.name, color: c.color, group: c.group }));
}

/** Engster Zeitraum, der alle gewählten Zentren umfasst, auf 10 Jahre gerundet. */
export function suggestedPeriod(ids: readonly string[]): { start: number; end: number } | null {
  const cs = CENTRES.filter((c) => ids.includes(c.id));
  if (!cs.length) return null;
  const start = Math.min(...cs.map((c) => c.from)), end = Math.max(...cs.map((c) => c.to));
  return { start: Math.floor(start / 10) * 10, end: Math.ceil(end / 10) * 10 };
}
