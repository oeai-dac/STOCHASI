/**
 * Beispieldatensatz beim Start: Terra Sigillata aus Insula XLI von Flavia Solva.
 *
 * Es sind echte Zahlen — dieselben, mit denen STOCHASI 1 ausgeliefert wurde
 * (`config/config_f-solva_insula-xli.json`). Ein erfundener Datensatz hätte den
 * Nachteil, dass man an ihm nicht sieht, wie das Modell mit den Unregelmäßigkeiten
 * echten Fundmaterials umgeht: 77 Scherben sind wenig, und die zehn italischen
 * Stücke passen nicht zum Modell — genau das soll man beim ersten Start sehen.
 *
 * Die Marktkurve ist die aus Version 1 übernommene Arbeitsannahme des Autors und
 * gilt für diesen Fundort. Sie ist keine allgemeine Lieferkurve.
 *
 * Erzeugt aus der v1-Konfiguration; bei Änderungen an der Datei neu erzeugen.
 */
import type { ProjectV2 } from "../core/model.js";

const DATA = {
  "categories": [
    {
      "id": "IT",
      "name": "Italian",
      "color": "#8B0000",
      "group": "Italisch"
    },
    {
      "id": "LG",
      "name": "La Graufesenque",
      "color": "#FF6666",
      "group": "Südgallisch"
    },
    {
      "id": "BA",
      "name": "Banassac",
      "color": "#1E90FF",
      "group": "Südgallisch"
    },
    {
      "id": "MG",
      "name": "Central Gaulish",
      "color": "#228B22",
      "group": "Mittelgallisch"
    },
    {
      "id": "RZ",
      "name": "Rheinzabern",
      "color": "#FF8C00",
      "group": "Ostgallisch"
    }
  ],
  "market": {
    "years": [
      120,
      130,
      140,
      150,
      160,
      170,
      180,
      190,
      200,
      210,
      220,
      230,
      240,
      250,
      260,
      270,
      280,
      290
    ],
    "shares": {
      "IT": [
        30,
        10,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      "LG": [
        20,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      "BA": [
        50,
        60,
        10,
        10,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      "MG": [
        0,
        30,
        80,
        80,
        80,
        80,
        50,
        10,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      "RZ": [
        0,
        0,
        10,
        10,
        20,
        20,
        50,
        90,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100
      ]
    }
  },
  "params": {
    "startYear": 125,
    "endYear": 290,
    "replacement": {},
    "replacementDefault": 0.1,
    "noiseSd": 2,
    "runs": 100,
    "seed": 0,
    "settlementMode": false,
    "residual": 0,
    "initial": {
      "IT": 60,
      "LG": 10,
      "BA": 30,
      "MG": 0,
      "RZ": 0
    }
  },
  "assemblages": [
    {
      "id": "A1",
      "name": "Insula XLI",
      "counts": {
        "IT": 10,
        "LG": 0,
        "BA": 1,
        "MG": 56,
        "RZ": 10
      }
    }
  ],
  "comparisonYear": 170
} as const;

/** Neue Instanz — der Datensatz wird in der Oberfläche bearbeitet. */
export function DEMO(): ProjectV2 {
  return {
    _meta: { app: "STOCHASI", version: "2.0", created: new Date().toISOString() },
    name: "Flavia-Solva-Beispiel",
    ...structuredClone(DATA as unknown as Omit<ProjectV2, "_meta" | "name">),
  };
}
