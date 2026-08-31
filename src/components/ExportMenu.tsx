/**
 * Exportmenü. Abbildung und Zahlen sind bewusst nebeneinander erreichbar: Wer
 * eine Abbildung exportiert, soll die zugehörige Tabelle mit einem zweiten
 * Klick danebenlegen können.
 */
import { useEffect, useRef, useState } from "react";
import type { ProjectV2 } from "../core/model.js";
import type { Scene } from "../charts/scene.js";
import { sceneToPDF, sceneToPNG, sceneToSVG } from "../charts/scene.js";
import { LIGHT } from "../charts/plot.js";
import { downloadBytes, downloadText, downloadBlob, safeFilename } from "../export/download.js";
import { toCSVForDownload, toXLSX, type Aoa } from "../export/exportTable.js";
import { writeProject } from "../core/io/project.js";
import { useT } from "../i18n/I18nContext.js";

export interface ExportPayload {
  /** Szene der aktuellen Ansicht — im hellen Schema, damit der Export druckbar ist. */
  scene: Scene | null;
  /** Kurzname der Ansicht für den Dateinamen. */
  viewName: string;
  /** Tabellen der aktuellen Ansicht. */
  sheets: Array<{ name: string; aoa: Aoa }>;
}

export function ExportMenu({ project, payload, onToast }: {
  project: ProjectV2; payload: ExportPayload; onToast: (msg: string) => void;
}) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => { if (!box.current?.contains(e.target as Node)) setOpen(false); };
    const onEsc = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onEsc);
    return () => { document.removeEventListener("mousedown", onDoc); document.removeEventListener("keydown", onEsc); };
  }, [open]);

  const base = safeFilename(project.name) + "_" + payload.viewName;

  async function run(kind: string) {
    setOpen(false);
    try {
      switch (kind) {
        case "png": {
          if (!payload.scene) return;
          downloadBlob(`${base}.png`, await sceneToPNG(payload.scene, 2));
          onToast(t("export.done", { file: `${base}.png` })); break;
        }
        case "svg": {
          if (!payload.scene) return;
          downloadText(`${base}.svg`, sceneToSVG(payload.scene), "image/svg+xml");
          onToast(t("export.done", { file: `${base}.svg` })); break;
        }
        case "pdf": {
          if (!payload.scene) return;
          downloadBytes(`${base}.pdf`, sceneToPDF(payload.scene), "application/pdf");
          onToast(t("export.done", { file: `${base}.pdf` })); break;
        }
        case "csv": {
          const s = payload.sheets[0]; if (!s) return;
          downloadText(`${base}.csv`, toCSVForDownload(s.aoa), "text/csv", false);
          onToast(t("export.done", { file: `${base}.csv` })); break;
        }
        case "xlsx": {
          if (!payload.sheets.length) return;
          downloadBytes(`${base}.xlsx`, await toXLSX(payload.sheets),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet");
          onToast(t("export.done", { file: `${base}.xlsx` })); break;
        }
        case "json": {
          const f = safeFilename(project.name) + ".stochasi.json";
          downloadText(f, writeProject(project), "application/json");
          onToast(t("export.done", { file: f })); break;
        }
      }
    } catch (e) {
      onToast(t("export.failed", { msg: e instanceof Error ? e.message : String(e) }));
    }
  }

  const item = (kind: string, label: string, sub: string, disabled = false) => (
    <button className="exp-item" onClick={() => run(kind)} disabled={disabled}>
      <strong>{label}</strong><span>{sub}</span>
    </button>
  );

  return (
    <div className="exp" ref={box}>
      <button className="sl" aria-expanded={open} aria-haspopup="menu" onClick={() => setOpen((o) => !o)}>
        <span aria-hidden="true">⤓</span> {t("export.title")}
      </button>
      {open && (
        <div className="exp-menu" role="menu">
          {item("png", t("export.png"), t("export.pngSub"), !payload.scene)}
          {item("svg", t("export.svg"), t("export.svgSub"), !payload.scene)}
          {item("pdf", t("export.pdf"), t("export.pdfSub"), !payload.scene)}
          <hr />
          {item("csv", t("export.csv"), t("export.csvSub"), !payload.sheets.length)}
          {item("xlsx", t("export.xlsx"), t("export.xlsxSub"), !payload.sheets.length)}
          <hr />
          {item("json", t("export.json"), t("export.jsonSub"))}
        </div>
      )}
    </div>
  );
}

/** Das helle Schema für Exporte — eine Abbildung im Dunkelmodus ist im Druck unbrauchbar. */
export const EXPORT_THEME = LIGHT;
