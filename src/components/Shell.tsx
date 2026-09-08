import { useState } from "react";
import type { ProjectV2 } from "../core/model.js";
import { currentTheme, toggleTheme, type Theme } from "../core/theme.js";
import { useI18n, useT } from "../i18n/I18nContext.js";
import type { Lang } from "../i18n/i18n.js";

function ThemeToggle() {
  const t = useT();
  const [theme, setTheme] = useState<Theme>(currentTheme());
  const dark = theme === "dark";
  return (
    <button className="sl theme-toggle" onClick={() => setTheme(toggleTheme())}
      aria-label={dark ? t("header.theme.toLight") : t("header.theme.toDark")}
      title={dark ? t("header.theme.toLight") : t("header.theme.toDark")} aria-pressed={dark}>
      <span aria-hidden="true">{dark ? "☀" : "☾"}</span>
    </button>
  );
}

function LangToggle() {
  const { lang, setLang, t } = useI18n();
  const next: Lang = lang === "de" ? "en" : "de";
  return (
    <button className="sl lang-toggle" onClick={() => setLang(next)}
      aria-label={next === "en" ? t("header.lang.toEnglish") : t("header.lang.toGerman")} title={t("header.lang.label")}>
      <span aria-hidden="true">{lang.toUpperCase()}</span>
    </button>
  );
}

/* Das Zeichen ist der Signaturplot: Mittelwertkurve mit Unsicherheitsband. */
const LOGO = (
  <svg viewBox="0 0 40 40" aria-hidden="true">
    <path d="M5.40,25.19 L6.13,24.94 L6.85,24.62 L7.58,24.22 L8.30,23.72 L9.03,23.13 L9.75,22.44 L10.48,21.65 L11.20,20.76 L11.93,19.79 L12.65,18.75 L13.38,17.67 L14.10,16.58 L14.83,15.51 L15.55,14.50 L16.28,13.58 L17.00,12.79 L17.73,12.15 L18.45,11.69 L19.18,11.41 L19.90,11.33 L20.63,11.44 L21.35,11.74 L22.08,12.21 L22.80,12.85 L23.53,13.62 L24.25,14.51 L24.98,15.49 L25.70,16.53 L26.43,17.60 L27.15,18.68 L27.88,19.73 L28.60,20.72 L29.33,21.65 L30.05,22.48 L30.78,23.21 L31.50,23.84 L32.23,24.36 L32.95,24.78 L33.68,25.11 L34.40,25.36 L34.40,29.20 L33.68,29.09 L32.95,28.94 L32.23,28.74 L31.50,28.48 L30.78,28.15 L30.05,27.74 L29.33,27.24 L28.60,26.65 L27.88,25.96 L27.15,25.19 L26.43,24.35 L25.70,23.46 L24.98,22.55 L24.25,21.66 L23.53,20.83 L22.80,20.10 L22.08,19.51 L21.35,19.08 L20.63,18.83 L19.90,18.76 L19.18,18.87 L18.45,19.15 L17.73,19.60 L17.00,20.19 L16.28,20.91 L15.55,21.72 L14.83,22.58 L14.10,23.46 L13.38,24.33 L12.65,25.16 L11.93,25.94 L11.20,26.65 L10.48,27.29 L9.75,27.85 L9.03,28.33 L8.30,28.74 L7.58,29.09 L6.85,29.36 L6.13,29.58 L5.40,29.74 Z" fill="#e48d91" />
    <path d="M5.40,27.46 L7.58,26.66 L9.75,24.99 L11.93,22.66 L14.10,20.02 L16.28,17.53 L18.45,15.72 L19.90,15.24 L21.35,15.56 L23.53,17.06 L25.70,19.44 L27.88,22.19 L30.05,24.65 L32.23,26.42 L34.40,27.38" fill="none" stroke="#d22630" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

export type TabId = "market" | "sim" | "year" | "compare" | "dating" | "data";
const TABS: Array<{ id: TabId; labelKey: string; icon: string }> = [
  { id: "market", labelKey: "tab.market", icon: "⌁" },
  { id: "sim", labelKey: "tab.sim", icon: "∿" },
  { id: "year", labelKey: "tab.year", icon: "▤" },
  { id: "compare", labelKey: "tab.compare", icon: "⇄" },
  { id: "dating", labelKey: "tab.dating", icon: "⌖" },
  { id: "data", labelKey: "tab.data", icon: "▦" },
];

/** Kurzanleitung auf GitHub, je Sprache. Im Desktop-Gehäuse öffnet sie der Browser. */
const GUIDE_URL: Record<Lang, string> = {
  de: "https://github.com/oeai-dac/STOCHASI/blob/main/docs/QUICKSTART.de.md",
  en: "https://github.com/oeai-dac/STOCHASI/blob/main/docs/QUICKSTART.md",
};

export function Shell({ project, tab, onTab, onPickFile, onShare, exportMenu }: {
  project: ProjectV2 | null; tab: TabId; onTab: (t: TabId) => void;
  onPickFile: () => void; onShare?: () => void; exportMenu?: React.ReactNode;
}) {
  const { t, lang } = useI18n();
  return (
    <>
      <header className="hdr">
        <div className="hdr-logo">{LOGO}</div>
        <div className="hdr-title">
          <h1>STOCHASI{project ? ` · ${project.name}` : ""}</h1>
          <p>{t("app.subtitle")}</p>
        </div>
        <span className="badge p2">v2.1</span>
        {project && <span className="badge">{t("cat.count", { n: project.categories.length })}</span>}
        {exportMenu}
        {project && onShare && (
          <button className="sl share-btn" onClick={onShare} title={t("share.title")} aria-label={t("share.button")}>
            <span aria-hidden="true">⇗</span> {t("share.button")}
          </button>
        )}
        <a className="sl" href={GUIDE_URL[lang]} target="_blank" rel="noreferrer"
          title={t("header.guideTitle")}>
          <span aria-hidden="true">?</span> {t("header.guide")}
        </a>
        <LangToggle />
        <ThemeToggle />
        <button className="file-btn" onClick={onPickFile}>{t("header.loadFile")}</button>
      </header>
      <nav className="tabs" role="tablist" aria-label={t("a11y.tabsLabel")}>
        {TABS.map((tb) => (
          <button key={tb.id} id={`tab-${tb.id}`} className={"tab" + (tab === tb.id ? " on" : "")}
            onClick={() => onTab(tb.id)} role="tab" aria-selected={tab === tb.id} aria-controls="main"
            tabIndex={tab === tb.id ? 0 : -1}
            onKeyDown={(e) => {
              const i = TABS.findIndex((x) => x.id === tab);
              let ni = -1;
              if (e.key === "ArrowRight") ni = (i + 1) % TABS.length;
              else if (e.key === "ArrowLeft") ni = (i - 1 + TABS.length) % TABS.length;
              else if (e.key === "Home") ni = 0;
              else if (e.key === "End") ni = TABS.length - 1;
              if (ni >= 0) { e.preventDefault(); const id = TABS[ni].id; onTab(id); requestAnimationFrame(() => document.getElementById(`tab-${id}`)?.focus()); }
            }}>
            <span aria-hidden="true">{tb.icon}</span> {t(tb.labelKey)}
            {tb.id === "data" && project && <span className="tab-cnt">{project.assemblages.length}</span>}
          </button>
        ))}
      </nav>
    </>
  );
}
