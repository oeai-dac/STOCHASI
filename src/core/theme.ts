/**
 * Theme-Verwaltung (Dark/Light).
 *
 * Einzige Farbquelle bleibt `theme.css`. Dieses Modul setzt nur das
 * `data-theme`-Attribut am `<html>`, persistiert die Wahl und liest die
 * Diagrammfarben aus den CSS-Tokens aus, damit die Diagramme beim Umschalten
 * mitziehen. Framework-frei.
 */

export type Theme = "light" | "dark";
const KEY = "stochasi.theme";
export const THEME_EVENT = "stochasi:theme";

/** Gespeicherte Wahl, sonst Systemeinstellung (prefers-color-scheme). */
export function getInitialTheme(): Theme {
  try {
    const s = localStorage.getItem(KEY);
    if (s === "light" || s === "dark") return s;
  } catch { /* localStorage kann fehlen (Privatmodus) */ }
  const m = typeof matchMedia === "function" && matchMedia("(prefers-color-scheme: dark)").matches;
  return m ? "dark" : "light";
}

export function currentTheme(): Theme {
  const t = document.documentElement.getAttribute("data-theme");
  return t === "dark" ? "dark" : "light";
}

/** Setzt das Theme (Attribut + Persistenz) und benachrichtigt Abonnenten. */
export function applyTheme(t: Theme): void {
  document.documentElement.setAttribute("data-theme", t);
  try { localStorage.setItem(KEY, t); } catch { /* ignorieren */ }
  window.dispatchEvent(new CustomEvent<Theme>(THEME_EVENT, { detail: t }));
}

export function toggleTheme(): Theme {
  const next: Theme = currentTheme() === "dark" ? "light" : "dark";
  applyTheme(next);
  return next;
}

/** Beim App-Start aufrufen (vor dem Rendern), verhindert ein helles Aufblitzen. */
export function initTheme(): Theme {
  const t = getInitialTheme();
  document.documentElement.setAttribute("data-theme", t);
  return t;
}

/** Abonniert Theme-Wechsel; liefert eine Abmeldefunktion. */
export function onThemeChange(cb: (t: Theme) => void): () => void {
  const h = (e: Event) => cb((e as CustomEvent<Theme>).detail);
  window.addEventListener(THEME_EVENT, h);
  return () => window.removeEventListener(THEME_EVENT, h);
}

/* ── CSS-Token → RGB ── */
/** Wandelt eine CSS-Farbe (`#rgb`, `#rrggbb` oder `rgb()/rgba()`) in 0..1-RGB. Exportiert für Tests. */
export function toRgb01(css: string): [number, number, number] {
  const s = css.trim();
  const hex = s.match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i);
  if (hex) {
    const h = hex[1].length === 3 ? hex[1].split("").map((c) => c + c).join("") : hex[1];
    return [parseInt(h.slice(0, 2), 16) / 255, parseInt(h.slice(2, 4), 16) / 255, parseInt(h.slice(4, 6), 16) / 255];
  }
  const m = s.match(/rgba?\(([^)]+)\)/i);
  if (m) {
    const p = m[1].split(/[,\s/]+/).map(Number);
    return [(p[0] || 0) / 255, (p[1] || 0) / 255, (p[2] || 0) / 255];
  }
  return [0.965, 0.957, 0.949];
}

/** Farben der Diagramme, aus den CSS-Tokens des `<html>` gelesen. */
export function readPlotColors(): { text: string; label: string; dim: string; grid: string; active: string } {
  const cs = getComputedStyle(document.documentElement);
  const v = (name: string, fallback: string) => (cs.getPropertyValue(name).trim() || fallback);
  return {
    text: v("--tx", "#1c1b1a"),
    label: v("--tx2", "#5d584f"),
    dim: v("--tx3", "#8b857c"),
    grid: v("--bd", "#dcd7d1"),
    active: v("--accent2", "#a81d26"),
  };
}
