/**
 * React-Anbindung des i18n-Kerns. Hält die aktuelle Sprache, rendert den
 * Teilbaum bei Sprachwechsel neu und stellt `t()` sowie `setLang` bereit.
 */
import { createContext, useContext, useEffect, useMemo, useState, useCallback, type ReactNode } from "react";
import { translate, getInitialLang, setLang as persistLang, onLangChange, type Lang } from "./i18n.js";

type T = (key: string, vars?: Record<string, string | number>) => string;
interface I18nApi { lang: Lang; t: T; setLang: (l: Lang) => void; }

const Ctx = createContext<I18nApi | null>(null);

export function I18nProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(getInitialLang);

  // Auch auf externe Wechsel (anderer Tab/Aufrufer) reagieren.
  useEffect(() => onLangChange((l) => setLangState(l)), []);
  useEffect(() => { document.documentElement.setAttribute("lang", lang); }, [lang]);

  const setLang = useCallback((l: Lang) => { persistLang(l); setLangState(l); }, []);
  const t = useCallback<T>((key, vars) => translate(lang, key, vars), [lang]);
  const api = useMemo<I18nApi>(() => ({ lang, t, setLang }), [lang, t, setLang]);
  return <Ctx.Provider value={api}>{children}</Ctx.Provider>;
}

export function useI18n(): I18nApi {
  const c = useContext(Ctx);
  if (!c) throw new Error("useI18n muss innerhalb von <I18nProvider> stehen");
  return c;
}

/** Kurzform: nur die Übersetzungsfunktion. */
export function useT(): T { return useI18n().t; }
