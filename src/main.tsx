import "./fonts.css"; // Lokal eingebettete Schriften — kein Aufruf an einen Fremdserver
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App.js";
import { initTheme } from "./core/theme.js";
import { I18nProvider } from "./i18n/I18nContext.js";

initTheme(); // Hell/Dunkel setzen, bevor gerendert wird (kein Aufblitzen)

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <I18nProvider>
      <App />
    </I18nProvider>
  </StrictMode>,
);

// Service Worker nur im Produktionsbau registrieren (offline- und installierbar).
// Entfällt in der Desktop-Fassung: dort liegt die App bereits lokal, und unter
// file:// stehen Service Worker ohnehin nicht zur Verfügung.
const isProd = (import.meta as { env?: { PROD?: boolean } }).env?.PROD === true;
const isDesktop = location.protocol === "file:" || Boolean((window as { stochasiDesktop?: unknown }).stochasiDesktop);
if (isProd && !isDesktop && "serviceWorker" in navigator) {
  const swUrl = new URL("sw.js", document.baseURI).href;
  window.addEventListener("load", () => {
    navigator.serviceWorker.register(swUrl).catch(() => { /* PWA ist optional */ });
  });
}
