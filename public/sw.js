/*
 * Service Worker (§13) — macht STOCHASI offline-/installierbar.
 *
 * Strategie ohne Build-Plugin (Vite-Assets sind gehasht, also nicht vorab
 * namentlich bekannt):
 *   - App-Shell (Startseite, Manifest, Icons) beim Install vorab cachen.
 *   - Navigationen: Network-first mit Rückfall auf die gecachte Startseite
 *     (SPA funktioniert offline).
 *   - Sonstige GET (gehashte JS/CSS, Schriften): stale-while-revalidate —
 *     sofort aus dem Cache, im Hintergrund aktualisieren.
 * Bei jedem Deploy CACHE hochzählen; alte Caches werden beim Activate entfernt.
 *
 * Alle Pfade sind RELATIV und werden gegen den Ort dieses Skripts aufgelöst.
 * Damit funktioniert der Worker gleichermaßen unter / (eigene Domain) wie unter
 * einem Unterpfad (GitHub Pages liefert unter /STOCHASI/ aus). Absolute Pfade
 * würden dort auf die Domainwurzel zeigen und der Install scheiterte.
 */
const CACHE = "stochasi-v2.0.0";
const BASE = new URL("./", self.location).href;
const SHELL = ["./", "./index.html", "./manifest.webmanifest", "./favicon.svg", "./icon-192.png", "./icon-512.png", "./icon-maskable-512.png"]
  .map((p) => new URL(p, BASE).href);
const INDEX = new URL("./index.html", BASE).href;

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()).catch(() => {}));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;

  // Navigationen: erst Netz, dann gecachte Shell (offline lauffähig).
  if (req.mode === "navigate") {
    e.respondWith(
      fetch(req).then((res) => { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(INDEX, copy)).catch(() => {}); return res; })
        .catch(() => caches.match(INDEX).then((r) => r || caches.match(BASE))),
    );
    return;
  }

  // Übrige GET: stale-while-revalidate.
  e.respondWith(
    caches.match(req).then((cached) => {
      const network = fetch(req).then((res) => {
        if (res && (res.ok || res.type === "opaque")) { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(req, copy)).catch(() => {}); }
        return res;
      }).catch(() => cached);
      return cached || network;
    }),
  );
});
