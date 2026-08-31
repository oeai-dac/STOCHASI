# Bauen und Veröffentlichen

Für Mitwirkende. Wer STOCHASI nur benutzen möchte, findet in
[INSTALLATION.md](INSTALLATION.md) das Passende.

## Voraussetzungen

Node.js 20 oder neuer. Sonst nichts — die Fachlogik kommt ohne Fremdbibliotheken
aus, und die Werkzeuge für den Paketbau lädt `electron-builder` selbst nach.

```bash
npm ci
```

> **Hinweis zu npm 12 und neuer:** npm blockiert Installations-Skripte von
> Abhängigkeiten inzwischen standardmäßig, und bei `npm ci` greift die
> `allowScripts`-Freigabe aus der `package.json` nicht zuverlässig. Electron
> käme dann ohne seine Programmdatei an. Deshalb holt das `postinstall`-Skript
> des Projekts (`scripts/ensure-electron.mjs`) den Download bei Bedarf nach.
> Sollte er einmal fehlschlagen: `npm run ensure-electron`.

## Alltag

```bash
npm run dev        # Entwicklungsserver im Browser
npm run build      # Typprüfung + Produktions-Build nach dist/
npm test           # gesamte Testsuite (453 Prüfungen, framework-frei)
npm run electron   # Build + Start im Desktop-Gehäuse
npm run smoke      # Selbsttest der Desktop-Fassung
```

### Der Selbsttest

`npm run smoke` startet ein unsichtbares Electron-Fenster und prüft genau die
Eigenschaften, die beim Wechsel vom Browser ins Gehäuse brechen können. Die
wichtigste davon: Die App erzeugt ihre Rechen-Worker als **Modul-Worker**
(`new Worker(url, { type: "module" })`). Unter `file://` blockiert Chromium
deren Laden, womit Simulation und inverse Datierung in den Haupt-Thread
zurückfielen und die Oberfläche bei jedem Reglerzug einfröre — ohne sichtbare
Fehlermeldung.

Deshalb lädt der Hauptprozess die Oberfläche **nicht** über `file://`, sondern
über ein eigenes, als sicher registriertes Schema `app://` (siehe
`electron/main.js`). Das liefert zusätzlich einen stabilen Origin und damit
verlässliches IndexedDB (Autosave), localStorage (Theme, Sprache) und einen
sicheren Kontext für `CompressionStream` (Teilen-Link).

## Installationspakete bauen

```bash
npm run dist          # für das aktuelle Betriebssystem
npm run dist:linux    # AppImage, deb, rpm
npm run dist:win      # NSIS-Installer, portable .exe
npm run dist:mac      # dmg für Apple Silicon (arm64); Intel wird nicht ausgeliefert
```

Die Ergebnisse liegen in `release/`.

Das deb ist gegen **Ubuntu 22.04 und neuer** gerichtet und dort lauffähig:

- Die Electron-Binärdateien verlangen höchstens **GLIBC 2.25** und keine
  `GLIBCXX`-Symbole (libstdc++ ist statisch gebunden). Ubuntu 22.04 liefert
  glibc 2.35.
- Alle Abhängigkeiten des Pakets sind unter 22.04, 24.04 und 25.04 auflösbar —
  teils über die `t64`-Nachfolgepakete, die den alten Namen weiterhin
  bereitstellen.
- Das `postinst` unterscheidet die Fälle selbst: `chrome-sandbox` erhält das
  setuid-Bit nur auf Systemen ohne User-Namespaces, und das mitgelieferte
  AppArmor-Profil wird nur dort eingespielt, wo AppArmor `abi/4.0` versteht
  (Ubuntu 24.04 und neuer). Ohne dieses Profil starten Electron-Anwendungen
  unter Ubuntu 24.04 nicht, weil dort unprivilegierte User-Namespaces
  eingeschränkt sind.

Ein Betriebssystem kann jeweils nur seine eigenen Pakete zuverlässig bauen.
Deshalb erledigt das die CI auf drei Runnern gleichzeitig; lokal ist der Bau
vor allem zum Prüfen gedacht.

### Werkzeuge für deb und rpm

Beide entstehen über `fpm`. Vorausgesetzt werden:

| Ziel | Bedarf | Debian/Ubuntu | Arch |
|---|---|---|---|
| `deb` | `libcrypt.so.1` für das Ruby von `fpm` | vorhanden | `sudo pacman -S libxcrypt-compat` |
| `rpm` | `rpmbuild` | `sudo apt install rpm` | `sudo pacman -S rpm-tools` |

Das AppImage braucht keines von beidem.

Auf Arch fehlt `libcrypt.so.1`, weil dort nur `libcrypt.so.2` ausgeliefert wird;
die Symbolversionen sind nicht austauschbar, ein Symlink hilft also nicht. Auf
dem Ubuntu-Runner der CI stellt sich nur die `rpmbuild`-Frage — dafür enthält
`.github/workflows/release.yml` einen eigenen Installationsschritt.

## Zur Code-Signierung

Die Pakete werden **bewusst nicht** mit einem Zertifikat signiert — weder bei
Apple noch bei Microsoft. Die Folgen für Nutzer sind in
[INSTALLATION.md](INSTALLATION.md) beschrieben.

Eine Feinheit ist dennoch nötig, und sie lässt sich in der `package.json` nicht
kommentieren, weshalb sie hier steht:

```json
"mac": { "identity": "-", "hardenedRuntime": false }
```

- **`identity: "-"`** erzwingt eine **Ad-hoc-Signatur**. Das ist nicht dasselbe
  wie „nicht signieren": Wäre `identity` auf `null` gesetzt, unterbliebe die
  Signierung vollständig — und eine gänzlich unsignierte Anwendung **startet auf
  Apple Silicon überhaupt nicht**, macOS beendet sie sofort. Die Ad-hoc-Signatur
  macht die App lauffähig, hebt die Gatekeeper-Warnung aber nicht auf.
- **`hardenedRuntime: false`** ist bei Ad-hoc-Signatur zwingend. Andernfalls
  verwirft die Library-Validierung das vorsignierte Electron-Framework, weil es
  eine andere Team-ID trägt, und die App stürzt beim Start ab.

Sobald Zertifikate vorliegen, genügt es, die Secrets in der CI zu hinterlegen
(`CSC_LINK`, `CSC_KEY_PASSWORD` für Windows und macOS; `APPLE_ID`,
`APPLE_APP_SPECIFIC_PASSWORD`, `APPLE_TEAM_ID` für die Notarisierung) und in
`.github/workflows/release.yml` `CSC_IDENTITY_AUTO_DISCOVERY` zu entfernen sowie
`mac.identity` auf den Zertifikatsnamen zu setzen.

## Veröffentlichen

```bash
npm version 2.0.1        # hebt package.json und legt den Tag v2.0.1 an
git push --follow-tags
```

Der Tag löst `.github/workflows/release.yml` aus. Der Ablauf baut auf drei
Betriebssystemen, hängt alle Dateien an einen **Release-Entwurf** und stoppt
dort. Prüfen Sie die Dateien, ergänzen Sie den Begleittext und veröffentlichen
Sie das Release erst dann von Hand.

Ein Push auf `main` aktualisiert außerdem die Web-Fassung auf GitHub Pages
(`.github/workflows/pages.yml`).

## Schriften

Cormorant Garamond, JetBrains Mono und Outfit liegen als WOFF2 unter
`src/fonts/` im Repository und werden mitgeliefert — die App ruft beim Start
keinen Fremdserver auf. Sollen Schnitte oder Stärken geändert werden:

```bash
npm run vendor-fonts
```

Das Skript lädt sie neu und schreibt `src/fonts.css`. Alle drei Familien stehen
unter der SIL Open Font License 1.1; ihre Weitergabe im Bundle ist zulässig.
