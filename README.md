# STOCHASI 2

**English** · [Deutsch ↓](#stochasi-2--deutsch)

Stochastic simulation of archaeological find spectra, and inverse dating of
assemblages from them — as a desktop application for Windows, macOS and Linux,
and as a web application that runs entirely on your own machine.

The working cycle: **market supply → simulated stock in circulation → compare
with a counted assemblage → date it back → export**. React + TypeScript + Vite
in the ÖAI design, with charts written from scratch so that screen and export
show the same figure.

MIT licence · © Christian Gugl / Austrian Archaeological Institute (ÖAW)

Developed with the assistance of Anthropic's Claude, in Claude Code: large parts
of the application and of its documentation were produced with its help. The
direction of the work, the archaeological and methodological decisions, the
testing and the responsibility for the result lie with the author.

---

## Download

**Try it without installing:** <https://oeai-dac.github.io/STOCHASI/>

**Install:** the finished packages are on the
[**Releases**](https://github.com/oeai-dac/STOCHASI/releases) page.

| Your system | File |
|---|---|
| Windows 10/11 | `STOCHASI-*-Setup-x64.exe` — or `*-portable-x64.exe`, no installation |
| macOS (Apple M1–M4) | `STOCHASI-*-arm64.dmg` |
| Linux, any distribution | `STOCHASI-*-x86_64.AppImage` |
| Ubuntu 22.04+, Debian 12+, Mint | `stochasi_*_amd64.deb` |
| Fedora, openSUSE, RHEL | `stochasi-*.x86_64.rpm` |

The packages are not digitally signed, so Windows and macOS show a warning on
first launch. **[docs/INSTALLATION.md](docs/INSTALLATION.md)** walks through it
step by step — no technical background required.

## Documentation

| Document | What it covers |
|---|---|
| [**Quick Start Guide**](docs/QUICKSTART.md) | Getting STOCHASI, the model in three paragraphs, the six views, a first analysis. Ten minutes. |
| [**Complete Guide**](docs/GUIDE.md) | The full reference: the mathematics of the forward model and of the Dirichlet-multinomial dating, the residuality model, exact file formats, and a frank account of the method's limits. |
| [**Installation**](docs/INSTALLATION.md) | Step-by-step installation per platform, security warnings, uninstalling *(German)*. |
| [**Build**](docs/BUILD.md) | Building packages, release process, code signing *(German)*. |

---

## What it does

The model separates two things that are routinely conflated. **Market supply**
is what arrives at a site in a given year. **The stock in circulation** is what
is actually in use there — pottery bought in 140 is still on the shelf in 155.
STOCHASI moves the stock forward year by year:

```
stock(t) = stock(t−1) · (1 − r) + market(t) · r + noise
```

where `r` is the annual replacement rate. The lag between the two curves is
exactly the size of the dating error you make by treating them as the same
thing.

Because excavation samples the stock, a counted assemblage should resemble the
simulated stock of the year its context was deposited. That gives the inverse
question: which year makes those counts most probable? STOCHASI answers it with
a probability distribution over the years, in which the sample size is properly
accounted for — 77 sherds and 770 sherds give visibly different intervals.

## New in version 2

- **Inverse dating.** From a counted assemblage to a probability curve over the
  years, with a 95 % interval. Version 1 could only run forwards and let you
  compare by eye.
- **Residuality.** Redeposited older material as a model parameter, and — more
  useful — a scan across several residual shares that shows whether your dating
  depends on that unknown at all.
- **Replacement rates per category.** Different wares leave circulation at
  different speeds.
- **Several assemblages at once**, compared in one table and, in the ranking
  view, sorted by estimated date with interval bars. Where a single year is
  consistent with every assemblage, that range is shaded — so a sorted list
  cannot pass itself off as a sequence the data do not support.
- **36 production centres** as a reference list, grouped by region with rough
  production periods — Montans, Trier, Sinzig, La Madeleine,
  Chémery-Faulquemont, the Pannonian and local wares, African Red Slip and the
  eastern sigillata families among them.
- **Desktop application** for Windows, macOS and Linux, plus an installable web
  version. Version 1 needed Python, a virtual environment and a terminal.
- **Bilingual interface**, light and dark theme, ÖAI design.

Version 1 configuration files load directly and are migrated. With residuality
at 0 and no per-category rates, version 2 reproduces version 1 exactly.

**Version 1 remains available** on the
[`v1` branch](https://github.com/oeai-dac/STOCHASI/tree/v1).

## Principles

- **Everything stays local.** There is no backend. Data is never uploaded; the
  app is installable and works offline after the first visit. The desktop
  edition goes further: a content-security policy forbids any outbound
  connection there, and even the fonts are bundled rather than loaded from a
  CDN.
- **No third-party libraries for the domain logic.** The Monte Carlo core, the
  Dirichlet-multinomial likelihood, the log-gamma function, the charts, the
  SVG/PNG/PDF writers, internationalisation and icon generation are written
  in-house. The only runtime dependencies are React and — loaded only if XLSX is
  actually used — SheetJS.
- **No invented data.** The reference list of production centres supplies names,
  colours and rough production periods. It deliberately supplies **no market
  shares**: those are regional, and inventing them would be worse than useless.
  The supply curve has to come from the literature on your site.
- **Limits are named, not glossed over.** The dating is conditional on the
  supply curve, the replacement rate and the residual share, and the interface
  says so where you read the result. [§14 of the Complete
  Guide](docs/GUIDE.md#14-limits-of-the-method) lists what the method cannot do.

## Building from source

```bash
npm ci
npm run dev        # development server (Vite)
npm run build      # type-check + production build into dist/
npm run preview    # view the build locally
npm test           # the full test suite
npm run electron   # build and launch inside the desktop shell
npm run smoke      # self-test of the desktop edition
npm run dist       # installer packages for the current system
npm run gen-icons  # regenerate icons (PWA + package icon)
```

Details on packaging, releasing and code signing:
**[docs/BUILD.md](docs/BUILD.md)**.

A real example dataset is loaded at startup — the terra sigillata from Insula
XLI at Flavia Solva, 77 sherds. Load your own files by **drag and drop** or via
"Load file…": a version 1 project (`.json`, migrated automatically), a version 2
project (`.stochasi.json`), or a raw table (`.csv`/`.tsv`/`.xlsx`) holding
either a market curve or assemblage counts. Four files to try it with are in
[`examples/`](examples/).

---
---

# STOCHASI 2 — Deutsch

[English ↑](#stochasi-2)

Stochastische Simulation archäologischer Fundspektren und inverse Datierung von
Fundkomplexen — als Desktop-Anwendung für Windows, macOS und Linux und als
Web-Fassung, die vollständig auf dem eigenen Rechner läuft.

Der Arbeitsgang: **Marktangebot → simulierter Umlaufbestand → Vergleich mit
einem gezählten Fundkomplex → Rückdatierung → Export.** React + TypeScript +
Vite im ÖAI-Layout, mit eigenen Diagrammen, damit Bildschirm und Export
dieselbe Abbildung zeigen.

MIT-Lizenz · © Christian Gugl / Österreichisches Archäologisches Institut (ÖAW)

Entwickelt mit Unterstützung von Anthropics Claude in Claude Code: Große Teile
der Anwendung und ihrer Dokumentation sind mit dessen Hilfe entstanden. Die
Richtung der Arbeit, die archäologischen und methodischen Entscheidungen, die
Prüfung und die Verantwortung für das Ergebnis liegen beim Autor.

## Herunterladen

**Ohne Installation ausprobieren:** <https://oeai-dac.github.io/STOCHASI/>

**Installieren:** Die fertigen Pakete liegen auf der Seite
[**Releases**](https://github.com/oeai-dac/STOCHASI/releases).

| Ihr System | Datei |
|---|---|
| Windows 10/11 | `STOCHASI-*-Setup-x64.exe` — oder `*-portable-x64.exe`, ohne Installation |
| macOS (Apple M1–M4) | `STOCHASI-*-arm64.dmg` |
| Linux, beliebige Distribution | `STOCHASI-*-x86_64.AppImage` |
| Ubuntu 22.04+, Debian 12+, Mint | `stochasi_*_amd64.deb` |
| Fedora, openSUSE, RHEL | `stochasi-*.x86_64.rpm` |

Die Pakete sind nicht digital signiert; Windows und macOS zeigen beim ersten
Start deshalb eine Warnung. **[docs/INSTALLATION.md](docs/INSTALLATION.md)**
führt Schritt für Schritt hindurch.

## Dokumentation

| Dokument | Inhalt |
|---|---|
| [**Quick Start Guide**](docs/QUICKSTART.md) | Bezug, das Modell in drei Absätzen, die sechs Ansichten, eine erste Auswertung. Zehn Minuten. *(englisch)* |
| [**Complete Guide**](docs/GUIDE.md) | Die vollständige Referenz: die Mathematik des Vorwärtsmodells und der Dirichlet-Multinomial-Datierung, das Residualitätsmodell, die genauen Dateiformate und eine offene Darstellung der Grenzen. *(englisch)* |
| [**Installation**](docs/INSTALLATION.md) | Installation je Plattform, Sicherheitswarnungen, Deinstallation. |
| [**Bauen**](docs/BUILD.md) | Pakete bauen, Release-Ablauf, Signierung. |

## Was das Programm tut

Das Modell trennt zwei Dinge, die regelmäßig vermengt werden. Das
**Marktangebot** ist das, was in einem Jahr an einen Ort geliefert wird. Der
**Umlaufbestand** ist das, was dort tatsächlich in Gebrauch ist — Geschirr, das
140 gekauft wurde, steht 155 noch im Regal. STOCHASI schreibt den Bestand Jahr
für Jahr fort:

```
Bestand(t) = Bestand(t−1) · (1 − r) + Markt(t) · r + Rauschen
```

mit der jährlichen Ersatzrate `r`. Der Abstand zwischen beiden Kurven ist genau
der Datierungsfehler, den man macht, wenn man sie gleichsetzt.

Weil eine Grabung den Umlaufbestand beprobt, sollte ein gezählter Fundkomplex
dem simulierten Bestand des Ablagerungsjahres ähneln. Daraus wird die umgekehrte
Frage: Welches Jahr macht diese Zählung am wahrscheinlichsten? STOCHASI
beantwortet sie mit einer Wahrscheinlichkeitsverteilung über die Jahre, in der
der Stichprobenumfang richtig eingeht — 77 Scherben und 770 Scherben liefern
sichtbar verschiedene Intervalle.

## Neu in Version 2

- **Inverse Datierung.** Vom gezählten Fundkomplex zur Wahrscheinlichkeitskurve
  über die Jahre, mit 95-%-Intervall. Version 1 konnte nur vorwärts rechnen und
  dann per Auge vergleichen.
- **Residualität.** Umgelagertes Altmaterial als Modellparameter und — der
  eigentliche Ertrag — eine Schar über mehrere Residualanteile, die zeigt, ob
  die Datierung von dieser Unbekannten überhaupt abhängt.
- **Ersatzraten je Kategorie.** Verschiedene Warenarten scheiden verschieden
  schnell aus dem Umlauf aus.
- **Mehrere Fundkomplexe gleichzeitig**, verglichen in einer Tabelle und, in der
  Rangfolge, nach geschätztem Datum sortiert mit Intervallbalken. Passt ein
  einziges Jahr zu allen Komplexen, wird dieser Bereich hinterlegt — damit eine
  Sortierung sich nicht als Abfolge ausgeben kann, die die Daten nicht tragen.
- **36 Produktionszentren** als Referenzliste, nach Regionen gruppiert und mit
  groben Produktionszeiträumen — darunter Montans, Trier, Sinzig, La Madeleine,
  Chémery-Faulquemont, die pannonisch-lokalen Waren, African Red Slip und die
  östlichen Sigillaten.
- **Desktop-Anwendung** für Windows, macOS und Linux, dazu eine installierbare
  Web-Fassung. Version 1 brauchte Python, eine virtuelle Umgebung und ein
  Terminal.
- **Zweisprachige Oberfläche**, helles und dunkles Design, ÖAI-Layout.

Konfigurationsdateien aus Version 1 lassen sich unmittelbar laden und werden
übernommen. Mit Residualität 0 und ohne kategoriespezifische Raten rechnet
Version 2 exakt wie Version 1.

**Version 1 bleibt verfügbar** im
[Branch `v1`](https://github.com/oeai-dac/STOCHASI/tree/v1).

## Grundsätze

- **Alles bleibt lokal.** Es gibt kein Backend. Daten werden nicht übertragen;
  die App ist installierbar und läuft nach dem ersten Aufruf offline. Die
  Desktop-Fassung geht weiter: Dort verbietet eine Content-Security-Policy jede
  ausgehende Verbindung, und selbst die Schriften sind mitgeliefert statt von
  einem CDN geladen.
- **Keine Fremdbibliotheken für die Fachlogik.** Monte-Carlo-Kern,
  Dirichlet-Multinomial-Likelihood, Log-Gamma-Funktion, Diagramme, SVG-, PNG-
  und PDF-Ausgabe, Zweisprachigkeit und Icon-Erzeugung sind selbst geschrieben.
  Zur Laufzeit hängt die App nur an React und — nur wenn XLSX tatsächlich
  gebraucht wird — an SheetJS.
- **Keine erfundenen Daten.** Die Referenzliste der Produktionszentren liefert
  Bezeichnungen, Farben und grobe Produktionszeiträume. Sie liefert bewusst
  **keine Marktanteile**: Die sind regional verschieden, und sie zu erfinden
  wäre schlimmer als nutzlos. Die Lieferkurve muss aus der Literatur zum eigenen
  Fundort kommen.
- **Grenzen werden benannt, nicht überspielt.** Die Datierung ist bedingt auf
  Marktkurve, Ersatzrate und Residualanteil, und das steht dort, wo man das
  Ergebnis abliest. [Kapitel 14 des Complete
  Guide](docs/GUIDE.md#14-limits-of-the-method) listet auf, was das Verfahren
  nicht kann.

## Aus dem Quelltext bauen

```bash
npm ci
npm run dev        # Entwicklungsserver (Vite)
npm run build      # Typprüfung + Produktions-Build nach dist/
npm run preview    # den Build lokal ansehen
npm test           # die gesamte Testsuite
npm run electron   # Bauen und im Desktop-Gehäuse starten
npm run smoke      # Selbsttest der Desktop-Fassung
npm run dist       # Installationspakete für das aktuelle System
npm run gen-icons  # Icons neu erzeugen (PWA + Paket)
```

Einzelheiten zu Paketen, Release und Signierung:
**[docs/BUILD.md](docs/BUILD.md)**.

Beim Start wird ein echter Beispieldatensatz geladen — die Terra Sigillata aus
Insula XLI von Flavia Solva, 77 Scherben. Eigene Dateien laden Sie per
**Ziehen und Ablegen** oder über „Datei laden…": ein Projekt aus Version 1
(`.json`, wird automatisch übernommen), ein Projekt aus Version 2
(`.stochasi.json`) oder eine Rohtabelle (`.csv`/`.tsv`/`.xlsx`) mit einer
Marktkurve oder mit Fundzahlen. Vier Dateien zum Ausprobieren liegen in
[`examples/`](examples/).
