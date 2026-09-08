# STOCHASI 2 — Kurzanleitung

STOCHASI simuliert, wie sich die Zusammensetzung eines Fundspektrums über die
Zeit verändert, und rechnet dieselbe Sache rückwärts: Zu einem gezählten
Fundkomplex fragt es, welches Ablagerungsjahr diese Zahlen erklärt.

Diese Anleitung braucht etwa zehn Minuten. Der [Complete Guide](GUIDE.md)
behandelt die Mathematik, die Dateiformate und die Grenzen des Verfahrens
*(englisch)*.

**Inhalt**

1. [Was das Programm rechnet](#1-was-das-programm-rechnet)
2. [STOCHASI bekommen](#2-stochasi-bekommen)
3. [Der Beispieldatensatz](#3-der-beispieldatensatz)
4. [Die sechs Ansichten](#4-die-sechs-ansichten)
5. [Eigene Daten laden](#5-eigene-daten-laden)
6. [Eine erste Auswertung](#6-eine-erste-auswertung)
7. [Die Datierungskurve ehrlich lesen](#7-die-datierungskurve-ehrlich-lesen)
8. [Was die Zahlen in der Übersicht bedeuten](#8-was-die-zahlen-in-der-übersicht-bedeuten)
9. [Kurven einfärben, ein- und ausblenden](#9-kurven-einfärben-ein--und-ausblenden)
10. [Export](#10-export)
11. [Von STOCHASI 1 kommend](#11-von-stochasi-1-kommend)

---

## 1. Was das Programm rechnet

Das Modell hat zwei Eingaben und eine Größe, die sich bewegt.

Das **Marktangebot** gibt für jedes Jahr an, welcher Anteil des neu erworbenen
Materials aus welchem Produktionszentrum stammt. Diese Angabe kommt von Ihnen:
Sie ist eine Aussage über Ihren Fundort, gewonnen aus der Literatur oder aus
eigenen Zählungen.

Der **Umlaufbestand** ist das, was in einem bestimmten Jahr tatsächlich in
Gebrauch steht. Er ist nicht dasselbe wie das Marktangebot, denn ein 140
gekaufter Teller steht 155 immer noch im Regal. STOCHASI schreibt den Bestand
Jahr für Jahr fort:

```
Bestand(t) = Bestand(t−1) · (1 − r) + Markt(t) · r + Rauschen
```

`r` ist die **Ersatzrate**: der Anteil des Bestands, der jedes Jahr ausscheidet
und durch neue Ware ersetzt wird. Eine niedrige Rate bedeutet langlebiges
Geschirr und einen Bestand, der dem Markt weit hinterherhinkt; eine hohe Rate
bedeutet, dass der Bestand dem Markt dicht folgt.

Der **Fundkomplex** sind die Scherben, die Sie in einem Befund gezählt haben.
Weil eine Grabung aus dem Umlaufbestand schöpft, sollte der gezählte Komplex dem
simulierten Bestand des Jahres ähneln, in dem der Befund entstanden ist.

Dieser letzte Satz ist der ganze Zweck des Programms. Vorwärts gerechnet zeigt
es, wie ein Spektrum in einem gegebenen Jahr aussehen müsste; rückwärts
gerechnet fragt es, auf welches Jahr Ihr Spektrum deutet.

Alles daran ist stochastisch. Jeder Lauf würfelt eigenes Rauschen, und STOCHASI
rechnet viele Läufe. Was Sie sehen, ist deshalb nie eine einzelne Linie, sondern
ein Mittelwert mit einem Unsicherheitsband, und die Datierung ist nie ein Jahr,
sondern eine Wahrscheinlichkeitsverteilung über Jahre.

---

## 2. STOCHASI bekommen

**Ohne Installation ausprobieren.** Unter
<https://oeai-dac.github.io/STOCHASI/> läuft das vollständige Programm im
Browser. Es gibt keinen Server: Jede Rechnung findet auf Ihrem eigenen Gerät
statt, es werden keine Daten übertragen. Das ist der richtige Weg, wenn Sie nur
hineinsehen wollen oder auf einem Dienstrechner nichts installieren dürfen.

**Als Programm installieren** für die regelmäßige Arbeit. Die Pakete liegen auf
der Seite [Releases](https://github.com/oeai-dac/STOCHASI/releases).
[INSTALLATION.md](INSTALLATION.md) führt Schritt für Schritt durch die
Installation und erklärt auch die Warnmeldungen, die Windows und macOS bei nicht
signierten Paketen zeigen.

**Aus dem Quelltext bauen**: `npm ci && npm run dev`, Näheres in
[BUILD.md](BUILD.md).

---

## 3. Der Beispieldatensatz

Beim Start stehen echte Zahlen im Programm: die Terra Sigillata aus der Insula
XLI von Flavia Solva, 77 Scherben, fünf Warenarten, dazu eine Marktkurve von 120
bis 290 n. Chr. Es sind dieselben Zahlen, mit denen STOCHASI 1 ausgeliefert
wurde.

Ein erfundener Datensatz wäre ordentlicher gewesen, und genau deshalb steht dort
keiner. 77 Scherben sind wenig, und die zehn italischen Stücke passen nicht zum
Modell. Beides ist in der Oberfläche zu sehen, und beides sollte man auf dem
ersten Bildschirm sehen.

---

## 4. Die sechs Ansichten

**Marktangebot** zeigt die Kurve, die Sie eingegeben haben, auf Einzeljahre
interpoliert. Kleine Marken auf der unteren Achse stehen bei den Stützjahren,
also dort, wo Sie tatsächlich Daten haben. Dazwischen wird linear interpoliert,
und das Programm schreibt das dazu.

**Simulation** ist die Hauptansicht: der Umlaufbestand, eine Linie je Kategorie,
dazu ein Band vom 10. bis zum 90. Perzentil über alle Läufe. Wo das Band breit
ist, ist das Modell unsicher.

**Jahresspektrum** hält ein Jahr fest und zeigt es als Balken: den simulierten
Mittelwert mit einem Fühler für das 10.–90. Perzentil und, wenn Sie einen
Fundkomplex wählen, darunter die gemessenen Anteile mit ihren Stückzahlen.

*(STOCHASI 1 zeichnete hier ein Tortendiagramm. Der Balken hat es aus zwei
Gründen ersetzt: Eine Torte mit zwanzig Produktionszentren ist unlesbar, und die
Unsicherheit der Simulation kann sie überhaupt nicht darstellen.)*

**Vergleich** stellt den Unterschied zwischen Fundkomplex und Simulation als
beidseitige Balken dar. Kategorien außerhalb des simulierten 10.–90. Perzentils
sind rot markiert. Diese Ansicht sagt, *wo* Modell und Befund auseinandergehen,
und die roten Marken sind der wichtige Teil: Ohne sie sieht jeder Balken nach
einem Befund aus, obwohl die meisten nur Rauschen sind.

**Datierung** ist die Rückwärtsrechnung, in zwei Formen. *Kurven* zeigt die
Wahrscheinlichkeitsverteilung über die Jahre, das wahrscheinlichste Jahr und ein
95-%-Intervall. *Rangfolge* sortiert die Fundkomplexe nach geschätztem Datum und
zeichnet ihre Intervalle als Balken.

In der Rangfolge steckt eine Falle, und die Ansicht ist so gebaut, dass sie sie
zeigt: Eine sortierte Liste sieht nach Abfolge aus, auch wenn sämtliche
Intervalle einander überlappen und die Daten gar keine Reihenfolge tragen. Gibt
es ein einziges Jahr, das zu *allen* Intervallen passt, wird dieser Bereich
hinter den Balken hinterlegt. Ist dieses Band zu sehen, ist die abgelesene
Reihenfolge nicht belegt.

**Daten** ist die Ansicht zum Bearbeiten: Projektname, Kategorien, die
Marktkurve an ihren Stützjahren und die Stückzahlen der Fundkomplexe.

---

## 5. Eigene Daten laden

**Datei laden…** in der Kopfzeile, oder die Datei ins Fenster ziehen. STOCHASI
erkennt selbst, um welche Art Tabelle es sich handelt.

Eine **Markttabelle** hat eine Jahresspalte (`Jahr`, `Year` oder eine erste
Spalte aus plausiblen Jahreszahlen) und daneben je Produktionszentrum eine
Spalte. Die Werte dürfen Prozente oder Rohzahlen sein, jede Zeile wird auf 100 %
normiert.

```
Jahr;IT;LG;BA;MG;RZ
120;30;20;50;0;0
130;10;0;60;30;0
140;0;0;10;80;10
```

Eine **Fundkomplextabelle** hat keine Jahresspalte. Die Kategorien dürfen in den
Spalten stehen (eine Zeile je Befund) oder in den Zeilen (eine Spalte je
Befund). STOCHASI erkennt die Ausrichtung und meldet, wenn es transponiert hat.
Die zweite Form ist die Pivot-Tabelle, die bei einer Kreuztabelle Provenienz ×
Befund herauskommt, deshalb werden beide gelesen.

```
Insula;IT;LG;BA;MG;RZ          Provenienz;405;403;802
405;10;0;1;56;10               IT;10;3;1
403;3;0;0;20;7                 MG;56;20;5
```

Auch das Langformat wird gelesen: Spalten `Typ`/`Kategorie`/`Type` und
`Anzahl`/`Count`, wahlweise mit einer Spalte für den Befundnamen.

**Stückzahlen, keine Prozente.** Die Datierung rechnet mit dem
Stichprobenumfang: 77 Scherben und 770 Scherben wiegen sehr verschieden, und
Prozentwerte werfen genau das weg. Findet STOCHASI gebrochene Werte, sagt es das.

Steht in der Datei eine Spalte, die zu keiner Kategorie des Projekts passt, legt
STOCHASI diese Kategorie an und meldet es. In der Marktkurve steht sie dann mit
Nullen, und dort müssen Sie sie füllen. Ohne diesen Schritt wären die Stücke
stillschweigend weggefallen und die Datierung liefe mit einem zu kleinen N.

Waren im Projekt schon Fundkomplexe, erscheint nach dem Laden ein Balken mit der
Wahl zwischen **Vorhandene ersetzen** und **An vorhandene anhängen**.
Voreingestellt ist Ersetzen; zum Anhängen genügt ein Klick, solange der Balken
steht. Beim Anhängen werden gleiche Namen durchnummeriert, damit zwei Grabungen
mit je einer „Insula I" unterscheidbar bleiben.

Gelesen werden `.csv`, `.tsv`, `.xlsx`, `.xls` sowie `.json` für Projektdateien
von STOCHASI 1 und 2. Vier echte Dateien zum Ausprobieren liegen in
[`examples/`](../examples/).

---

## 6. Eine erste Auswertung

1. **Marktkurve laden** oder im Reiter Daten von Hand anlegen. Fangen Sie grob
   an; fünf oder sechs Stützjahre über den ganzen Zeitraum reichen meist.
2. **Kategorien wählen.** Daten → *Produktionszentrum hinzufügen* öffnet eine
   Referenzliste von 36 Zentren, nach Regionen gruppiert, jeweils mit grober
   Laufzeit. Die Liste liefert Namen, Farben und Zeiträume, **keine**
   Marktanteile. Die müssen aus der Literatur zu Ihrem Fundort kommen.
3. **Zeitraum einstellen** in der Seitenleiste, dann in die Ansicht *Simulation*
   sehen.
4. **Ersatzrate setzen.** Beginnen Sie bei 10 % pro Jahr und beobachten Sie, was
   sich ändert: Eine niedrigere Rate lässt den Bestand weiter hinter dem Markt
   zurückbleiben. Wenn Sie für eine Warenart eine deutlich andere Lebensdauer
   kennen, öffnen Sie *Rate je Kategorie*.
5. **Fundkomplexe laden** und im *Vergleich* das erwartete Jahr prüfen. Die
   roten Marken zeigen, wo der Befund vom Modell abweicht.
6. **In die Datierung wechseln.** Wahrscheinlichstes Jahr und Intervall ablesen,
   dann *Über mehrere Residualanteile rechnen* anhaken; warum, steht im nächsten
   Abschnitt.
7. **Seed festsetzen** (Seitenleiste, *Random seed*), bevor Sie etwas für eine
   Publikation exportieren. Bei 0 fällt jeder Lauf anders aus.

---

## 7. Die Datierungskurve ehrlich lesen

Drei Dinge sollte man wissen, bevor man eine Zahl aus diesem Programm zitiert.

**Die Kurve ist bedingt.** Sie gibt die Wahrscheinlichkeit des
Ablagerungsjahres *unter* Ihrer Marktkurve, Ihrer Ersatzrate und Ihrem
Residualanteil an. Ändert sich eine dieser Annahmen, ändert sich das Ergebnis.
Es ist keine absolute Datierung, und es wird auch keine dadurch, dass es genau
gezeichnet ist.

**Kleine Komplexe geben breite Intervalle, und das zu Recht.** Bei zwölf
Scherben ist das Intervall weit. Das ist keine Vorsicht des Programms, sondern
die Datenlage. Ein Verfahren, das aus zwölf Scherben ein enges Intervall macht,
würde Sie belügen.

**Residualität verschiebt die Antwort.** Umgelagertes Altmaterial lässt einen
Komplex älter erscheinen, und das angepasste Jahr wandert zum Ausgleich nach
hinten. Den Residualanteil können Sie nicht wissen, aber Sie können
herausfinden, ob er eine Rolle spielt: Das Häkchen *Über mehrere Residualanteile
rechnen* lässt STOCHASI dieselbe Datierung viermal rechnen, mit 0, 10, 20 und
30 %, und zeichnet die vier Kurven für den ausgewählten Fundkomplex. Der
Residual-Regler in der Seitenleiste gilt in diesem Modus nicht.

- **Die Kurven liegen übereinander** → die Datierung hängt nicht von der
  Annahme ab. So kann man sie berichten.
- **Die Kurven wandern auseinander** → die Datierung ist eine Funktion einer
  unbekannten Größe. Berichten Sie die Spanne über alle vier Kurven, nicht das
  Intervall aus einer davon.

Die Übersichtstabelle und die Rangfolge nehmen bei eingeschalteter Schar von
selbst die weitere Spanne. Was dort steht, ist also die belastbare Zahl und
nicht die aus einer einzelnen Annahme.

Beim Beispieldatensatz liegen die vier Kurven fast übereinander, und die
Übersicht weist entsprechend die Spanne über alle aus.

---

## 8. Was die Zahlen in der Übersicht bedeuten

Die Tabelle unter dem Diagramm nennt zu jedem Fundkomplex zwei Jahreszahlen, und
die werden leicht verwechselt.

Das **wahrscheinlichste Jahr** ist der höchste Punkt der Kurve, also das Jahr,
für das die gezählten Stückzahlen am wahrscheinlichsten sind.

Der **Erwartungswert** ist der Schwerpunkt der Kurve, die mit den
Wahrscheinlichkeiten gewichtete Summe aller Jahre. Bei einer symmetrischen Kurve
fallen beide Zahlen fast zusammen. Bei einer schiefen nicht: Ein langer
Ausläufer nach spät zieht den Schwerpunkt mit sich, während der Gipfel bleibt,
wo er ist. Der Abstand der beiden Zahlen ist damit ein Maß dafür, wie schief die
Kurve ist.

Zum Berichten eignet sich das wahrscheinlichste Jahr zusammen mit dem
Intervall. Der Erwartungswert hängt zusätzlich am eingestellten Zeitraum, weil
die Kurve an dessen Rändern abgeschnitten wird; ein zu eng gewählter Zeitraum
verschiebt ihn.

Der **Random seed** in der Seitenleiste steuert den Zufall. Bei 0 würfelt jeder
Lauf neu, und dieselbe Rechnung ergibt beim nächsten Mal leicht andere Kurven.
Ein fester Wert macht denselben Zufall wiederholbar. Für eine Publikation gehört
er in die Abbildungslegende, damit andere die Abbildung nachrechnen können; die
Fußnote jedes Diagramms schreibt ihn ohnehin mit.

---

## 9. Kurven einfärben, ein- und ausblenden

Jeder Fundkomplex hat eine eigene Farbe. Sie steht im Reiter Daten in der ersten
Spalte der Fundkomplextabelle und gilt für die Kurve in der Datierung, für den
Punkt vor der Beschriftung in der Rangfolge und für die vier Kurven der
Residualschar, die aus ihr abgeleitet werden. Neue Komplexe bekommen der Reihe
nach eine Farbe aus einer Palette, die auch bei den gängigen Farbsehschwächen
unterscheidbar bleibt.

Diese Farbe ist etwas anderes als die Farbe einer Kategorie. Die Kategorienfarbe
steht für ein Produktionszentrum, die Komplexfarbe für einen Befund; beide
kommen nie im selben Diagramm vor.

Über dem Diagramm steht eine Reihe von Schaltern, einer je Kurve, dahinter
*Alle* und *Keine*. Ausgeschaltete Kurven verschwinden aus dem Diagramm, bleiben
in der Tabelle darunter aber stehen, nur blass — die Zahlen sind auch dann noch
das, was man ablesen will.

Ist die Residualschar eingeschaltet, schalten dieselben Knöpfe die vier
Residualanteile statt der Fundkomplexe. So lässt sich zeigen, dass 0 % und 30 %
auseinanderlaufen, ohne dass die beiden mittleren Kurven im Weg sind.

Das Häkchen *Modus-Linien* zeichnet zu jeder sichtbaren Kurve eine strichlierte
Senkrechte am wahrscheinlichsten Jahr, in der Farbe der Kurve. Die Textzeile mit
Jahr und Intervall erscheint nur, wenn eine einzige Kurve zu sehen ist; bei
sieben Fundkomplexen lägen sieben Beschriftungen übereinander.

Beim Ausblenden ist eine Sache zu bedenken. Wer aus einer Rangfolge die beiden
Komplexe wegklickt, die nicht ins Bild passen, bekommt eine Abbildung, die eine
Abfolge behauptet, welche die Daten nicht tragen — und der hinterlegte
Schnittbereich rechnet dann nur noch über die sichtbaren Zeilen. Deshalb nennt
die Fußnote jeder Abbildung die Zahl der ausgeblendeten Kurven, und sie wandert
so auch in den Export. Die Auswahl gilt nur für die laufende Sitzung und wird
nicht in der Projektdatei gespeichert.

---

## 10. Export

Das Menü **Export** bietet zur gerade sichtbaren Ansicht:

- **PNG** — Rasterbild in doppelter Auflösung
- **SVG** — Vektorgrafik, in Illustrator oder Inkscape weiterzubearbeiten
- **PDF** — Vektorgrafik für den Satz
- **CSV / XLSX** — die Zahlen hinter der Abbildung, dazu ein Parameterblatt
- **Projektdatei** — alles, was zur Wiederholung der Rechnung nötig ist

Zwei Dinge sind dabei zu wissen. Exportierte Abbildungen nehmen immer die helle
Palette, auch wenn Sie im dunklen Schema arbeiten, denn eine dunkle Abbildung
ist im Druck unbrauchbar. Und die exportierte Abbildung ist dieselbe Szene, die
auf dem Bildschirm steht: Es gibt keinen zweiten, hübscheren Zeichenweg.

Die Tabellenblätter bleiben vollständig, auch wenn im Diagramm Kurven
ausgeblendet sind. Die Abbildung ist eine Aussage, die Tabelle der Anhang, in
dem jemand nachrechnet; die Spalte „im Diagramm" hält beides zusammen.

Jeder Tabellenexport enthält ein Blatt **Parameter** mit Zeitraum, Raten,
Streuung, Residualanteil, Zahl der Läufe und Seed. Legt man es zu einer
Abbildung, lässt sich der Lauf genau wiederholen.

---

## 11. Von STOCHASI 1 kommend

Konfigurationsdateien der Version 1 lassen sich direkt laden: die `.json`
einfach ins Fenster ziehen. Kategorien, Marktkurve, Zeitraum, Startverteilung,
Streuung, Läufe, Seed, Neugründungsmodus und das Grabungsspektrum werden
übernommen, und STOCHASI meldet, was es dabei anpassen musste.

Zwei Einstellungen starten bewusst neutral, damit Version 2 zunächst genau das
rechnet, was Version 1 gerechnet hat: Der Residualanteil steht auf 0, und
kategoriespezifische Ersatzraten gibt es erst, wenn Sie eine setzen.

Der umgekehrte Weg wird absichtlich nicht angeboten. Version 1 kennt weder
kategoriespezifische Raten noch Residualität und würde eine v2-Datei
stillschweigend falsch rechnen.

Version 1 bleibt im [Zweig `v1`](https://github.com/oeai-dac/STOCHASI/tree/v1)
dieses Repositoriums verfügbar.

---

## Weiter

- **[Complete Guide](GUIDE.md)** — die Mathematik des Vorwärtsmodells, die
  Dirichlet-Multinomial-Likelihood hinter der Datierung, das Residualmodell, die
  Dateiformate im Einzelnen und eine offene Darstellung dessen, was das
  Verfahren nicht kann *(englisch)*.
- **[INSTALLATION.md](INSTALLATION.md)** — Installation je Betriebssystem.
- **[BUILD.md](BUILD.md)** — Bauen und Veröffentlichen.
- Fragen und Fehlermeldungen:
  [Issues](https://github.com/oeai-dac/STOCHASI/issues).
