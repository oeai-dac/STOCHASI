# STOCHASI installieren

Diese Anleitung führt Schritt für Schritt durch die Installation. Sie setzt
keine technischen Vorkenntnisse voraus.

> **Ganz ohne Installation:** STOCHASI läuft auch direkt im Browser unter
> **<https://oeai-dac.github.io/STOCHASI/>**. Alle Berechnungen finden dabei auf
> Ihrem eigenen Rechner statt; es werden keine Daten übertragen. Wenn Sie die
> Software nur ausprobieren möchten oder auf einem Dienstrechner nichts
> installieren dürfen, ist das der schnellste Weg.

---

## Welche Datei brauche ich?

Alle Dateien finden Sie auf der Seite
[**Releases**](https://github.com/oeai-dac/STOCHASI/releases) unter dem
Abschnitt „Assets" der neuesten Version.

| Ihr System | Datei |
|---|---|
| **Windows 10 oder 11** | `STOCHASI-2.1.0-Setup-x64.exe` |
| Windows, ohne Installation | `STOCHASI-2.1.0-portable-x64.exe` |
| **macOS** mit Apple-Chip (M1–M4) | `STOCHASI-2.1.0-arm64.dmg` |
| **Linux**, beliebige Distribution | `STOCHASI-2.1.0-x86_64.AppImage` |
| Linux: Ubuntu 22.04+, Debian 12+, Mint | `stochasi_2.1.0_amd64.deb` |
| Linux: Fedora, openSUSE, RHEL | `stochasi-2.1.0.x86_64.rpm` |

**Welchen Mac habe ich?** Apfel-Menü → „Über diesen Mac". Steht dort „Apple M1",
„M2", „M3" oder „M4", ist die `arm64`-Datei die richtige. **Für Macs mit
Intel-Prozessor wird kein Paket ausgeliefert** — nutzen Sie dort die
Web-Fassung: <https://oeai-dac.github.io/STOCHASI/>. Sie rechnet ebenso
vollständig auf dem eigenen Rechner, es werden keine Daten übertragen.

---

## Wichtig vorab: die Sicherheitswarnung

STOCHASI wird als freie Software eines Forschungsinstituts herausgegeben und ist
**nicht digital signiert**. Ein solches Zertifikat kostet mehrere hundert Euro
pro Jahr und würde nichts an der Software selbst verbessern — es bestätigt nur
gegenüber Microsoft und Apple, wer der Herausgeber ist.

Deshalb zeigen Windows und macOS beim **ersten** Start eine Warnung. Das ist
kein Hinweis auf Schadsoftware, sondern lediglich die Feststellung, dass das
Betriebssystem den Herausgeber nicht bei Microsoft oder Apple registriert
vorfindet. Die folgenden Abschnitte zeigen, wie Sie die Warnung bestätigen.

Wenn Sie sichergehen möchten: Der vollständige Quelltext ist einsehbar, und die
Installationspakete werden automatisch von GitHub aus genau diesem Quelltext
gebaut — nachvollziehbar unter
[Actions](https://github.com/oeai-dac/STOCHASI/actions).

---

## Windows

### Mit Installation (empfohlen)

1. Laden Sie `STOCHASI-2.1.0-Setup-x64.exe` herunter.
2. Doppelklicken Sie die Datei.
3. **Es erscheint ein blaues Fenster „Der Computer wurde durch Windows
   geschützt".** Klicken Sie auf den kleinen Text **„Weitere Informationen"** —
   erst dann erscheint die Schaltfläche **„Trotzdem ausführen"**. Klicken Sie
   darauf.
4. Folgen Sie dem Installationsprogramm. Sie können den Zielordner ändern oder
   die Voreinstellung übernehmen.
5. STOCHASI liegt anschließend im Startmenü und auf dem Desktop.

Es werden **keine Administratorrechte** benötigt: Die Installation erfolgt in
Ihr Benutzerprofil. Das funktioniert auch auf verwalteten Dienstrechnern.

### Ohne Installation

Laden Sie `STOCHASI-2.1.0-portable-x64.exe` herunter und doppelklicken Sie sie.
Das Programm startet unmittelbar, ohne etwas zu installieren. Auch hier erscheint
beim ersten Mal die oben beschriebene Windows-Meldung. Diese Variante lässt sich
auf einen USB-Stick legen.

---

## macOS

1. Laden Sie `STOCHASI-2.1.0-arm64.dmg` herunter (nur Macs mit Apple-Chip, siehe oben).
2. Doppelklicken Sie die Datei. Es öffnet sich ein Fenster mit dem
   STOCHASI-Symbol und dem Ordner „Programme".
3. Ziehen Sie das STOCHASI-Symbol auf den Ordner **Programme**.
4. Öffnen Sie den Ordner „Programme" im Finder.
5. **Wichtig beim ersten Start:** Doppelklicken Sie STOCHASI **nicht**, sondern
   klicken Sie mit der **rechten Maustaste** (oder Ctrl-Taste + Klick) darauf und
   wählen Sie **„Öffnen"**.
6. Es erscheint ein Hinweis, dass die App von einem nicht verifizierten
   Entwickler stammt. Nur über diesen Weg gibt es dort die Schaltfläche
   **„Öffnen"**. Klicken Sie darauf.

Ab dem zweiten Start genügt ein normaler Doppelklick.

### Falls „STOCHASI ist beschädigt und kann nicht geöffnet werden" erscheint

Diese Meldung ist irreführend — die Datei ist in Ordnung. macOS versieht
heruntergeladene Dateien mit einer Quarantäne-Markierung und verweigert bei
nicht notariell beglaubigten Programmen den Start. So entfernen Sie die
Markierung:

1. Öffnen Sie das Programm **Terminal** (Spotlight mit ⌘ + Leertaste, dann
   „Terminal" eintippen).
2. Tippen Sie folgende Zeile ein und drücken Sie Enter:

   ```
   xattr -cr /Applications/STOCHASI.app
   ```

3. Starten Sie STOCHASI nun normal.

---

## Linux

### AppImage — funktioniert auf jeder Distribution

Ein AppImage ist eine einzelne Datei, die ohne Installation läuft.

1. Laden Sie `STOCHASI-2.1.0-x86_64.AppImage` herunter.
2. Machen Sie die Datei ausführbar. Entweder im Dateimanager
   (Rechtsklick → Eigenschaften → Berechtigungen → „Als Programm ausführen"),
   oder im Terminal:

   ```bash
   chmod +x STOCHASI-2.1.0-x86_64.AppImage
   ```

3. Doppelklicken Sie die Datei, oder starten Sie sie im Terminal:

   ```bash
   ./STOCHASI-2.1.0-x86_64.AppImage
   ```

Möchten Sie STOCHASI im Anwendungsmenü sehen, hilft das Werkzeug
[AppImageLauncher](https://github.com/TheAssassin/AppImageLauncher); es richtet
den Menüeintrag beim ersten Start selbsttätig ein.

#### Auf Ubuntu 24.04 und neuer: zwei Handgriffe vorab

Ubuntu hat ab 24.04 zwei Dinge geändert, die alle AppImages betreffen — nicht
nur STOCHASI.

**1. Fehlermeldung „dlopen(): error loading libfuse.so.2"**

AppImages benötigen FUSE 2, Ubuntu liefert seit 24.04 nur noch FUSE 3. Entweder
Sie rüsten die alte Bibliothek nach:

```bash
sudo apt install libfuse2t64
```

Oder Sie starten das AppImage ohne sie, indem es sich selbst entpackt:

```bash
./STOCHASI-2.1.0-x86_64.AppImage --appimage-extract-and-run
```

**2. Fehlermeldung zur Sandbox oder ein Fenster, das nicht erscheint**

Ubuntu schränkt seit 24.04 unprivilegierte User-Namespaces ein. Programme wie
STOCHASI, die auf Chromium aufbauen, brauchen dafür ein AppArmor-Profil — das
lässt sich aus einem AppImage heraus aber nicht installieren. Behelf:

```bash
./STOCHASI-2.1.0-x86_64.AppImage --no-sandbox
```

**Beide Punkte entfallen beim `.deb`-Paket.** Es bringt das AppArmor-Profil mit
und richtet es bei der Installation selbsttätig ein. Auf Ubuntu, Kubuntu und
Linux Mint ist das `.deb` deshalb der bequemere Weg.

### Ubuntu 22.04+, Debian 12+, Linux Mint

```bash
sudo apt install ./stochasi_2.1.0_amd64.deb
```

Das vorangestellte `./` ist wichtig — ohne den Punkt sucht `apt` ein Paket
dieses Namens in den Paketquellen und findet es nicht.

### Fedora, RHEL, openSUSE

```bash
sudo dnf install ./stochasi-2.1.0.x86_64.rpm     # Fedora, RHEL
sudo zypper install ./stochasi-2.1.0.x86_64.rpm  # openSUSE
```

Nach der Installation erscheint STOCHASI im Anwendungsmenü unter
„Wissenschaft" bzw. „Bildung".

---

## Erste Schritte

Beim Start lädt STOCHASI einen Beispieldatensatz, an dem Sie alle Funktionen
gefahrlos ausprobieren können.

Eigene Daten laden Sie über **„Datei laden…"** oder indem Sie eine Datei
einfach in das Fenster **ziehen**. Erkannt werden:

- Rohtabellen als `.csv`, `.tsv` oder `.xlsx`
- STOCHASI-v2-Projekte (`.stochasi.json`)
- Projekte aus STOCHASI Version 1 (`.json`) — sie werden automatisch übernommen

---

## Datenschutz

STOCHASI arbeitet vollständig auf Ihrem Rechner. Es gibt keinen Server, keine
Registrierung und keine Telemetrie. Die Desktop-Fassung ist so eingerichtet,
dass sie technisch **keine** Verbindung ins Internet aufbauen kann; selbst die
verwendeten Schriften sind mitgeliefert statt nachgeladen.

Ausnahme: Wenn Sie im Menü **Hilfe** einen Verweis anklicken, öffnet sich Ihr
Browser mit der GitHub-Seite. Das geschieht nur auf ausdrücklichen Klick.

---

## Deinstallieren

| System | Vorgehen |
|---|---|
| Windows | Einstellungen → Apps → STOCHASI → Deinstallieren |
| Windows (portabel) | Die `.exe`-Datei löschen |
| macOS | `STOCHASI.app` aus „Programme" in den Papierkorb ziehen |
| Linux (AppImage) | Die `.AppImage`-Datei löschen |
| Debian/Ubuntu | `sudo apt remove stochasi` |
| Fedora/RHEL | `sudo dnf remove stochasi` |

Ihre gespeicherten Projektdateien bleiben dabei erhalten — sie liegen dort, wo
Sie sie abgelegt haben.

---

## Es klappt nicht — was nun?

Bitte melden Sie das Problem unter
[github.com/oeai-dac/STOCHASI/issues](https://github.com/oeai-dac/STOCHASI/issues).
Hilfreich sind dabei:

- Ihr Betriebssystem und dessen Version
- Welche Datei Sie heruntergeladen haben
- Der genaue Wortlaut der Fehlermeldung (gern als Bildschirmfoto)
