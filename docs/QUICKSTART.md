# STOCHASI 2 — Quick Start Guide

STOCHASI simulates how the composition of an archaeological find spectrum
changes over time, and it works the same calculation backwards: given a counted
assemblage, it asks which year of deposition would produce those counts.

This guide takes about ten minutes. The [Complete Guide](GUIDE.md) covers the
mathematics, the file formats and the limits of the method. In German:
[QUICKSTART.de.md](QUICKSTART.de.md).

**Contents**

1. [What STOCHASI does](#1-what-stochasi-does)
2. [Getting STOCHASI](#2-getting-stochasi)
3. [Your first look: the example dataset](#3-your-first-look-the-example-dataset)
4. [The six views](#4-the-six-views)
5. [Getting your own data in](#5-getting-your-own-data-in)
6. [A first analysis, step by step](#6-a-first-analysis-step-by-step)
7. [Reading the dating curve honestly](#7-reading-the-dating-curve-honestly)
8. [Export](#8-export)
9. [Coming from STOCHASI 1](#9-coming-from-stochasi-1)
10. [Where to go next](#10-where-to-go-next)

---

## 1. What STOCHASI does

The model has two inputs and one moving quantity.

**Market supply** — for each year, the share of newly acquired material coming
from each production centre. This is what you supply: it is a statement about
your site, drawn from the literature or from your own counts.

**The stock in circulation** — what is actually in use at a site in a given
year. It is not the market supply: pottery bought in 140 is still on the shelf
in 155. STOCHASI moves this stock forward year by year:

```
stock(t) = stock(t−1) · (1 − r) + market(t) · r + noise
```

where `r` is the **replacement rate**: the fraction of the stock that drops out
each year and is replaced by new supply. A low rate means a conservative
household with long-lived vessels and a stock that lags the market; a high rate
means the stock tracks the market closely.

**The assemblage** — the sherds you actually counted in a context. Because
excavation samples the stock, the counted assemblage should resemble the
simulated stock of the year the context was deposited.

That last sentence is the whole point of the program. Run it forwards to see
what a spectrum *should* look like in a given year; run it backwards to ask
which year your spectrum points to.

Everything is stochastic: each run adds random scatter, and STOCHASI performs
many runs. What you see is never a single line but a mean with an uncertainty
band, and the dating is never a year but a probability distribution over years.

---

## 2. Getting STOCHASI

**Try it without installing.** <https://oeai-dac.github.io/STOCHASI/> runs the
complete application in your browser. There is no server: every calculation
happens on your own machine and no data is transmitted. This is the right route
if you only want a look, or if you are not allowed to install software on a work
computer.

**Install the desktop application** for regular work. The packages are on the
[Releases](https://github.com/oeai-dac/STOCHASI/releases) page.

| Your system | File |
|---|---|
| Windows 10/11 | `STOCHASI-*-Setup-x64.exe`, or `*-portable-x64.exe` for no installation |
| macOS (Apple M1–M4) | `STOCHASI-*-arm64.dmg` |
| Linux, any distribution | `STOCHASI-*-x86_64.AppImage` |
| Ubuntu 22.04+, Debian 12+, Mint | `stochasi_*_amd64.deb` |
| Fedora, openSUSE, RHEL | `stochasi-*.x86_64.rpm` |

The packages are not digitally signed, so Windows and macOS show a warning on
first launch. **[INSTALLATION.md](INSTALLATION.md)** walks through it step by
step *(in German)*. No package is shipped for Intel Macs; use the web version
there.

**Build from source** if you want to: `npm ci && npm run dev`. See
[BUILD.md](BUILD.md) *(in German)*.

---

## 3. Your first look: the example dataset

STOCHASI opens with real data: the terra sigillata from Insula XLI at Flavia
Solva — 77 sherds, five wares, a market curve running from AD 120 to 290. These
are the same numbers that shipped with STOCHASI 1.

An invented dataset would have been tidier, and that is exactly why it is not
used. Seventy-seven sherds is a small sample, and the ten Italian pieces do not
fit the model at all. Both of those show up in the interface, and both are worth
seeing on the first screen.

---

## 4. The six views

**Market supply** shows the curve you have supplied, interpolated to single
years. Small ticks on the bottom axis mark the reference years — the places
where you actually have data. Between them, STOCHASI interpolates linearly and
says so.

**Simulation** is the main view: the stock in circulation, one line per
category, with a shaded band from the 10th to the 90th percentile across all
runs. Where the band is wide, the model is uncertain.

**Year spectrum** freezes one year and shows it as a bar chart: the simulated
mean with a whisker for the 10th–90th percentile, and, if you pick an
assemblage, the measured shares underneath with their counts.

*(STOCHASI 1 drew a pie here. The bar chart replaced it for two reasons: a pie
with twenty production centres is unreadable, and a pie cannot show the
simulation's uncertainty at all.)*

**Comparison** shows the difference between assemblage and simulation as
diverging bars. Categories that fall outside the simulated 10th–90th percentile
are marked in red. This is the view that says *where* the model and the evidence
disagree — and the red marks matter, because without them every bar looks like a
finding when most of them are just noise.

**Dating** is the inverse calculation, in two forms. *Curves* gives the
probability distribution over the years, the most likely year and a 95 %
interval, one curve per assemblage in its own colour. *Ranking* sorts your
assemblages by estimated date and draws their intervals as bars — the view that
turns several contexts into a sequence.

A row of buttons above the figure hides and shows individual curves, with *All*
and *None* beside them; hidden assemblages stay in the table below, only faint.
Each visible curve carries a dashed vertical at its most likely year, which the
*Mode lines* tickbox turns off. The footnote of every figure names how many
curves are hidden, so a figure cannot quietly claim more than it shows.

The ranking has a trap built into it, and the view is built to show it: a
sorted list looks like a sequence even when every interval overlaps and the
data support no order at all. So if there is a single year consistent with
*all* the intervals, that range is shaded behind the bars. If you can see that
band, the ordering you are reading off is not evidence.

**Data** is where you edit: the project name, categories, the market curve at its
reference years, and the assemblage counts with their colours.

---

## 5. Getting your own data in

Use **Load file…** in the header, or drag a file into the window. STOCHASI works
out what kind of table it is:

**A market table** has a year column (`Year`, `Jahr`, or a first column of
plausible years) and one further column per production centre. Values may be
percentages or raw counts — each row is normalised to 100 %.

```
Year;IT;LG;BA;MG;RZ
120;30;20;50;0;0
130;10;0;60;30;0
140;0;0;10;80;10
```

**An assemblage table** has no year column. Categories may be in the columns
(one row per context) or in the rows (one column per context) — STOCHASI detects
which and says so if it transposed. The second shape is the pivot table you get
from a provenance × context cross-tabulation, so both are supported.

```
Insula;IT;LG;BA;MG;RZ          Provenance;405;403;802
405;10;0;1;56;10               IT;10;3;1
403;3;0;0;20;7                 MG;56;20;5
```

The long format also works: columns `Type`/`Typ`/`Category` and
`Count`/`Anzahl`, optionally with a context column.

**Counts, not percentages.** The dating uses the sample size: 77 sherds and 770
sherds carry very different weight, and percentages throw that away. If STOCHASI
sees fractional values it will say so.

A column that matches none of the project's categories creates that category and
is reported. It then sits in the market curve with zeros, waiting to be filled;
without this the sherds would have dropped out silently and the dating would run
on too small an N.

If the project already held assemblages, a bar appears after loading offering
**Replace existing** or **Append to existing**. Replacing is the default;
appending is one click away while the bar is up, and it renumbers duplicate
names so that two excavations with an "Insula I" each stay apart.

Accepted file types: `.csv`, `.tsv`, `.xlsx`, `.xls`, and `.json` for STOCHASI
projects (version 1 or 2).

Four real files to try this with are in
[`examples/`](../examples/): a version 1 project, a market table and two
assemblages.

---

## 6. A first analysis, step by step

1. **Load your market curve** (Data → or drag the file in). If you do not have
   one yet, add reference years by hand in the Data view. Start coarse: five or
   six reference years across the period is usually enough.
2. **Pick your categories.** Data → *Add production centre* offers a reference
   list of 36 centres grouped by region, each with a rough production period.
   The list supplies names, colours and periods — **not** market shares. Those
   have to come from the literature on your site.
3. **Set the period** in the sidebar, then look at the *Simulation* view.
4. **Set the replacement rate.** Start at 10 % per year. Watch what changes: a
   lower rate makes the stock lag the market further behind. If you know that
   some ware had a very different service life, open *Rate per category*.
5. **Load your assemblages** and check the *Comparison* view for the year you
   expect. Red marks show where the evidence departs from the model.
6. **Go to Dating.** Read the most likely year and the interval, then tick
   *Compute across several residual shares* — see the next section for why.
7. **Fix the seed** (sidebar, *Random seed*) before you export anything for
   publication. With 0 the run is different every time.

---

## 7. Reading the dating curve honestly

Three things are worth knowing before you quote a number from this program.

**The curve is conditional.** It is the probability of the deposition year
*given* your market curve, your replacement rate and your residual share. Change
any of those and the answer changes. It is not an absolute date, and it does not
become one by being drawn precisely.

**Small samples give wide intervals — correctly.** With twelve sherds the
interval will be broad. That is not the program being cautious; it is the
evidence being thin. A method that gave you a narrow interval from twelve sherds
would be lying to you.

**Residuality moves the answer.** Redeposited older material makes an assemblage
look older, so the fitted year moves later to compensate. You cannot know the
residual share, but you can find out whether it matters: tick *Compute across
several residual shares* and STOCHASI draws the curve for 0, 10, 20 and 30 %.

- **The curves lie on top of each other** → the dating does not depend on the
  assumption. Report it.
- **The curves drift apart** → the dating is a function of an unknown. Report
  the span across all four curves, not the interval from one of them.

The overview table and the ranking view both use that wider span automatically
when the scan is on, so what you read there is the defensible figure rather
than the one from a single assumption.

For the example dataset the four curves sit almost on top of each other, and the
overview table therefore reports the span across all of them rather than a
single interval.

---

## 8. Export

The **Export** menu offers, for the view you are currently looking at:

- **PNG** — raster image at twice the resolution
- **SVG** — vector graphic, editable in Illustrator or Inkscape
- **PDF** — vector graphic for typesetting
- **CSV / XLSX** — the numbers behind the figure, plus a parameter sheet
- **Project file** — everything needed to reproduce the run

Two things worth knowing. Exported figures always use the light palette, even if
you are working in dark mode — a dark figure is useless in print. And the
exported figure is the same scene that is on your screen: there is no second,
prettier rendering path.

The table sheets stay complete even when curves are hidden in the figure; a
column *im Diagramm* records which assemblages were shown.

Every table export includes a **Parameter** sheet with the period, rates,
scatter, residual share, number of runs and seed. Attach it to a figure and the
run can be reproduced exactly.

---

## 9. Coming from STOCHASI 1

Version 1 configuration files load directly — drag the `.json` in. Categories,
market curve, period, initial distribution, scatter, runs, seed, settlement mode
and the excavation spectrum all carry over. STOCHASI reports anything it had to
adjust.

Two settings start neutral so that version 2 initially computes exactly what
version 1 computed: the residual share starts at 0, and there are no
category-specific replacement rates until you set one.

The reverse direction is deliberately not offered. Version 1 knows neither
category-specific rates nor residuality and would silently compute a v2 file
wrongly.

Version 1 itself remains available on the
[`v1` branch](https://github.com/oeai-dac/STOCHASI/tree/v1) of this repository.

---

## 10. Where to go next

- **[Complete Guide](GUIDE.md)** — the mathematics of the forward model, the
  Dirichlet-multinomial likelihood behind the dating, the residuality model,
  exact file formats, and a frank account of what the method cannot do.
- **[QUICKSTART.de.md](QUICKSTART.de.md)** — this guide in German.
- **[INSTALLATION.md](INSTALLATION.md)** — installation per platform *(German)*.
- **[BUILD.md](BUILD.md)** — building and releasing *(German)*.
- Questions and bug reports:
  [Issues](https://github.com/oeai-dac/STOCHASI/issues).
