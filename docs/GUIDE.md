# STOCHASI 2 — Complete Guide

Stochastic simulation of archaeological find spectra, and inverse dating of
assemblages from them.

MIT licence · © Christian Gugl / Austrian Archaeological Institute (ÖAW)

This is the reference document. For a first orientation, read the
[Quick Start Guide](QUICKSTART.md) instead — it takes ten minutes and covers the
normal working cycle.

---

## Table of Contents

1. [What the program claims, and what it does not](#1-what-the-program-claims-and-what-it-does-not)
2. [The forward model](#2-the-forward-model)
3. [Replacement rates](#3-replacement-rates)
4. [Stochastic scatter and the ensemble](#4-stochastic-scatter-and-the-ensemble)
5. [Residuality](#5-residuality)
6. [Inverse dating](#6-inverse-dating)
7. [Reading a dating result](#7-reading-a-dating-result)
8. [The data model](#8-the-data-model)
9. [File formats](#9-file-formats)
10. [The reference list of production centres](#10-the-reference-list-of-production-centres)
11. [The interface, view by view](#11-the-interface-view-by-view)
12. [Export](#12-export)
13. [Reproducibility](#13-reproducibility)
14. [Limits of the method](#14-limits-of-the-method)
15. [Performance](#15-performance)
16. [Architecture](#16-architecture)
17. [Migrating from version 1](#17-migrating-from-version-1)
18. [Troubleshooting](#18-troubleshooting)
19. [Citing STOCHASI](#19-citing-stochasi)

---

## 1. What the program claims, and what it does not

STOCHASI answers one question in two directions.

**Forwards:** given a market supply curve and a replacement rate, what should
the stock of pottery in circulation look like in each year?

**Backwards:** given a counted assemblage, which year of deposition makes those
counts most probable?

Both answers are **conditional on the model and its inputs**. STOCHASI does not
produce absolute dates. It produces the date that follows *if* the supply curve
you entered is right for your site, *if* the replacement rate is roughly right,
and *if* the residual share is roughly right. Change one of those and the answer
changes. The program is a way of making an argument explicit and testing how
much it depends on each assumption — not a way of avoiding the argument.

What it is genuinely good for:

- Making the difference between *market supply* and *stock in circulation*
  visible. These are routinely conflated, and the lag between them is exactly
  the size of the dating error that follows from conflating them.
- Turning a spectrum into a probability distribution over years, with the
  sample size properly accounted for. Twelve sherds and twelve hundred sherds
  produce visibly different intervals.
- Testing whether a conclusion survives the assumptions it rests on. The
  residual scan is the clearest case: if the four curves lie on top of each
  other, the conclusion is robust; if they drift apart, it is not.

What it cannot do is listed in [§14](#14-limits-of-the-method), and that section
is worth reading before the method appears in a publication.

---

## 2. The forward model

### 2.1 The three quantities

**Market supply** `m_c(t)` — the share of newly acquired material in year `t`
coming from production centre `c`, with `Σ_c m_c(t) = 100`. This is an input.

**Stock in circulation** `s_c(t)` — the composition of the material actually in
use at the site in year `t`, again normalised to 100. This is what the model
computes.

**Assemblage** `n_c` — counted sherds per category from an excavated context.
This is an input, and it must be counts.

### 2.2 The recursion

For every year in the period, for every category:

```
s_c(t) = s_c(t−1) · (1 − r_c)  +  m_c(t) · r_c  +  ε
```

then clipped at zero and renormalised so the year sums to 100. Here `ε` is drawn
from `N(0, σ)` in percentage points, and `r_c` is the annual replacement rate of
category `c`.

The interpretation: each year, a fraction `r_c` of the existing stock of
category `c` drops out (breakage, discard, wear) and is replaced by material
drawn from that year's market. With `r_c = 0` the stock never changes; with
`r_c = 1` the stock *is* the market supply, with no lag at all. Realistic values
for Roman fine ware are of the order of 5–20 % per year.

If all `r_c` are equal, this reduces exactly to the model of STOCHASI 1.

### 2.3 The initial distribution

`s_c(t₀ − 1)` — the stock in the year *before* the period starts — is an input
(sidebar, *Initial distribution*). It is normalised to 100.

The choice matters most at the start of the period and washes out over roughly
`3/r` years. With `r = 0.1` the influence of the initial distribution is
negligible after about thirty years. If your period of interest begins well
inside the modelled span, the initial distribution is not something to agonise
over.

**New foundation mode** replaces this: in the start year the stock is set equal
to that year's market supply, with no older material at all. Use it for sites
founded in the start year, where by construction there is no inherited stock.

### 2.4 Interpolation of the market curve

The market curve is given at reference years. Between them STOCHASI interpolates
linearly. **Outside** them it holds the edge value rather than extrapolating —
extrapolating supply shares beyond the documented span would be a claim the data
do not support. Every interpolated year is renormalised to 100.

The Market view marks the reference years with small ticks on the axis, so it is
always visible where you have data and where the program is filling in.

---

## 3. Replacement rates

### 3.1 The default rate

One slider, 0–50 % per year, applied to every category that has no rate of its
own. For most work this is the only rate you need.

### 3.2 Rates per category

Different wares have different service lives. A thin-walled beaker and a heavy
mortarium do not leave circulation at the same rate, and neither do a table
service in daily use and a display piece. STOCHASI therefore allows a rate per
category (sidebar → *Rate per category*).

A caution: this is a parameter you will rarely be able to justify from evidence.
Use it when you have a specific reason — a documented difference in vessel form
or function — and leave it alone otherwise. Every category-specific rate you set
is another free parameter, and free parameters make a model fit better without
making it more true.

Any rate you set is written into the exported parameter sheet, so a reader can
see what you assumed.

---

## 4. Stochastic scatter and the ensemble

### 4.1 What σ represents

The noise term `ε ~ N(0, σ)` is additive, in percentage points, applied
independently to each category in each year. It stands for everything the model
does not represent: irregular deliveries, uneven breakage, the chance of what
ended up in the ground and what was recovered.

It is explicitly **not** the sampling error of your excavated material. That
enters only in the inverse dating, through the sample size (§6).

σ = 2 is a reasonable starting value. Setting σ = 0 makes the model
deterministic — the band collapses to the line, and the dating loses the
mechanism by which simulation uncertainty is carried through.

### 4.2 The ensemble

STOCHASI runs the recursion `M` times (default 100, maximum 500), each with its
own random draws. The result is an **ensemble** of `M` trajectories. From it:

- the **mean** at each (year, category), drawn as a line;
- the **10th and 90th percentiles**, drawn as a band.

The band is the honest statement of what the model does and does not pin down.
Where it is wide, small differences between simulated and observed spectra mean
nothing.

More runs make the mean and the percentiles smoother; they do not make the
underlying uncertainty smaller. Beyond about 200 runs the visible improvement is
small.

---

## 5. Residuality

### 5.1 Why it is in the model

In settlement contexts a part of the finds is older than the layer containing
them: material worked up from levelling, pit fills, older deposits. This is the
largest systematic error in the whole exercise, and STOCHASI 1 did not model it
at all. Redeposited old material makes an assemblage look older than it is, and
the fitted date moves later to compensate.

### 5.2 The model

One parameter, `r_res ∈ [0, 0.5]`:

```
observed(t) = (1 − r_res) · s(t)  +  r_res · a(t)
a(t)        = mean of s(u) over all years u < t
```

The "old stock" `a(t)` is the equally weighted mean of the circulating stock in
all previous years — the simplest assumption that does not introduce a further
unknowable quantity such as a decay length. In the start year there are no
previous years, and the stock is left unchanged.

The mixture is applied to **every run individually**, not to the mean. This is
not a detail: averaging over previous years smooths the spectra, so residuality
makes the uncertainty band *narrower*, not wider. Applying the mixture to the
mean would have got the band wrong.

Two properties follow, and both are covered by tests:

- In equilibrium (a constant market and a stock that has converged to it),
  residuality changes nothing — the old stock equals the current one.
- With a market that shifts over time, residuality raises the share of the
  earlier categories and lowers that of the later ones.

### 5.3 How to use it

You cannot know `r_res` for your context. What you can find out is whether it
matters. In the Dating view, tick *Compute across several residual shares*:
STOCHASI computes the dating curve for 0, 10, 20 and 30 % and draws all four.

If the curves lie on top of each other, your dating does not depend on the
assumption — say so and report the interval. If they drift apart, your dating is
a function of an unknown, and the honest report is the span across all four
curves. The overview table gives you that span directly.

---

## 6. Inverse dating

### 6.1 The question

Given counts `n_c` with `N = Σ_c n_c`, and given the simulated stock for each
candidate year, which year makes those counts most probable?

### 6.2 The likelihood

For a known composition `p` (shares summing to 1), the probability of drawing
exactly the counts `n` is multinomial:

```
log L(t | p) = Σ_c n_c · log p_c(t)      + a constant independent of t
```

This is the right likelihood, and the reason for using it rather than a distance
measure is the sample size. A χ² distance on percentages cannot tell 12 sherds
from 1200: both give the same percentages. The multinomial can, and it does —
which is why a small assemblage produces a wide interval in STOCHASI rather than
a falsely precise one.

But `p(t)` is not known: the simulation gives an *ensemble* of possible
compositions for each year. The simulation's own uncertainty has to be
integrated out:

```
L(t) = ∫ L(t | p) dP(p | t)
```

STOCHASI offers two ways of doing that integral.

### 6.3 The analytic route (default)

Fit a Dirichlet distribution to the ensemble at year `t` by moment matching. For
a Dirichlet with concentration `α₀` and mean `m_c`:

```
Var_c = m_c(1 − m_c) / (α₀ + 1)
```

so `α₀` is estimated by averaging `m_c(1 − m_c)/Var_c − 1` across the categories
that actually vary, and `α_c = m_c · α₀`. The integral then has a closed form —
the Dirichlet-multinomial (Pólya) distribution:

```
log L(t) = log Γ(α₀) − log Γ(α₀ + N) + Σ_c [ log Γ(n_c + α_c) − log Γ(α_c) ]
```

Categories with `n_c = 0` contribute nothing and are skipped.

### 6.4 The Monte Carlo route

The direct approximation, averaging over the `M` runs:

```
L(t) = (1/M) Σ_m exp( log L(t | p⁽ᵐ⁾) )
```

computed with a log-sum-exp for numerical stability.

### 6.5 Why the analytic route is the default

The Monte Carlo sum looks more obvious, and for small assemblages it behaves
well. For larger ones it does not. The likelihood becomes sharply peaked, and
the sum comes to be dominated by whichever single run happens to fit best. The
resulting curve is jagged, and the jaggedness looks like structure: you read off
a peak, change the seed, and the peak has moved three years.

The Dirichlet fit integrates the same uncertainty analytically. It gives the
same location with a smooth curve, and the result is stable across seeds. The
test suite checks both properties — that the two methods agree on the date, and
that the analytic one produces fewer direction changes in the body of the curve
and less spread across seeds.

The Monte Carlo route stays available for comparison, and it is worth looking at
occasionally: if the two disagree noticeably, the ensemble is not well described
by a Dirichlet, which usually means too few runs or a strongly bimodal stock.

### 6.6 The floor on simulated shares

A simulated share is floored at `10⁻⁴` (0.01 %) before the logarithm. Without
this, a single sherd of a category that the model puts at exactly zero for some
year would exclude that year completely.

That would be a harshness the model does not earn. Misidentifications, heirloom
pieces and deliveries outside the modelled range all happen. At the floor, one
such sherd costs about 9 log units — substantial, but not disqualifying.

### 6.7 Posterior and interval

With a flat prior over the years, the normalised likelihood is the posterior
distribution of the deposition year. STOCHASI reports:

- the **mode** — the most probable single year;
- the **expected value** — the posterior mean, which differs from the mode when
  the distribution is skewed;
- the **95 % interval**, formed by taking years in order of descending
  probability until 95 % of the mass is covered, then reporting the first and
  last year of that set.

Note the last definition carefully. For a bimodal posterior the reported
interval also spans the low-probability years in between, so it is
**conservative** — wider rather than narrower than the strict highest-density
region. The curve itself shows the difference, which is why the curve and not
just the interval belongs in a publication.

---

## 7. Reading a dating result

A worked example, using the dataset that ships with the program.

Insula XLI at Flavia Solva: 77 sherds — 10 Italian, 0 La Graufesenque, 1
Banassac, 56 Central Gaulish, 10 Rheinzabern. With the market curve and a
replacement rate of 10 %, STOCHASI dates the assemblage to about AD 165, with a
95 % interval of roughly AD 157–178 and all four residual curves nearly
coincident.

Three observations follow, and they are the kind you should be making.

**The interval is 21 years wide from 77 sherds.** That is the correct order of
magnitude. Anyone quoting a decade from a sample this size is over-reading.

**The dating is robust against residuality here.** The four curves sit on top of
one another, so the answer does not hinge on an unknowable parameter. Report the
interval.

**The Italian ware does not fit.** Ten Italian sherds in the 160s is far more
than the model predicts; the Comparison view flags it in red. This is not a
failure of the program but a finding: either the market curve is wrong for
Italian ware at this site, or those ten sherds are residual, or they are
misattributed. The model has done its job by making the discrepancy explicit
instead of absorbing it.

---

## 8. The data model

A project consists of:

**Categories** — code (as used in table headers), name, colour, optional group.

**Market table** — reference years and, per category, a share at each of them.

**Parameters** — period, default replacement rate, per-category rates, scatter
σ, residual share, number of runs, seed, new-foundation flag, initial
distribution.

**Assemblages** — name and counts per category.

**Comparison year** — the year used by the Year spectrum and Comparison views.

All of it lives in one file. Attach that file to a publication and the figures
can be reproduced exactly.

---

## 9. File formats

### 9.1 Project file (`*.stochasi.json`)

JSON with `_meta.app = "STOCHASI"` and `_meta.version = "2.0"`.

```json
{
  "_meta": { "app": "STOCHASI", "version": "2.0", "created": "2026-08-31T…" },
  "name": "Flavia Solva, Insula XLI",
  "categories": [
    { "id": "MG", "name": "Central Gaulish", "color": "#228B22", "group": "Central Gaulish" }
  ],
  "market": { "years": [120, 130], "shares": { "MG": [0, 30] } },
  "params": {
    "startYear": 125, "endYear": 290,
    "replacementDefault": 0.1, "replacement": { "RZ": 0.14 },
    "noiseSd": 2, "residual": 0, "runs": 100, "seed": 42,
    "settlementMode": false,
    "initial": { "MG": 0 }
  },
  "assemblages": [{ "id": "A1", "name": "Insula XLI", "counts": { "MG": 56 } }],
  "comparisonYear": 170
}
```

Reading is tolerant. Duplicate or empty category codes are dropped, invalid
colours become grey, negative shares become zero, out-of-range parameters are
clamped, an end year before the start year is corrected, and assemblages with no
finds are skipped. Every correction is reported in the interface rather than
applied silently.

### 9.2 Market table (CSV, TSV, XLSX)

Detected when there is a year header (`Year`, `Jahr`, `Years`, `Date`, `Datum`,
…) or when the first column consists of strictly increasing plausible years.

- One column per category; the header is the category code.
- Values may be percentages or raw counts — each row is normalised to 100 %.
- Columns headed `Summe`, `Total`, `Sum`, `Gesamt` are ignored as marginals.
- Rows with an unreadable year are skipped and reported.
- Duplicate years: the first wins, and you are told.
- Rows are sorted by year on import.

### 9.3 Assemblage table (CSV, TSV, XLSX)

Detected when there is no year column. Three shapes are accepted.

**Categories in columns** — one row per context:

```
Insula;IT;LG;BA;MG;RZ
405;10;0;1;56;10
403;3;0;0;20;7
```

**Categories in rows** — one column per context. This is the shape of a
provenance × context pivot table:

```
Provenance;405;403;802
IT;10;3;1
MG;56;20;5
```

**Long format** — one row per (category, count), optionally with a context
column:

```
Insula;Type;Count
405;IT;10
405;MG;56
403;IT;3
```

Orientation for the wide shapes is detected by matching the headers and the
first column against the codes already in the project; whichever side matches
more wins. Without a project to compare against, short code-like labels are
taken as the categories. The interface says when it transposed. Rows and columns
headed `Summe`/`Total`/`Gesamt`/`FSt?` are treated as marginals and ignored.

Numbers are parsed tolerantly: decimal commas, thousands separators, and cells
of the form `56 (72,7 %)` from formatted pivot exports all work — the first
number in the cell is taken.

### 9.4 Counts versus percentages

The dating uses `N`. Percentages discard it, and an interval computed from
percentages read as counts is far too narrow. If STOCHASI sees fractional
values in an assemblage table it says so, and it repeats the point in the Data
view. Give it counts.

---

## 10. The reference list of production centres

The Data view offers 36 terra sigillata production centres, grouped by region,
each with a code, a name, a colour and a rough production period.

**What the list is:** a convenience. It saves you creating categories by hand and
keeps the colours consistent across projects, so two figures from different sites
can sit on the same page.

**What the list is not:** a data source for the simulation. STOCHASI computes
with *market shares per year* — the share of a specific site's newly acquired
material coming from each centre. Those curves are regional, they differ between
sites, and they are deliberately **not** shipped. Inventing them would be worse
than useless. The market curve has to come from the literature on your site or
from your own counts.

The periods below are rounded orientation values from the standard handbook
literature, used only to pre-fill a plausible time window and to order the list.
`approximate` marks centres whose span is debated or only roughly outlined in
regional research. **Check both against the literature for your own study area
before using them in a publication.**

| Code | Centre | Production period | Basis |
|---|---|---|---|
| | **Italian** | | |
| `AR` | Arezzo | 40 BC – AD 40 | established |
| `PI` | Pisa | 20 BC – AD 60 | established |
| `MI` | Central Italy (Mittelitalien) | 20 BC – AD 50 | approximate |
| `PA` | Padana | AD 1 – 150 | established |
| `TP` | Tardopadana | AD 150 – 400 | approximate |
| | **South Gaulish** | | |
| `LG` | La Graufesenque | AD 20 – 120 | established |
| `MT` | Montans | AD 20 – 150 | established |
| `BA` | Banassac | AD 80 – 150 | established |
| `ES` | Espalion | AD 40 – 70 | approximate |
| | **Central Gaulish** | | |
| `MV` | Les Martres-de-Veyre | AD 100 – 165 | established |
| `LZ` | Lezoux | AD 120 – 200 | established |
| `TF` | Terre-Franche | AD 120 – 200 | approximate |
| | **East Gaulish** | | |
| `CH` | Chémery-Faulquemont | AD 50 – 120 | approximate |
| `LM` | La Madeleine | AD 100 – 160 | established |
| `BW` | Blickweiler | AD 100 – 180 | established |
| `HB` | Heiligenberg | AD 120 – 180 | established |
| `LV` | Lavoye | AD 130 – 260 | established |
| `TR` | Trier | AD 130 – 300 | established |
| `RZ` | Rheinzabern | AD 140 – 260 | established |
| `IW` | Ittenweiler | AD 150 – 200 | approximate |
| `SZ` | Sinzig | AD 150 – 250 | approximate |
| `WB` | Waiblingen | AD 150 – 200 | approximate |
| `AG` | Argonnen | AD 300 – 450 | established |
| | **Raetian** | | |
| `WD` | Westerndorf | AD 180 – 240 | established |
| `PF` | Pfaffenhofen | AD 220 – 270 | established |
| | **Pannonian / local** | | |
| `AQ` | Aquincum | AD 100 – 250 | approximate |
| `SI` | Siscia | AD 150 – 250 | approximate |
| `NP` | Noric-Pannonian ware | AD 50 – 250 | approximate |
| | **North African** | | |
| `AA` | African Red Slip A | AD 80 – 200 | established |
| `AC` | African Red Slip C | AD 200 – 320 | established |
| `AD` | African Red Slip D | AD 320 – 600 | established |
| | **Eastern** | | |
| `EA` | Eastern Sigillata A | 150 BC – AD 150 | established |
| `EB` | Eastern Sigillata B | 25 BC – AD 150 | established |
| `EC` | Çandarlı (ESC) | AD 50 – 200 | established |
| `PS` | Pontische Sigillata | 50 BC – AD 150 | approximate |
| `PH` | Phokäische Ware | AD 400 – 700 | approximate |

Compared with the nineteen provenances used in the Flavia Solva analysis, this
list adds Montans, Trier, Sinzig, La Madeleine, Chémery-Faulquemont,
Terre-Franche, Espalion, the Pannonian and local wares, the African Red Slip
groups and the eastern sigillata families. Anything missing can be added as a
custom category with its own code and colour.

---

## 11. The interface, view by view

### 11.1 The sidebar

Every parameter that drives the simulation lives here, with a short note on what
it means in the model. That is not decoration: a replacement rate of 30 % read
without knowing that it means a third of the stock leaving circulation every
year will be read wrongly.

*Period* · *Replacement rate per year* (and, folded away, *Rate per category*) ·
*Scatter σ* · *Residual share* · *Runs* · *Random seed* · *New foundation* ·
*Initial distribution*.

Two conveniences on the initial distribution: **Take from the market in the
start year** fills it with that year's supply, which is the right choice for a
site with no inherited stock but a normal (non-foundation) history; **Normalise
to 100 %** rescales what you typed.

### 11.2 Market supply

The interpolated curve, one line per category, with ticks marking the reference
years. Where there are no ticks, the program is interpolating.

### 11.3 Simulation

The stock in circulation: mean lines with a 10th–90th percentile band. The band
can be switched off for a cleaner figure, but leaving it on is usually the
honest choice.

The parameter note in the top right of the figure carries the rate, σ, the
number of runs, the residual share if non-zero, and the seed — so a screenshot
still says how it was produced.

### 11.4 Year spectrum

One year as horizontal bars: the simulated mean with a 10th–90th percentile
whisker, and, if an assemblage is selected, the measured shares beneath with
their counts.

Version 1 drew a pie chart here. It was replaced because a pie with twenty
production centres cannot be read, and because a pie cannot show the
simulation's uncertainty at all.

### 11.5 Comparison

The difference between assemblage and simulation, in percentage points,
as diverging bars. Categories whose measured share falls **outside** the
simulated 10th–90th percentile are marked in red at the bar tip and in the
label.

The marking is the point of the view. Without it every bar looks like a finding,
when most of them are within the range the model itself considers ordinary.

### 11.6 Dating

The posterior curve over the years, with the 95 % interval shaded and the mode
marked. Below it, a table across all assemblages: sample size, mode, expected
value, interval and width.

Two modes. Unticked, one curve per assemblage — this is the direct comparison
between contexts. Ticked (*Compute across several residual shares*), four curves
for one assemblage at 0, 10, 20 and 30 % residuality. In the second mode the
overview table reports the span across all four curves, because that, not the
interval from any single assumption, is the defensible statement.

A warning appears when any assemblage has fewer than 30 finds. It says the
interval is wide because the evidence is thin, not because the computation is
being timid.

### 11.7 Data

Categories (code, name, colour; the code can be renamed and the change
propagates through market curve, initial values, rates and all assemblages), the
market curve at its reference years, and the assemblage counts.

The sum column in the market table doubles as a *normalise this row* button. It
turns red only when the row actually departs from 100.

### 11.8 Interface conveniences

- **Theme** — light and dark, following the system setting until you choose.
  Category colours are automatically lightened in dark mode when they would
  otherwise vanish into the background; exported figures are unaffected.
- **Language** — German and English, following the browser language until you
  choose.
- **Share** — copies a link containing the whole project, compressed into the
  URL fragment. The fragment is never sent to a server. Large projects do not
  fit, and you are told so rather than handed a broken link.
- **Autosave** — the session is stored in IndexedDB when the page is hidden or
  closed, and offered for restoration on the next start.
- **Drag and drop** — a file dropped anywhere on the window is imported.

---

## 12. Export

The Export menu always acts on the view you are currently looking at.

| Format | Contents |
|---|---|
| PNG | The figure, rasterised at twice the resolution |
| SVG | The figure as vectors, editable in Illustrator or Inkscape |
| PDF | The figure as vectors, for typesetting |
| CSV | The first table of the current view, with a UTF-8 BOM so Excel on Windows reads it correctly |
| XLSX | All tables of the current view as separate sheets |
| Project file | The complete project |

Two guarantees are worth stating.

**One scene, three formats.** The figure on screen and the figure in the
exported file are built from the same scene description. There is no second
rendering path, so what you check on screen is what lands in the file. The one
deliberate difference: exports always use the light palette, because a dark
figure is unusable in print.

**No transparency.** Uncertainty bands are pre-blended against the background
rather than drawn with an alpha channel. Alpha in PDF requires an ExtGState and
renders differently in different viewers; pre-blending makes SVG, PNG and PDF
pixel-identical.

Every table export includes a **Parameter** sheet listing period, rates, σ,
residual share, runs, seed, foundation mode, comparison year and initial
distribution. Attach it to a figure and the run is reproducible from the
document alone.

### 12.1 The tables in detail

- **Simulation** — one row per year, three columns per category (mean, P10, P90).
- **Market supply** — one row per year, interpolated, with a column marking which
  years are reference years.
- **Assemblages** — one row per assemblage, counts per category, with totals.
- **Dating (overview)** — per assemblage: N, mode, expected value, interval,
  width, level, method, and which categories were actually present.
- **Dating (curves)** — one row per year, one column per assemblage, giving the
  posterior probability. Sums to 1 down each column.

---

## 13. Reproducibility

Three things determine whether a run can be repeated exactly.

**The seed.** With seed 0 every run draws fresh randomness. Set a fixed seed
before producing anything for publication, and quote it in the figure caption.
The parameter note in the corner of each figure says `Seed random` when it is
not fixed.

**The parameters.** All of them are in the exported parameter sheet and in the
project file.

**The version.** Numerical results are tied to the version of the program. The
release the figure was made with belongs in the caption. Version 2.0 with
residuality at 0 and no category-specific rates reproduces the results of
version 1 exactly.

A defensible figure caption looks roughly like this:

> Simulated stock in circulation, Insula XLI. Market curve after [reference],
> replacement rate 10 %/year, σ = 2, 200 runs, seed 42, residual share 0.
> STOCHASI 2.0.0.

---

## 14. Limits of the method

This section is the one to read before the method appears in print.

**The market curve is an assumption, not a measurement.** Everything downstream
inherits its errors. If the supply curve for a ware is wrong by twenty years,
the dating is wrong by roughly twenty years, and nothing in the program will
tell you.

**The replacement rate is barely constrained by evidence.** Values between 5 and
20 % are all defensible for Roman fine ware, and they give visibly different
answers. The right response is to run the range and report what changes, not to
pick one and present the result as if the choice were free of consequence.

**Residuality is modelled crudely.** One parameter, an equally weighted mean
over previous years. Real residuality is neither uniform nor equally weighted
across all earlier periods. The model is good enough to reveal whether a
conclusion is sensitive to residuality at all; it is not good enough to
*correct* for it.

**The multinomial assumes independent draws.** Sherds from a single vessel are
not independent, and a context dominated by a few smashed pots will produce an
interval that is too narrow. Counting by minimum number of vessels rather than
sherd count reduces the problem but does not remove it.

**Deposition is not the same as production, use or discard.** STOCHASI dates the
composition of the material in circulation at the moment of deposition. What
that moment means archaeologically — a levelling event, gradual accumulation, a
destruction horizon — is your interpretation, and gradual accumulation in
particular is not what the model describes.

**A flat prior over years is a choice.** If you have independent dating evidence
— coins, stratigraphy, dendrochronology — it is *not* in the curve. Combining
them is your job, and the curve is one input to that argument rather than its
conclusion.

**The interval is conservative for bimodal posteriors.** See [§6.7](#67-posterior-and-interval).
Show the curve, not only the interval.

**What is deliberately absent.** Weibull or gamma service-life distributions
instead of a constant rate; aoristic analysis of individually dated finds; coins
as a second dating source; formal sensitivity analysis; ABC estimation of the
rate and residual share alongside the date. Each was considered for version 2
and left out on purpose — the first three because they need a different kind of
input data, the last two because they are worth doing properly rather than
quickly.

---

## 15. Performance

Measured on an ordinary desktop machine:

| Size | Simulation | Residuality | Dating |
|---|---|---|---|
| 5 categories × 175 years × 100 runs | 25 ms | 13 ms | 6 ms |
| 19 categories × 400 years × 200 runs | 205 ms | 123 ms | 8 ms |
| 36 categories × 700 years × 500 runs | 1.7 s | 1.1 s | 45 ms |

The simulation and the dating run in a **web worker**, so the interface stays
responsive even at the top of that range. Parameter changes are debounced: while
you drag a slider, intermediate values are discarded and only the value you
settle on is computed.

Two optimisations are worth knowing about. The percentiles — a sort per
(year, category) across all runs — are the single largest cost, and they are
skipped entirely for the dating, which reads the runs directly. And the worker
returns only the summary (mean and percentiles) rather than the full ensemble,
which at the top of the table would be tens of megabytes.

If workers are unavailable, the same code runs on the main thread. The results
are identical; the interface stutters.

---

## 16. Architecture

React 18 + TypeScript + Vite, packaged for the desktop with Electron and
electron-builder. Runtime dependencies: React, and SheetJS loaded only when
XLSX is actually touched.

**No third-party libraries for the domain logic.** The Monte Carlo core, the
Dirichlet-multinomial likelihood, the log-gamma function, the chart rendering,
the SVG/PNG/PDF writers, the internationalisation and the icon generation are
all written in-house and covered by tests. The reason is auditability: a
published date should rest on code a reader can follow.

**Everything stays local.** There is no backend. In the desktop edition a
content security policy forbids outbound connections entirely, the interface is
served over a private `app://` scheme rather than `file://` (which would block
module workers), and the fonts are bundled rather than fetched from a CDN.

Layout of the source:

```
src/core/      data model, project and table IO, theme, share link, autosave
src/sim/       simulation, residuality, inverse dating, statistics
src/charts/    scene description and the five chart builders; SVG/PNG/PDF writers
src/data/      the reference list of centres, the example dataset
src/export/    tables, downloads, text metrics
src/workers/   the compute worker and its client
src/components/ the interface
src/i18n/      German and English
electron/      main process, preload, self-test
```

The test suite is framework-free — plain TypeScript files run with `tsx`, each
printing its own results. `npm test` runs all of them.

---

## 17. Migrating from version 1

Drop a version 1 `.json` onto the window. Carried over: categories with their
codes, names and colours; the market curve (renormalised to 100 per row if it
was not); period; initial distribution; scatter; runs; seed; settlement mode;
and the excavation spectrum as an assemblage.

Not carried over: display settings (band opacity, line width, auto-normalise),
which belonged to the old interface, and the version 1 configuration URLs, which
are replaced by the share link.

Two settings start neutral so that version 2 initially reproduces version 1
exactly: residuality at 0, and no category-specific replacement rates.

One case needs attention. If the version 1 file stored the excavation spectrum
only as percentages, the sample size was never recorded. STOCHASI imports the
percentages, but says clearly that the dating interval will be too narrow until
you enter the actual counts.

Writing version 1 files is deliberately not supported: version 1 knows neither
category-specific rates nor residuality and would compute a version 2 file
silently wrongly.

Version 1 remains available on the
[`v1` branch](https://github.com/oeai-dac/STOCHASI/tree/v1), including its
original documentation.

---

## 18. Troubleshooting

**"The simulation needs a market curve."** No reference years yet. Load a market
table, or add years by hand in the Data view.

**The dating curve is jagged.** Increase the number of runs. If it stays jagged
with 300+ runs, check whether you have switched to the Monte Carlo method —
the analytic one is the default for exactly this reason (§6.5).

**The dating interval is implausibly narrow.** Almost always percentages
imported as counts. Check the Data view: the totals should be your actual sherd
counts.

**All categories are red in the Comparison view.** The simulation and the
evidence disagree across the board. Usually the comparison year is wrong, or the
market curve does not fit the site. Try the Dating view first: it will tell you
which year the assemblage actually points to.

**The import put the categories in the wrong direction.** The orientation guess
failed, which happens with unfamiliar codes. Load the market curve first so the
project knows its category codes, then load the assemblages — with codes to
match against, the detection is reliable.

**The share link says the project is too large.** Market curves with hundreds of
reference years or dozens of assemblages exceed what a URL can hold. Pass on the
project file instead.

**Windows or macOS refuses to open the application.** The packages are not
signed. [INSTALLATION.md](INSTALLATION.md) explains the two clicks needed
*(German)*.

**Nothing computes at all after loading a file.** Check the period: if the end
year is not after the start year, the simulation does not run, and the sidebar
says so in red.

---

## 19. Citing STOCHASI

The repository contains a `CITATION.cff`, from which GitHub generates a *Cite
this repository* button. Please quote the version you used — numerical results
are tied to it.

> Gugl, Christian (2026): *STOCHASI 2 — Stochastic simulation of archaeological
> find spectra*. Austrian Archaeological Institute, Austrian Academy of
> Sciences. <https://github.com/oeai-dac/STOCHASI>

And please quote the parameters. A date from this program without the market
curve, the replacement rate and the seed behind it cannot be checked by anyone,
which rather defeats the purpose.
