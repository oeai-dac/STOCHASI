# Example files

Real data, taken over from STOCHASI 1. Drag any of them onto the application
window.

| File | What it is |
|---|---|
| `stochasi-1_flavia-solva_insula-xli.json` | A **version 1 project**: categories, market curve, parameters and the excavation spectrum in one file. Loading it demonstrates the migration path, and it is the dataset the application opens with. |
| `market-supply_terra-sigillata.xlsx` | A **market table** — a year column plus one column per ware. |
| `assemblage_flavia-solva_insula-xli.xlsx` | An **assemblage**: Insula XLI at Flavia Solva, 77 sherds. |
| `assemblage_carnuntum_auxiliary-fort.xlsx` | An **assemblage**: the auxiliary fort at Carnuntum. |

The market curve in these files is the working assumption used in the original
version 1 analysis and applies to those sites. It is not a general supply curve
for terra sigillata — see [§10 of the Complete Guide](../docs/GUIDE.md#10-the-reference-list-of-production-centres).

The version 1 project file is also used as a test fixture: `npm test` checks
that every value in it survives the migration to version 2 unchanged.
