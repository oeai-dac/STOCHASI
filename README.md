# STOCHASI

**Stochastic Chronological Artifact Simulation**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-FF4B4B.svg)](https://streamlit.io/)

A Streamlit-based tool for archaeological research that models and simulates the temporal transformation of artifact spectra using stochastic modeling and Monte Carlo methods.

## Overview

STOCHASI enables archaeologists to simulate how artifact distributions evolve over time under various market and cultural conditions. Originally developed for Terra Sigillata (Roman fine pottery) analysis, the tool supports any artifact classification system with temporal market supply data.

The fundamental premise: archaeological assemblages are dynamic entities whose composition changes over time due to continuous processes of acquisition, use, breakage, and replacement.

## Key Features

### 🎲 Monte Carlo Simulation
- Stochastic modeling with configurable noise parameters
- 10–500 simulation iterations for robust uncertainty quantification
- 80% confidence intervals (P10/P90) for all predictions
- Reproducible results via random seed control

### 📊 Replacement Model
Core mathematical model for artifact evolution:

```
S(t+1) = (1 - r) · S(t) + r · M(t)
```

Where:
- `S(t)` = artifact spectrum at time t
- `r` = replacement rate (0–50% annually)
- `M(t)` = market supply vector at time t

### 📈 Interactive Visualizations
- Market supply evolution plots
- Simulation timeline with confidence bands
- Year spectrum pie charts
- Comparison bar charts (simulation vs. excavation)
- Deviation analysis

### 🔧 Flexible Data Management
- Excel import for market supply data (.xlsx, .xls)
- Built-in market data editor with real-time validation
- JSON configuration export/import for full session persistence
- CSV export (full statistics or simplified)

### 📐 Statistical Analysis
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Maximum deviation identification
- Category-wise comparison metrics

### 🖼️ Publication-Ready Export

| Preset | Resolution | Use Case |
|--------|------------|----------|
| Screen | 72 DPI | Web, presentations |
| Print | 300 DPI | Publications |
| Print HD | 600 DPI | High-quality prints |
| Poster | A3/A2 | Large displays |

Available in PNG and SVG formats.

## Applications

- **Chronological Modeling**: Simulate artifact spectrum evolution over time
- **Settlement Dating**: Compare simulations with excavation data to estimate occupation periods
- **Uncertainty Quantification**: Confidence intervals via Monte Carlo methods
- **Hypothesis Testing**: Evaluate if assemblages match hypothesized settlement histories
- **Sensitivity Analysis**: Explore parameter effects on outcomes

## Terra Sigillata Categories (Default)

| Code | Production Center | Period | Region |
|------|-------------------|--------|--------|
| IT | Italian | 30 BC – 100 AD | Italy |
| LG | La Graufesenque | 20 – 120 AD | Southern Gaul |
| BA | Banassac | 80 – 160 AD | Southern Gaul |
| MG | Central Gaulish | 120 – 200 AD | Central Gaul |
| RZ | Rheinzabern | 150 – 280 AD | Germania Superior |

Custom categories can be defined for any artifact classification system.

## Platform Support

| OS | Status |
|----|--------|
| Windows 10/11 | ✅ Supported |
| macOS 12+ | ✅ Supported |
| Linux (Ubuntu 20.04+) | ✅ Supported |

## Documentation

Full technical documentation including mathematical foundations is available in the `docs/` folder.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Developed at the Austrian Academy of Sciences for archaeological chronology research.

---

**STOCHASI** — Bringing stochastic modeling to archaeological research
