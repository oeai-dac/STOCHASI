"""
STOCHASI - Stochastic Chronological Artifact Simulation
Version: 1.0
Designed by Christian Gugl (christian.gugl@oeaw.ac.at)
Austrian Academy of Sciences (ÖAW)
Coded with the assistance of Anthropic Claude
License: MIT License
Copyright (c) 2026 Christian Gugl
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
"""

from __future__ import annotations

import sys

# Check if running in test mode
RUNNING_TESTS = len(sys.argv) > 1 and sys.argv[1] == "--test"

# Core imports
import hashlib
import io
import json
import logging
import urllib.parse
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional imports (only for app, not for tests)
if not RUNNING_TESTS:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*keyword arguments.*")
    warnings.filterwarnings("ignore", message=".*deprecated.*")

    import base64
    from concurrent.futures import ProcessPoolExecutor, as_completed

    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st

    import streamlit.logger
    streamlit.logger.get_logger("streamlit").setLevel(logging.ERROR)

    def cache_data(show_spinner=False):
        return st.cache_data(show_spinner=show_spinner)
else:
    def cache_data(show_spinner=False):
        def decorator(func):
            return func
        return decorator


# =========================================================
# LOGGING CONFIGURATION
# =========================================================
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =========================================================
# DYNAMIC CATEGORY MANAGEMENT
# =========================================================
class CategoryManager:
    """Manages dynamically loaded categories from the data file."""

    def __init__(self):
        self._categories: List[str] = []
        self._category_names: Dict[str, str] = {}
        self._colors: Dict[str, str] = {}
        self._is_initialized: bool = False

    def initialize_from_dataframe(self, df: pd.DataFrame, year_column: str = "Year") -> None:
        """Initialize categories from a DataFrame."""
        excluded_cols = {year_column.lower(), 'summe', 'summe ', 'sum', 'total', 'jahr', 'year'}
        self._categories = [
            col for col in df.columns
            if col.lower().strip() not in excluded_cols
            and not col.startswith('_')
        ]
        self._colors = self._generate_colors(self._categories)
        self._category_names = {cat: cat for cat in self._categories}
        self._is_initialized = True
        logger.info(f"Categories initialized: {self._categories}")

    def add_category(self, name: str) -> None:
        """Add a new category."""
        if name not in self._categories:
            self._categories.append(name)
            self._colors = self._generate_colors(self._categories)
            self._category_names[name] = name

    def _generate_colors(self, categories: List[str]) -> Dict[str, str]:
        """Generate a color palette for the categories."""
        known_colors = {
            "IT": "#8B0000", "LG": "#FF6666", "BA": "#1E90FF",
            "MG": "#228B22", "RZ": "#FF8C00",
        }
        color_palette = [
            "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
            "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
            "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000",
            "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080",
        ]
        colors = {}
        color_idx = 0
        for cat in categories:
            if cat in known_colors:
                colors[cat] = known_colors[cat]
            else:
                colors[cat] = color_palette[color_idx % len(color_palette)]
                color_idx += 1
        return colors

    def set_category_names(self, names: Dict[str, str]) -> None:
        self._category_names.update(names)

    @property
    def categories(self) -> List[str]:
        return self._categories

    @property
    def category_names(self) -> Dict[str, str]:
        return self._category_names

    @property
    def colors(self) -> Dict[str, str]:
        return self._colors

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized


category_manager = CategoryManager()


# =========================================================
# PRESET CONFIGURATIONS
# =========================================================
PRESET_CONFIGS = {
    "Terra Sigillata": {
        "category_names": {
            "IT": "Italian", "LG": "La Graufesenque", "BA": "Banassac",
            "MG": "Central Gaulish", "RZ": "Rheinzabern",
        },
        "colors": {
            "IT": "#8B0000", "LG": "#FF6666", "BA": "#1E90FF",
            "MG": "#228B22", "RZ": "#FF8C00",
        },
        "min_year": 1, "max_year": 500, "default_start": 125, "default_end": 300,
        "title": "Terra Sigillata Spectra", "icon": "⚱️",
    },
    "Custom": {
        "category_names": {}, "colors": {},
        "min_year": -5000, "max_year": 2000, "default_start": 0, "default_end": 300,
        "title": "Artifact Spectra", "icon": "📊",
    },
}


# =========================================================
# CONSTANTS
# =========================================================
MIN_RUNS, MAX_RUNS, DEFAULT_RUNS = 10, 500, 100
MAX_REPLACEMENT_RATE, DEFAULT_REPLACEMENT_RATE = 0.5, 0.1
MAX_NOISE_SD, DEFAULT_NOISE_SD = 10.0, 2.0

EXPORT_PRESETS: Dict[str, Dict[str, Any]] = {
    "screen": {"name": "Screen (72 DPI)", "width": 1200, "height": 800, "scale": 1.0,
               "description": "For screen display and web"},
    "print_300dpi": {"name": "Print (300 DPI)", "width": 2400, "height": 1600, "scale": 2.5,
                    "description": "High resolution for publications"},
    "print_600dpi": {"name": "Print (600 DPI)", "width": 3600, "height": 2400, "scale": 3.0,
                    "description": "Very high resolution for large format"},
    "poster": {"name": "Poster (A3/A2)", "width": 4800, "height": 3200, "scale": 4.0,
               "description": "For posters and large prints"},
}


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def normalize(arr: np.ndarray) -> np.ndarray:
    total = arr.sum()
    return (arr / total) * 100 if total != 0 else arr


def hex_to_rgba(hex_color: str, opacity: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{opacity})"


def validate_numeric_columns(df: pd.DataFrame, columns: List[str]) -> Tuple[bool, str]:
    for col in columns:
        if col not in df.columns:
            return False, f"Column '{col}' missing"
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                pd.to_numeric(df[col])
            except (ValueError, TypeError):
                return False, f"Column '{col}' contains non-numeric values"
    return True, ""


def create_data_hash(*args: Any) -> str:
    return hashlib.md5(str(args).encode()).hexdigest()


def normalize_market_row(row: pd.Series, categories: List[str]) -> pd.Series:
    """Normalize a row so that the sum of category values equals 1."""
    row = row.copy()
    total = sum(row[cat] for cat in categories if cat in row and pd.notna(row[cat]))
    if total > 0:
        for cat in categories:
            if cat in row:
                row[cat] = row[cat] / total
    return row


def normalize_market_dataframe(df: pd.DataFrame, categories: List[str]) -> pd.DataFrame:
    """Normalize all rows in a market DataFrame so each row sums to 1."""
    df = df.copy()
    for idx in df.index:
        total = sum(df.loc[idx, cat] for cat in categories if cat in df.columns and pd.notna(df.loc[idx, cat]))
        if total > 0:
            for cat in categories:
                if cat in df.columns:
                    df.loc[idx, cat] = df.loc[idx, cat] / total
    return df


# =========================================================
# URL PARAMETER MANAGEMENT
# =========================================================
def get_config_from_url_params() -> Dict[str, Any]:
    if RUNNING_TESTS:
        return {}
    params = st.query_params
    config = {}

    param_mappings = [
        ("start", "start_year", int), ("end", "end_year", int),
        ("rate", "replacement_rate", float), ("noise", "noise_sd", float),
        ("runs", "n_runs", int), ("seed", "seed", int),
        ("opacity", "uncertainty_opacity", float), ("linewidth", "line_width", int),
    ]

    for url_key, config_key, converter in param_mappings:
        if url_key in params:
            try:
                config[config_key] = converter(params[url_key])
            except (ValueError, TypeError):
                pass

    if "settlement" in params:
        config["settlement_mode"] = params["settlement"].lower() == "true"
    if "normalize" in params:
        config["auto_normalize"] = params["normalize"].lower() == "true"
    if "demo" in params:
        config["use_demo"] = params["demo"].lower() == "true"
    if "preset" in params:
        config["preset_choice"] = params["preset"]

    if "init" in params:
        try:
            init_dict = {}
            for pair in params["init"].split(","):
                if ":" in pair:
                    cat, val = pair.split(":")
                    init_dict[cat.strip()] = int(val.strip())
            config["initial_values"] = init_dict
        except (ValueError, TypeError):
            pass

    return config


def set_url_params_from_config(config: Dict[str, Any]) -> None:
    if RUNNING_TESTS:
        return
    params = {}
    if "start_year" in config:
        params["start"] = str(config["start_year"])
    if "end_year" in config:
        params["end"] = str(config["end_year"])
    if "replacement_rate" in config:
        params["rate"] = f"{config['replacement_rate']:.2f}"
    if "noise_sd" in config:
        params["noise"] = f"{config['noise_sd']:.1f}"
    if "n_runs" in config:
        params["runs"] = str(config["n_runs"])
    if "seed" in config and config["seed"] != 0:
        params["seed"] = str(config["seed"])
    if "uncertainty_opacity" in config:
        params["opacity"] = f"{config['uncertainty_opacity']:.2f}"
    if "line_width" in config:
        params["linewidth"] = str(config["line_width"])
    if "settlement_mode" in config:
        params["settlement"] = str(config["settlement_mode"]).lower()
    if "auto_normalize" in config:
        params["normalize"] = str(config["auto_normalize"]).lower()
    if "use_demo" in config and config["use_demo"]:
        params["demo"] = "true"
    if "preset_choice" in config:
        params["preset"] = config["preset_choice"]
    if "initial_values" in config and config["initial_values"]:
        params["init"] = ",".join(f"{k}:{v}" for k, v in config["initial_values"].items())
    st.query_params.update(params)


def create_shareable_url(config: Dict[str, Any], base_url: str = "") -> str:
    params = {}
    if "start_year" in config:
        params["start"] = config["start_year"]
    if "end_year" in config:
        params["end"] = config["end_year"]
    if "replacement_rate" in config:
        params["rate"] = f"{config['replacement_rate']:.2f}"
    if "noise_sd" in config:
        params["noise"] = f"{config['noise_sd']:.1f}"
    if "n_runs" in config:
        params["runs"] = config["n_runs"]
    if "seed" in config and config["seed"] != 0:
        params["seed"] = config["seed"]
    if "uncertainty_opacity" in config:
        params["opacity"] = f"{config['uncertainty_opacity']:.2f}"
    if "line_width" in config:
        params["linewidth"] = config["line_width"]
    if "settlement_mode" in config:
        params["settlement"] = str(config["settlement_mode"]).lower()
    if "auto_normalize" in config:
        params["normalize"] = str(config["auto_normalize"]).lower()
    if "use_demo" in config and config["use_demo"]:
        params["demo"] = "true"
    if "preset_choice" in config:
        params["preset"] = config["preset_choice"]
    if "initial_values" in config and config["initial_values"]:
        params["init"] = ",".join(f"{k}:{v}" for k, v in config["initial_values"].items())
    query_string = urllib.parse.urlencode(params)
    return f"{base_url}?{query_string}" if base_url else f"?{query_string}"


# =========================================================
# JSON CONFIGURATION
# =========================================================
def config_to_json(config: Dict[str, Any], pretty: bool = True,
                   market_data: Optional[pd.DataFrame] = None,
                   excavation_data: Optional[Dict[str, float]] = None,
                   excavation_data_absolute: Optional[Dict[str, int]] = None,
                   categories: Optional[List[str]] = None,
                   category_names: Optional[Dict[str, str]] = None,
                   edited_market_data: Optional[pd.DataFrame] = None,
                   excavation_year: Optional[int] = None) -> str:
    """Export configuration to JSON, optionally including edited market data."""
    export_config = {
        "_meta": {
            "app": "STOCHASI", "version": "1.0",
            "description": "Stochastic Chronological Artifact Simulation",
            "includes_data": market_data is not None or edited_market_data is not None,
            "market_data_edited": edited_market_data is not None,
        },
        "parameters": config,
    }
    if categories:
        export_config["categories"] = {"list": categories, "names": category_names or {}}
    
    # Use edited market data if available, otherwise use original
    export_market = edited_market_data if edited_market_data is not None else market_data
    if export_market is not None:
        year_col = "Year" if "Year" in export_market.columns else "Jahr"
        market_dict = {"years": export_market[year_col].tolist()}
        for cat in (categories or []):
            if cat in export_market.columns:
                market_dict[cat] = export_market[cat].tolist()
        export_config["market_data"] = market_dict
        
    if excavation_data is not None:
        export_config["excavation_data"] = {
            "percent": excavation_data, 
            "absolute": excavation_data_absolute or {},
            "comparison_year": excavation_year
        }
    return json.dumps(export_config, indent=2 if pretty else None, ensure_ascii=False)


def config_to_json_simple(config: Dict[str, Any], pretty: bool = True) -> str:
    export_config = {
        "_meta": {
            "app": "STOCHASI", "version": "1.0",
            "description": "Stochastic Chronological Artifact Simulation",
            "includes_data": False,
        },
        "parameters": config,
    }
    return json.dumps(export_config, indent=2 if pretty else None, ensure_ascii=False)


def json_to_config(json_str: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        data = json.loads(json_str)
        if "_meta" in data and "parameters" in data:
            return data["parameters"], None
        return data, None
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {str(e)}"
    except Exception as e:
        return None, f"Error: {str(e)}"


def json_to_full_config(json_str: str) -> Tuple[Optional[Dict], Optional[pd.DataFrame], Optional[Dict], Optional[Dict], Optional[List], Optional[int], Optional[str]]:
    """Parse JSON config and return: parameters, market_df, excavation_percent, excavation_absolute, categories, excavation_year, error"""
    try:
        data = json.loads(json_str)
        parameters = data.get("parameters", data) if "_meta" in data else data

        market_df = None
        if "market_data" in data:
            market_dict = data["market_data"]
            if "years" in market_dict:
                df_data = {"Year": market_dict["years"]}
                for key, values in market_dict.items():
                    if key != "years":
                        df_data[key] = values
                market_df = pd.DataFrame(df_data)

        excavation_percent = data.get("excavation_data", {}).get("percent") if "excavation_data" in data else None
        excavation_absolute = data.get("excavation_data", {}).get("absolute") if "excavation_data" in data else None
        excavation_year = data.get("excavation_data", {}).get("comparison_year") if "excavation_data" in data else None
        categories = data.get("categories", {}).get("list") if "categories" in data else (
            [col for col in market_df.columns if col != "Year"] if market_df is not None else None
        )

        return parameters, market_df, excavation_percent, excavation_absolute, categories, excavation_year, None
    except json.JSONDecodeError as e:
        return None, None, None, None, None, None, f"Invalid JSON: {str(e)}"
    except Exception as e:
        return None, None, None, None, None, None, f"Error: {str(e)}"


def check_json_has_data(json_str: str) -> Tuple[bool, bool]:
    try:
        data = json.loads(json_str)
        return bool(data.get("market_data")), bool(data.get("excavation_data"))
    except:
        return False, False


def get_current_config(start_year, end_year, replacement_rate, noise_sd, n_runs, seed,
                       uncertainty_opacity, line_width, settlement_mode, auto_normalize,
                       use_demo, preset_choice, initial_values) -> Dict[str, Any]:
    return {
        "start_year": start_year, "end_year": end_year, "replacement_rate": replacement_rate,
        "noise_sd": noise_sd, "n_runs": n_runs, "seed": seed,
        "uncertainty_opacity": uncertainty_opacity, "line_width": line_width,
        "settlement_mode": settlement_mode, "auto_normalize": auto_normalize,
        "use_demo": use_demo, "preset_choice": preset_choice, "initial_values": initial_values,
    }


# =========================================================
# DATA LOADING
# =========================================================
def detect_categories_from_file(file) -> Tuple[Optional[pd.DataFrame], List[str], Optional[str]]:
    try:
        df = pd.read_excel(file)
        year_col = None
        for col in df.columns:
            if col.lower().strip() in ['jahr', 'year', 'date', 'zeit', 'time']:
                year_col = col
                break
        if year_col is None:
            first_col = df.columns[0]
            if pd.api.types.is_numeric_dtype(df[first_col]):
                year_col = first_col
            else:
                return None, [], "No 'Year' column found"

        excluded = {year_col.lower(), 'summe', 'summe ', 'sum', 'total'}
        categories = []
        for col in df.columns:
            if col.lower().strip() not in excluded:
                if pd.api.types.is_numeric_dtype(df[col]):
                    categories.append(col)
                else:
                    try:
                        df[col] = pd.to_numeric(df[col])
                        categories.append(col)
                    except:
                        pass

        if not categories:
            return None, [], "No numeric category columns found"
        df = df.rename(columns={year_col: "Year"})
        return df, categories, None
    except Exception as e:
        return None, [], f"Error: {str(e)}"


def load_market_data_generic(file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        df, categories, error = detect_categories_from_file(file)
        if error:
            return None, error
        category_manager.initialize_from_dataframe(df)
        required_cols = ["Year"] + categories
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[required_cols].isna().any().any():
            return None, "File contains missing or invalid values"
        df = auto_scale_to_percent(df, categories)
        return df, None
    except Exception as e:
        return None, f"Error: {str(e)}"


def auto_scale_to_percent(df: pd.DataFrame, categories: List[str]) -> pd.DataFrame:
    df = df.copy()
    row_sums = df[categories].sum(axis=1)
    max_val = df[categories].max().max()
    avg_sum = row_sums.mean()

    if max_val <= 1.0 and 0.9 <= avg_sum <= 1.1:
        logger.info("Detected: Values as proportions (0-1). Scaling to percent.")
        for cat in categories:
            df[cat] = df[cat] * 100
    elif 90 <= avg_sum <= 110:
        logger.info("Detected: Values already as percent (0-100).")
    else:
        logger.info(f"Normalizing rows to 100% (sum was: {avg_sum:.1f}).")
        for i in range(len(df)):
            row_sum = sum(df.loc[i, cat] for cat in categories)
            if row_sum > 0:
                for cat in categories:
                    df.loc[i, cat] = (df.loc[i, cat] / row_sum) * 100
    return df


def load_excavation_data_generic(file) -> Tuple[Optional[Dict], Optional[Dict], Optional[str]]:
    categories = category_manager.categories
    try:
        df = pd.read_excel(file)
        result_percent, result_absolute = {}, {}

        type_col = next((c for c in ['Typ', 'typ', 'Kategorie', 'kategorie', 'Category', 'category', 'Type', 'type'] if c in df.columns), None)
        count_col = next((c for c in ['Anzahl', 'anzahl', 'Count', 'count', 'Menge', 'menge', 'n', 'N', 'Number', 'number'] if c in df.columns), None)

        if type_col and count_col:
            df[count_col] = pd.to_numeric(df[count_col], errors="coerce")
            total = 0
            for cat in categories:
                mask = df[type_col].str.upper() == cat.upper()
                count = int(df.loc[mask, count_col].sum()) if mask.any() else 0
                result_absolute[cat] = count
                total += count
            if total <= 0:
                return None, None, f"No valid categories found. Expected: {', '.join(categories)}"
            for cat in categories:
                result_percent[cat] = (result_absolute[cat] / total) * 100
            result_absolute["_total"] = total
            return result_percent, result_absolute, None

        found_cats = [col for col in df.columns if col in categories]
        if found_cats:
            total = 0
            for cat in categories:
                if cat in df.columns:
                    val = pd.to_numeric(df.loc[0, cat], errors="coerce")
                    val = 0 if pd.isna(val) else val
                else:
                    val = 0
                result_absolute[cat] = int(val) if val >= 1 else 0
                total += val
            if total <= 0:
                return None, None, "No valid values found"
            if total <= len(categories):
                for cat in categories:
                    result_percent[cat] = (result_absolute.get(cat, 0) / total) * 100 if total > 0 else 0
            elif 90 <= total <= 110:
                for cat in categories:
                    result_percent[cat] = float(result_absolute.get(cat, 0))
            else:
                for cat in categories:
                    result_percent[cat] = (result_absolute[cat] / total) * 100 if total > 0 else 0
            result_absolute["_total"] = int(total) if total >= 1 else int(total * 100)
            return result_percent, result_absolute, None

        if len(df.columns) >= 2:
            cat_col, val_col = df.columns[0], df.columns[1]
            df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
            total = 0
            for cat in categories:
                mask = df[cat_col].astype(str).str.upper() == cat.upper()
                count = int(df.loc[mask, val_col].sum()) if mask.any() else 0
                result_absolute[cat] = count
                total += count
            if total > 0:
                for cat in categories:
                    result_percent[cat] = (result_absolute[cat] / total) * 100
                result_absolute["_total"] = total
                return result_percent, result_absolute, None

        return None, None, f"Format not recognized. Expected: 'Type'+'Count' columns or {categories} as columns"
    except Exception as e:
        return None, None, f"Error: {str(e)}"


@cache_data(show_spinner=False)
def interpolate_market_data(market_df: pd.DataFrame, years: List[int]) -> pd.DataFrame:
    categories = category_manager.categories
    year_col = "Year" if "Year" in market_df.columns else "Jahr"
    data_years = market_df[year_col].values
    result = {"Year": years}
    for cat in categories:
        result[cat] = np.interp(years, data_years, market_df[cat].values)
    return pd.DataFrame(result)


def interpolate_market_data_nocache(market_df: pd.DataFrame, years: List[int], categories: List[str]) -> pd.DataFrame:
    """Non-cached version for edited data."""
    year_col = "Year" if "Year" in market_df.columns else "Jahr"
    data_years = market_df[year_col].values
    result = {"Year": years}
    for cat in categories:
        if cat in market_df.columns:
            result[cat] = np.interp(years, data_years, market_df[cat].values)
    return pd.DataFrame(result)


# =========================================================
# SIMULATION
# =========================================================
def simulate_single_run_vectorized(market_array, initial, replacement_rates, noise_sd, seed=None, settlement_mode=False):
    n_years, n_categories = len(replacement_rates), len(initial)
    rng = np.random.default_rng(seed)
    result = np.zeros((n_years, n_categories))
    current = initial.copy()

    for i in range(n_years):
        if settlement_mode and i == 0:
            result[i] = current.copy()
            continue
        rate, market_share = replacement_rates[i], market_array[i]
        exchanged = current * (1 - rate) + market_share * rate
        if noise_sd > 0:
            exchanged = np.maximum(exchanged + rng.normal(0, noise_sd, n_categories), 0)
        total = exchanged.sum()
        if total > 0:
            exchanged = (exchanged / total) * 100
        result[i] = exchanged
        current = exchanged.copy()
    return result


def run_simulation(market_df, initial, replacement_rates, noise_sd, n_runs, base_seed=0, settlement_mode=False):
    categories = category_manager.categories
    market_array = market_df[categories].values
    n_years, n_cats = len(replacement_rates), len(categories)
    all_runs = np.zeros((n_runs, n_years, n_cats))

    for run in range(n_runs):
        seed = None if base_seed == 0 else base_seed + run
        all_runs[run] = simulate_single_run_vectorized(market_array, initial, replacement_rates, noise_sd, seed, settlement_mode)

    return np.mean(all_runs, axis=0), np.percentile(all_runs, 10, axis=0), np.percentile(all_runs, 90, axis=0)


# =========================================================
# VISUALIZATION
# =========================================================
def create_simulation_plot(years, mean, p10, p90, uncertainty_opacity=0.12, line_width=3):
    categories, colors, names = category_manager.categories, category_manager.colors, category_manager.category_names
    fig = go.Figure()
    for i, cat in enumerate(categories):
        color, name = colors.get(cat, "#888888"), names.get(cat, cat)
        fig.add_trace(go.Scatter(x=list(years) + list(years)[::-1], y=list(p90[:, i]) + list(p10[:, i])[::-1],
                                  fill="toself", fillcolor=hex_to_rgba(color, uncertainty_opacity),
                                  line=dict(width=0), name=f"{name} (10-90%)", showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=years, y=mean[:, i], mode="lines", name=name,
                                  line=dict(color=color, width=line_width, dash="dot"),
                                  hovertemplate=f"<b>{name}</b><br>Year: %{{x}}<br>Share: %{{y:.1f}}%<extra></extra>"))
    fig.update_layout(xaxis_title="Year AD", yaxis_title="Share (%)", yaxis_range=[0, 100],
                      hovermode="x unified", height=600, legend=dict(orientation="h", y=-0.15, xanchor="center", x=0.5),
                      margin=dict(l=60, r=40, t=40, b=100))
    return fig


def create_market_plot(market_df, categories=None):
    if categories is None:
        categories = category_manager.categories
    colors = category_manager.colors
    fig = go.Figure()
    for cat in categories:
        if cat in market_df.columns:
            fig.add_trace(go.Scatter(x=market_df["Year"], y=market_df[cat], mode="lines+markers",
                                     name=category_manager.category_names.get(cat, cat),
                                     line=dict(color=colors.get(cat, "#888888")),
                                     hovertemplate=f"<b>{cat}</b><br>Year: %{{x}}<br>Share: %{{y:.1f}}%<extra></extra>"))
    fig.update_layout(xaxis_title="Year AD", yaxis_title="Market Share (%)", yaxis_range=[0, 100],
                      hovermode="x unified", height=500, legend=dict(orientation="h", y=-0.15, xanchor="center", x=0.5))
    return fig


def create_comparison_bar_chart(sim_values, excavation_data, year):
    categories, colors, names = category_manager.categories, category_manager.colors, category_manager.category_names
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Simulation", x=[names.get(c, c) for c in categories],
                          y=[sim_values.get(c, 0) for c in categories],
                          marker_color=[colors.get(c, "#888888") for c in categories],
                          text=[f"{sim_values.get(c, 0):.1f}%" for c in categories], textposition="auto"))
    fig.add_trace(go.Bar(name="Excavation", x=[names.get(c, c) for c in categories],
                          y=[excavation_data.get(c, 0) for c in categories],
                          marker_color="lightgray", marker_line_color="black", marker_line_width=2,
                          text=[f"{excavation_data.get(c, 0):.1f}%" for c in categories], textposition="auto"))
    fig.update_layout(title=f"Comparison: Simulation vs. Excavation (Year {year})",
                      xaxis_title="Category", yaxis_title="Share (%)", yaxis_range=[0, 100],
                      barmode="group", height=500, showlegend=True)
    return fig


def create_deviation_plot(sim_values, excavation_data):
    categories, names = category_manager.categories, category_manager.category_names
    deviations = [sim_values.get(c, 0) - excavation_data.get(c, 0) for c in categories]
    bar_colors = ["#d62728" if d < 0 else "#2ca02c" for d in deviations]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[names.get(c, c) for c in categories], y=deviations, marker_color=bar_colors,
                          text=[f"{d:+.1f}%" for d in deviations], textposition="outside"))
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.update_layout(title="Deviation: Simulation − Excavation", xaxis_title="Category",
                      yaxis_title="Difference (%)", height=400, showlegend=False)
    return fig


def create_simulation_plot_with_excavation(years, mean, p10, p90, excavation_data, excavation_year, uncertainty_opacity=0.12, line_width=3):
    categories, colors, names = category_manager.categories, category_manager.colors, category_manager.category_names
    fig = go.Figure()
    for i, cat in enumerate(categories):
        color, name = colors.get(cat, "#888888"), names.get(cat, cat)
        fig.add_trace(go.Scatter(x=list(years) + list(years)[::-1], y=list(p90[:, i]) + list(p10[:, i])[::-1],
                                  fill="toself", fillcolor=hex_to_rgba(color, uncertainty_opacity),
                                  line=dict(width=0), name=f"{name} (10-90%)", showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=years, y=mean[:, i], mode="lines", name=name,
                                  line=dict(color=color, width=line_width, dash="dot"),
                                  hovertemplate=f"<b>{name}</b><br>Year: %{{x}}<br>Share: %{{y:.1f}}%<extra></extra>"))
        exc_val = excavation_data.get(cat, 0)
        fig.add_trace(go.Scatter(x=[excavation_year], y=[exc_val], mode="markers",
                                  marker=dict(size=15, color=color, symbol="diamond", line=dict(width=2, color="black")),
                                  name=f"{name} (Excavation)", showlegend=False,
                                  hovertemplate=f"<b>{name} (Excavation)</b><br>Year: {excavation_year}<br>Share: {exc_val:.1f}%<extra></extra>"))
    fig.add_vline(x=excavation_year, line_dash="dash", line_color="gray",
                  annotation_text=f"Excavation ({excavation_year})", annotation_position="top")
    fig.update_layout(xaxis_title="Year AD", yaxis_title="Share (%)", yaxis_range=[0, 100],
                      hovermode="x unified", height=600, legend=dict(orientation="h", y=-0.15, xanchor="center", x=0.5),
                      margin=dict(l=60, r=40, t=40, b=100))
    return fig


def create_pie_chart(values, title):
    categories, colors, names = category_manager.categories, category_manager.colors, category_manager.category_names
    labels = [names.get(cat, cat) for cat in categories]
    fig = go.Figure(data=[go.Pie(labels=labels, values=[values.get(cat, 0) for cat in categories],
                                  marker_colors=[colors.get(cat, "#888888") for cat in categories],
                                  textinfo="label+percent", hovertemplate="<b>%{label}</b><br>Share: %{value:.1f}%<extra></extra>")])
    fig.update_layout(title=title, height=400)
    return fig


# =========================================================
# EXPORT FUNCTIONS
# =========================================================
def fig_to_image_bytes(fig, format="png", width=1200, height=800, scale=1.0):
    """Convert figure to image bytes with improved error handling."""
    try:
        # Use engine parameter explicitly to ensure fresh kaleido process
        return fig.to_image(format=format, width=width, height=height, scale=scale, engine="kaleido")
    except Exception as e:
        # Try to restart kaleido scope on error
        try:
            import kaleido.scopes.plotly
            kaleido.scopes.plotly.PlotlyScope._shutdown_kaleido()
        except:
            pass
        # Retry once after cleanup
        return fig.to_image(format=format, width=width, height=height, scale=scale, engine="kaleido")


def get_cached_image_bytes(fig_json: str, format: str, width: int, height: int, scale: float) -> bytes:
    """Generate image bytes from figure JSON - separated for caching."""
    fig = go.Figure(json.loads(fig_json))
    return fig_to_image_bytes(fig, format, width, height, scale)


def create_print_ready_figure(fig, title=None, font_scale=1.0):
    fig_copy = go.Figure(fig)
    base_font_size, title_font_size = 14 * font_scale, 18 * font_scale
    update_dict = dict(
        font=dict(size=base_font_size, family="Arial, sans-serif"),
        xaxis=dict(title_font=dict(size=base_font_size), tickfont=dict(size=base_font_size * 0.85)),
        yaxis=dict(title_font=dict(size=base_font_size), tickfont=dict(size=base_font_size * 0.85)),
        legend=dict(font=dict(size=base_font_size * 0.9)),
        paper_bgcolor="white", plot_bgcolor="white",
    )
    if title:
        update_dict["title"] = dict(text=title, font=dict(size=title_font_size))
    fig_copy.update_layout(**update_dict)
    return fig_copy


def create_pie_chart_for_export(values, title, font_scale=1.0, p10_values=None, p90_values=None):
    from plotly.subplots import make_subplots
    categories, colors, names = category_manager.categories, category_manager.colors, category_manager.category_names
    labels = [names.get(cat, cat) for cat in categories]
    base_font_size = 14 * font_scale

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "table"}]], column_widths=[0.55, 0.45])
    fig.add_trace(go.Pie(labels=labels, values=[values.get(cat, 0) for cat in categories],
                          marker_colors=[colors.get(cat, "#888888") for cat in categories],
                          textinfo="label+percent", textfont=dict(size=base_font_size),
                          hovertemplate="<b>%{label}</b><br>Share: %{value:.1f}%<extra></extra>",
                          domain=dict(x=[0, 0.5])), row=1, col=1)

    table_headers = ["Category", "Share (%)"]
    table_col1 = [f"{cat} ({names.get(cat, cat)})" for cat in categories]
    table_col2 = [f"{values.get(cat, 0):.1f}%" for cat in categories]
    table_values = [table_col1, table_col2]

    if p10_values and p90_values:
        table_headers.append("80% Interval")
        table_values.append([f"{p10_values.get(cat, 0):.1f}–{p90_values.get(cat, 0):.1f}%" for cat in categories])

    fig.add_trace(go.Table(
        header=dict(values=table_headers, fill_color='lightgray', align='left',
                    font=dict(size=base_font_size, color='black'), height=35),
        cells=dict(values=table_values,
                   fill_color=[[hex_to_rgba(colors.get(cat, "#888888"), 0.3) for cat in categories]] + ['white'] * (len(table_values) - 1),
                   align='left', font=dict(size=base_font_size * 0.9, color='black'), height=30)
    ), row=1, col=2)

    fig.update_layout(title=dict(text=title, font=dict(size=base_font_size * 1.3), x=0.5),
                      height=600, width=1200, paper_bgcolor="white",
                      font=dict(size=base_font_size, family="Arial, sans-serif"), showlegend=False)
    return fig


# =========================================================
# DEMO DATA
# =========================================================
def create_demo_market_data():
    return pd.DataFrame({
        "Year": list(range(50, 310, 10)),
        "IT": [80, 70, 55, 40, 25, 15, 8, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "LG": [15, 25, 35, 45, 50, 45, 35, 25, 15, 8, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "BA": [3, 4, 7, 10, 15, 25, 35, 40, 35, 25, 15, 8, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "MG": [2, 1, 2, 4, 8, 12, 18, 25, 38, 51, 61, 65, 60, 50, 35, 20, 10, 5, 2, 1, 0, 0, 0, 0, 0, 0],
        "RZ": [0, 0, 1, 1, 2, 3, 4, 6, 10, 15, 20, 25, 35, 48, 64, 80, 90, 95, 98, 99, 100, 100, 100, 100, 100, 100],
    })


def create_demo_excavation_data():
    return {"IT": 5.2, "LG": 12.8, "BA": 8.6, "MG": 35.6, "RZ": 37.8}, {"IT": 26, "LG": 64, "BA": 43, "MG": 178, "RZ": 189, "_total": 500}


# =========================================================
# MARKET DATA EDITOR FUNCTIONS
# =========================================================
def create_editable_market_df(market_raw: pd.DataFrame, categories: List[str]) -> pd.DataFrame:
    """Create a DataFrame suitable for the data editor with proportions (0-1)."""
    df = market_raw.copy()
    year_col = "Year" if "Year" in df.columns else "Jahr"
    
    # Convert to proportions (0-1) if in percent
    for cat in categories:
        if cat in df.columns:
            max_val = df[cat].max()
            if max_val > 1:
                df[cat] = df[cat] / 100
    
    # Keep only Year and category columns
    cols_to_keep = [year_col] + [c for c in categories if c in df.columns]
    df = df[cols_to_keep]
    
    # Rename to Year if needed
    if year_col != "Year":
        df = df.rename(columns={year_col: "Year"})
    
    # Add Sum column for reference
    df["Summe"] = df[categories].sum(axis=1)
    
    return df


def apply_market_edits(edited_df: pd.DataFrame, categories: List[str], auto_normalize: bool = True) -> pd.DataFrame:
    """Apply edits and optionally normalize rows to sum to 1."""
    df = edited_df.copy()
    
    # Remove Sum column if present
    if "Summe" in df.columns:
        df = df.drop(columns=["Summe"])
    
    # Ensure all values are non-negative
    for cat in categories:
        if cat in df.columns:
            df[cat] = df[cat].clip(lower=0)
    
    # Normalize if requested
    if auto_normalize:
        for idx in df.index:
            total = sum(df.loc[idx, cat] for cat in categories if cat in df.columns and pd.notna(df.loc[idx, cat]))
            if total > 0:
                for cat in categories:
                    if cat in df.columns:
                        df.loc[idx, cat] = df.loc[idx, cat] / total
    
    return df


def market_df_to_percent(df: pd.DataFrame, categories: List[str]) -> pd.DataFrame:
    """Convert market data from proportions (0-1) to percent (0-100)."""
    df = df.copy()
    for cat in categories:
        if cat in df.columns:
            df[cat] = df[cat] * 100
    return df


# =========================================================
# MAIN APPLICATION
# =========================================================
if RUNNING_TESTS:
    print("Tests not implemented")
    sys.exit(0)

st.set_page_config(page_title="STOCHASI – Artifact Simulation", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
st.markdown("<style>.stMetric{background-color:rgba(128,128,128,0.1);padding:10px;border-radius:5px}</style>", unsafe_allow_html=True)

st.title("📊 STOCHASI – Stochastic Chronological Artifact Simulation")
st.markdown("**Generalized version** for arbitrary artifact categories. Simulation of temporal transformation of artifact spectra considering **market influence, replacement rate, and stochastics**.")

url_config = get_config_from_url_params()

for key in ["imported_market_data", "imported_excavation_data", "imported_excavation_absolute", 
            "imported_categories", "data_source", "edited_market_data", "original_market_data",
            "market_data_modified", "pending_market_edits", "editor_base_data", "comparison_year",
            "uploaded_market_file", "uploaded_excavation_file"]:
    if key not in st.session_state:
        st.session_state[key] = None

# Keys for file uploaders (incremented to reset uploader)
if "market_upload_key" not in st.session_state:
    st.session_state.market_upload_key = 0
if "excavation_upload_key" not in st.session_state:
    st.session_state.excavation_upload_key = 0

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("📂 Load Configuration")

if url_config:
    st.sidebar.success(f"✅ {len([k for k in url_config if not k.startswith('_')])} parameters from URL")
    with st.sidebar.expander("📋 Loaded Parameters", expanded=False):
        for k, v in url_config.items():
            st.write(f"**{k}:** {v}")

if st.session_state.data_source == "json":
    st.sidebar.info("📊 Data from JSON import active")
    
    # Show what's imported and provide separate delete buttons
    json_has_market = st.session_state.imported_market_data is not None
    json_has_excavation = st.session_state.imported_excavation_data is not None
    
    if json_has_market:
        col1, col2 = st.sidebar.columns([3, 1])
        col1.caption("✅ Market data from JSON")
        if col2.button("🗑️", key="delete_json_market", help="Delete imported market data"):
            st.session_state.imported_market_data = None
            st.session_state.imported_categories = None
            st.session_state.edited_market_data = None
            st.session_state.original_market_data = None
            st.session_state.market_data_modified = None
            # If no excavation data either, clear JSON source
            if st.session_state.imported_excavation_data is None:
                st.session_state.data_source = None
            st.rerun()
    
    if json_has_excavation:
        col1, col2 = st.sidebar.columns([3, 1])
        col1.caption("✅ Excavation from JSON")
        if col2.button("🗑️", key="delete_json_excavation", help="Delete imported excavation data"):
            st.session_state.imported_excavation_data = None
            st.session_state.imported_excavation_absolute = None
            st.session_state.comparison_year = None
            # If no market data either, clear JSON source
            if st.session_state.imported_market_data is None:
                st.session_state.data_source = None
            st.rerun()
    
    # Show "Discard All" button only if both types are present
    if json_has_market and json_has_excavation:
        if st.sidebar.button("🗑️ Discard All Imported Data", type="secondary"):
            for key in ["imported_market_data", "imported_excavation_data", "imported_excavation_absolute", "imported_categories"]:
                st.session_state[key] = None
            st.session_state.data_source = None
            st.session_state.edited_market_data = None
            st.session_state.original_market_data = None
            st.session_state.market_data_modified = None
            st.session_state.comparison_year = None
            st.session_state.uploaded_market_file = None
            st.session_state.uploaded_excavation_file = None
            st.rerun()

with st.sidebar.expander("📥 Import Configuration", expanded=False):
    uploaded_config = st.file_uploader("Upload JSON", type=["json"], key="config_upload", label_visibility="collapsed")
    if uploaded_config:
        try:
            config_str = uploaded_config.read().decode("utf-8")
            has_market, has_excavation = check_json_has_data(config_str)
            imported_params, imported_market, imported_exc_pct, imported_exc_abs, imported_cats, imported_exc_year, import_error = json_to_full_config(config_str)
            if import_error:
                st.error(f"❌ {import_error}")
            else:
                st.success("✅ Configuration recognized!")
                # Preview section (without expander to avoid nesting)
                st.caption("📋 **Preview:**")
                st.caption(f"Period: {imported_params.get('start_year', '?')}–{imported_params.get('end_year', '?')} | Rate: {imported_params.get('replacement_rate', 0)*100:.0f}% | Runs: {imported_params.get('n_runs', '?')}")
                st.caption(f"{'✅' if has_market else '❌'} Market | {'✅' if has_excavation else '❌'} Excavation" + (f" (Year {imported_exc_year})" if has_excavation and imported_exc_year else ""))
                if st.button("✅ Apply Configuration", type="primary"):
                    if has_market and imported_market is not None:
                        st.session_state.imported_market_data = imported_market
                        st.session_state.imported_categories = imported_cats
                        st.session_state.data_source = "json"
                        st.session_state.edited_market_data = None
                        st.session_state.original_market_data = None
                    if has_excavation:
                        st.session_state.imported_excavation_data = imported_exc_pct
                        st.session_state.imported_excavation_absolute = imported_exc_abs
                        if imported_exc_year:
                            st.session_state.comparison_year = imported_exc_year
                    set_url_params_from_config(imported_params)
                    st.rerun()
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

st.sidebar.markdown("---")
st.sidebar.header("📁 Artifact Type & Data")

use_imported_data = st.session_state.data_source == "json" and st.session_state.imported_market_data is not None
preset_options = list(PRESET_CONFIGS.keys())
preset_default_idx = preset_options.index(url_config["preset_choice"]) if url_config.get("preset_choice") in preset_options else 0
preset_choice = st.sidebar.selectbox("Artifact Type Preset", preset_options, index=preset_default_idx, disabled=use_imported_data)
preset = PRESET_CONFIGS[preset_choice]

if use_imported_data:
    market_raw = st.session_state.imported_market_data
    category_manager.initialize_from_dataframe(market_raw)
    if preset_choice == "Terra Sigillata" and set(category_manager.categories).issubset({"IT", "LG", "BA", "MG", "RZ"}):
        category_manager._category_names = preset["category_names"]
        category_manager._colors = preset["colors"]
    st.sidebar.success("✅ Market data from JSON")
    use_demo, market_file = False, None
else:
    use_demo = st.sidebar.checkbox("🎮 Use Demo Data", value=url_config.get("use_demo", False))
    if use_demo:
        market_raw = create_demo_market_data()
        category_manager.initialize_from_dataframe(market_raw)
        if preset_choice == "Terra Sigillata":
            category_manager._category_names = preset["category_names"]
            category_manager._colors = preset["colors"]
        market_raw = auto_scale_to_percent(market_raw, category_manager.categories)
        st.sidebar.success("✅ Demo data loaded")
        market_file = None
    else:
        market_file = st.sidebar.file_uploader(
            "Upload Market Data Excel", 
            type=["xlsx", "xls"], 
            key=f"market_upload_{st.session_state.market_upload_key}"
        )
        # Store uploaded file in session state for persistence
        if market_file is not None:
            st.session_state.uploaded_market_file = market_file
        
        # Show delete button if a file was uploaded
        if st.session_state.uploaded_market_file is not None:
            if st.sidebar.button("🗑️ Delete Market Data", key="delete_market", type="secondary"):
                st.session_state.uploaded_market_file = None
                st.session_state.edited_market_data = None
                st.session_state.original_market_data = None
                st.session_state.market_data_modified = None
                st.session_state.market_upload_key += 1  # Reset uploader
                st.rerun()
        
        # Use stored file if current upload is None
        if market_file is None and st.session_state.uploaded_market_file is not None:
            market_file = st.session_state.uploaded_market_file

if not use_imported_data and not use_demo and market_file is None:
    st.info("👈 Please upload **Market Data Excel**, activate **Demo Data**, or import **JSON configuration**.")
    st.markdown("""
    ### File Format
    | Year | Category1 | Category2 | ... |
    |------|-----------|-----------|-----|
    | 100  | 0.5       | 0.3       | ... |

    Values can be percentages (0-100) or proportions (0-1).
    """)
    st.stop()

if not use_imported_data and not use_demo:
    market_raw, error = load_market_data_generic(market_file)
    if error:
        st.error(f"❌ Error: {error}")
        st.stop()
    if preset_choice == "Terra Sigillata" and set(category_manager.categories).issubset({"IT", "LG", "BA", "MG", "RZ"}):
        category_manager._category_names = preset["category_names"]
        category_manager._colors = preset["colors"]

# Sync categories from edited market data if present (in case new categories were added)
if st.session_state.edited_market_data is not None:
    edited_cats = [c for c in st.session_state.edited_market_data.columns if c != "Year"]
    for cat in edited_cats:
        if cat not in category_manager.categories:
            category_manager.add_category(cat)

categories = category_manager.categories
n_cats = len(categories)
st.sidebar.success(f"✅ {n_cats} categories detected")

with st.sidebar.expander("📋 Detected Categories", expanded=False):
    for cat in categories:
        st.markdown(f"<span style='color:{category_manager.colors.get(cat, '#888')}'>●</span> **{cat}** = {category_manager.category_names.get(cat, cat)}", unsafe_allow_html=True)

st.sidebar.header("📅 Time Period")
# Use edited market data for year range if available
if st.session_state.edited_market_data is not None:
    year_source = st.session_state.edited_market_data
else:
    year_source = market_raw
year_col = "Year" if "Year" in year_source.columns else "Jahr"
min_year, max_year = int(year_source[year_col].min()), int(year_source[year_col].max())
start_year_default = max(min_year, min(max_year - 1, url_config.get("start_year", preset.get("default_start", min_year))))
end_year_default = max(start_year_default + 1, min(max_year, url_config.get("end_year", preset.get("default_end", max_year))))

col1, col2 = st.sidebar.columns(2)
start_year = col1.number_input("Start Year", min_year, max_year - 1, start_year_default)
end_year = col2.number_input("End Year", start_year + 1, max_year, max(start_year + 1, end_year_default))
if end_year <= start_year:
    st.sidebar.error("⚠️ End > Start required!")
    st.stop()
years = list(range(start_year, end_year + 1))

st.sidebar.header("⚙️ Simulation")
replacement_rate = st.sidebar.slider("Replacement Rate (annual)", 0.0, MAX_REPLACEMENT_RATE, url_config.get("replacement_rate", DEFAULT_REPLACEMENT_RATE), 0.01)
noise_sd = st.sidebar.slider("Stochastic Scatter (σ)", 0.0, MAX_NOISE_SD, url_config.get("noise_sd", DEFAULT_NOISE_SD), 0.5)
n_runs = st.sidebar.slider("Simulation Runs", MIN_RUNS, MAX_RUNS, url_config.get("n_runs", DEFAULT_RUNS), 10)
seed = st.sidebar.number_input("Random Seed (0=random)", 0, 2**31-1, url_config.get("seed", 0))

st.sidebar.header("🎨 Visualization")
uncertainty_opacity = st.sidebar.slider("Uncertainty Opacity", 0.0, 1.0, url_config.get("uncertainty_opacity", 0.12), 0.02)
line_width = st.sidebar.slider("Line Width", 1, 5, url_config.get("line_width", 3))

st.sidebar.header("🎯 Initial Distribution")
settlement_mode = st.sidebar.checkbox("🏘️ New Settlement Mode", url_config.get("settlement_mode", False),
                                       help="Initial distribution applies FOR start year (settlers bring pottery)")
if settlement_mode:
    st.sidebar.info(f"📍 Initial for **year {start_year}**")
else:
    st.sidebar.caption(f"Initial for year {start_year - 1}")

auto_normalize = st.sidebar.checkbox("🔄 Auto-normalize", url_config.get("auto_normalize", True))
url_init = url_config.get("initial_values", {})
init_vals, init_vals_dict = [], {}
for cat in categories:
    val = st.sidebar.number_input(f"{cat} ({category_manager.category_names.get(cat, cat)})", 0, 100, url_init.get(cat, 100 // n_cats), key=f"init_{cat}")
    init_vals.append(val)
    init_vals_dict[cat] = val
initial_distribution = normalize(np.array(init_vals, dtype=float)) if auto_normalize and sum(init_vals) > 0 else np.array(init_vals, dtype=float)

st.sidebar.header("⛏️ Excavation Spectrum")
use_imported_exc = st.session_state.imported_excavation_data is not None and st.session_state.data_source == "json"
use_demo_exc = st.sidebar.checkbox("🎮 Demo Excavation", False) if use_demo and not use_imported_exc else False

if use_demo_exc or use_imported_exc:
    excavation_file = None
else:
    excavation_file = st.sidebar.file_uploader(
        "Upload Excavation", 
        type=["xlsx", "xls"], 
        key=f"exc_upload_{st.session_state.excavation_upload_key}"
    )
    # Store uploaded file in session state for persistence
    if excavation_file is not None:
        st.session_state.uploaded_excavation_file = excavation_file
    
    # Show delete button if a file was uploaded
    if st.session_state.uploaded_excavation_file is not None:
        if st.sidebar.button("🗑️ Delete Excavation Data", key="delete_excavation", type="secondary"):
            st.session_state.uploaded_excavation_file = None
            st.session_state.excavation_upload_key += 1  # Reset uploader
            st.rerun()
    
    # Use stored file if current upload is None
    if excavation_file is None and st.session_state.uploaded_excavation_file is not None:
        excavation_file = st.session_state.uploaded_excavation_file

excavation_data, excavation_data_absolute = None, None
if use_imported_exc:
    excavation_data, excavation_data_absolute = st.session_state.imported_excavation_data, st.session_state.imported_excavation_absolute
    st.sidebar.success(f"✅ Excavation from JSON ({excavation_data_absolute.get('_total', 0)} finds)")
elif use_demo_exc:
    excavation_data, excavation_data_absolute = create_demo_excavation_data()
    st.sidebar.success("✅ Demo excavation loaded")
elif excavation_file:
    excavation_data, excavation_data_absolute, exc_error = load_excavation_data_generic(excavation_file)
    if exc_error:
        st.sidebar.error(f"❌ {exc_error}")
    else:
        st.sidebar.success(f"✅ Excavation loaded ({excavation_data_absolute.get('_total', 0)} finds)")

if excavation_data:
    with st.sidebar.expander("📊 Excavation Spectrum", expanded=False):
        st.write(f"**Total:** {excavation_data_absolute.get('_total', 0)} finds")
        for cat in categories:
            st.write(f"**{cat}**: {excavation_data_absolute.get(cat, 0)} ({excavation_data.get(cat, 0):.1f}%)")

st.sidebar.header("🔗 Share Configuration")
current_config = get_current_config(start_year, end_year, replacement_rate, noise_sd, n_runs, seed,
                                     uncertainty_opacity, line_width, settlement_mode, auto_normalize,
                                     use_demo, preset_choice, init_vals_dict)

# Determine which market data to use for simulation and export
if st.session_state.edited_market_data is not None:
    market_for_simulation = market_df_to_percent(st.session_state.edited_market_data, categories)
    market_data_is_edited = True
else:
    market_for_simulation = market_raw
    market_data_is_edited = False

with st.sidebar.expander("🔗 Link & Export", expanded=False):
    st.markdown("**Shareable Link:**")
    st.code(create_shareable_url(current_config), language=None)
    if st.button("🔄 Update URL"):
        set_url_params_from_config(current_config)
        st.success("✅ URL updated!")
    st.markdown("---")
    include_data = st.checkbox("📊 Export with data", True)
    
    if market_data_is_edited:
        st.info("📝 Edited market data will be exported")
    
    if excavation_data and st.session_state.comparison_year:
        st.info(f"📅 Comparison year: {st.session_state.comparison_year}")
    
    if include_data:
        # Use edited market data if available
        export_market = st.session_state.edited_market_data if market_data_is_edited else None
        config_json = config_to_json(
            current_config, 
            market_data=market_raw,
            excavation_data=excavation_data,
            excavation_data_absolute=excavation_data_absolute, 
            categories=categories,
            category_names=category_manager.category_names,
            edited_market_data=market_for_simulation if market_data_is_edited else None,
            excavation_year=st.session_state.comparison_year
        )
        st.download_button("💾 Complete Export", config_json, "stochasi_complete.json", "application/json")
    else:
        st.download_button("💾 Parameters Only", config_to_json_simple(current_config), "stochas_config.json", "application/json")

# =========================================================
# RUN SIMULATION
# =========================================================
market_df = interpolate_market_data_nocache(market_for_simulation, years, categories)
replacement_rates = np.full(len(years), replacement_rate)
mean, p10, p90 = run_simulation(market_df, initial_distribution, replacement_rates, noise_sd, n_runs, seed, settlement_mode)

# =========================================================
# MAIN TABS
# =========================================================
tab_market, tab_year, tab_sim, tab_compare, tab_export = st.tabs(["📈 Market Supply", "🥧 Year Spectrum", "🔄 Simulation", "⚖️ Comparison", "💾 Export"])

with tab_market:
    st.subheader("📈 Market Supply Over Time")
    st.markdown("This shows the **market supply composition** used as exchange values. You can **edit the data** below.")
    
    # Show current plot
    st.plotly_chart(create_market_plot(market_for_simulation, categories), width="stretch")
    
    # Status indicator
    if market_data_is_edited:
        st.success("✅ **Using edited market data** – Changes are active and will be used in simulation & export")
    
    st.markdown("---")
    
    # Market Data Editor Section
    st.subheader("✏️ Edit Market Data")
    st.markdown("""
    Edit the market share proportions below (values between 0 and 1).
    
    **Workflow:** 
    1. Edit values in the table at your own pace
    2. Click **"✅ Apply Changes"** to update the simulation
    """)
    
    # Initialize pending data in session state if needed
    if "pending_market_edits" not in st.session_state:
        st.session_state.pending_market_edits = None
    if "editor_base_data" not in st.session_state:
        st.session_state.editor_base_data = None
    
    # Prepare base data for editor (what's currently active)
    if st.session_state.edited_market_data is not None:
        base_df = st.session_state.edited_market_data.copy()
    else:
        base_df = create_editable_market_df(market_raw, categories)
        # Remove Summe if present from create_editable_market_df
        if "Summe" in base_df.columns:
            base_df = base_df.drop(columns=["Summe"])
    
    # Store base data for comparison
    st.session_state.editor_base_data = base_df.copy()
    
    # Add Summe column for display
    current_categories = [c for c in base_df.columns if c != "Year"]
    editor_df = base_df.copy()
    editor_df["Summe"] = editor_df[current_categories].sum(axis=1)
    
    # ===== CONTROL BUTTONS ROW =====
    st.markdown("#### ⚙️ Editor Controls")
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.5, 1.5, 1.5, 1.5])
    
    with ctrl_col1:
        auto_norm_editor = st.checkbox("🔄 Normalize on apply", value=True, key="auto_norm_editor",
                                       help="Normalize each row to sum = 1 when applying changes")
    
    with ctrl_col2:
        if st.button("🔄 Reset to Original", type="secondary", width="stretch"):
            st.session_state.edited_market_data = None
            st.session_state.pending_market_edits = None
            st.session_state.market_data_modified = None
            st.rerun()
    
    # ===== ADD CATEGORY / YEAR =====
    add_col1, add_col2 = st.columns(2)
    
    with add_col1:
        with st.expander("➕ Add New Category", expanded=False):
            new_cat_name = st.text_input("Category Name", placeholder="e.g., NEW", key="new_cat_name")
            if st.button("➕ Add Category", type="primary", key="add_cat_btn"):
                if new_cat_name and new_cat_name.strip():
                    new_cat = new_cat_name.strip().upper()
                    if new_cat not in editor_df.columns and new_cat != "Year" and new_cat != "Summe":
                        # Add to base_df and save
                        base_df[new_cat] = 0.0
                        category_manager.add_category(new_cat)
                        st.session_state.edited_market_data = base_df.copy()
                        st.success(f"✅ Category '{new_cat}' added!")
                        st.rerun()
                    else:
                        st.error(f"❌ Category '{new_cat}' already exists or is invalid")
                else:
                    st.error("❌ Please enter a category name")
    
    with add_col2:
        with st.expander("➕ Add New Year", expanded=False):
            existing_years = sorted(editor_df["Year"].tolist())
            min_new_year = int(existing_years[0]) - 100
            max_new_year = int(existing_years[-1]) + 100
            new_year = st.number_input("Year", min_value=min_new_year, max_value=max_new_year, 
                                       value=int(existing_years[-1]) + 10, step=10, key="new_year_input")
            if st.button("➕ Add Year", type="primary", key="add_year_btn"):
                if new_year not in existing_years:
                    new_row = {"Year": new_year}
                    for cat in current_categories:
                        new_row[cat] = 0.0
                    base_df = pd.concat([base_df, pd.DataFrame([new_row])], ignore_index=True)
                    base_df = base_df.sort_values("Year").reset_index(drop=True)
                    st.session_state.edited_market_data = base_df.copy()
                    st.success(f"✅ Year {new_year} added!")
                    st.rerun()
                else:
                    st.error(f"❌ Year {new_year} already exists")
    
    st.markdown("---")
    
    # ===== DATA EDITOR =====
    st.markdown("#### 📝 Edit Values")
    st.caption("Edit the values below, then click **'✅ Apply Changes'** to update the simulation.")
    
    # Configure column types for editor
    column_config = {
        "Year": st.column_config.NumberColumn("Year", format="%d", disabled=False),
        "Summe": st.column_config.NumberColumn("Σ Sum", format="%.3f", disabled=True,
                                                help="Current sum of row (should be 1.0 after normalization)")
    }
    for cat in current_categories:
        column_config[cat] = st.column_config.NumberColumn(
            cat,
            format="%.3f",
            min_value=0.0,
            max_value=10.0,  # Allow larger values, will be normalized
            step=0.01,
            help=f"Market share for {category_manager.category_names.get(cat, cat)}"
        )
    
    # Data Editor - changes are stored but NOT immediately applied
    edited_df = st.data_editor(
        editor_df,
        column_config=column_config,
        width="stretch",
        num_rows="dynamic",
        key="market_editor",
        hide_index=True
    )
    
    # ===== APPLY CHANGES SECTION =====
    st.markdown("---")
    
    # Check if there are pending changes
    if edited_df is not None:
        # Get categories from edited dataframe (excluding Year and Summe)
        edit_categories = [c for c in edited_df.columns if c not in ["Year", "Summe"]]
        
        # Calculate current sums for display
        edited_df_clean = edited_df.drop(columns=["Summe"], errors="ignore").copy()
        row_sums = edited_df_clean[edit_categories].sum(axis=1)
        
        # Check for differences from base data
        has_changes = False
        if st.session_state.editor_base_data is not None:
            base_cats = [c for c in st.session_state.editor_base_data.columns if c != "Year"]
            # Compare only if same categories
            if set(edit_categories) == set(base_cats) and len(edited_df_clean) == len(st.session_state.editor_base_data):
                try:
                    diff = (edited_df_clean[edit_categories].values - st.session_state.editor_base_data[edit_categories].values)
                    has_changes = np.abs(diff).sum() > 0.0001
                except:
                    has_changes = True
            else:
                has_changes = True
        
        # Show validation summary
        valid_rows = ((row_sums >= 0.99) & (row_sums <= 1.01)).sum()
        invalid_rows = len(row_sums) - valid_rows
        
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        with summary_col1:
            if has_changes:
                st.warning("🔶 **Unsaved changes** in editor")
            else:
                st.info("ℹ️ No changes detected")
        with summary_col2:
            if invalid_rows > 0 and not auto_norm_editor:
                st.warning(f"⚠️ {invalid_rows} rows with sum ≠ 1.0")
            else:
                st.success(f"✅ {len(row_sums)} rows ready")
        with summary_col3:
            if auto_norm_editor:
                st.info("🔄 Will normalize on apply")
        
        # APPLY BUTTON
        st.markdown("")
        apply_col1, apply_col2, apply_col3 = st.columns([1, 2, 1])
        with apply_col2:
            if st.button("✅ Apply Changes to Simulation", type="primary", width="stretch",
                        disabled=not has_changes):
                # Process and apply the edits
                processed_df = edited_df_clean.copy()
                
                # Ensure non-negative values
                for cat in edit_categories:
                    if cat in processed_df.columns:
                        processed_df[cat] = processed_df[cat].clip(lower=0)
                
                # Normalize if requested
                if auto_norm_editor:
                    for idx in processed_df.index:
                        total = sum(processed_df.loc[idx, cat] for cat in edit_categories if pd.notna(processed_df.loc[idx, cat]))
                        if total > 0:
                            for cat in edit_categories:
                                processed_df.loc[idx, cat] = processed_df.loc[idx, cat] / total
                
                # Save to session state
                st.session_state.edited_market_data = processed_df
                
                # Update category manager if new categories
                for cat in edit_categories:
                    if cat not in category_manager.categories:
                        category_manager.add_category(cat)
                
                st.success("✅ Changes applied! Simulation updated.")
                st.rerun()
    
    st.markdown("---")
    
    # Data Preview / Download
    with st.expander("📊 Data Preview & Download (Active Data)"):
        display_df = st.session_state.edited_market_data if st.session_state.edited_market_data is not None else base_df
        st.dataframe(display_df, width="stretch")
        
        col1, col2 = st.columns(2)
        with col1:
            csv_data = display_df.to_csv(index=False)
            st.download_button("📥 Download as CSV", csv_data, "market_data.csv", "text/csv")
        with col2:
            # Excel download
            buffer = io.BytesIO()
            display_df.to_excel(buffer, index=False, engine='openpyxl')
            st.download_button("📥 Download as Excel", buffer.getvalue(), "market_data.xlsx", 
                             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with tab_year:
    st.subheader("🥧 Interactive Year View")
    if settlement_mode:
        st.success(f"🏘️ **New Settlement Mode:** Year {start_year} = initial distribution")
    selected_year = st.slider("Select Year", years[0], years[-1], years[len(years)//2], key="year_slider")
    year_idx = years.index(selected_year)
    year_values = {cat: mean[year_idx, i] for i, cat in enumerate(categories)}
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_pie_chart(year_values, f"Distribution in Year {selected_year}"), width="stretch")
    with col2:
        st.markdown(f"### Values for Year {selected_year}")
        if settlement_mode and selected_year == start_year:
            st.caption("📍 *Settlement year = initial distribution*")
        for i, cat in enumerate(categories):
            color = category_manager.colors.get(cat, "#888")
            st.markdown(f"<span style='color:{color}'>●</span> **{cat}**: {year_values[cat]:.1f}% (80%: {p10[year_idx,i]:.1f}–{p90[year_idx,i]:.1f}%)", unsafe_allow_html=True)
        st.metric("Replacement Rate", f"{replacement_rates[year_idx]*100:.1f}%")

with tab_sim:
    st.subheader("Simulation Results")
    if market_data_is_edited:
        st.info("📝 **Using edited market data** for simulation")
    if settlement_mode:
        st.info(f"🏘️ **New Settlement Mode:** Starts year **{start_year}** with initial distribution. Exchange from **{start_year+1}**.")
    else:
        st.caption(f"*Classic mode: Initial for year {start_year-1}, exchange from {start_year}.*")
    st.plotly_chart(create_simulation_plot(years, mean, p10, p90, uncertainty_opacity, line_width), width="stretch")
    st.markdown("### Statistics")
    cols = st.columns(n_cats)
    for i, cat in enumerate(categories):
        with cols[i % n_cats]:
            st.metric(f"{category_manager.category_names.get(cat, cat)} (End)", f"{mean[-1,i]:.1f}%", f"{mean[-1,i]-mean[0,i]:+.1f}%")

with tab_compare:
    st.subheader("🔍 Comparison: Simulation vs. Excavation")
    if excavation_data is None:
        st.info("👈 Upload **excavation spectrum** in sidebar to compare.")
    else:
        # Use session_state comparison_year if available, otherwise default to middle of range
        default_exc_year = st.session_state.comparison_year if st.session_state.comparison_year is not None else years[len(years)//2]
        # Ensure default is within valid range
        default_exc_year = max(years[0], min(years[-1], default_exc_year))
        
        excavation_year = st.number_input("Excavation Year", years[0], years[-1], default_exc_year, key="excavation_year_input")
        
        # Store in session_state for export
        st.session_state.comparison_year = excavation_year
        
        total_finds = excavation_data_absolute.get("_total", 0) if excavation_data_absolute else 0
        st.info(f"📅 **Comparison Year:** {excavation_year} AD | **Finds:** {total_finds}")
        if excavation_year not in years:
            st.error(f"⚠️ Year {excavation_year} outside range ({years[0]}–{years[-1]})")
        else:
            year_idx = years.index(excavation_year)
            sim_values = {cat: mean[year_idx, i] for i, cat in enumerate(categories)}

            st.markdown("### 📊 Comparison Table")
            comparison_df = pd.DataFrame([{
                "Category": f"{cat} ({category_manager.category_names.get(cat, cat)})",
                "Simulation (%)": sim_values.get(cat, 0),
                "Excavation (%)": excavation_data.get(cat, 0),
                "Difference (%)": sim_values.get(cat, 0) - excavation_data.get(cat, 0),
            } for cat in categories])
            st.dataframe(comparison_df.style.format({"Simulation (%)": "{:.1f}", "Excavation (%)": "{:.1f}", "Difference (%)": "{:+.1f}"}), width="stretch", hide_index=True)

            st.markdown("### 📊 Bar Chart")
            st.plotly_chart(create_comparison_bar_chart(sim_values, excavation_data, excavation_year), width="stretch")

            st.markdown("### 📈 Deviation Analysis")
            differences = [sim_values.get(c, 0) - excavation_data.get(c, 0) for c in categories]
            mae, rmse = np.mean([abs(d) for d in differences]), np.sqrt(np.mean([d**2 for d in differences]))
            max_dev = max(differences, key=abs)
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE", f"{mae:.2f}%")
            c2.metric("RMSE", f"{rmse:.2f}%")
            c3.metric(f"Max ({categories[differences.index(max_dev)]})", f"{max_dev:+.1f}%")
            st.plotly_chart(create_deviation_plot(sim_values, excavation_data), width="stretch")
            st.info("🟢 **Positive:** Simulation overestimates | 🔴 **Negative:** Simulation underestimates")

            st.markdown("### 📈 Timeline with Excavation")
            st.plotly_chart(create_simulation_plot_with_excavation(years, mean, p10, p90, excavation_data, excavation_year, uncertainty_opacity, line_width), width="stretch")
            st.caption("◆ = Excavation | Lines = Simulation | Shading = 80% confidence")

with tab_export:
    st.subheader("💾 Data & Graphics Export")
    
    if market_data_is_edited:
        st.success("📝 **Edited market data** will be included in exports")
    
    st.markdown("### 📊 Data Export")
    export_df = pd.DataFrame(mean, columns=categories)
    export_df.insert(0, "Year", years)
    export_df["ReplacementRate"] = replacement_rates
    for i, cat in enumerate(categories):
        export_df[f"{cat}_P10"], export_df[f"{cat}_P90"] = p10[:, i], p90[:, i]
    with st.expander("📋 Data Preview", expanded=False):
        st.dataframe(export_df.head(15))
    c1, c2 = st.columns(2)
    c1.download_button("📥 CSV (with percentiles)", export_df.to_csv(index=False), "simulation_results.csv", "text/csv")
    simple_df = pd.DataFrame(mean, columns=categories)
    simple_df.insert(0, "Year", years)
    c2.download_button("📥 CSV (means only)", simple_df.to_csv(index=False), "simulation_means.csv", "text/csv")

    st.markdown("---")
    st.markdown("### 🖼️ Graphics Export (High Resolution)")
    c1, c2 = st.columns([1, 2])
    export_preset = c1.selectbox("Resolution", list(EXPORT_PRESETS.keys()), format_func=lambda x: EXPORT_PRESETS[x]["name"], index=1)
    preset_info = EXPORT_PRESETS[export_preset]
    c2.info(f"**{preset_info['name']}**: {preset_info['description']} ({preset_info['width']}×{preset_info['height']})")

    c1, c2 = st.columns(2)
    include_title = c1.checkbox("Embed title", True)
    white_background = c2.checkbox("White background", True)
    font_scale = 1.2 if export_preset in ["print_300dpi", "print_600dpi", "poster"] else 1.0

    st.markdown("---")
    try:
        import kaleido
        kaleido_available = True
    except ImportError:
        kaleido_available = False

    if not kaleido_available:
        st.error("⚠️ **Graphics export unavailable!** Install `kaleido`: `pip install kaleido`")
    else:
        # Initialize export cache in session state
        if "export_cache" not in st.session_state:
            st.session_state.export_cache = {}
        
        st.markdown("#### 🥧 Year Spectrum (Pie Chart)")
        export_year = st.slider("Year for export", years[0], years[-1], years[len(years)//2], key="export_year")
        ey_idx = years.index(export_year)
        ey_vals = {cat: mean[ey_idx, i] for i, cat in enumerate(categories)}
        ey_p10 = {cat: p10[ey_idx, i] for i, cat in enumerate(categories)}
        ey_p90 = {cat: p90[ey_idx, i] for i, cat in enumerate(categories)}
        pie_title = f"Artifact Spectrum Year {export_year}" if include_title else ""
        
        # Create unique cache key for pie chart
        pie_cache_key = f"pie_{export_year}_{include_title}_{white_background}_{export_preset}"
        
        c1, c2, c3 = st.columns([1, 1, 1])
        if c1.button("🔄 Generate Pie Chart", key="gen_pie"):
            try:
                fig_pie_exp = create_pie_chart_for_export(ey_vals, pie_title, font_scale, ey_p10, ey_p90)
                if white_background:
                    fig_pie_exp.update_layout(paper_bgcolor="white")
                st.session_state.export_cache[f"{pie_cache_key}_png"] = fig_to_image_bytes(fig_pie_exp, "png", preset_info["width"], preset_info["height"], preset_info["scale"])
                st.session_state.export_cache[f"{pie_cache_key}_svg"] = fig_to_image_bytes(fig_pie_exp, "svg", preset_info["width"], preset_info["height"], 1.0)
                st.success("✅ Generated!")
            except Exception as e:
                st.error(f"Export error: {e}")
        
        # Show download buttons only if data is cached
        if f"{pie_cache_key}_png" in st.session_state.export_cache:
            png = st.session_state.export_cache[f"{pie_cache_key}_png"]
            svg = st.session_state.export_cache[f"{pie_cache_key}_svg"]
            c2.download_button(f"🖼️ PNG ({len(png)/(1024*1024):.1f} MB)", png, f"YearSpectrum_{export_year}.png", "image/png", key="pie_png")
            c3.download_button("🖼️ SVG (Vector)", svg, f"YearSpectrum_{export_year}.svg", "image/svg+xml", key="pie_svg")
        else:
            c2.caption("Click Generate first")

        st.markdown("---")
        st.markdown("#### 🔄 Simulation (Timeline)")
        
        # Create unique cache key for simulation
        sim_cache_key = f"sim_{start_year}_{end_year}_{include_title}_{white_background}_{export_preset}_{replacement_rate}_{noise_sd}"
        
        c1, c2, c3 = st.columns([1, 1, 1])
        if c1.button("🔄 Generate Timeline", key="gen_sim"):
            try:
                fig_sim_exp = create_simulation_plot(years, mean, p10, p90, uncertainty_opacity, line_width)
                if white_background or include_title:
                    sim_title = f"Simulation ({start_year}–{end_year} AD)" if include_title else None
                    fig_sim_exp = create_print_ready_figure(fig_sim_exp, sim_title, font_scale)
                st.session_state.export_cache[f"{sim_cache_key}_png"] = fig_to_image_bytes(fig_sim_exp, "png", preset_info["width"], preset_info["height"], preset_info["scale"])
                st.session_state.export_cache[f"{sim_cache_key}_svg"] = fig_to_image_bytes(fig_sim_exp, "svg", preset_info["width"], preset_info["height"], 1.0)
                st.success("✅ Generated!")
            except Exception as e:
                st.error(f"Export error: {e}")
        
        if f"{sim_cache_key}_png" in st.session_state.export_cache:
            png = st.session_state.export_cache[f"{sim_cache_key}_png"]
            svg = st.session_state.export_cache[f"{sim_cache_key}_svg"]
            c2.download_button(f"🖼️ PNG ({len(png)/(1024*1024):.1f} MB)", png, f"Simulation_{export_preset}.png", "image/png", key="sim_png")
            c3.download_button("🖼️ SVG (Vector)", svg, f"Simulation_{export_preset}.svg", "image/svg+xml", key="sim_svg")
        else:
            c2.caption("Click Generate first")

        if excavation_data:
            st.markdown("---")
            st.markdown("#### ⚖️ Timeline with Excavation")
            exc_year_exp = st.slider("Excavation year for export", years[0], years[-1], years[len(years)//2], key="exc_year_exp")
            
            # Create unique cache key for comparison
            exc_cache_key = f"exc_{exc_year_exp}_{start_year}_{end_year}_{include_title}_{white_background}_{export_preset}"
            
            c1, c2, c3 = st.columns([1, 1, 1])
            if c1.button("🔄 Generate Comparison", key="gen_exc"):
                try:
                    fig_exc_exp = create_simulation_plot_with_excavation(years, mean, p10, p90, excavation_data, exc_year_exp, uncertainty_opacity, line_width)
                    if white_background or include_title:
                        exc_title = f"Simulation vs. Excavation ({exc_year_exp} AD)" if include_title else None
                        fig_exc_exp = create_print_ready_figure(fig_exc_exp, exc_title, font_scale)
                    st.session_state.export_cache[f"{exc_cache_key}_png"] = fig_to_image_bytes(fig_exc_exp, "png", preset_info["width"], preset_info["height"], preset_info["scale"])
                    st.session_state.export_cache[f"{exc_cache_key}_svg"] = fig_to_image_bytes(fig_exc_exp, "svg", preset_info["width"], preset_info["height"], 1.0)
                    st.success("✅ Generated!")
                except Exception as e:
                    st.error(f"Export error: {e}")
            
            if f"{exc_cache_key}_png" in st.session_state.export_cache:
                png = st.session_state.export_cache[f"{exc_cache_key}_png"]
                svg = st.session_state.export_cache[f"{exc_cache_key}_svg"]
                c2.download_button(f"🖼️ PNG ({len(png)/(1024*1024):.1f} MB)", png, f"Comparison_{exc_year_exp}.png", "image/png", key="exc_png")
                c3.download_button("🖼️ SVG (Vector)", svg, f"Comparison_{exc_year_exp}.svg", "image/svg+xml", key="exc_svg")
            else:
                c2.caption("Click Generate first")

        st.markdown("---")
        st.markdown("**Export Notes:** **PNG** = pixel-based (presentations, web) | **SVG** = vector (publications, scalable)")

st.markdown("---")
data_status = " | 📝 **Edited Market Data Active**" if market_data_is_edited else ""
st.caption(f"**STOCHASI – Stochastic Chronological Artifact Simulation** | Monte-Carlo modeling of archaeological artifact spectra | {n_cats} categories: {', '.join(categories)}{data_status}")
