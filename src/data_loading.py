"""
Data Loading and Validation Module
----------------------------------
Stage 1 & 2 of the CNB Breast Cancer Prediction Pipeline:
- Loads raw CNB clinical/pathology datasets (CSV/Excel).
- Cleans messy raw text numbers (percentages, ranges like '30-40%').
- Validates data schema, column types, integrity, and missing value profiles.
- Outputs summary statistics and data understanding reports.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DataLoading")


def load_config(config_path: str | Path = "config/config.yaml") -> Dict[str, Any]:
    """
    Loads and parses the YAML configuration file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Dict containing configuration parameters.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def parse_numeric_cell(val: Any) -> Optional[float]:
    """
    Converts various clinical text entries (e.g. '30-40%', '0.15', '5%', '>2') to a float.
    """
    if pd.isna(val) or val is None:
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)

    s = str(val).strip()
    if not s or s.lower() in ["nan", "null", "none", "", "?"]:
        return np.nan

    # Check for percentage range e.g. "30-40%" or "10-20"
    range_match = re.match(r"^(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\%?$", s)
    if range_match:
        low = float(range_match.group(1))
        high = float(range_match.group(2))
        avg = (low + high) / 2.0
        return avg / 100.0 if ("%" in s or avg > 1.0) else avg

    # Check for percentage single value e.g. "40%"
    pct_match = re.match(r"^(\d+(?:\.\d+)?)\%$", s)
    if pct_match:
        return float(pct_match.group(1)) / 100.0

    # Try regular float conversion
    try:
        f_val = float(s)
        return f_val
    except ValueError:
        return np.nan


def clean_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans known clinical column formatting quirks in raw tabular data.
    """
    cleaned = df.copy()

    # Strip column names
    cleaned.columns = [c.strip() for c in cleaned.columns]

    # Clean Ki67 column if present
    for ki_col in ["Ki67", "ki67", "ki67_index", "Ki-67"]:
        if ki_col in cleaned.columns:
            cleaned[ki_col] = cleaned[ki_col].apply(parse_numeric_cell)

    # Clean Age, Tumour Size, etc.
    for col in cleaned.columns:
        col_lower = col.lower()
        if "age" in col_lower or "size" in col_lower or "metastases" in col_lower:
            cleaned[col] = pd.to_numeric(cleaned[col].astype(str).str.extract(r"(\d+\.?\d*)")[0], errors="coerce")

    return cleaned


def load_raw_data(data_path: str | Path) -> pd.DataFrame:
    """
    Loads raw dataset from file (CSV or Excel) and applies initial sanitization.

    Args:
        data_path: Path to raw data file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at {path.resolve()}.")

    if path.suffix.lower() in [".csv", ".txt"]:
        df = pd.read_csv(path)
    elif path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Expected .csv or .xlsx")

    logger.info("Loaded raw dataset from %s | Shape: %s", path.resolve(), df.shape)
    cleaned_df = clean_raw_dataframe(df)
    return cleaned_df


def validate_raw_data(df: pd.DataFrame, target_col: str = "Molecular subtype") -> Dict[str, Any]:
    """
    Validates dataset integrity, checks for target column presence, duplicate rows,
    and missing values.
    """
    report: Dict[str, Any] = {
        "num_rows": int(df.shape[0]),
        "num_cols": int(df.shape[1]),
        "target_present": target_col in df.columns,
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values_per_col": df.isnull().sum().to_dict(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }

    if not report["target_present"]:
        raise ValueError(f"Target column '{target_col}' is missing from the dataset columns: {list(df.columns)}")

    if df[target_col].isnull().any():
        n_missing = int(df[target_col].isnull().sum())
        logger.warning("Found %d rows with missing target value in '%s'.", n_missing, target_col)
        report["target_missing_count"] = n_missing

    target_counts = df[target_col].value_counts(dropna=False).to_dict()
    report["target_distribution"] = target_counts

    logger.info("Validation Complete: %d rows, %d cols | Target distribution: %s",
                report["num_rows"], report["num_cols"], target_counts)
    return report


def get_data_summary(df: pd.DataFrame, target_col: str = "Molecular subtype") -> str:
    """
    Generates a comprehensive readable text summary of the dataset.
    """
    summary_lines = [
        "=" * 60,
        "CNB BREAST CANCER DATASET - SUMMARY REPORT",
        "=" * 60,
        f"Total Records: {len(df)}",
        f"Total Columns: {len(df.columns)}",
        "",
        "Column Information & Missing Values:",
        "-" * 60,
    ]
    for col in df.columns:
        n_miss = df[col].isnull().sum()
        pct_miss = (n_miss / len(df)) * 100
        dtype = df[col].dtype
        unique_vals = df[col].nunique()
        summary_lines.append(
            f" - {col:32s} | Type: {str(dtype):10s} | Missing: {n_miss:3d} ({pct_miss:4.1f}%) | Unique: {unique_vals}"
        )

    if target_col in df.columns:
        summary_lines.extend([
            "",
            "Target Class Distribution:",
            "-" * 60,
        ])
        for val, count in df[target_col].value_counts(dropna=False).items():
            pct = (count / len(df)) * 100
            summary_lines.append(f" - {str(val):25s}: {count:4d} ({pct:5.2f}%)")

    summary_lines.append("=" * 60)
    return "\n".join(summary_lines)


def main() -> None:
    """CLI execution for Data Loading module."""
    parser = argparse.ArgumentParser(description="CNB Breast Cancer Pipeline - Data Loading & Validation")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    raw_path = config["paths"]["raw_data"]
    target_col = config["data"]["target_col"]

    df = load_raw_data(raw_path)
    validate_raw_data(df, target_col=target_col)
    print("\n" + get_data_summary(df, target_col=target_col) + "\n")


if __name__ == "__main__":
    main()
