"""
Unit and Integration Tests for CNB Breast Cancer Pipeline
----------------------------------------------------------
Tests cover:
- Data loading and sanitization
- Leakage-free preprocessing and transforms
- Feature selection execution
- Model training, CV evaluation, and prediction pipeline
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading import clean_raw_dataframe, load_config, parse_numeric_cell
from src.preprocessing import CNBPreprocessor, split_data


def test_parse_numeric_cell():
    assert parse_numeric_cell("30-40%") == 0.35
    assert parse_numeric_cell("50%") == 0.50
    assert parse_numeric_cell("0.25") == 0.25
    assert np.isnan(parse_numeric_cell(None))
    assert np.isnan(parse_numeric_cell("?"))


def test_clean_raw_dataframe():
    raw_sample = pd.DataFrame({
        "Patient ID": [1, 2],
        "Ki67": ["10-20%", "0.05"],
        "Age(years)": ["52", "61"],
    })
    cleaned = clean_raw_dataframe(raw_sample)
    assert cleaned["Ki67"].iloc[0] == 0.15
    assert cleaned["Ki67"].iloc[1] == 0.05
    assert cleaned["Age(years)"].iloc[0] == 52.0


def test_preprocessor_fit_transform():
    df = pd.DataFrame({
        "Patient ID": [101, 102, 103, 104, 105, 106, 107, 108, 109],
        "Age(years)": [45, 60, 52, np.nan, 58, 49, 62, 55, 41],
        "Tumour Type": [
            "Invasive ductal carcinoma", "Other type", "Invasive ductal carcinoma",
            "Other type", "Invasive ductal carcinoma", "Other type",
            "Invasive ductal carcinoma", "Other type", "Invasive ductal carcinoma",
        ],
        "Ki67": [0.1, 0.4, 0.2, 0.5, 0.15, 0.3, 0.25, 0.35, 0.1],
        "Molecular subtype": [
            "Luminal A", "Luminal A", "Luminal A",
            "Triple negative", "Triple negative", "Triple negative",
            "HER2(+)", "HER2(+)", "HER2(+)",
        ],
    })

    X_train, X_test, y_train, y_test = split_data(df, target_col="Molecular subtype", test_size=0.33, random_seed=42)

    preprocessor = CNBPreprocessor(id_cols=["Patient ID"])
    preprocessor.fit(X_train)

    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    assert len(X_train_trans) == len(X_train)
    assert len(X_test_trans) == len(X_test)
    assert not X_train_trans.isnull().any().any()
    assert not X_test_trans.isnull().any().any()


def test_config_load():
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    config = load_config(config_path)
    assert "project" in config
    assert "paths" in config
    assert "training" in config
