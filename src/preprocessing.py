"""
Data Preprocessing and Transformation Module
---------------------------------------------
Stage 3 & 4 of the CNB Breast Cancer Prediction Pipeline:
- Cleans data: handles missing values, invalid entries, duplicates, and outliers.
- Strict anti-leakage guarantee: fits transformers strictly on training data.
- Handles categorical encoding and numerical scaling.
- Implements stratified train/test split.
- Supports both binary and multi-class target label encoding.
- Persists processed datasets and fitted preprocessor & label encoder artifacts.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler, StandardScaler

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading import load_config, load_raw_data, validate_raw_data

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Preprocessing")


class CNBPreprocessor(BaseEstimator, TransformerMixin):
    """
    Composite transformer for CNB patient clinical data.
    Fits all imputation, outlier clipping bounds, encodings, and scalers strictly on training data.
    """
    __module__ = "src.preprocessing"

    def __init__(
        self,
        id_cols: Optional[List[str]] = None,
        numeric_strategy: str = "median",
        categorical_strategy: str = "most_frequent",
        categorical_fill_value: str = "Unknown",
        scaling: str = "standard",
        outlier_clipping: bool = True,
        lower_percentile: float = 0.01,
        upper_percentile: float = 0.99,
    ) -> None:
        self.id_cols = id_cols or ["Patient ID", "patient_id", "sample_id"]
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.categorical_fill_value = categorical_fill_value
        self.scaling = scaling
        self.outlier_clipping = outlier_clipping
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

        # Fitted attributes
        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.feature_names_out_: List[str] = []
        self.num_imputer_: Optional[SimpleImputer] = None
        self.cat_imputer_: Optional[SimpleImputer] = None
        self.scaler_: Optional[Any] = None
        self.encoder_: Optional[OneHotEncoder] = None
        self.outlier_bounds_: Dict[str, Tuple[float, float]] = {}

    def _drop_unneeded_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [c for c in self.id_cols if c in df.columns]
        return df.drop(columns=cols_to_drop, errors="ignore")

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CNBPreprocessor":
        """
        Fits transformers on the training dataframe.
        """
        X_clean = self._drop_unneeded_cols(X.copy())

        self.numeric_cols_ = [
            c for c in X_clean.columns if pd.api.types.is_numeric_dtype(X_clean[c])
        ]
        self.categorical_cols_ = [
            c for c in X_clean.columns if c not in self.numeric_cols_
        ]

        logger.info(
            "Fitting Preprocessor | Numeric features (%d): %s | Categorical features (%d): %s",
            len(self.numeric_cols_),
            self.numeric_cols_,
            len(self.categorical_cols_),
            self.categorical_cols_,
        )

        # 1. Numeric Outlier Bounds & Imputation
        if self.numeric_cols_:
            if self.outlier_clipping:
                for col in self.numeric_cols_:
                    valid_series = X_clean[col].dropna()
                    if len(valid_series) > 0:
                        low = float(valid_series.quantile(self.lower_percentile))
                        high = float(valid_series.quantile(self.upper_percentile))
                        self.outlier_bounds_[col] = (low, high)

            self.num_imputer_ = SimpleImputer(strategy=self.numeric_strategy)
            num_data = X_clean[self.numeric_cols_].to_numpy(dtype=float, copy=True)
            if self.outlier_clipping:
                for i, col in enumerate(self.numeric_cols_):
                    low, high = self.outlier_bounds_.get(col, (-np.inf, np.inf))
                    num_data[:, i] = np.clip(num_data[:, i], low, high)
            num_imputed = self.num_imputer_.fit_transform(num_data)

            # Scaler
            if self.scaling == "standard":
                self.scaler_ = StandardScaler().fit(num_imputed)
            elif self.scaling == "robust":
                self.scaler_ = RobustScaler().fit(num_imputed)
            elif self.scaling == "none":
                self.scaler_ = None
            else:
                self.scaler_ = StandardScaler().fit(num_imputed)

        # 2. Categorical Imputation & Encoding
        if self.categorical_cols_:
            if self.categorical_strategy == "constant":
                self.cat_imputer_ = SimpleImputer(
                    strategy="constant", fill_value=self.categorical_fill_value
                )
            else:
                self.cat_imputer_ = SimpleImputer(strategy="most_frequent")

            cat_raw = X_clean[self.categorical_cols_].fillna(self.categorical_fill_value).astype(str)
            cat_imputed = self.cat_imputer_.fit_transform(cat_raw)
            self.encoder_ = OneHotEncoder(
                sparse_output=False, handle_unknown="ignore"
            ).fit(cat_imputed)

        # Compute output feature names
        out_features: List[str] = []
        if self.numeric_cols_:
            out_features.extend(self.numeric_cols_)
        if self.categorical_cols_ and self.encoder_ is not None:
            encoded_names = list(
                self.encoder_.get_feature_names_out(self.categorical_cols_)
            )
            out_features.extend(encoded_names)

        self.feature_names_out_ = out_features
        logger.info("Total transformed output features: %d", len(self.feature_names_out_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms input features using fitted rules.
        """
        X_clean = self._drop_unneeded_cols(X.copy())
        arrays_to_concat: List[np.ndarray] = []

        # 1. Transform Numeric
        if self.numeric_cols_:
            num_data = X_clean[self.numeric_cols_].to_numpy(dtype=float, copy=True)
            if self.outlier_clipping:
                for i, col in enumerate(self.numeric_cols_):
                    low, high = self.outlier_bounds_.get(col, (-np.inf, np.inf))
                    num_data[:, i] = np.clip(num_data[:, i], low, high)

            if self.num_imputer_ is not None:
                num_imputed = self.num_imputer_.transform(num_data)
            else:
                num_imputed = num_data

            if self.scaler_ is not None:
                num_scaled = self.scaler_.transform(num_imputed)
            else:
                num_scaled = num_imputed

            arrays_to_concat.append(num_scaled)

        # 2. Transform Categorical
        if self.categorical_cols_ and self.cat_imputer_ is not None and self.encoder_ is not None:
            cat_raw = X_clean[self.categorical_cols_].fillna(self.categorical_fill_value).astype(str)
            cat_imputed = self.cat_imputer_.transform(cat_raw)
            cat_encoded = self.encoder_.transform(cat_imputed)
            arrays_to_concat.append(cat_encoded)

        if not arrays_to_concat:
            return pd.DataFrame(index=X.index)

        X_transformed = np.hstack(arrays_to_concat)
        return pd.DataFrame(
            X_transformed, columns=self.feature_names_out_, index=X.index
        )

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(self.feature_names_out_)


def split_data(
    df: pd.DataFrame,
    target_col: str = "Molecular subtype",
    test_size: float = 0.2,
    random_seed: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the dataset into train and test sets prior to any feature transformation.
    """
    df_valid = df.dropna(subset=[target_col]).copy()
    X = df_valid.drop(columns=[target_col])
    y = df_valid[target_col]

    strat = None
    if stratify:
        class_counts = y.value_counts()
        n_samples = len(y)
        n_classes = len(class_counts)
        n_test = int(np.ceil(n_samples * test_size)) if isinstance(test_size, float) else test_size
        n_train = n_samples - n_test

        if (class_counts >= 2).all() and n_classes > 1 and n_test >= n_classes and n_train >= n_classes:
            strat = y
        else:
            logger.warning(
                "Stratified split requirements not met (n_train=%d, n_test=%d for %d classes). "
                "Falling back to unstratified train_test_split.",
                n_train,
                n_test,
                n_classes,
            )
            strat = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=strat
    )

    logger.info(
        "Split dataset: Train size = %d, Test size = %d (Test ratio: %.2f)",
        len(X_train),
        len(X_test),
        test_size,
    )
    return X_train, X_test, y_train, y_test


def apply_imbalance_handling(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = "none",
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Applies synthetic oversampling (SMOTE) strictly to the training fold if requested.
    """
    if method == "smote":
        try:
            from imblearn.over_sampling import SMOTE

            logger.info("Applying SMOTE oversampling to training data...")
            smote = SMOTE(random_state=random_seed)
            X_res, y_res = smote.fit_resample(X_train, y_train)
            logger.info(
                "SMOTE Resampling complete: %d samples (Original: %d)",
                len(X_res),
                len(X_train),
            )
            return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(
                y_res, name=y_train.name
            )
        except ImportError:
            logger.warning("imblearn not installed. Skipping SMOTE.")
            return X_train, y_train
        except Exception as exc:
            logger.warning("SMOTE failed (%s). Continuing with original train set.", exc)
            return X_train, y_train
    return X_train, y_train


def run_preprocessing_pipeline(
    config_path: str | Path = "config/config.yaml",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, CNBPreprocessor, LabelEncoder]:
    """
    Executes the complete preprocessing pipeline based on configuration settings.
    """
    config = load_config(config_path)

    raw_path = config["paths"]["raw_data"]
    target_col = config["data"]["target_col"]
    test_size = config["data"].get("test_size", 0.20)
    stratify = config["data"].get("stratify", True)
    seed = config["project"]["random_seed"]

    df = load_raw_data(raw_path)
    validate_raw_data(df, target_col=target_col)

    # 1. Stratified Split BEFORE fitting any transformers
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = split_data(
        df,
        target_col=target_col,
        test_size=test_size,
        random_seed=seed,
        stratify=stratify,
    )

    # 2. Encode Target with LabelEncoder
    le = LabelEncoder()
    y_train_arr = le.fit_transform(y_train_raw.astype(str))
    y_test_arr = le.transform(y_test_raw.astype(str))
    y_train = pd.Series(y_train_arr, index=X_train_raw.index, name=target_col)
    y_test = pd.Series(y_test_arr, index=X_test_raw.index, name=target_col)

    logger.info("Target classes mapped: %s", {idx: cls for idx, cls in enumerate(le.classes_)})

    # 3. Fit Preprocessor ONLY on Training Data
    prep_cfg = config.get("preprocessing", {})
    id_cols = config["data"].get("id_cols", ["Patient ID", "patient_id"])
    preprocessor = CNBPreprocessor(
        id_cols=id_cols,
        numeric_strategy=prep_cfg.get("missing_numeric_strategy", "median"),
        categorical_strategy=prep_cfg.get("missing_categorical_strategy", "most_frequent"),
        categorical_fill_value=prep_cfg.get("categorical_fill_value", "Unknown"),
        scaling=prep_cfg.get("numeric_scaling", "standard"),
        outlier_clipping=prep_cfg.get("outlier_clipping", True),
        lower_percentile=prep_cfg.get("outlier_lower_percentile", 0.01),
        upper_percentile=prep_cfg.get("outlier_upper_percentile", 0.99),
    )

    preprocessor.fit(X_train_raw)

    # 4. Transform Train and Test
    X_train_trans = preprocessor.transform(X_train_raw)
    X_test_trans = preprocessor.transform(X_test_raw)

    # 5. Handle Imbalance on Train set only
    imbalance_method = prep_cfg.get("handle_imbalance", "none")
    X_train_final, y_train_final = apply_imbalance_handling(
        X_train_trans, y_train, method=imbalance_method, random_seed=seed
    )

    # 6. Save Artifacts and Processed Data
    processed_dir = Path(config["paths"]["processed_dir"])
    models_dir = Path(config["paths"]["models_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    train_out = X_train_final.copy()
    train_out[target_col] = y_train_final.values
    train_out.to_csv(config["paths"]["train_data"], index=False)

    test_out = X_test_trans.copy()
    test_out[target_col] = y_test.values
    test_out.to_csv(config["paths"]["test_data"], index=False)

    preprocessor_artifact_path = Path(config["paths"]["preprocessor_artifact"])
    joblib.dump(preprocessor, preprocessor_artifact_path)

    le_artifact_path = Path(config["paths"].get("label_encoder_artifact", "models/label_encoder.joblib"))
    joblib.dump(le, le_artifact_path)

    logger.info("Saved fitted preprocessor artifact to %s", preprocessor_artifact_path.resolve())
    logger.info("Saved fitted label encoder artifact to %s", le_artifact_path.resolve())
    logger.info("Saved processed train & test datasets to %s", processed_dir.resolve())

    return X_train_final, X_test_trans, y_train_final, y_test, preprocessor, le


def main() -> None:
    """CLI entry point for data preprocessing."""
    parser = argparse.ArgumentParser(
        description="CNB Breast Cancer Pipeline - Preprocessing & Splitting"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to YAML config"
    )
    args = parser.parse_args()

    run_preprocessing_pipeline(args.config)


if __name__ == "__main__":
    main()
