"""
Inference and Prediction Pipeline Module
----------------------------------------
Stage 8 of the CNB Breast Cancer Prediction Pipeline:
- Loads trained model, preprocessor artifact, label encoder, and selected features list.
- Performs end-to-end inference on single patient records or batch CSV datasets.
- Returns predicted class label, confidence probabilities, and classification report.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading import clean_raw_dataframe, load_config
from src.preprocessing import CNBPreprocessor

# Register CNBPreprocessor into __main__ for unpickling backward compatibility
if "__main__" in sys.modules:
    setattr(sys.modules["__main__"], "CNBPreprocessor", CNBPreprocessor)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Predictor")


class CNBPredictor:
    """
    Inference engine for CNB Breast Cancer Molecular Subtype and Clinical Predictions.
    """

    def __init__(
        self,
        config_path: str | Path = "config/config.yaml",
        model_path: Optional[str | Path] = None,
        preprocessor_path: Optional[str | Path] = None,
        label_encoder_path: Optional[str | Path] = None,
        selected_features_path: Optional[str | Path] = None,
    ) -> None:
        self.config = load_config(config_path)

        m_path = Path(model_path or self.config["paths"]["best_model"])
        p_path = Path(preprocessor_path or self.config["paths"]["preprocessor_artifact"])
        le_path = Path(label_encoder_path or self.config["paths"].get("label_encoder_artifact", "models/label_encoder.joblib"))
        sf_path = Path(
            selected_features_path
            or self.config["paths"]["selected_features_artifact"]
        )

        if not m_path.exists():
            raise FileNotFoundError(f"Model artifact not found at: {m_path.resolve()}")
        if not p_path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at: {p_path.resolve()}")

        self.model = joblib.load(m_path)
        self.preprocessor = joblib.load(p_path)

        if le_path.exists():
            self.label_encoder = joblib.load(le_path)
            self.classes = list(self.label_encoder.classes_)
        else:
            self.label_encoder = None
            self.classes = getattr(self.model, "classes_", [0, 1])

        if sf_path.exists():
            with open(sf_path, "r", encoding="utf-8") as f:
                self.selected_features = json.load(f).get("selected_features", [])
        else:
            self.selected_features = []

        logger.info("CNB Predictor initialized with model: %s | Classes: %s",
                    type(self.model).__name__, self.classes)

    def predict_record(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs prediction on a single patient record dictionary.
        """
        df = pd.DataFrame([patient_data])
        results_df = self.predict_dataframe(df)
        return results_df.iloc[0].to_dict()

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs batch prediction on a dataframe of patient records.
        """
        df_clean = clean_raw_dataframe(df)
        X_trans = self.preprocessor.transform(df_clean)

        if self.selected_features:
            valid_feats = [f for f in self.selected_features if f in X_trans.columns]
            X_model = X_trans[valid_feats]
        else:
            X_model = X_trans

        preds = self.model.predict(X_model)

        output_df = df.copy()
        if self.label_encoder is not None:
            output_df["predicted_label"] = self.label_encoder.inverse_transform(preds)
        else:
            output_df["predicted_label"] = preds

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_model)
            output_df["prediction_confidence"] = [round(float(np.max(p)), 4) for p in probs]
            for idx, cls_name in enumerate(self.classes):
                output_df[f"prob_{cls_name}"] = [round(float(p[idx]), 4) for p in probs]

        return output_df

    def predict_csv(
        self, input_csv_path: str | Path, output_csv_path: Optional[str | Path] = None
    ) -> pd.DataFrame:
        """
        Reads CSV, performs batch inference, and optionally saves results.
        """
        input_path = Path(input_csv_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {input_path.resolve()}")

        df = pd.read_csv(input_path)
        logger.info("Running batch prediction on %d records from %s", len(df), input_path)
        results = self.predict_dataframe(df)

        if output_csv_path:
            out_p = Path(output_csv_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            results.to_csv(out_p, index=False)
            logger.info("Saved prediction results to %s", out_p.resolve())

        return results


def run_interactive_demo(predictor: CNBPredictor) -> None:
    """Runs a demonstration with representative cases."""
    print("\n" + "=" * 60)
    print("CNB BREAST CANCER DIAGNOSTIC PREDICTOR - DEMO MODE")
    print("=" * 60)

    sample_case_1 = {
        "Patient ID": "DEMO_001",
        "Age(years)": 52,
        "Tumour Size(cm)": 2.5,
        "Tumour Type": "Invasive ductal carcinoma",
        "ER": "Positive",
        "PR": "Positive",
        "HER2": "Negative",
        "HER2 Expression": "1+",
        "Histological grading": 2,
        "Surgical": "Axillary lymph node dissection",
        "Ki67": 0.08,
        "Number of lymph node metastases": 1,
        "ALN status": "N+(1-2)",
    }

    sample_case_2 = {
        "Patient ID": "DEMO_002",
        "Age(years)": 45,
        "Tumour Size(cm)": 2.2,
        "Tumour Type": "Invasive ductal carcinoma",
        "ER": "Negative",
        "PR": "Negative",
        "HER2": "Negative",
        "HER2 Expression": "0",
        "Histological grading": 3,
        "Surgical": "Axillary lymph node dissection",
        "Ki67": 0.60,
        "Number of lymph node metastases": 0,
        "ALN status": "N0",
    }

    for name, sample in [("Case 1 (Hormone-Receptor Positive)", sample_case_1), ("Case 2 (High Ki67, Triple Negative)", sample_case_2)]:
        res = predictor.predict_record(sample)
        print(f"\n--- {name} ---")
        print(f"  Patient ID:       {res.get('Patient ID')}")
        print(f"  Predicted Class:  {res.get('predicted_label')}")
        if "prediction_confidence" in res:
            print(f"  Confidence:       {res['prediction_confidence'] * 100:.1f}%")

    print("\n" + "=" * 60 + "\n")


def main() -> None:
    """CLI execution for prediction."""
    parser = argparse.ArgumentParser(
        description="CNB Breast Cancer Pipeline - Inference and Prediction"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to YAML config"
    )
    parser.add_argument("--input-csv", type=str, help="Path to input CSV for batch prediction")
    parser.add_argument("--output-csv", type=str, help="Path to save prediction output CSV")
    parser.add_argument("--demo", action="store_true", help="Run interactive demo cases")
    args = parser.parse_args()

    predictor = CNBPredictor(config_path=args.config)

    if args.input_csv:
        results = predictor.predict_csv(args.input_csv, args.output_csv)
        print(results[["predicted_label", "prediction_confidence"]].head(10))
    else:
        run_interactive_demo(predictor)


if __name__ == "__main__":
    main()
