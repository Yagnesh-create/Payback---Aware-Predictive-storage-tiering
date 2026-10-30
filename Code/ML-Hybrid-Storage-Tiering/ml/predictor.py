"""
Prediction Module for ML-Driven Hybrid Storage Staging and Hotspot Migration.

Loads the trained LinearRegression model and StandardScaler from disk,
accepts input features, and produces:
  1. Predicted future access count
  2. Hotspot / not-hotspot binary decision
"""

from pathlib import Path
from typing import Tuple, Union
import numpy as np
import pandas as pd
import joblib

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


class HotspotPredictor:
    """
    Loads trained model and scaler artifacts and provides a prediction interface.

    Usage:
        predictor = HotspotPredictor()
        pred_count, is_hotspot = predictor.predict_single(feature_dict)
        pred_df = predictor.predict_batch(feature_dataframe)
    """

    def __init__(
        self,
        model_path: Path = config.MODEL_PATH,
        scaler_path: Path = config.SCALER_PATH,
        threshold: int = config.HOTSPOT_THRESHOLD,
    ):
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found at: {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Fitted scaler not found at: {scaler_path}")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.threshold = threshold
        self.feature_columns = config.FEATURE_COLUMNS

        print(f"[INFO] Loaded model from: {model_path}")
        print(f"[INFO] Loaded scaler from: {scaler_path}")
        print(f"[INFO] Hotspot threshold: >= {threshold} predicted accesses")

    def predict_single(self, features: dict) -> Tuple[float, bool]:
        """
        Predict for a single file/window observation.

        Args:
            features: Dictionary with keys matching config.FEATURE_COLUMNS.
                      Must NOT contain true_hotspot_next_6h or future_access_count.

        Returns:
            (predicted_future_access_count, is_hotspot)
        """
        # Leakage guard
        assert "true_hotspot_next_6h" not in features, "Leakage: true_hotspot_next_6h in input!"
        assert "future_access_count" not in features, "Leakage: future_access_count in input!"

        row = pd.DataFrame([features])[self.feature_columns]
        scaled = self.scaler.transform(row)
        pred = float(self.model.predict(scaled)[0])
        pred = max(pred, 0.0)  # Access count >= 0
        is_hotspot = pred >= self.threshold
        return pred, is_hotspot

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict for a batch of file/window observations.

        Args:
            df: DataFrame containing at least config.FEATURE_COLUMNS.
                Must NOT contain true_hotspot_next_6h as a feature.

        Returns:
            DataFrame with added columns:
              - predicted_future_access_count
              - predicted_hotspot
        """
        # Leakage guard
        assert "true_hotspot_next_6h" not in self.feature_columns

        X = df[self.feature_columns].copy()
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        preds = np.maximum(preds, 0.0)

        result = df.copy()
        result["predicted_future_access_count"] = np.round(preds, 2)
        result["predicted_hotspot"] = (preds >= self.threshold).astype(int)
        return result


if __name__ == "__main__":
    print("Testing Predictor Module...")
    predictor = HotspotPredictor()

    # Load a few test rows
    test_df = pd.read_csv(config.TEST_DATASET_PATH)
    sample = test_df.head(5)

    results = predictor.predict_batch(sample)
    print("\nSample Predictions:")
    display_cols = [
        "file_id",
        "prediction_timestamp",
        config.TARGET_COLUMN,
        "predicted_future_access_count",
        "predicted_hotspot",
    ]
    if config.EVAL_GROUND_TRUTH_COLUMN in results.columns:
        display_cols.append(config.EVAL_GROUND_TRUTH_COLUMN)
    print(results[display_cols].to_string(index=False))
