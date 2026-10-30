"""
Data Preparation Module for Machine Learning.

Handles reproducible chronological train/val/test temporal splitting,
feature/target matrix extraction, strict data leakage assertions,
and feature scaling (fit exclusively on the training set).
"""

import json
from pathlib import Path
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def load_ml_dataset(dataset_path: Path = config.PROCESSED_DATASET_PATH) -> pd.DataFrame:
    """Load the processed ML dataset."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Processed ML dataset not found at: {dataset_path}")
    df = pd.read_csv(dataset_path)
    return df


def perform_temporal_split(
    df: pd.DataFrame,
    train_ratio: float = config.TRAIN_SPLIT_RATIO,
    val_ratio: float = config.VAL_SPLIT_RATIO,
    test_ratio: float = config.TEST_SPLIT_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Perform a strictly chronological split based on unique prediction timestamps.
    Preserves all files at each prediction timestamp together to prevent cross-file timestamp leakage.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"

    # Sort chronologically by prediction_timestamp
    df_sorted = df.sort_values(by=["prediction_timestamp", "file_id"]).reset_index(drop=True)

    unique_ts = sorted(df_sorted["prediction_timestamp"].unique())
    n_ts = len(unique_ts)

    n_train_ts = int(round(n_ts * train_ratio))
    n_val_ts = int(round(n_ts * val_ratio))
    # Remaining timestamps to test to ensure exact match
    n_test_ts = n_ts - n_train_ts - n_val_ts

    train_ts = unique_ts[:n_train_ts]
    val_ts = unique_ts[n_train_ts : n_train_ts + n_val_ts]
    test_ts = unique_ts[n_train_ts + n_val_ts :]

    df_train = df_sorted[df_sorted["prediction_timestamp"].isin(train_ts)].copy().reset_index(drop=True)
    df_val = df_sorted[df_sorted["prediction_timestamp"].isin(val_ts)].copy().reset_index(drop=True)
    df_test = df_sorted[df_sorted["prediction_timestamp"].isin(test_ts)].copy().reset_index(drop=True)

    # Validate strict chronological order
    max_train_ts = df_train["prediction_timestamp"].max()
    min_val_ts = df_val["prediction_timestamp"].min()
    max_val_ts = df_val["prediction_timestamp"].max()
    min_test_ts = df_test["prediction_timestamp"].min()

    assert max_train_ts < min_val_ts, (
        f"Temporal boundary violated between train and val! {max_train_ts} >= {min_val_ts}"
    )
    assert max_val_ts < min_test_ts, (
        f"Temporal boundary violated between val and test! {max_val_ts} >= {min_test_ts}"
    )

    split_info = {
        "total_rows": len(df_sorted),
        "total_timestamps": n_ts,
        "train": {
            "rows": len(df_train),
            "percentage": round(len(df_train) / len(df_sorted) * 100, 2),
            "timestamp_count": len(train_ts),
            "min_timestamp": df_train["prediction_timestamp"].min(),
            "max_timestamp": max_train_ts,
            "unique_files": int(df_train["file_id"].nunique()),
            "target_mean": round(float(df_train[config.TARGET_COLUMN].mean()), 4),
            "target_median": round(float(df_train[config.TARGET_COLUMN].median()), 4),
            "target_min": int(df_train[config.TARGET_COLUMN].min()),
            "target_max": int(df_train[config.TARGET_COLUMN].max()),
        },
        "validation": {
            "rows": len(df_val),
            "percentage": round(len(df_val) / len(df_sorted) * 100, 2),
            "timestamp_count": len(val_ts),
            "min_timestamp": min_val_ts,
            "max_timestamp": max_val_ts,
            "unique_files": int(df_val["file_id"].nunique()),
            "target_mean": round(float(df_val[config.TARGET_COLUMN].mean()), 4),
            "target_median": round(float(df_val[config.TARGET_COLUMN].median()), 4),
            "target_min": int(df_val[config.TARGET_COLUMN].min()),
            "target_max": int(df_val[config.TARGET_COLUMN].max()),
        },
        "test": {
            "rows": len(df_test),
            "percentage": round(len(df_test) / len(df_sorted) * 100, 2),
            "timestamp_count": len(test_ts),
            "min_timestamp": min_test_ts,
            "max_timestamp": df_test["prediction_timestamp"].max(),
            "unique_files": int(df_test["file_id"].nunique()),
            "target_mean": round(float(df_test[config.TARGET_COLUMN].mean()), 4),
            "target_median": round(float(df_test[config.TARGET_COLUMN].median()), 4),
            "target_min": int(df_test[config.TARGET_COLUMN].min()),
            "target_max": int(df_test[config.TARGET_COLUMN].max()),
        },
    }

    return df_train, df_val, df_test, split_info


def extract_X_y(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: List[str] = config.FEATURE_COLUMNS,
    target_col: str = config.TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Extract feature matrices X and target vectors y for Train, Validation, and Test.
    Strictly asserts that target and evaluation columns are excluded from X.
    """
    # Safeguard checks
    forbidden = [target_col, "true_hotspot_next_6h", "eval_true_hotspot_next_6h", "prediction_timestamp"]
    for col in forbidden:
        if col in feature_cols:
            raise ValueError(f"Data leakage detected! Forbidden column '{col}' is present in feature_cols.")

    X_train = df_train[feature_cols].copy()
    y_train = df_train[target_col].copy()

    X_val = df_val[feature_cols].copy()
    y_val = df_val[target_col].copy()

    X_test = df_test[feature_cols].copy()
    y_test = df_test[target_col].copy()

    # Sanity checks
    for split_name, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
        for f in forbidden:
            assert f not in X.columns, f"Leakage: '{f}' found in X_{split_name}!"

    return X_train, y_train, X_val, y_val, X_test, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_save_path: Path = config.SCALER_PATH,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit a StandardScaler ONLY on the training features X_train,
    then transform X_train, X_val, and X_test.
    Saves the fitted scaler artifact to disk.
    """
    scaler = StandardScaler()

    # Fit ONLY on train
    scaler.fit(X_train)

    # Transform all splits
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Persist scaler artifact
    scaler_save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_save_path)
    print(f"[INFO] Fitted StandardScaler saved to: {scaler_save_path}")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def save_splits_and_metadata(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    split_info: Dict[str, Any],
    metadata_path: Path = config.SPLIT_METADATA_PATH,
) -> None:
    """Save train.csv, validation.csv, test.csv and data_split_metadata.json."""
    config.TRAIN_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

    df_train.to_csv(config.TRAIN_DATASET_PATH, index=False)
    df_val.to_csv(config.VAL_DATASET_PATH, index=False)
    df_test.to_csv(config.TEST_DATASET_PATH, index=False)

    metadata = {
        "description": "Temporal 70/15/15 train/validation/test split for hybrid storage tiering ML",
        "split_ratios": {
            "train": config.TRAIN_SPLIT_RATIO,
            "validation": config.VAL_SPLIT_RATIO,
            "test": config.TEST_SPLIT_RATIO,
        },
        "feature_columns": config.FEATURE_COLUMNS,
        "target_column": config.TARGET_COLUMN,
        "evaluation_only_column": config.EVAL_GROUND_TRUTH_COLUMN,
        "scaling_method": "StandardScaler (fitted exclusively on X_train)",
        "scaler_artifact": str(config.SCALER_PATH.name),
        "split_statistics": split_info,
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"[INFO] Saved splits to:\n  - {config.TRAIN_DATASET_PATH}\n  - {config.VAL_DATASET_PATH}\n  - {config.TEST_DATASET_PATH}")
    print(f"[INFO] Saved metadata to: {metadata_path}")


def prepare_ml_data():
    """Execute complete reproducible ML data preparation pipeline."""
    print("=" * 80)
    print("PHASE 5: ML DATA PREPARATION & TEMPORAL SPLITTING")
    print("=" * 80)

    # 1. Load dataset
    df = load_ml_dataset()
    print(f"[1] Loaded dataset with {len(df)} rows and {df.shape[1]} columns.")

    # 2. Temporal split
    df_train, df_val, df_test, split_info = perform_temporal_split(df)

    # 3. Extract X and y
    X_train, y_train, X_val, y_val, X_test, y_test = extract_X_y(
        df_train, df_val, df_test, config.FEATURE_COLUMNS, config.TARGET_COLUMN
    )

    # 4. Feature scaling (fit on train only)
    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)

    # 5. Save splits and metadata
    save_splits_and_metadata(df_train, df_val, df_test, split_info)

    # 6. Print detailed verification report
    print("\n" + "=" * 80)
    print("TEMPORAL SPLIT VERIFICATION SUMMARY")
    print("=" * 80)
    for split_key in ["train", "validation", "test"]:
        info = split_info[split_key]
        print(f"[{split_key.upper()}]")
        print(f"  Row Count         : {info['rows']:,} ({info['percentage']}%)")
        print(f"  Timestamps Count  : {info['timestamp_count']}")
        print(f"  Timestamp Range   : {info['min_timestamp']}  -->  {info['max_timestamp']}")
        print(f"  Unique Files      : {info['unique_files']}")
        print(f"  Target Stats      : Mean={info['target_mean']:.4f} | Median={info['target_median']} | Min={info['target_min']} | Max={info['target_max']}")
        print("-" * 80)

    print("\nStrict Temporal Disjointness Assertions:")
    print(f"  max(train) < min(val)  : {split_info['train']['max_timestamp']} < {split_info['validation']['min_timestamp']} [PASSED]")
    print(f"  max(val) < min(test)   : {split_info['validation']['max_timestamp']} < {split_info['test']['min_timestamp']} [PASSED]")
    print(f"\nFeature Columns ({len(config.FEATURE_COLUMNS)}):")
    for i, col in enumerate(config.FEATURE_COLUMNS, 1):
        print(f"  {i}. {col}")
    print(f"\nTarget Column: {config.TARGET_COLUMN}")
    print(f"Evaluation Column: {config.EVAL_GROUND_TRUTH_COLUMN} (Strictly isolated from training)")
    print("\nData Preparation Pipeline Completed Successfully.")

    return X_train_s, y_train, X_val_s, y_val, X_test_s, y_test


if __name__ == "__main__":
    prepare_ml_data()
