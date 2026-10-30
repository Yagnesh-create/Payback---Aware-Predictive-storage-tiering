"""
Preprocessing Module for ML-Driven Hybrid Storage Staging and Hotspot Migration.

Loads, validates, and cleans raw access trace and metadata datasets without
modifying the raw CSV files on disk.
"""

from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


REQUIRED_TRACE_COLUMNS = [
    "event_id",
    "timestamp",
    "file_id",
    "operation",
    "file_size_bytes",
    "offset_bytes",
    "request_size_bytes",
    "response_time_us",
    "tier",
    "application_id",
    "true_hotspot_next_6h",
]

REQUIRED_METADATA_COLUMNS = [
    "file_id",
    "file_size_bytes",
    "application_id",
    "file_type",
]


def load_raw_data(
    trace_path: Path = config.TRACE_DATASET_PATH,
    meta_path: Path = config.METADATA_DATASET_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw CSV trace and metadata files from disk without altering original files.

    Returns:
        Tuple of (df_trace, df_meta)
    """
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace dataset not found at: {trace_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata dataset not found at: {meta_path}")

    df_trace = pd.read_csv(trace_path)
    df_meta = pd.read_csv(meta_path)
    return df_trace, df_meta


def validate_schema(df_trace: pd.DataFrame, df_meta: pd.DataFrame) -> None:
    """Validate presence of all required columns."""
    missing_trace_cols = [c for c in REQUIRED_TRACE_COLUMNS if c not in df_trace.columns]
    if missing_trace_cols:
        raise ValueError(f"Trace dataset missing required columns: {missing_trace_cols}")

    missing_meta_cols = [c for c in REQUIRED_METADATA_COLUMNS if c not in df_meta.columns]
    if missing_meta_cols:
        raise ValueError(f"Metadata dataset missing required columns: {missing_meta_cols}")


def run_data_validation(df_trace: pd.DataFrame, df_meta: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform comprehensive data quality and validation checks.

    Returns a dictionary of audit results.
    """
    validate_schema(df_trace, df_meta)

    # 1. Missing values
    trace_nulls = int(df_trace.isnull().sum().sum())
    meta_nulls = int(df_meta.isnull().sum().sum())

    # 2. Duplicate rows
    trace_dups = int(df_trace.duplicated().sum())
    meta_dups = int(df_meta.duplicated().sum())

    # 3. Numerical checks
    neg_file_size = int((df_trace["file_size_bytes"] < 0).sum())
    neg_offset = int((df_trace["offset_bytes"] < 0).sum())
    nonpos_req_size = int((df_trace["request_size_bytes"] <= 0).sum())
    neg_resp_time = int((df_trace["response_time_us"] < 0).sum())

    # 4. Categorical checks
    valid_ops = {"READ", "WRITE"}
    invalid_ops = int((~df_trace["operation"].str.upper().isin(valid_ops)).sum())

    valid_tiers = {"HDD", "SSD"}
    invalid_tiers = int((~df_trace["tier"].str.upper().isin(valid_tiers)).sum())

    return {
        "trace_rows": len(df_trace),
        "meta_rows": len(df_meta),
        "trace_nulls": trace_nulls,
        "meta_nulls": meta_nulls,
        "trace_duplicates": trace_dups,
        "meta_duplicates": meta_dups,
        "neg_file_size": neg_file_size,
        "neg_offset": neg_offset,
        "nonpos_req_size": nonpos_req_size,
        "neg_resp_time": neg_resp_time,
        "invalid_ops": invalid_ops,
        "invalid_tiers": invalid_tiers,
    }


def preprocess_trace(
    df_trace: pd.DataFrame, df_meta: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Standardize timestamps, sort events chronologically, filter corruptions (if any),
    and return prepared datasets along with an audit report.

    Strictly preserves raw data without silent modifications.
    """
    audit = run_data_validation(df_trace, df_meta)

    initial_len = len(df_trace)

    # Convert timestamp
    df_processed = df_trace.copy()
    df_processed["dt"] = pd.to_datetime(df_processed["timestamp"], errors="coerce")

    # Detect malformed timestamps
    malformed_ts = int(df_processed["dt"].isnull().sum())
    audit["malformed_timestamps"] = malformed_ts

    # Identify any invalid rows
    invalid_mask = (
        df_processed["dt"].isnull()
        | (df_processed["file_size_bytes"] < 0)
        | (df_processed["offset_bytes"] < 0)
        | (df_processed["request_size_bytes"] <= 0)
        | (df_processed["response_time_us"] < 0)
        | (~df_processed["operation"].str.upper().isin({"READ", "WRITE"}))
        | (~df_processed["tier"].str.upper().isin({"HDD", "SSD"}))
    )

    removed_count = int(invalid_mask.sum())
    audit["removed_records_count"] = removed_count

    if removed_count > 0:
        print(f"[WARNING] Removing {removed_count} invalid records from working trace.")
        df_processed = df_processed[~invalid_mask].copy()
    else:
        print("[INFO] 0 records removed. Working trace data is 100% valid.")

    # Chronological sort
    df_processed.sort_values(by="dt", inplace=True)
    df_processed.reset_index(drop=True, inplace=True)

    # Standardize uppercase for categories
    df_processed["operation"] = df_processed["operation"].str.upper()
    df_processed["tier"] = df_processed["tier"].str.upper()

    audit["final_trace_rows"] = len(df_processed)
    audit["is_monotonic"] = bool(df_processed["dt"].is_monotonic_increasing)

    return df_processed, df_meta.copy(), audit


if __name__ == "__main__":
    print("Testing Preprocessing Module...")
    raw_trace, raw_meta = load_raw_data()
    clean_trace, clean_meta, audit_report = preprocess_trace(raw_trace, raw_meta)
    print("\n--- Audit Report ---")
    for k, v in audit_report.items():
        print(f"  {k:<26}: {v}")
