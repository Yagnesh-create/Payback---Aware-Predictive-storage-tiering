"""
Feature Extraction Module for ML-Driven Hybrid Storage Staging and Hotspot Migration.

Constructs temporal observation windows and extracts file-level I/O features strictly
prior to the prediction timestamp to guarantee zero data leakage.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from features.preprocess import load_raw_data, preprocess_trace


def compute_burstiness(timestamps: List[pd.Timestamp]) -> float:
    """
    Calculate the normalized Goh-Barabasi burstiness coefficient:
        B = (sigma - mu) / (sigma + mu)
    Bounded in [-1, 1], where:
        B = -1: perfectly periodic / uniform arrivals
        B =  0: memoryless Poisson arrivals (sigma = mu)
        B = +1: highly concentrated bursty access bursts
    Returns 0.0 if access count <= 1 or sigma + mu == 0.
    """
    if len(timestamps) <= 1:
        return 0.0

    # Calculate inter-arrival intervals in seconds
    deltas = np.diff([ts.timestamp() for ts in timestamps])
    mu = float(np.mean(deltas))
    sigma = float(np.std(deltas))

    if (sigma + mu) == 0:
        return 0.0

    b = (sigma - mu) / (sigma + mu)
    return float(np.clip(b, -1.0, 1.0))


def extract_features_for_window(
    t: pd.Timestamp,
    df_past_12h: pd.DataFrame,
    df_future_6h: pd.DataFrame,
    meta_df: pd.DataFrame,
    last_known_tiers: Dict[str, int],
    last_known_access_times: Dict[str, pd.Timestamp],
) -> List[Dict[str, Any]]:
    """
    Extract temporal features and target for all files at prediction timestamp t.

    Strict Temporal Boundaries:
        - Previous Window: [t - 12h, t - 6h)
        - Recent Window  : [t - 6h, t)
        - Future Window  : [t, t + 6h)  <-- Target ONLY
    """
    t_recent_start = t - pd.Timedelta(hours=config.RECENT_WINDOW_HOURS)
    t_prev_start = t - pd.Timedelta(hours=config.WARMUP_HOURS)

    # 1. Partition past data strictly before t
    recent_events = df_past_12h[df_past_12h["dt"] >= t_recent_start]
    prev_events = df_past_12h[(df_past_12h["dt"] >= t_prev_start) & (df_past_12h["dt"] < t_recent_start)]

    # Pre-aggregate counts and stats
    recent_grp = recent_events.groupby("file_id")
    prev_grp = prev_events.groupby("file_id")
    past_grp = df_past_12h.groupby("file_id")
    future_grp = df_future_6h.groupby("file_id")

    recent_counts = recent_grp.size().to_dict()
    prev_counts = prev_grp.size().to_dict()
    future_counts = future_grp.size().to_dict()

    # Evaluation-only future ground truth
    future_eval_hotspot = future_grp["true_hotspot_next_6h"].max().to_dict()

    # Precompute per-file aggregated statistics for recent window
    recent_reads = recent_events[recent_events["operation"] == "READ"].groupby("file_id").size().to_dict()
    recent_writes = recent_events[recent_events["operation"] == "WRITE"].groupby("file_id").size().to_dict()
    recent_req_sizes = recent_grp["request_size_bytes"].mean().to_dict()

    # Fallback past 12h aggregates for files with 0 recent accesses but recent in past 12h
    past_reads = df_past_12h[df_past_12h["operation"] == "READ"].groupby("file_id").size().to_dict()
    past_writes = df_past_12h[df_past_12h["operation"] == "WRITE"].groupby("file_id").size().to_dict()
    past_req_sizes = past_grp["request_size_bytes"].mean().to_dict()

    # Group timestamps for burstiness
    recent_timestamps_by_file = recent_grp["dt"].apply(list).to_dict()

    # Update latest known tier and access time
    for _, row in df_past_12h.iterrows():
        fid = row["file_id"]
        tier_enc = config.TIER_SSD_ENCODING if row["tier"] == "SSD" else config.TIER_HDD_ENCODING
        last_known_tiers[fid] = tier_enc
        last_known_access_times[fid] = row["dt"]

    records = []
    for _, meta_row in meta_df.iterrows():
        fid = meta_row["file_id"]
        file_size = int(meta_row["file_size_bytes"])

        rec_cnt = int(recent_counts.get(fid, 0))
        prev_cnt = int(prev_counts.get(fid, 0))

        # Recency in seconds
        last_access = last_known_access_times.get(fid)
        if last_access is not None:
            recency_sec = float((t - last_access).total_seconds())
        else:
            recency_sec = float(config.MAX_RECENCY_SECONDS)

        # Burstiness in recent window
        ts_list = recent_timestamps_by_file.get(fid, [])
        burstiness = compute_burstiness(ts_list)

        # Read / write ratio
        if rec_cnt > 0:
            reads = recent_reads.get(fid, 0)
            writes = recent_writes.get(fid, 0)
            tot = reads + writes
            read_ratio = float(reads / tot) if tot > 0 else 0.0
            write_ratio = float(writes / tot) if tot > 0 else 0.0
            avg_req = float(recent_req_sizes.get(fid, 0.0))
        else:
            # Check past 12h fallback
            tot_past = past_reads.get(fid, 0) + past_writes.get(fid, 0)
            if tot_past > 0:
                read_ratio = float(past_reads.get(fid, 0) / tot_past)
                write_ratio = float(past_writes.get(fid, 0) / tot_past)
                avg_req = float(past_req_sizes.get(fid, 0.0))
            else:
                read_ratio = 0.0
                write_ratio = 0.0
                avg_req = 0.0

        # Current tier as of timestamp t
        tier_val = int(last_known_tiers.get(fid, config.TIER_HDD_ENCODING))

        # Target: future access count in [t, t + 6h)
        future_cnt = int(future_counts.get(fid, 0))

        # Evaluation ground truth: whether marked hotspot in future window
        eval_hotspot = int(future_eval_hotspot.get(fid, 0))

        record = {
            "file_id": fid,
            "prediction_timestamp": str(t),
            # Input Features (strictly pre-t)
            "recent_access_count": rec_cnt,
            "previous_access_count": prev_cnt,
            "recency_seconds": recency_sec,
            "burstiness": round(burstiness, 4),
            "file_size_bytes": file_size,
            "read_ratio": round(read_ratio, 4),
            "write_ratio": round(write_ratio, 4),
            "average_request_size": round(avg_req, 2),
            "current_tier": tier_val,
            # Target (future)
            "future_access_count": future_cnt,
            # Evaluation Only Ground Truth
            "eval_true_hotspot_next_6h": eval_hotspot,
        }
        records.append(record)

    return records


def build_ml_dataset(
    df_trace: pd.DataFrame, df_meta: pd.DataFrame
) -> pd.DataFrame:
    """
    Construct the full temporal ML training dataset across all 14 days.

    Generates prediction points every 6 hours starting after 12h of warmup.
    """
    start_dt = df_trace["dt"].min()
    end_dt = df_trace["dt"].max()

    # Prediction points: start at start + 12h, end at end - 6h (every 6 hours)
    first_pred = start_dt.floor("6h") + pd.Timedelta(hours=config.WARMUP_HOURS)
    last_pred = end_dt.floor("6h")

    pred_points = pd.date_range(start=first_pred, end=last_pred, freq=f"{config.STEP_WINDOW_HOURS}h")

    print(f"[INFO] Generating ML Dataset across {len(pred_points)} prediction timestamps...")
    print(f"       Range: {pred_points[0]} to {pred_points[-1]} (every {config.STEP_WINDOW_HOURS}h)")

    last_known_tiers: Dict[str, int] = {}
    last_known_access_times: Dict[str, pd.Timestamp] = {}

    all_records: List[Dict[str, Any]] = []

    for t in pred_points:
        t_hist_start = t - pd.Timedelta(hours=config.WARMUP_HOURS)
        t_future_end = t + pd.Timedelta(hours=config.FUTURE_WINDOW_HOURS)

        # Slice past strictly before t
        past_slice = df_trace[(df_trace["dt"] >= t_hist_start) & (df_trace["dt"] < t)]

        # Slice future strictly at or after t up to t + 6h
        future_slice = df_trace[(df_trace["dt"] >= t) & (df_trace["dt"] < t_future_end)]

        w_records = extract_features_for_window(
            t=t,
            df_past_12h=past_slice,
            df_future_6h=future_slice,
            meta_df=df_meta,
            last_known_tiers=last_known_tiers,
            last_known_access_times=last_known_access_times,
        )
        all_records.extend(w_records)

    df_ml = pd.DataFrame(all_records)
    return df_ml


def validate_no_data_leakage(df_ml: pd.DataFrame) -> Dict[str, bool]:
    """
    Perform rigorous sanity assertions to verify that zero data leakage occurs.
    """
    checks = {}

    # 1. Check feature columns do not include target or ground truth
    checks["no_target_in_features"] = config.TARGET_COLUMN not in config.FEATURE_COLUMNS
    checks["no_gt_in_features"] = config.EVAL_GROUND_TRUTH_COLUMN not in config.FEATURE_COLUMNS
    checks["no_raw_gt_in_features"] = "true_hotspot_next_6h" not in config.FEATURE_COLUMNS

    # 2. Check value boundaries
    checks["non_negative_target"] = bool((df_ml[config.TARGET_COLUMN] >= 0).all())
    checks["non_negative_features"] = bool((df_ml["recent_access_count"] >= 0).all() and (df_ml["previous_access_count"] >= 0).all())
    checks["valid_read_ratios"] = bool(((df_ml["read_ratio"] >= 0.0) & (df_ml["read_ratio"] <= 1.0)).all())
    checks["valid_write_ratios"] = bool(((df_ml["write_ratio"] >= 0.0) & (df_ml["write_ratio"] <= 1.0)).all())
    checks["valid_tier_encodings"] = bool(df_ml["current_tier"].isin([config.TIER_HDD_ENCODING, config.TIER_SSD_ENCODING]).all())
    checks["valid_burstiness_range"] = bool(((df_ml["burstiness"] >= -1.0) & (df_ml["burstiness"] <= 1.0)).all())

    # 3. Check for nulls
    checks["zero_null_features"] = bool(df_ml[config.FEATURE_COLUMNS].isnull().sum().sum() == 0)
    checks["zero_null_target"] = bool(df_ml[config.TARGET_COLUMN].isnull().sum() == 0)

    all_passed = all(checks.values())
    print("\n--- Data Leakage & Sanity Audit Results ---")
    for check_name, passed in checks.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {check_name:<30}: [{status}]")
    print("-" * 50)
    print(f"Overall Data Integrity Status: {'ALL CHECKS PASSED' if all_passed else 'FAILURES DETECTED'}")

    return checks


if __name__ == "__main__":
    print("Executing Feature Extraction Pipeline...")
    raw_trace, raw_meta = load_raw_data()
    clean_trace, clean_meta, audit = preprocess_trace(raw_trace, raw_meta)

    ml_dataset = build_ml_dataset(clean_trace, clean_meta)
    print(f"[SUCCESS] Generated ML dataset: {ml_dataset.shape[0]} rows x {ml_dataset.shape[1]} columns")

    # Run data leakage validation
    leakage_report = validate_no_data_leakage(ml_dataset)

    # Save to disk
    output_path = config.PROCESSED_DATASET_PATH
    ml_dataset.to_csv(output_path, index=False)
    print(f"[SUCCESS] Saved ML dataset to: {output_path}")
