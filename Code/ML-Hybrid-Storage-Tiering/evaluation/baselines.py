"""
Baseline Systems (Phase 17) for ML-Driven Hybrid Storage Staging.

Implements baseline comparison methods to validate the proposed
ML-driven migration approach. All baselines use the same underlying
dataset, evaluation period, file population, simulated storage
assumptions, latency assumptions, and migration bandwidth assumptions.

Baseline 1 — Static Tiering:
    Files remain in their initial tier. No migration occurs.

Baseline 2 — Frequency-Based Migration:
    Migration decisions based solely on recent access frequency.
    Does NOT use the trained ML model.

Baseline 3 — Rule-Based Hotspot (optional):
    Simple threshold rule without payback filtering.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


# ============================================================
# BASELINE 1 — STATIC TIERING
# ============================================================

def run_static_tiering_baseline(
    df: pd.DataFrame,
    hdd_latency_us: float = config.HDD_READ_LATENCY_US,
    ssd_latency_us: float = config.SSD_READ_LATENCY_US,
) -> Dict:
    """
    Static Tiering Baseline: Files remain in their initial tier.
    No predictive migration occurs.

    For each file at each prediction timestamp, the estimated latency
    depends solely on the file's current tier assignment.

    Args:
        df: DataFrame with 'current_tier' and 'future_access_count' columns.
        hdd_latency_us: Simulated HDD latency in microseconds.
        ssd_latency_us: Simulated SSD latency in microseconds.

    Returns:
        Dictionary of baseline metrics.
    """
    df_eval = df.copy()

    # Assign latency based on current tier
    df_eval["access_latency_us"] = df_eval["current_tier"].apply(
        lambda t: ssd_latency_us if t == config.TIER_SSD_ENCODING else hdd_latency_us
    )

    # Total estimated latency = accesses × per-access latency
    df_eval["total_latency_us"] = df_eval["future_access_count"] * df_eval["access_latency_us"]

    total_accesses = df_eval["future_access_count"].sum()
    total_latency = df_eval["total_latency_us"].sum()

    # Average latency per access
    avg_latency_us = total_latency / total_accesses if total_accesses > 0 else 0.0

    # SSD utilization
    total_files_per_ts = df_eval.groupby("prediction_timestamp").size()
    ssd_files_per_ts = df_eval[df_eval["current_tier"] == config.TIER_SSD_ENCODING].groupby(
        "prediction_timestamp"
    ).size()
    avg_ssd_count = ssd_files_per_ts.mean() if len(ssd_files_per_ts) > 0 else 0
    avg_total_count = total_files_per_ts.mean() if len(total_files_per_ts) > 0 else 1
    ssd_utilization = avg_ssd_count / avg_total_count if avg_total_count > 0 else 0.0

    # Count actual hotspots on HDD (missed by static policy)
    if "future_access_count" in df_eval.columns:
        hdd_hotspots = df_eval[
            (df_eval["current_tier"] == config.TIER_HDD_ENCODING)
            & (df_eval["future_access_count"] >= config.HOTSPOT_THRESHOLD)
        ]
        missed_hotspots = len(hdd_hotspots)
    else:
        missed_hotspots = 0

    return {
        "method": "Static Tiering",
        "avg_latency_us": avg_latency_us,
        "avg_latency_ms": avg_latency_us / 1000.0,
        "total_latency_us": total_latency,
        "total_accesses": int(total_accesses),
        "migrations": 0,
        "migration_time_sec": 0.0,
        "migration_volume_bytes": 0,
        "false_hotspot_rate": None,  # N/A — no predictions made
        "useful_migration_rate": None,  # N/A — no migrations
        "ssd_utilization": ssd_utilization,
        "missed_hotspots": missed_hotspots,
        "unnecessary_migrations": 0,
    }


# ============================================================
# BASELINE 2 — FREQUENCY-BASED MIGRATION
# ============================================================

def run_frequency_baseline(
    df: pd.DataFrame,
    threshold: int = config.HOTSPOT_THRESHOLD,
    max_migrations: int = config.MAX_MIGRATIONS_PER_RUN,
    min_payback: float = config.MIN_PAYBACK_RATIO,
    hdd_latency_us: float = config.HDD_READ_LATENCY_US,
    ssd_latency_us: float = config.SSD_READ_LATENCY_US,
    bandwidth: float = config.MIGRATION_BANDWIDTH_BYTES_PER_SEC,
) -> Dict:
    """
    Frequency-Based Migration Baseline.

    This baseline does NOT use the trained ML model. It makes migration
    decisions using only the recent access frequency (recent_access_count),
    which is a historical feature — NOT a future prediction.

    Policy:
    1. Calculate recent access frequency for each file (already in dataset).
    2. Rank files by recent_access_count descending.
    3. Files with recent_access_count >= threshold AND on HDD are candidates.
    4. Respect migration constraints (max_migrations, payback check).
    5. Estimate latency assuming migrated files serve from SSD.

    This threshold is the SAME as the ML system's HOTSPOT_THRESHOLD
    for fair comparison. This is a clearly documented, reproducible policy.

    Args:
        df: DataFrame with feature columns including recent_access_count,
            current_tier, file_size_bytes, and future_access_count.
        threshold: Access count threshold for frequency-based migration.
        max_migrations: Maximum number of migrations allowed.
        min_payback: Minimum payback ratio for migration approval.
        hdd_latency_us: Simulated HDD latency.
        ssd_latency_us: Simulated SSD latency.
        bandwidth: Simulated migration bandwidth (bytes/sec).

    Returns:
        Dictionary of baseline metrics.
    """
    df_eval = df.copy()

    # Identify migration candidates: on HDD + high recent frequency
    candidates = df_eval[
        (df_eval["current_tier"] == config.TIER_HDD_ENCODING)
        & (df_eval["recent_access_count"] >= threshold)
    ].copy()

    # Rank by recent_access_count descending
    candidates = candidates.sort_values("recent_access_count", ascending=False)

    # Apply payback check using recent_access_count as a proxy for expected benefit
    # (This is the frequency-based estimate, NOT an ML prediction)
    latency_saving_per_access_sec = (hdd_latency_us - ssd_latency_us) / 1_000_000.0

    migrated_file_ids = set()
    total_migration_time = 0.0
    total_migration_volume = 0
    migrations_performed = 0

    for _, row in candidates.iterrows():
        if migrations_performed >= max_migrations:
            break

        file_id = row["file_id"]
        pred_ts = row.get("prediction_timestamp", "")

        # Unique key: only migrate each file once per timestamp batch
        key = (file_id, pred_ts)
        if key in migrated_file_ids:
            continue

        file_size = float(row["file_size_bytes"])
        freq_accesses = float(row["recent_access_count"])

        # Simple payback: frequency_accesses × latency_saving / migration_cost
        mig_cost_sec = file_size / bandwidth if bandwidth > 0 else float("inf")
        expected_benefit_sec = freq_accesses * latency_saving_per_access_sec
        payback = expected_benefit_sec / mig_cost_sec if mig_cost_sec > 0 else 0.0

        if payback >= min_payback:
            migrated_file_ids.add(key)
            migrations_performed += 1
            total_migration_time += mig_cost_sec
            total_migration_volume += int(file_size)

    # Compute estimated latency with migrations applied
    # For migrated files on HDD, assume they are now on SSD
    df_eval["effective_tier"] = df_eval["current_tier"].copy()
    for fid, pts in migrated_file_ids:
        mask = (df_eval["file_id"] == fid) & (df_eval["prediction_timestamp"] == pts)
        df_eval.loc[mask, "effective_tier"] = config.TIER_SSD_ENCODING

    df_eval["access_latency_us"] = df_eval["effective_tier"].apply(
        lambda t: ssd_latency_us if t == config.TIER_SSD_ENCODING else hdd_latency_us
    )
    df_eval["total_latency_us"] = df_eval["future_access_count"] * df_eval["access_latency_us"]

    total_accesses = df_eval["future_access_count"].sum()
    total_latency = df_eval["total_latency_us"].sum()
    avg_latency_us = total_latency / total_accesses if total_accesses > 0 else 0.0

    # SSD utilization after migration
    ssd_count = (df_eval["effective_tier"] == config.TIER_SSD_ENCODING).sum()
    total_count = len(df_eval)
    ssd_utilization = ssd_count / total_count if total_count > 0 else 0.0

    # Evaluate prediction quality of frequency baseline
    # "Predicted hotspot" = recent_access_count >= threshold
    df_eval["freq_predicted_hotspot"] = (
        df_eval["recent_access_count"] >= threshold
    ).astype(int)
    df_eval["actual_hotspot"] = (
        df_eval["future_access_count"] >= threshold
    ).astype(int)

    tp = ((df_eval["freq_predicted_hotspot"] == 1) & (df_eval["actual_hotspot"] == 1)).sum()
    fp = ((df_eval["freq_predicted_hotspot"] == 1) & (df_eval["actual_hotspot"] == 0)).sum()
    fn = ((df_eval["freq_predicted_hotspot"] == 0) & (df_eval["actual_hotspot"] == 1)).sum()

    total_predicted_hot = tp + fp
    false_hotspot_rate = fp / total_predicted_hot if total_predicted_hot > 0 else None

    # Useful migrations: migrated files that were actually hotspots
    useful_count = 0
    unnecessary_count = 0
    for fid, pts in migrated_file_ids:
        mask = (df_eval["file_id"] == fid) & (df_eval["prediction_timestamp"] == pts)
        actual = df_eval.loc[mask, "actual_hotspot"].values
        if len(actual) > 0 and actual[0] == 1:
            useful_count += 1
        else:
            unnecessary_count += 1

    useful_migration_rate = (
        useful_count / migrations_performed if migrations_performed > 0 else None
    )

    return {
        "method": "Frequency-Based",
        "avg_latency_us": avg_latency_us,
        "avg_latency_ms": avg_latency_us / 1000.0,
        "total_latency_us": total_latency,
        "total_accesses": int(total_accesses),
        "migrations": migrations_performed,
        "migration_time_sec": total_migration_time,
        "migration_volume_bytes": total_migration_volume,
        "false_hotspot_rate": false_hotspot_rate,
        "useful_migration_rate": useful_migration_rate,
        "ssd_utilization": ssd_utilization,
        "missed_hotspots": int(fn),
        "unnecessary_migrations": unnecessary_count,
    }


# ============================================================
# BASELINE 3 — RULE-BASED HOTSPOT (No Payback Filter)
# ============================================================

def run_rule_based_baseline(
    df: pd.DataFrame,
    threshold: int = config.HOTSPOT_THRESHOLD,
    max_migrations: int = config.MAX_MIGRATIONS_PER_RUN,
    hdd_latency_us: float = config.HDD_READ_LATENCY_US,
    ssd_latency_us: float = config.SSD_READ_LATENCY_US,
    bandwidth: float = config.MIGRATION_BANDWIDTH_BYTES_PER_SEC,
) -> Dict:
    """
    Rule-Based Hotspot Baseline.

    Simple rule: if recent_access_count >= threshold AND file is on HDD,
    migrate it. No payback filtering.

    This is Baseline 2 WITHOUT the payback constraint, to isolate the
    effect of payback filtering.

    Args:
        df: DataFrame with feature columns.
        threshold: Access count threshold.
        max_migrations: Maximum migrations.
        hdd_latency_us: Simulated HDD latency.
        ssd_latency_us: Simulated SSD latency.
        bandwidth: Simulated migration bandwidth.

    Returns:
        Dictionary of baseline metrics.
    """
    # Re-use frequency baseline with min_payback=0 (no payback filter)
    result = run_frequency_baseline(
        df,
        threshold=threshold,
        max_migrations=max_migrations,
        min_payback=0.0,  # No payback filter
        hdd_latency_us=hdd_latency_us,
        ssd_latency_us=ssd_latency_us,
        bandwidth=bandwidth,
    )
    result["method"] = "Rule-Based Hotspot"
    return result


# ============================================================
# Standalone test / demonstration
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Phase 17 — Baseline Systems Test")
    print("=" * 70)

    df_test = pd.read_csv(config.TEST_DATASET_PATH)
    print(f"Loaded {len(df_test)} test rows across "
          f"{df_test['prediction_timestamp'].nunique()} timestamps.")

    print("\n--- Baseline 1: Static Tiering ---")
    static = run_static_tiering_baseline(df_test)
    for k, v in static.items():
        print(f"  {k:<30}: {v}")

    print("\n--- Baseline 2: Frequency-Based ---")
    freq = run_frequency_baseline(df_test, min_payback=0.05)
    for k, v in freq.items():
        print(f"  {k:<30}: {v}")

    print("\n--- Baseline 3: Rule-Based Hotspot ---")
    rule = run_rule_based_baseline(df_test)
    for k, v in rule.items():
        print(f"  {k:<30}: {v}")

    print("\n[OK] All baselines operational.")
