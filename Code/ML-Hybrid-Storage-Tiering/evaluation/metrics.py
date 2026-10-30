"""
Evaluation Metrics Module (Phase 18) for ML-Driven Hybrid Storage Staging.

Provides all evaluation metric calculations for:
- ML prediction quality (MAE, RMSE, R², classification metrics)
- Storage/tiering performance (latency, utilisation, migration stats)
- Payback analysis
- Improvement calculations
- Comparison table generation

All functions handle zero denominators safely.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


# ============================================================
# ML PREDICTION METRICS (Part 18.1)
# ============================================================

def compute_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(actual - predicted)))


def compute_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def compute_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Coefficient of Determination (R²)."""
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def compute_classification_metrics(
    actual_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> Dict:
    """
    Compute hotspot classification metrics.

    Returns:
        Dict with accuracy, precision, recall, f1, confusion_matrix.
    """
    tp = int(((predicted_labels == 1) & (actual_labels == 1)).sum())
    tn = int(((predicted_labels == 0) & (actual_labels == 0)).sum())
    fp = int(((predicted_labels == 1) & (actual_labels == 0)).sum())
    fn = int(((predicted_labels == 0) & (actual_labels == 1)).sum())

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def compute_ml_metrics(
    df: pd.DataFrame,
    actual_col: str = "future_access_count",
    predicted_col: str = "predicted_future_access_count",
    threshold: int = config.HOTSPOT_THRESHOLD,
) -> Dict:
    """
    Compute all ML prediction metrics from a DataFrame.

    Args:
        df: DataFrame with actual and predicted columns.
        actual_col: Column name for actual values.
        predicted_col: Column name for predicted values.
        threshold: Hotspot threshold for classification metrics.

    Returns:
        Dict with regression and classification metrics.
    """
    actual = df[actual_col].values.astype(float)
    predicted = df[predicted_col].values.astype(float)

    regression = {
        "mae": compute_mae(actual, predicted),
        "rmse": compute_rmse(actual, predicted),
        "r2": compute_r2(actual, predicted),
    }

    actual_labels = (actual >= threshold).astype(int)
    predicted_labels = (predicted >= threshold).astype(int)
    classification = compute_classification_metrics(actual_labels, predicted_labels)

    return {**regression, **classification}


# ============================================================
# STORAGE/TIERING METRICS (Part 18.2)
# ============================================================

def compute_estimated_avg_latency(
    df: pd.DataFrame,
    tier_col: str = "current_tier",
    access_col: str = "future_access_count",
    hdd_latency_us: float = config.HDD_READ_LATENCY_US,
    ssd_latency_us: float = config.SSD_READ_LATENCY_US,
) -> float:
    """
    Compute average estimated access latency.

    Returns average latency in microseconds per access.
    """
    latencies = df[tier_col].apply(
        lambda t: ssd_latency_us if t == config.TIER_SSD_ENCODING else hdd_latency_us
    )
    total_latency = (df[access_col] * latencies).sum()
    total_accesses = df[access_col].sum()
    if total_accesses == 0:
        return 0.0
    return float(total_latency / total_accesses)


def compute_ssd_utilization(df: pd.DataFrame, tier_col: str = "current_tier") -> float:
    """Compute SSD utilization as fraction of files on SSD."""
    total = len(df)
    if total == 0:
        return 0.0
    ssd_count = (df[tier_col] == config.TIER_SSD_ENCODING).sum()
    return float(ssd_count / total)


def compute_hdd_utilization(df: pd.DataFrame, tier_col: str = "current_tier") -> float:
    """Compute HDD utilization as fraction of files on HDD."""
    total = len(df)
    if total == 0:
        return 0.0
    hdd_count = (df[tier_col] == config.TIER_HDD_ENCODING).sum()
    return float(hdd_count / total)


# ============================================================
# MIGRATION METRICS (Part 18.4)
# ============================================================

def compute_migration_metrics(migration_log_df: pd.DataFrame) -> Dict:
    """
    Compute migration metrics from the migration log.

    Returns dict with migration count, volume, overhead, success rate, etc.
    """
    if migration_log_df is None or len(migration_log_df) == 0:
        return {
            "total_migrations": 0,
            "successful_migrations": 0,
            "failed_migrations": 0,
            "migration_success_rate": None,
            "total_migration_time_sec": 0.0,
            "avg_migration_time_sec": 0.0,
            "total_data_migrated_bytes": 0,
        }

    total = len(migration_log_df)
    successes = migration_log_df[migration_log_df["migration_status"] == "SUCCESS"]
    successful = len(successes)
    failed = total - successful

    success_rate = successful / total if total > 0 else None

    # Use actual migration duration, not theoretical cost
    duration_col = "actual_migration_duration_seconds"
    if duration_col in migration_log_df.columns:
        total_time = pd.to_numeric(migration_log_df[duration_col], errors="coerce").sum()
        avg_time = total_time / total if total > 0 else 0.0
    else:
        total_time = 0.0
        avg_time = 0.0

    size_col = "logical_file_size_bytes"
    if size_col in migration_log_df.columns:
        total_volume = pd.to_numeric(migration_log_df[size_col], errors="coerce").sum()
    else:
        total_volume = 0

    return {
        "total_migrations": total,
        "successful_migrations": successful,
        "failed_migrations": failed,
        "migration_success_rate": success_rate,
        "total_migration_time_sec": float(total_time),
        "avg_migration_time_sec": float(avg_time),
        "total_data_migrated_bytes": int(total_volume),
    }


# ============================================================
# DECISION QUALITY METRICS
# ============================================================

def compute_decision_quality_metrics(feedback_df: pd.DataFrame) -> Dict:
    """
    Compute decision quality metrics from feedback-enriched data.
    """
    if len(feedback_df) == 0:
        return {}

    from feedback.migration_feedback import (
        PRED_TRUE_POSITIVE, PRED_FALSE_POSITIVE,
        PRED_FALSE_NEGATIVE, PRED_TRUE_NEGATIVE,
        OUTCOME_SUCCESSFUL_USEFUL, OUTCOME_FALSE_HOTSPOT_MIG,
        OUTCOME_MISSED_HOTSPOT, OUTCOME_CORRECT_NON_MIG,
    )

    outcomes = feedback_df["prediction_outcome"].value_counts().to_dict()
    tp = outcomes.get(PRED_TRUE_POSITIVE, 0)
    fp = outcomes.get(PRED_FALSE_POSITIVE, 0)
    fn = outcomes.get(PRED_FALSE_NEGATIVE, 0)
    tn = outcomes.get(PRED_TRUE_NEGATIVE, 0)

    total_predicted_hot = tp + fp
    total_actual_hot = tp + fn

    false_hotspot_rate = fp / total_predicted_hot if total_predicted_hot > 0 else None
    missed_hotspot_rate = fn / total_actual_hot if total_actual_hot > 0 else None

    # Migration-specific
    mig_outcomes = feedback_df.get("migration_outcome", pd.Series())
    if len(mig_outcomes) > 0:
        mig_counts = mig_outcomes.value_counts().to_dict()
        useful = mig_counts.get(OUTCOME_SUCCESSFUL_USEFUL, 0)
        false_mig = mig_counts.get(OUTCOME_FALSE_HOTSPOT_MIG, 0)
        total_migrated = useful + false_mig
        useful_rate = useful / total_migrated if total_migrated > 0 else None
    else:
        useful = 0
        false_mig = 0
        total_migrated = 0
        useful_rate = None

    return {
        "false_hotspot_rate": false_hotspot_rate,
        "missed_hotspot_rate": missed_hotspot_rate,
        "useful_migration_rate": useful_rate,
        "useful_migrations": useful,
        "unnecessary_migrations": false_mig,
        "total_migrated": total_migrated,
        "total_predicted_hotspots": total_predicted_hot,
        "total_actual_hotspots": total_actual_hot,
    }


# ============================================================
# PAYBACK METRICS (Part 18.5)
# ============================================================

def compute_payback_metrics(
    df: pd.DataFrame,
    payback_col: str = "payback_ratio",
    min_payback: float = config.MIN_PAYBACK_RATIO,
) -> Dict:
    """
    Compute payback analysis metrics.
    """
    if payback_col not in df.columns or len(df) == 0:
        return {}

    paybacks = df[payback_col].replace([np.inf, -np.inf], np.nan).dropna()

    if len(paybacks) == 0:
        return {
            "avg_payback": None,
            "median_payback": None,
            "min_payback": None,
            "max_payback": None,
            "above_threshold": 0,
            "below_threshold": 0,
        }

    above = int((paybacks >= min_payback).sum())
    below = int((paybacks < min_payback).sum())

    return {
        "avg_payback": float(paybacks.mean()),
        "median_payback": float(paybacks.median()),
        "min_payback": float(paybacks.min()),
        "max_payback": float(paybacks.max()),
        "above_threshold": above,
        "below_threshold": below,
    }


# ============================================================
# IMPROVEMENT CALCULATIONS (Part 18.7)
# ============================================================

def compute_latency_improvement(
    baseline_latency: float,
    proposed_latency: float,
) -> Optional[float]:
    """
    Latency improvement percentage.

    Returns None if baseline_latency is zero.
    """
    if baseline_latency == 0:
        return None
    return ((baseline_latency - proposed_latency) / baseline_latency) * 100.0


def compute_migration_reduction(
    baseline_migrations: int,
    proposed_migrations: int,
) -> Optional[float]:
    """
    Migration reduction percentage.

    Returns None if baseline_migrations is zero.
    """
    if baseline_migrations == 0:
        return None
    return ((baseline_migrations - proposed_migrations) / baseline_migrations) * 100.0


# ============================================================
# COMPARISON TABLE (Part 18.6)
# ============================================================

def build_comparison_table(results: List[Dict]) -> pd.DataFrame:
    """
    Build a comparison table from a list of method result dictionaries.

    Each dict should have:
        method, avg_latency_ms, migrations, migration_time_sec,
        false_hotspot_rate, useful_migration_rate, ssd_utilization
    """
    rows = []
    for r in results:
        rows.append({
            "Method": r.get("method", "Unknown"),
            "Avg Latency (ms)": round(r.get("avg_latency_ms", 0), 4),
            "Migrations": r.get("migrations", 0),
            "Migration Time (s)": round(r.get("migration_time_sec", 0), 4),
            "False Hotspot Rate": _format_rate(r.get("false_hotspot_rate")),
            "Useful Migration Rate": _format_rate(r.get("useful_migration_rate")),
            "SSD Utilization": round(r.get("ssd_utilization", 0), 4),
        })
    return pd.DataFrame(rows)


def _format_rate(val) -> str:
    """Format a rate value, handling None as 'N/A'."""
    if val is None:
        return "N/A"
    return f"{val:.4f}"
