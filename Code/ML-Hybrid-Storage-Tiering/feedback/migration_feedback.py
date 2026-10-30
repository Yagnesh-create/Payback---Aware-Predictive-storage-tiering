"""
Migration Feedback Module (Phase 16) for ML-Driven Hybrid Storage Staging.

Closes the prediction–migration loop:

    Prediction → Migration → Observe actual behaviour → Compare →
    Feedback signal → Bounded adjustment → Update heat score

IMPORTANT:
    This module does NOT retrain the ML model (hotspot_predictor.pkl).
    It only adjusts the heat-map / decision state via an adaptive
    heuristic feedback mechanism.

The feedback mechanism is an adaptive heuristic, NOT a mathematically
proven optimal learning algorithm. Its effectiveness is evaluated
experimentally in Phase 18.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math
import uuid
import datetime

import numpy as np
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


# ============================================================
# Migration-Specific Outcome Constants
# ============================================================
OUTCOME_SUCCESSFUL_USEFUL = "SUCCESSFUL_USEFUL_MIGRATION"
OUTCOME_FALSE_HOTSPOT_MIG = "FALSE_HOTSPOT_MIGRATION"
OUTCOME_MISSED_HOTSPOT = "MISSED_HOTSPOT"
OUTCOME_CORRECT_NON_MIG = "CORRECT_NON_MIGRATION"

# Prediction Classification Constants
PRED_TRUE_POSITIVE = "TRUE_POSITIVE"
PRED_TRUE_NEGATIVE = "TRUE_NEGATIVE"
PRED_FALSE_POSITIVE = "FALSE_POSITIVE"
PRED_FALSE_NEGATIVE = "FALSE_NEGATIVE"

# Feedback Log Schema
FEEDBACK_LOG_COLUMNS = [
    "feedback_id",
    "timestamp",
    "file_id",
    "prediction_timestamp",
    "predicted_future_access_count",
    "actual_future_access_count",
    "prediction_error",
    "absolute_prediction_error",
    "predicted_hotspot",
    "actual_hotspot",
    "prediction_outcome",
    "migration_status",
    "migration_was_useful",
    "old_heat_score",
    "feedback_signal",
    "bounded_feedback",
    "updated_heat_score",
]


# ============================================================
# PART 16.2 — Observe Actual Future Access
# ============================================================

def count_actual_future_accesses(
    trace_df: pd.DataFrame,
    file_id: str,
    prediction_timestamp: str,
    future_window_hours: float = config.FUTURE_WINDOW_HOURS,
) -> int:
    """
    Count actual accesses for a file in the future window after
    the prediction timestamp, using the raw access trace.

    This derives the ground-truth access count from the raw trace
    data itself, NOT from the ML prediction or pre-computed labels.

    Args:
        trace_df: Raw storage access trace DataFrame with 'timestamp'
                  and 'file_id' columns.
        file_id: The file to count accesses for.
        prediction_timestamp: The prediction decision point (string).
        future_window_hours: Duration of the future window in hours.

    Returns:
        Number of actual accesses in the future window.
    """
    pred_ts = pd.Timestamp(prediction_timestamp)
    window_end = pred_ts + pd.Timedelta(hours=future_window_hours)

    # Filter trace for this file in the future window
    mask = (
        (trace_df["file_id"] == file_id)
        & (trace_df["_parsed_ts"] > pred_ts)
        & (trace_df["_parsed_ts"] <= window_end)
    )
    return int(mask.sum())


def count_actual_future_accesses_batch(
    trace_df: pd.DataFrame,
    records_df: pd.DataFrame,
    future_window_hours: float = config.FUTURE_WINDOW_HOURS,
) -> pd.Series:
    """
    Count actual future accesses for a batch of (file_id, prediction_timestamp)
    pairs using the raw access trace.

    This is the vectorised batch version. Pre-parses trace timestamps once.

    Args:
        trace_df: Raw storage access trace with 'timestamp' and 'file_id'.
        records_df: DataFrame with 'file_id' and 'prediction_timestamp' columns.
        future_window_hours: Duration of the future window in hours.

    Returns:
        pd.Series of integer access counts aligned with records_df index.
    """
    # Ensure trace timestamps are parsed
    if "_parsed_ts" not in trace_df.columns:
        trace_df = trace_df.copy()
        trace_df["_parsed_ts"] = pd.to_datetime(trace_df["timestamp"])

    counts = []
    for _, row in records_df.iterrows():
        cnt = count_actual_future_accesses(
            trace_df, row["file_id"], row["prediction_timestamp"],
            future_window_hours,
        )
        counts.append(cnt)

    return pd.Series(counts, index=records_df.index, name="actual_future_access_count")


# ============================================================
# PART 16.3 — Compare Prediction With Reality
# ============================================================

def compare_prediction_with_reality(
    predicted_access: float,
    actual_access: int,
    threshold: int = config.HOTSPOT_THRESHOLD,
) -> Dict:
    """
    Compare a single prediction against actual outcome.

    Returns dict with:
        predicted_future_access_count, actual_future_access_count,
        prediction_error, absolute_prediction_error,
        predicted_hotspot, actual_hotspot, prediction_outcome
    """
    prediction_error = actual_access - predicted_access
    absolute_error = abs(prediction_error)

    predicted_hotspot = int(predicted_access >= threshold)
    actual_hotspot = int(actual_access >= threshold)

    # Classify prediction outcome (Part 16.4)
    if predicted_hotspot == 1 and actual_hotspot == 1:
        outcome = PRED_TRUE_POSITIVE
    elif predicted_hotspot == 0 and actual_hotspot == 0:
        outcome = PRED_TRUE_NEGATIVE
    elif predicted_hotspot == 1 and actual_hotspot == 0:
        outcome = PRED_FALSE_POSITIVE
    else:  # predicted_hotspot == 0 and actual_hotspot == 1
        outcome = PRED_FALSE_NEGATIVE

    return {
        "predicted_future_access_count": predicted_access,
        "actual_future_access_count": actual_access,
        "prediction_error": prediction_error,
        "absolute_prediction_error": absolute_error,
        "predicted_hotspot": predicted_hotspot,
        "actual_hotspot": actual_hotspot,
        "prediction_outcome": outcome,
    }


# ============================================================
# PART 16.5 — Migration-Specific Outcomes
# ============================================================

def classify_migration_outcome(
    was_migrated: bool,
    predicted_hotspot: int,
    actual_hotspot: int,
) -> str:
    """
    Classify the migration outcome into one of four categories:

    SUCCESSFUL_USEFUL_MIGRATION:
        Migration executed + predicted hot + actually hot.
    FALSE_HOTSPOT_MIGRATION:
        Migration executed + predicted hot + NOT actually hot.
    MISSED_HOTSPOT:
        Not migrated + actually hot.
    CORRECT_NON_MIGRATION:
        Not migrated + NOT actually hot.
    """
    if was_migrated:
        if actual_hotspot == 1:
            return OUTCOME_SUCCESSFUL_USEFUL
        else:
            return OUTCOME_FALSE_HOTSPOT_MIG
    else:
        if actual_hotspot == 1:
            return OUTCOME_MISSED_HOTSPOT
        else:
            return OUTCOME_CORRECT_NON_MIG


# ============================================================
# PART 16.6 — False Hotspot Rate
# ============================================================

def compute_false_hotspot_rate(
    false_positive_count: int,
    total_predicted_hotspot: int,
) -> Optional[float]:
    """
    False Hotspot Rate = FP / Total Predicted Hotspots.

    Returns None ("N/A") if there were no predicted hotspots.
    """
    if total_predicted_hotspot == 0:
        return None
    return false_positive_count / total_predicted_hotspot


def compute_false_hotspot_migration_rate(
    migrated_not_hotspot: int,
    total_migrated: int,
) -> Optional[float]:
    """
    False Hotspot Migration Rate =
        Migrated files that did NOT become hotspots /
        Total successful hotspot migrations.

    Returns None ("N/A") if there were no migrations.
    """
    if total_migrated == 0:
        return None
    return migrated_not_hotspot / total_migrated


# ============================================================
# PART 16.7 — Feedback Adjustment
# ============================================================

def compute_bounded_feedback(
    feedback_signal: float,
    scale: float = config.FEEDBACK_SCALE,
) -> float:
    """
    Apply bounded transformation to the feedback signal using tanh.

    bounded_feedback = tanh(feedback_signal / scale)

    This prevents a single extreme workload event from completely
    dominating the heat score. The output is bounded to [-1, 1].
    """
    if scale <= 0:
        return 0.0
    return math.tanh(feedback_signal / scale)


def update_heat_score(
    old_heat: float,
    bounded_feedback: float,
    weight: float = config.FEEDBACK_WEIGHT,
) -> float:
    """
    Update heat score with bounded feedback.

    updated_heat = old_heat + weight * bounded_feedback

    The result is clipped to [0, 1] to maintain valid heat score bounds.
    """
    new_heat = old_heat + weight * bounded_feedback
    return float(np.clip(new_heat, 0.0, 1.0))


# ============================================================
# PART 16.10 — Feedback Logging
# ============================================================

def _initialize_feedback_log(log_path: Path):
    """Create the feedback log file with headers if it doesn't exist."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        df_empty = pd.DataFrame(columns=FEEDBACK_LOG_COLUMNS)
        df_empty.to_csv(log_path, index=False)


def log_feedback_record(record: Dict, log_path: Path = config.FEEDBACK_LOG_PATH) -> str:
    """
    Append a single feedback record to the feedback log CSV.

    Returns the feedback_id.
    """
    _initialize_feedback_log(log_path)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    feedback_id = f"FB_{ts}_{short_uuid}"

    record["feedback_id"] = feedback_id
    record["timestamp"] = datetime.datetime.now().isoformat()

    df_record = pd.DataFrame([record], columns=FEEDBACK_LOG_COLUMNS)
    df_record.to_csv(log_path, mode="a", header=False, index=False)

    return feedback_id


# ============================================================
# MAIN — Run Feedback Loop
# ============================================================

def run_feedback_loop(
    decisions_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    threshold: int = config.HOTSPOT_THRESHOLD,
    feedback_weight: float = config.FEEDBACK_WEIGHT,
    feedback_scale: float = config.FEEDBACK_SCALE,
    log_path: Path = config.FEEDBACK_LOG_PATH,
) -> pd.DataFrame:
    """
    Run the complete feedback loop for all decision records.

    For each file/prediction record:
    1. Count actual future accesses from the raw trace.
    2. Compare prediction vs reality.
    3. Classify prediction outcome (TP/TN/FP/FN).
    4. Classify migration outcome (useful/false hotspot/missed/correct).
    5. Compute bounded feedback signal.
    6. Update heat score.
    7. Log the feedback.

    Args:
        decisions_df: DataFrame from the decision pipeline containing
            file_id, prediction_timestamp, predicted_future_access_count,
            predicted_hotspot, heat_score, decision, etc.
        trace_df: Raw storage access trace DataFrame.
        threshold: Hotspot access threshold.
        feedback_weight: Weight for heat score adjustment.
        feedback_scale: Scale for tanh bounding.
        log_path: Path to write the feedback log CSV.

    Returns:
        DataFrame with all feedback columns added, including
        updated_heat_score.

    IMPORTANT: This function does NOT retrain the ML model.
    """
    # Pre-parse trace timestamps for efficiency
    trace = trace_df.copy()
    if "_parsed_ts" not in trace.columns:
        trace["_parsed_ts"] = pd.to_datetime(trace["timestamp"])

    # Count actual future accesses for all records
    actual_counts = count_actual_future_accesses_batch(
        trace, decisions_df, config.FUTURE_WINDOW_HOURS,
    )

    result = decisions_df.copy()
    result["actual_future_access_count"] = actual_counts.values

    # Lists for batch processing
    prediction_errors = []
    absolute_errors = []
    actual_hotspots = []
    prediction_outcomes = []
    migration_outcomes = []
    feedback_signals = []
    bounded_feedbacks = []
    old_heat_scores = []
    updated_heat_scores = []

    for idx, row in result.iterrows():
        predicted = float(row["predicted_future_access_count"])
        actual = int(row["actual_future_access_count"])
        old_heat = float(row.get("heat_score", 0.5))

        # Compare prediction with reality
        comparison = compare_prediction_with_reality(predicted, actual, threshold)

        prediction_errors.append(comparison["prediction_error"])
        absolute_errors.append(comparison["absolute_prediction_error"])
        actual_hotspots.append(comparison["actual_hotspot"])
        prediction_outcomes.append(comparison["prediction_outcome"])

        # Classify migration outcome
        decision = str(row.get("decision", ""))
        was_migrated = decision == "MIGRATE_TO_SSD"

        mig_outcome = classify_migration_outcome(
            was_migrated, comparison["predicted_hotspot"], comparison["actual_hotspot"],
        )
        migration_outcomes.append(mig_outcome)

        # Compute feedback
        feedback_signal = comparison["prediction_error"]  # actual - predicted
        bounded_fb = compute_bounded_feedback(feedback_signal, feedback_scale)
        new_heat = update_heat_score(old_heat, bounded_fb, feedback_weight)

        feedback_signals.append(feedback_signal)
        bounded_feedbacks.append(bounded_fb)
        old_heat_scores.append(old_heat)
        updated_heat_scores.append(new_heat)

        # Log the feedback record
        log_record = {
            "file_id": row["file_id"],
            "prediction_timestamp": row["prediction_timestamp"],
            "predicted_future_access_count": predicted,
            "actual_future_access_count": actual,
            "prediction_error": comparison["prediction_error"],
            "absolute_prediction_error": comparison["absolute_prediction_error"],
            "predicted_hotspot": comparison["predicted_hotspot"],
            "actual_hotspot": comparison["actual_hotspot"],
            "prediction_outcome": comparison["prediction_outcome"],
            "migration_status": decision,
            "migration_was_useful": mig_outcome,
            "old_heat_score": old_heat,
            "feedback_signal": feedback_signal,
            "bounded_feedback": round(bounded_fb, 6),
            "updated_heat_score": round(new_heat, 6),
        }
        log_feedback_record(log_record, log_path)

    # Add all computed columns to the result DataFrame
    result["prediction_error"] = prediction_errors
    result["absolute_prediction_error"] = absolute_errors
    result["actual_hotspot"] = actual_hotspots
    result["prediction_outcome"] = prediction_outcomes
    result["migration_outcome"] = migration_outcomes
    result["feedback_signal"] = feedback_signals
    result["bounded_feedback"] = bounded_feedbacks
    result["old_heat_score"] = old_heat_scores
    result["updated_heat_score"] = updated_heat_scores

    return result


def compute_feedback_summary(feedback_df: pd.DataFrame) -> Dict:
    """
    Compute summary statistics from a feedback-enriched DataFrame.

    Returns a dictionary of aggregate metrics.
    """
    total = len(feedback_df)
    if total == 0:
        return {"total_records": 0}

    pred_outcomes = feedback_df["prediction_outcome"].value_counts().to_dict()
    mig_outcomes = feedback_df["migration_outcome"].value_counts().to_dict()

    tp = pred_outcomes.get(PRED_TRUE_POSITIVE, 0)
    tn = pred_outcomes.get(PRED_TRUE_NEGATIVE, 0)
    fp = pred_outcomes.get(PRED_FALSE_POSITIVE, 0)
    fn = pred_outcomes.get(PRED_FALSE_NEGATIVE, 0)

    total_predicted_hot = tp + fp
    total_migrated = mig_outcomes.get(OUTCOME_SUCCESSFUL_USEFUL, 0) + \
                     mig_outcomes.get(OUTCOME_FALSE_HOTSPOT_MIG, 0)

    false_hotspot_rate = compute_false_hotspot_rate(fp, total_predicted_hot)
    false_hotspot_mig_rate = compute_false_hotspot_migration_rate(
        mig_outcomes.get(OUTCOME_FALSE_HOTSPOT_MIG, 0), total_migrated,
    )

    summary = {
        "total_records": total,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "total_predicted_hotspots": total_predicted_hot,
        "total_actual_hotspots": tp + fn,
        "false_hotspot_rate": false_hotspot_rate,
        "false_hotspot_migration_rate": false_hotspot_mig_rate,
        "mean_prediction_error": float(feedback_df["prediction_error"].mean()),
        "mean_absolute_error": float(feedback_df["absolute_prediction_error"].mean()),
        "mean_old_heat": float(feedback_df["old_heat_score"].mean()),
        "mean_updated_heat": float(feedback_df["updated_heat_score"].mean()),
        "migration_outcomes": mig_outcomes,
        "useful_migrations": mig_outcomes.get(OUTCOME_SUCCESSFUL_USEFUL, 0),
        "false_hotspot_migrations": mig_outcomes.get(OUTCOME_FALSE_HOTSPOT_MIG, 0),
        "missed_hotspots": mig_outcomes.get(OUTCOME_MISSED_HOTSPOT, 0),
        "correct_non_migrations": mig_outcomes.get(OUTCOME_CORRECT_NON_MIG, 0),
    }
    return summary


# ============================================================
# Standalone test / demonstration
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Phase 16 — Migration Feedback Module Test")
    print("=" * 70)

    # Load test data and trace
    df_test = pd.read_csv(config.TEST_DATASET_PATH)
    df_trace = pd.read_csv(config.TRACE_DATASET_PATH)
    df_trace["_parsed_ts"] = pd.to_datetime(df_trace["timestamp"])

    # Run decision pipeline on a single timestamp
    from migration.decision_engine import run_decision_pipeline
    last_ts = sorted(df_test["prediction_timestamp"].unique())[-1]
    df_slice = df_test[df_test["prediction_timestamp"] == last_ts].copy()

    print(f"\nPrediction timestamp: {last_ts}")
    print(f"Files: {len(df_slice)}")

    df_decisions = run_decision_pipeline(df_slice, min_payback=0.05)

    # Run feedback
    import tempfile
    tmp_log = Path(tempfile.mktemp(suffix="_feedback_test.csv"))
    feedback_result = run_feedback_loop(
        df_decisions, df_trace, log_path=tmp_log,
    )

    summary = compute_feedback_summary(feedback_result)
    print("\n--- Feedback Summary ---")
    for k, v in summary.items():
        if k != "migration_outcomes":
            display_v = f"{v:.4f}" if isinstance(v, float) and v is not None else str(v)
            print(f"  {k:<35}: {display_v}")

    # Show a few sample rows
    print("\n--- Sample Feedback Records ---")
    cols = ["file_id", "predicted_future_access_count",
            "actual_future_access_count", "prediction_outcome",
            "old_heat_score", "bounded_feedback", "updated_heat_score"]
    sample = feedback_result.head(5)
    avail = [c for c in cols if c in sample.columns]
    print(sample[avail].to_string(index=False))

    # Clean up
    if tmp_log.exists():
        import os
        os.remove(tmp_log)

    print("\n[OK] Feedback module operational.")
