"""
Migration Decision Engine (Phase 9) for ML-Driven Hybrid Storage Staging.

Combines ML predictions, heat scores, and payback analysis to produce
a recommendation for each file:

    MIGRATE_TO_SSD   – Predicted hotspot on HDD with sufficient payback
    ALREADY_ON_SSD   – Already on the faster tier
    NOT_HOT          – Not predicted as a hotspot
    KEEP_CURRENT_TIER – Hotspot on HDD but payback is insufficient

No actual file movement occurs at this stage. The engine only produces
recommendations.
"""

from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from ml.heatmap import compute_heat_scores
from ml.predictor import HotspotPredictor
from migration.payback import compute_payback_for_batch


# Decision constants
DECISION_MIGRATE = "MIGRATE_TO_SSD"
DECISION_ALREADY_SSD = "ALREADY_ON_SSD"
DECISION_NOT_HOT = "NOT_HOT"
DECISION_KEEP = "KEEP_CURRENT_TIER"


def make_migration_decisions(
    df: pd.DataFrame,
    threshold: int = config.HOTSPOT_THRESHOLD,
    min_payback: float = config.MIN_PAYBACK_RATIO,
) -> pd.DataFrame:
    """
    Apply the migration decision policy to a DataFrame that already contains
    predicted_hotspot, current_tier, and payback_ratio columns.

    Decision rules:
        1. current_tier == SSD            → ALREADY_ON_SSD
        2. predicted_hotspot == 0         → NOT_HOT
        3. payback_ratio >= min_payback   → MIGRATE_TO_SSD
        4. otherwise                      → KEEP_CURRENT_TIER

    Returns:
        DataFrame with added columns: decision, decision_reason
    """
    result = df.copy()
    decisions = []
    reasons = []

    for _, row in result.iterrows():
        tier = int(row["current_tier"])
        is_hot = int(row["predicted_hotspot"])
        payback = float(row["payback_ratio"])
        pred_acc = float(row["predicted_future_access_count"])

        if tier == config.TIER_SSD_ENCODING:
            decisions.append(DECISION_ALREADY_SSD)
            reasons.append("File is already on SSD tier.")
        elif is_hot == 0:
            decisions.append(DECISION_NOT_HOT)
            reasons.append(
                f"Predicted future accesses ({pred_acc:.1f}) below "
                f"hotspot threshold ({threshold})."
            )
        elif payback >= min_payback:
            decisions.append(DECISION_MIGRATE)
            reasons.append(
                f"Predicted hotspot ({pred_acc:.1f} accesses) and "
                f"payback ratio ({payback:.4f}) >= minimum ({min_payback})."
            )
        else:
            decisions.append(DECISION_KEEP)
            reasons.append(
                f"Predicted hotspot ({pred_acc:.1f} accesses) but "
                f"payback ratio ({payback:.4f}) < minimum ({min_payback}). "
                f"Migration cost exceeds expected benefit."
            )

    result["decision"] = decisions
    result["decision_reason"] = reasons
    return result


def run_decision_pipeline(
    df: pd.DataFrame,
    predictor: HotspotPredictor = None,
    threshold: int = config.HOTSPOT_THRESHOLD,
    min_payback: float = config.MIN_PAYBACK_RATIO,
    alpha: float = config.HEAT_FREQUENCY_WEIGHT,
    beta: float = config.HEAT_RECENCY_WEIGHT,
) -> pd.DataFrame:
    """
    Execute the full integrated pipeline: heat → predict → payback → decide.

    Args:
        df: DataFrame with feature columns (from ml_training_dataset or test.csv).
        predictor: Optional pre-loaded HotspotPredictor. If None, loads from disk.
        threshold: Hotspot access threshold.
        min_payback: Minimum payback ratio for migration approval.
        alpha: Heat frequency weight.
        beta: Heat recency weight.

    Returns:
        DataFrame with all computed columns and the final decision.
    """
    # Step 1: Heat scores
    df_h = compute_heat_scores(df, alpha=alpha, beta=beta)

    # Step 2: ML predictions
    if predictor is None:
        predictor = HotspotPredictor(threshold=threshold)
    df_p = predictor.predict_batch(df_h)

    # Step 3: Payback analysis
    df_pb = compute_payback_for_batch(df_p)

    # Step 4: Migration decisions
    df_d = make_migration_decisions(df_pb, threshold=threshold, min_payback=min_payback)

    return df_d


def generate_decision_summary(df_decided: pd.DataFrame) -> dict:
    """Summarize the migration decision counts."""
    counts = df_decided["decision"].value_counts().to_dict()
    total = len(df_decided)
    return {
        "total_files": total,
        "MIGRATE_TO_SSD": counts.get(DECISION_MIGRATE, 0),
        "ALREADY_ON_SSD": counts.get(DECISION_ALREADY_SSD, 0),
        "NOT_HOT": counts.get(DECISION_NOT_HOT, 0),
        "KEEP_CURRENT_TIER": counts.get(DECISION_KEEP, 0),
    }


DECISION_TABLE_COLUMNS = [
    "file_id",
    "prediction_timestamp",
    "current_tier",
    "heat_score",
    "predicted_future_access_count",
    "predicted_hotspot",
    "migration_cost_sec",
    "expected_benefit_sec",
    "payback_ratio",
    "decision",
    "decision_reason",
]


if __name__ == "__main__":
    print("Testing Migration Decision Engine...")
    df_test = pd.read_csv(config.TEST_DATASET_PATH)

    # Use last prediction timestamp
    last_ts = sorted(df_test["prediction_timestamp"].unique())[-1]
    df_slice = df_test[df_test["prediction_timestamp"] == last_ts].copy()

    result = run_decision_pipeline(df_slice)
    summary = generate_decision_summary(result)

    print(f"\nPrediction timestamp: {last_ts}")
    print(f"Total files evaluated: {summary['total_files']}")
    for k, v in summary.items():
        if k != "total_files":
            print(f"  {k:<20}: {v}")

    # Show files recommended for migration
    migrants = result[result["decision"] == DECISION_MIGRATE]
    avail_cols = [c for c in DECISION_TABLE_COLUMNS if c in result.columns]
    if len(migrants) > 0:
        print(f"\nFiles recommended for migration ({len(migrants)}):")
        print(migrants[avail_cols].head(10).to_string(index=False))
    else:
        print("\nNo files recommended for migration at this timestamp.")
