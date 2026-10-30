"""
Heat-Map Engine (Phase 7) for ML-Driven Hybrid Storage Staging and Hotspot Migration.

Computes a composite heat score for each file at each prediction point:

    H = alpha * frequency_score + beta * recency_score

where:
    frequency_score : min-max normalized recent access count  [0, 1]
    recency_score   : inverted & min-max normalized recency   [0, 1]
                      (higher = more recent, lower = older)
    alpha           : config.HEAT_FREQUENCY_WEIGHT (default 0.7)
    beta            : config.HEAT_RECENCY_WEIGHT   (default 0.3)
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def _min_max_normalize(series: pd.Series) -> pd.Series:
    """
    Min-max normalize a pandas Series to the [0, 1] range.
    If all values are identical (max == min), returns 0.0 for all entries.
    """
    s_min = series.min()
    s_max = series.max()
    if s_max == s_min:
        return pd.Series(0.0, index=series.index)
    return (series - s_min) / (s_max - s_min)


def compute_frequency_score(recent_access_count: pd.Series) -> pd.Series:
    """
    Compute frequency score via min-max normalization of recent_access_count.

    Higher recent_access_count  →  higher frequency_score  →  hotter file.
    Range: [0.0, 1.0]
    """
    return _min_max_normalize(recent_access_count)


def compute_recency_score(recency_seconds: pd.Series) -> pd.Series:
    """
    Compute recency score from recency_seconds.

    Raw recency_seconds semantics:
        smaller value = more recently accessed
        larger value  = accessed longer ago

    We INVERT the normalization so that:
        recently accessed (small recency_seconds) → HIGH recency_score
        long-ago accessed (large recency_seconds) → LOW recency_score

    Method:
        recency_score = 1.0 - min_max_normalize(recency_seconds)

    Range: [0.0, 1.0]
    """
    return 1.0 - _min_max_normalize(recency_seconds)


def compute_heat_scores(
    df: pd.DataFrame,
    alpha: float = config.HEAT_FREQUENCY_WEIGHT,
    beta: float = config.HEAT_RECENCY_WEIGHT,
) -> pd.DataFrame:
    """
    Compute heat scores for all file–window rows in the DataFrame.

    Args:
        df: DataFrame with at least 'recent_access_count' and 'recency_seconds'.
        alpha: Weight for frequency component (default from config).
        beta: Weight for recency component (default from config).

    Returns:
        Copy of df with added columns:
            frequency_score, recency_score, heat_score
    """
    result = df.copy()

    result["frequency_score"] = compute_frequency_score(result["recent_access_count"])
    result["recency_score"] = compute_recency_score(result["recency_seconds"])
    result["heat_score"] = (
        alpha * result["frequency_score"] + beta * result["recency_score"]
    ).round(4)

    return result


def generate_heat_table(
    df: pd.DataFrame,
    alpha: float = config.HEAT_FREQUENCY_WEIGHT,
    beta: float = config.HEAT_RECENCY_WEIGHT,
) -> pd.DataFrame:
    """
    Generate a sorted heat-map table for reporting.

    Returns a DataFrame sorted by heat_score descending with columns:
        file_id, prediction_timestamp, frequency_score, recency_score,
        heat_score, current_tier
    """
    scored = compute_heat_scores(df, alpha, beta)
    cols = [
        "file_id",
        "prediction_timestamp",
        "frequency_score",
        "recency_score",
        "heat_score",
        "current_tier",
    ]
    available_cols = [c for c in cols if c in scored.columns]
    return scored[available_cols].sort_values("heat_score", ascending=False).reset_index(drop=True)


def validate_heat_scores(df_scored: pd.DataFrame) -> dict:
    """
    Run sanity checks on computed heat scores.

    Returns dict of check_name -> passed (bool).
    """
    checks = {}
    checks["freq_score_min_ge_0"] = bool((df_scored["frequency_score"] >= 0.0).all())
    checks["freq_score_max_le_1"] = bool((df_scored["frequency_score"] <= 1.0).all())
    checks["rec_score_min_ge_0"] = bool((df_scored["recency_score"] >= 0.0).all())
    checks["rec_score_max_le_1"] = bool((df_scored["recency_score"] <= 1.0).all())
    checks["heat_score_min_ge_0"] = bool((df_scored["heat_score"] >= 0.0).all())
    checks["heat_score_max_le_1"] = bool((df_scored["heat_score"] <= 1.0).all())

    # Directional checks: highest recent_access_count file should have freq_score=1.0
    max_freq_idx = df_scored["recent_access_count"].idxmax()
    checks["max_freq_has_score_1"] = bool(
        np.isclose(df_scored.loc[max_freq_idx, "frequency_score"], 1.0)
    )

    # Smallest recency_seconds (most recent) should have highest recency_score
    min_rec_idx = df_scored["recency_seconds"].idxmin()
    checks["most_recent_has_high_rec_score"] = bool(
        df_scored.loc[min_rec_idx, "recency_score"] >= 0.9
    )

    return checks


if __name__ == "__main__":
    print("Testing Heat-Map Engine...")

    df_test = pd.read_csv(config.TEST_DATASET_PATH)

    # Pick last prediction timestamp for a focused view
    last_ts = sorted(df_test["prediction_timestamp"].unique())[-1]
    df_slice = df_test[df_test["prediction_timestamp"] == last_ts].copy()

    scored = compute_heat_scores(df_slice)
    checks = validate_heat_scores(scored)

    print(f"\nPrediction timestamp: {last_ts}")
    print(f"Files scored: {len(scored)}")
    print("\nValidation checks:")
    for name, passed in checks.items():
        print(f"  {name:<35}: {'PASSED' if passed else 'FAILED'}")

    top = scored.nlargest(5, "heat_score")
    bottom = scored.nsmallest(5, "heat_score")
    print("\nTop 5 hottest files:")
    print(top[["file_id", "recent_access_count", "recency_seconds",
               "frequency_score", "recency_score", "heat_score"]].to_string(index=False))
    print("\nBottom 5 coldest files:")
    print(bottom[["file_id", "recent_access_count", "recency_seconds",
                   "frequency_score", "recency_score", "heat_score"]].to_string(index=False))
