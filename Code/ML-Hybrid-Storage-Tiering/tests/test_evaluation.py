"""
Evaluation Metrics Tests (Phase 18) for ML-Driven Hybrid Storage Staging.

Tests:
- Metric calculations
- Zero-denominator handling
- Result generation
- Reproducibility
- Comparison table structure
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from evaluation.metrics import (
    compute_mae,
    compute_rmse,
    compute_r2,
    compute_classification_metrics,
    compute_ml_metrics,
    compute_estimated_avg_latency,
    compute_ssd_utilization,
    compute_hdd_utilization,
    compute_payback_metrics,
    compute_latency_improvement,
    compute_migration_reduction,
    build_comparison_table,
)


PASSED = 0
FAILED = 0
TOTAL = 0


def check(name: str, condition: bool):
    global PASSED, FAILED, TOTAL
    TOTAL += 1
    status = "PASSED" if condition else "FAILED"
    if condition:
        PASSED += 1
    else:
        FAILED += 1
    print(f"  [{status}] {name}")


def test_regression_metrics():
    """Test MAE, RMSE, R² calculations."""
    print("\n--- TEST 1: Regression Metrics ---")

    actual = np.array([1, 2, 3, 4, 5], dtype=float)
    predicted = np.array([1, 2, 3, 4, 5], dtype=float)

    check("Perfect prediction MAE = 0", compute_mae(actual, predicted) == 0.0)
    check("Perfect prediction RMSE = 0", compute_rmse(actual, predicted) == 0.0)
    check("Perfect prediction R² = 1", compute_r2(actual, predicted) == 1.0)

    predicted2 = np.array([2, 3, 4, 5, 6], dtype=float)
    check("Off-by-1 MAE = 1", compute_mae(actual, predicted2) == 1.0)
    check("Off-by-1 RMSE = 1", compute_rmse(actual, predicted2) == 1.0)


def test_classification_metrics():
    """Test classification metric calculations."""
    print("\n--- TEST 2: Classification Metrics ---")

    actual = np.array([1, 1, 0, 0, 1, 0])
    predicted = np.array([1, 0, 0, 1, 1, 0])

    result = compute_classification_metrics(actual, predicted)
    check("TP = 2", result["true_positives"] == 2)
    check("TN = 2", result["true_negatives"] == 2)
    check("FP = 1", result["false_positives"] == 1)
    check("FN = 1", result["false_negatives"] == 1)
    check("Accuracy = 4/6", abs(result["accuracy"] - 4/6) < 1e-10)
    check("Precision = 2/3", abs(result["precision"] - 2/3) < 1e-10)
    check("Recall = 2/3", abs(result["recall"] - 2/3) < 1e-10)


def test_classification_zero_denominator():
    """Test classification with no positives."""
    print("\n--- TEST 3: Classification Zero Denominator ---")

    actual = np.array([0, 0, 0])
    predicted = np.array([0, 0, 0])

    result = compute_classification_metrics(actual, predicted)
    check("All negative: precision = 0", result["precision"] == 0.0)
    check("All negative: recall = 0", result["recall"] == 0.0)
    check("All negative: accuracy = 1", result["accuracy"] == 1.0)

    empty = compute_classification_metrics(np.array([]), np.array([]))
    check("Empty arrays: accuracy = 0", empty["accuracy"] == 0.0)


def test_latency_calculation():
    """Test estimated average latency calculation."""
    print("\n--- TEST 4: Latency Calculation ---")

    df = pd.DataFrame({
        "current_tier": [0, 1, 0],  # HDD, SSD, HDD
        "future_access_count": [10, 10, 0],
    })
    lat = compute_estimated_avg_latency(df, hdd_latency_us=10000, ssd_latency_us=200)
    # 10*10000 + 10*200 + 0 = 102000, total accesses = 20, avg = 5100
    check("Latency calculation correct", abs(lat - 5100.0) < 0.01)

    # Zero accesses
    df_zero = pd.DataFrame({
        "current_tier": [0],
        "future_access_count": [0],
    })
    lat_zero = compute_estimated_avg_latency(df_zero)
    check("Zero accesses latency = 0", lat_zero == 0.0)


def test_utilization():
    """Test SSD/HDD utilization calculations."""
    print("\n--- TEST 5: Utilization ---")

    df = pd.DataFrame({"current_tier": [0, 0, 1, 1, 1]})
    check("SSD utilization = 3/5 = 0.6", abs(compute_ssd_utilization(df) - 0.6) < 1e-10)
    check("HDD utilization = 2/5 = 0.4", abs(compute_hdd_utilization(df) - 0.4) < 1e-10)

    # Empty
    df_empty = pd.DataFrame({"current_tier": []})
    check("Empty SSD util = 0", compute_ssd_utilization(df_empty) == 0.0)


def test_payback_metrics():
    """Test payback metric calculations."""
    print("\n--- TEST 6: Payback Metrics ---")

    df = pd.DataFrame({
        "payback_ratio": [0.5, 1.5, 2.0, 0.1, 3.0],
    })
    result = compute_payback_metrics(df, min_payback=1.0)
    check("Avg payback correct", abs(result["avg_payback"] - 1.42) < 0.01)
    check("Above threshold = 3", result["above_threshold"] == 3)
    check("Below threshold = 2", result["below_threshold"] == 2)
    check("Min payback = 0.1", abs(result["min_payback"] - 0.1) < 0.01)
    check("Max payback = 3.0", abs(result["max_payback"] - 3.0) < 0.01)


def test_improvement_calculations():
    """Test improvement percentage calculations."""
    print("\n--- TEST 7: Improvement Calculations ---")

    imp = compute_latency_improvement(10.0, 8.0)
    check("Latency improvement 10->8 = 20%", abs(imp - 20.0) < 0.01)

    imp_neg = compute_latency_improvement(10.0, 12.0)
    check("Latency worsening: negative %", imp_neg < 0)

    imp_zero = compute_latency_improvement(0.0, 5.0)
    check("Zero baseline -> None", imp_zero is None)

    mig_red = compute_migration_reduction(50, 30)
    check("Migration reduction 50->30 = 40%", abs(mig_red - 40.0) < 0.01)

    mig_red_zero = compute_migration_reduction(0, 10)
    check("Zero baseline migrations -> None", mig_red_zero is None)


def test_comparison_table():
    """Test comparison table structure."""
    print("\n--- TEST 8: Comparison Table ---")

    results = [
        {"method": "A", "avg_latency_ms": 10.0, "migrations": 0,
         "migration_time_sec": 0.0, "false_hotspot_rate": None,
         "useful_migration_rate": None, "ssd_utilization": 0.2},
        {"method": "B", "avg_latency_ms": 8.0, "migrations": 5,
         "migration_time_sec": 1.0, "false_hotspot_rate": 0.3,
         "useful_migration_rate": 0.7, "ssd_utilization": 0.4},
    ]
    table = build_comparison_table(results)
    check("Table has 2 rows", len(table) == 2)
    check("Table has Method column", "Method" in table.columns)
    check("Table has Avg Latency column", "Avg Latency (ms)" in table.columns)
    check("Static method in table", "A" in table["Method"].values)


def test_reproducibility():
    """Test that metrics are deterministic."""
    print("\n--- TEST 9: Reproducibility ---")

    actual = np.array([1.0, 5.0, 10.0, 3.0, 7.0])
    predicted = np.array([2.0, 4.0, 12.0, 2.0, 6.0])

    mae1 = compute_mae(actual, predicted)
    mae2 = compute_mae(actual, predicted)
    check("MAE is deterministic", mae1 == mae2)

    rmse1 = compute_rmse(actual, predicted)
    rmse2 = compute_rmse(actual, predicted)
    check("RMSE is deterministic", rmse1 == rmse2)


def test_ml_metrics_integration():
    """Test compute_ml_metrics on a small DataFrame."""
    print("\n--- TEST 10: ML Metrics Integration ---")

    df = pd.DataFrame({
        "future_access_count": [0, 5, 10, 15, 20],
        "predicted_future_access_count": [1, 4, 12, 14, 18],
    })
    result = compute_ml_metrics(df, threshold=8)
    check("Has mae", "mae" in result)
    check("Has rmse", "rmse" in result)
    check("Has r2", "r2" in result)
    check("Has accuracy", "accuracy" in result)
    check("Has confusion_matrix", "confusion_matrix" in result)
    check("MAE > 0", result["mae"] > 0)


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 18 — EVALUATION METRICS TESTS")
    print("=" * 70)

    test_regression_metrics()
    test_classification_metrics()
    test_classification_zero_denominator()
    test_latency_calculation()
    test_utilization()
    test_payback_metrics()
    test_improvement_calculations()
    test_comparison_table()
    test_reproducibility()
    test_ml_metrics_integration()

    print(f"\n{'=' * 70}")
    print(f"EVALUATION TEST SUMMARY: {PASSED}/{TOTAL} passed, {FAILED}/{TOTAL} failed")
    print(f"{'=' * 70}")

    if FAILED > 0:
        sys.exit(1)
    else:
        sys.exit(0)
