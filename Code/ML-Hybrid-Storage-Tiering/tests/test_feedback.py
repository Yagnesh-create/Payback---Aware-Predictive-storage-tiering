"""
Feedback Module Tests (Phase 16) for ML-Driven Hybrid Storage Staging.

Tests covering all 10+ specified cases for the feedback mechanism:
1. Correct prediction (True Positive)
2. False hotspot (False Positive)
3. Missed hotspot (False Negative)
4. Correct non-hotspot (True Negative)
5. Large positive error
6. Large negative error
7. Zero prediction
8. Zero actual accesses
9. Feedback clipping
10. Heat score remaining within [0, 1]
"""

from pathlib import Path
import sys
import math
import tempfile
import os

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from feedback.migration_feedback import (
    compare_prediction_with_reality,
    classify_migration_outcome,
    compute_false_hotspot_rate,
    compute_false_hotspot_migration_rate,
    compute_bounded_feedback,
    update_heat_score,
    count_actual_future_accesses,
    run_feedback_loop,
    compute_feedback_summary,
    PRED_TRUE_POSITIVE,
    PRED_TRUE_NEGATIVE,
    PRED_FALSE_POSITIVE,
    PRED_FALSE_NEGATIVE,
    OUTCOME_SUCCESSFUL_USEFUL,
    OUTCOME_FALSE_HOTSPOT_MIG,
    OUTCOME_MISSED_HOTSPOT,
    OUTCOME_CORRECT_NON_MIG,
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


def test_correct_prediction():
    """Test 1: Correct prediction — True Positive."""
    print("\n--- TEST 1: Correct Prediction (True Positive) ---")
    # Predicted 12 accesses, actual 15 — both >= threshold (8)
    result = compare_prediction_with_reality(12.0, 15, threshold=8)
    check("Prediction outcome is TRUE_POSITIVE", result["prediction_outcome"] == PRED_TRUE_POSITIVE)
    check("Predicted hotspot is 1", result["predicted_hotspot"] == 1)
    check("Actual hotspot is 1", result["actual_hotspot"] == 1)
    check("Prediction error = 3", result["prediction_error"] == 3.0)
    check("Absolute error = 3", result["absolute_prediction_error"] == 3.0)


def test_false_hotspot():
    """Test 2: False hotspot — False Positive."""
    print("\n--- TEST 2: False Hotspot (False Positive) ---")
    # Predicted 10 (hotspot), actual 2 (not hotspot)
    result = compare_prediction_with_reality(10.0, 2, threshold=8)
    check("Prediction outcome is FALSE_POSITIVE", result["prediction_outcome"] == PRED_FALSE_POSITIVE)
    check("Predicted hotspot is 1", result["predicted_hotspot"] == 1)
    check("Actual hotspot is 0", result["actual_hotspot"] == 0)
    check("Prediction error = -8", result["prediction_error"] == -8.0)


def test_missed_hotspot():
    """Test 3: Missed hotspot — False Negative."""
    print("\n--- TEST 3: Missed Hotspot (False Negative) ---")
    # Predicted 3 (not hotspot), actual 20 (hotspot)
    result = compare_prediction_with_reality(3.0, 20, threshold=8)
    check("Prediction outcome is FALSE_NEGATIVE", result["prediction_outcome"] == PRED_FALSE_NEGATIVE)
    check("Predicted hotspot is 0", result["predicted_hotspot"] == 0)
    check("Actual hotspot is 1", result["actual_hotspot"] == 1)
    check("Prediction error = 17", result["prediction_error"] == 17.0)


def test_correct_non_hotspot():
    """Test 4: Correct non-hotspot — True Negative."""
    print("\n--- TEST 4: Correct Non-Hotspot (True Negative) ---")
    # Predicted 2, actual 1 — both < threshold (8)
    result = compare_prediction_with_reality(2.0, 1, threshold=8)
    check("Prediction outcome is TRUE_NEGATIVE", result["prediction_outcome"] == PRED_TRUE_NEGATIVE)
    check("Predicted hotspot is 0", result["predicted_hotspot"] == 0)
    check("Actual hotspot is 0", result["actual_hotspot"] == 0)


def test_large_positive_error():
    """Test 5: Large positive error (actual >> predicted)."""
    print("\n--- TEST 5: Large Positive Error ---")
    result = compare_prediction_with_reality(5.0, 100, threshold=8)
    check("Prediction error = 95", result["prediction_error"] == 95.0)
    check("Absolute error = 95", result["absolute_prediction_error"] == 95.0)

    # Bounded feedback should saturate near +1
    bounded = compute_bounded_feedback(95.0, scale=5.0)
    check("Bounded feedback near +1", bounded > 0.99)
    check("Bounded feedback <= 1.0", bounded <= 1.0)


def test_large_negative_error():
    """Test 6: Large negative error (actual << predicted)."""
    print("\n--- TEST 6: Large Negative Error ---")
    result = compare_prediction_with_reality(50.0, 0, threshold=8)
    check("Prediction error = -50", result["prediction_error"] == -50.0)

    # Bounded feedback should saturate near -1
    bounded = compute_bounded_feedback(-50.0, scale=5.0)
    check("Bounded feedback near -1", bounded < -0.99)
    check("Bounded feedback >= -1.0", bounded >= -1.0)


def test_zero_prediction():
    """Test 7: Zero prediction."""
    print("\n--- TEST 7: Zero Prediction ---")
    result = compare_prediction_with_reality(0.0, 5, threshold=8)
    check("Prediction error = 5", result["prediction_error"] == 5.0)
    check("Predicted hotspot is 0", result["predicted_hotspot"] == 0)
    check("Actual hotspot is 0", result["actual_hotspot"] == 0)

    bounded = compute_bounded_feedback(5.0, scale=5.0)
    check("Bounded feedback is tanh(1)", abs(bounded - math.tanh(1.0)) < 1e-10)


def test_zero_actual_accesses():
    """Test 8: Zero actual accesses."""
    print("\n--- TEST 8: Zero Actual Accesses ---")
    result = compare_prediction_with_reality(10.0, 0, threshold=8)
    check("Prediction error = -10", result["prediction_error"] == -10.0)
    check("Actual hotspot is 0", result["actual_hotspot"] == 0)
    check("Predicted hotspot is 1", result["predicted_hotspot"] == 1)
    check("Outcome is FALSE_POSITIVE", result["prediction_outcome"] == PRED_FALSE_POSITIVE)


def test_feedback_clipping():
    """Test 9: Feedback clipping — heat score stays in [0, 1]."""
    print("\n--- TEST 9: Feedback Clipping ---")

    # Case A: Very large positive feedback on already high heat
    new_heat = update_heat_score(0.95, 0.99, weight=0.5)
    check("High heat + positive feedback clipped to <= 1.0", new_heat <= 1.0)
    check("High heat + positive feedback >= 0.0", new_heat >= 0.0)

    # Case B: Very large negative feedback on already low heat
    new_heat2 = update_heat_score(0.05, -0.99, weight=0.5)
    check("Low heat + negative feedback clipped to >= 0.0", new_heat2 >= 0.0)
    check("Low heat + negative feedback <= 1.0", new_heat2 <= 1.0)

    # Case C: Extreme weight
    new_heat3 = update_heat_score(1.0, 1.0, weight=1.0)
    check("Max heat + max feedback clipped to 1.0", new_heat3 == 1.0)

    new_heat4 = update_heat_score(0.0, -1.0, weight=1.0)
    check("Min heat + min feedback clipped to 0.0", new_heat4 == 0.0)


def test_heat_score_bounds():
    """Test 10: Heat score remains within [0, 1] for many random inputs."""
    print("\n--- TEST 10: Heat Score Bounds (Random Stress) ---")
    np.random.seed(config.RANDOM_SEED)

    all_valid = True
    for _ in range(1000):
        old_heat = np.random.uniform(0, 1)
        feedback_signal = np.random.uniform(-100, 100)
        bounded = compute_bounded_feedback(feedback_signal, scale=5.0)
        new_heat = update_heat_score(old_heat, bounded, weight=0.3)
        if new_heat < 0.0 or new_heat > 1.0:
            all_valid = False
            break

    check("All 1000 random trials: heat in [0, 1]", all_valid)


def test_migration_outcome_classification():
    """Test migration-specific outcome classification."""
    print("\n--- TEST 11: Migration Outcome Classification ---")

    check("Migrated + actually hot = SUCCESSFUL_USEFUL",
          classify_migration_outcome(True, 1, 1) == OUTCOME_SUCCESSFUL_USEFUL)
    check("Migrated + NOT actually hot = FALSE_HOTSPOT_MIG",
          classify_migration_outcome(True, 1, 0) == OUTCOME_FALSE_HOTSPOT_MIG)
    check("Not migrated + actually hot = MISSED_HOTSPOT",
          classify_migration_outcome(False, 0, 1) == OUTCOME_MISSED_HOTSPOT)
    check("Not migrated + NOT hot = CORRECT_NON_MIG",
          classify_migration_outcome(False, 0, 0) == OUTCOME_CORRECT_NON_MIG)


def test_false_hotspot_rates():
    """Test false hotspot rate calculations with zero denominators."""
    print("\n--- TEST 12: False Hotspot Rates ---")

    check("FP rate: 2/10 = 0.2", compute_false_hotspot_rate(2, 10) == 0.2)
    check("FP rate: 0 total predicted -> None", compute_false_hotspot_rate(0, 0) is None)
    check("FP rate: 0/5 = 0.0", compute_false_hotspot_rate(0, 5) == 0.0)

    check("FP migration rate: 3/10 = 0.3", compute_false_hotspot_migration_rate(3, 10) == 0.3)
    check("FP migration rate: 0 total -> None", compute_false_hotspot_migration_rate(0, 0) is None)


def test_bounded_feedback_properties():
    """Test mathematical properties of bounded feedback."""
    print("\n--- TEST 13: Bounded Feedback Properties ---")

    check("tanh(0) = 0", compute_bounded_feedback(0.0) == 0.0)
    check("Positive input -> positive output", compute_bounded_feedback(5.0) > 0)
    check("Negative input -> negative output", compute_bounded_feedback(-5.0) < 0)
    check("Output in [-1, 1] for large input", -1.0 <= compute_bounded_feedback(1000.0) <= 1.0)
    check("Output in [-1, 1] for large negative", -1.0 <= compute_bounded_feedback(-1000.0) <= 1.0)
    check("Zero scale -> 0.0", compute_bounded_feedback(10.0, scale=0) == 0.0)


def test_actual_access_counting():
    """Test actual future access counting from trace data."""
    print("\n--- TEST 14: Actual Access Counting ---")

    # Create a small synthetic trace
    trace_data = {
        "file_id": ["f1", "f1", "f1", "f2", "f1"],
        "timestamp": [
            "2026-01-13 01:00:00",  # within 6h of 2026-01-13 00:00:00
            "2026-01-13 03:00:00",  # within
            "2026-01-13 07:00:00",  # outside (> 6h)
            "2026-01-13 02:00:00",  # f2, not f1
            "2026-01-13 05:59:59",  # within
        ],
    }
    df_trace = pd.DataFrame(trace_data)
    df_trace["_parsed_ts"] = pd.to_datetime(df_trace["timestamp"])

    count = count_actual_future_accesses(
        df_trace, "f1", "2026-01-13 00:00:00", future_window_hours=6,
    )
    check("f1 has 3 accesses in 6h window", count == 3)

    count_f2 = count_actual_future_accesses(
        df_trace, "f2", "2026-01-13 00:00:00", future_window_hours=6,
    )
    check("f2 has 1 access in 6h window", count_f2 == 1)

    count_none = count_actual_future_accesses(
        df_trace, "f3", "2026-01-13 00:00:00", future_window_hours=6,
    )
    check("Non-existent file has 0 accesses", count_none == 0)


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 16 — FEEDBACK MODULE TESTS")
    print("=" * 70)

    test_correct_prediction()
    test_false_hotspot()
    test_missed_hotspot()
    test_correct_non_hotspot()
    test_large_positive_error()
    test_large_negative_error()
    test_zero_prediction()
    test_zero_actual_accesses()
    test_feedback_clipping()
    test_heat_score_bounds()
    test_migration_outcome_classification()
    test_false_hotspot_rates()
    test_bounded_feedback_properties()
    test_actual_access_counting()

    print(f"\n{'=' * 70}")
    print(f"FEEDBACK TEST SUMMARY: {PASSED}/{TOTAL} passed, {FAILED}/{TOTAL} failed")
    print(f"{'=' * 70}")

    if FAILED > 0:
        sys.exit(1)
    else:
        sys.exit(0)
