"""
Baseline Systems Tests (Phase 17) for ML-Driven Hybrid Storage Staging.

Tests:
- Static baseline returns 0 migrations
- Frequency baseline has no ML model dependency
- Fair comparison (same data/params)
- Correct latency calculation
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from evaluation.baselines import (
    run_static_tiering_baseline,
    run_frequency_baseline,
    run_rule_based_baseline,
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


def make_test_df():
    """Create a small synthetic dataset for baseline testing."""
    return pd.DataFrame({
        "file_id": ["f1", "f2", "f3", "f4", "f5"],
        "prediction_timestamp": ["2026-01-13 00:00:00"] * 5,
        "recent_access_count": [20, 3, 15, 1, 50],
        "previous_access_count": [10, 2, 8, 0, 30],
        "recency_seconds": [100, 5000, 200, 40000, 50],
        "burstiness": [0.5, 0.1, 0.3, -1.0, 0.8],
        "file_size_bytes": [10_000_000, 50_000_000, 20_000_000, 100_000_000, 5_000_000],
        "read_ratio": [0.8, 0.5, 0.7, 0.9, 0.6],
        "write_ratio": [0.2, 0.5, 0.3, 0.1, 0.4],
        "average_request_size": [32768, 16384, 32768, 8192, 65536],
        "current_tier": [0, 0, 0, 0, 1],  # f5 is on SSD
        "future_access_count": [25, 1, 10, 0, 40],
        "eval_true_hotspot_next_6h": [1, 0, 1, 0, 1],
    })


def test_static_baseline():
    """Test that static tiering baseline has zero migrations."""
    print("\n--- TEST 1: Static Tiering ---")
    df = make_test_df()
    result = run_static_tiering_baseline(df)

    check("Method name is 'Static Tiering'", result["method"] == "Static Tiering")
    check("Zero migrations", result["migrations"] == 0)
    check("Zero migration time", result["migration_time_sec"] == 0.0)
    check("Zero migration volume", result["migration_volume_bytes"] == 0)
    check("Zero unnecessary migrations", result["unnecessary_migrations"] == 0)
    check("Average latency > 0", result["avg_latency_us"] > 0)
    check("SSD utilization in [0, 1]", 0 <= result["ssd_utilization"] <= 1)
    check("False hotspot rate is None (N/A)", result["false_hotspot_rate"] is None)
    check("Useful migration rate is None (N/A)", result["useful_migration_rate"] is None)


def test_static_latency_correctness():
    """Test that static latency is correctly computed from tier assignments."""
    print("\n--- TEST 2: Static Latency Correctness ---")
    df = pd.DataFrame({
        "file_id": ["f1", "f2"],
        "prediction_timestamp": ["ts1", "ts1"],
        "current_tier": [0, 1],  # HDD, SSD
        "future_access_count": [10, 10],
    })
    result = run_static_tiering_baseline(df, hdd_latency_us=10000, ssd_latency_us=200)

    # f1: 10 * 10000 = 100000, f2: 10 * 200 = 2000, total = 102000, avg = 102000/20 = 5100
    expected_avg = (10 * 10000 + 10 * 200) / 20
    check("Average latency matches manual calculation",
          abs(result["avg_latency_us"] - expected_avg) < 0.01)


def test_frequency_no_ml_dependency():
    """Test that frequency baseline does not use the ML model."""
    print("\n--- TEST 3: Frequency Baseline — No ML Dependency ---")
    df = make_test_df()

    # Run frequency baseline — should work even if the model file doesn't exist
    # (it shouldn't try to load it)
    result = run_frequency_baseline(df, min_payback=0.0)

    check("Method name is 'Frequency-Based'", result["method"] == "Frequency-Based")
    check("Has migrations key", "migrations" in result)
    check("Has avg_latency_us key", "avg_latency_us" in result)


def test_frequency_migration_candidates():
    """Test that frequency baseline selects HDD files above threshold."""
    print("\n--- TEST 4: Frequency Migration Candidates ---")
    df = make_test_df()

    # With threshold=8, files f1 (20 acc, HDD), f3 (15 acc, HDD), f5 (50 acc, SSD=skip)
    result = run_frequency_baseline(df, threshold=8, min_payback=0.0,
                                     max_migrations=100)

    check("At least 1 migration occurred", result["migrations"] >= 1)
    # f5 is already on SSD, so shouldn't be migrated
    # f1 (20) and f3 (15) qualify
    check("At most 2 HDD hotspots migrated", result["migrations"] <= 2)


def test_frequency_respects_max_migrations():
    """Test that frequency baseline respects max_migrations limit."""
    print("\n--- TEST 5: Max Migrations Limit ---")
    df = make_test_df()

    result = run_frequency_baseline(df, threshold=8, min_payback=0.0,
                                     max_migrations=1)
    check("Migrations limited to 1", result["migrations"] <= 1)


def test_fair_comparison():
    """Test that all baselines use the same data and params."""
    print("\n--- TEST 6: Fair Comparison ---")
    df = make_test_df()

    static = run_static_tiering_baseline(df)
    freq = run_frequency_baseline(df, min_payback=0.05)
    rule = run_rule_based_baseline(df)

    # All use the same total accesses
    check("Same total accesses (static vs freq)",
          static["total_accesses"] == freq["total_accesses"])
    check("Same total accesses (static vs rule)",
          static["total_accesses"] == rule["total_accesses"])

    # Frequency/rule latency should be <= static latency
    # (migrations move files to faster tier)
    check("Freq latency <= Static latency",
          freq["avg_latency_us"] <= static["avg_latency_us"] + 0.01)
    check("Rule latency <= Static latency",
          rule["avg_latency_us"] <= static["avg_latency_us"] + 0.01)


def test_rule_based_no_payback():
    """Test that rule-based baseline has no payback filtering."""
    print("\n--- TEST 7: Rule-Based Has No Payback Filter ---")
    df = make_test_df()

    # Rule-based should migrate more than frequency w/ high min_payback
    freq_strict = run_frequency_baseline(df, threshold=8, min_payback=100.0)
    rule = run_rule_based_baseline(df, threshold=8)

    check("Rule-based has >= migrations than strict-payback frequency",
          rule["migrations"] >= freq_strict["migrations"])


def test_zero_accesses_handling():
    """Test baselines handle zero total accesses safely."""
    print("\n--- TEST 8: Zero Accesses Handling ---")
    df = pd.DataFrame({
        "file_id": ["f1"],
        "prediction_timestamp": ["ts1"],
        "current_tier": [0],
        "recent_access_count": [0],
        "file_size_bytes": [1000],
        "future_access_count": [0],
    })
    static = run_static_tiering_baseline(df)
    check("Static handles zero accesses without error", static["avg_latency_us"] == 0.0)


def test_real_dataset():
    """Test baselines on the actual test dataset."""
    print("\n--- TEST 9: Real Dataset ---")
    df_test = pd.read_csv(config.TEST_DATASET_PATH)

    static = run_static_tiering_baseline(df_test)
    check("Static on real data: avg latency > 0", static["avg_latency_us"] > 0)
    check("Static on real data: 0 migrations", static["migrations"] == 0)

    freq = run_frequency_baseline(df_test, min_payback=0.05)
    check("Freq on real data: has result", freq["method"] == "Frequency-Based")
    check("Freq on real data: avg latency > 0", freq["avg_latency_us"] > 0)


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 17 — BASELINE SYSTEMS TESTS")
    print("=" * 70)

    test_static_baseline()
    test_static_latency_correctness()
    test_frequency_no_ml_dependency()
    test_frequency_migration_candidates()
    test_frequency_respects_max_migrations()
    test_fair_comparison()
    test_rule_based_no_payback()
    test_zero_accesses_handling()
    test_real_dataset()

    print(f"\n{'=' * 70}")
    print(f"BASELINE TEST SUMMARY: {PASSED}/{TOTAL} passed, {FAILED}/{TOTAL} failed")
    print(f"{'=' * 70}")

    if FAILED > 0:
        sys.exit(1)
    else:
        sys.exit(0)
