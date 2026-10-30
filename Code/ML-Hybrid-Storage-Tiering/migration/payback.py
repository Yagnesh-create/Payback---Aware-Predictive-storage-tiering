"""
Migration Payback Calculator (Phase 10) for ML-Driven Hybrid Storage Staging.

Estimates whether migrating a file from HDD to SSD is performance-wise
worthwhile by computing:

    migration_cost     = file_size_bytes / bandwidth   (seconds)
    expected_benefit   = predicted_accesses * latency_saving_per_access  (seconds)
    payback_ratio      = expected_benefit / migration_cost

All latency and bandwidth values are SIMULATION PARAMETERS, not measured
hardware performance. They are configurable in config.py.
"""

from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def compute_migration_cost_seconds(
    file_size_bytes: float,
    bandwidth_bytes_per_sec: float = config.MIGRATION_BANDWIDTH_BYTES_PER_SEC,
) -> float:
    """
    Estimate the time cost to migrate a file between tiers.

    Args:
        file_size_bytes: Size of the file in bytes.
        bandwidth_bytes_per_sec: Simulated transfer bandwidth (bytes/sec).

    Returns:
        Migration cost in seconds.
    """
    if bandwidth_bytes_per_sec <= 0:
        return float("inf")
    return file_size_bytes / bandwidth_bytes_per_sec


def compute_expected_benefit_seconds(
    predicted_future_accesses: float,
    hdd_latency_us: float = config.HDD_READ_LATENCY_US,
    ssd_latency_us: float = config.SSD_READ_LATENCY_US,
) -> float:
    """
    Estimate the total latency savings from serving predicted future accesses
    from SSD instead of HDD during the next prediction window.

    Args:
        predicted_future_accesses: ML-predicted access count for the next 6h.
        hdd_latency_us: Simulated HDD per-access latency in microseconds.
        ssd_latency_us: Simulated SSD per-access latency in microseconds.

    Returns:
        Expected latency benefit in seconds.
    """
    latency_saving_per_access_us = hdd_latency_us - ssd_latency_us
    latency_saving_per_access_sec = latency_saving_per_access_us / 1_000_000.0
    return max(predicted_future_accesses, 0) * latency_saving_per_access_sec


def compute_payback_ratio(
    expected_benefit_sec: float,
    migration_cost_sec: float,
) -> float:
    """
    Compute the payback ratio: expected_benefit / migration_cost.

    Returns:
        Payback ratio (>1 means benefit exceeds cost).
        Returns float('inf') if migration_cost is 0 and benefit > 0.
        Returns 0.0 if both are 0.
    """
    if migration_cost_sec <= 0:
        if expected_benefit_sec > 0:
            return float("inf")
        return 0.0
    return expected_benefit_sec / migration_cost_sec


def compute_payback_for_batch(
    df: pd.DataFrame,
    predicted_col: str = "predicted_future_access_count",
    file_size_col: str = "file_size_bytes",
) -> pd.DataFrame:
    """
    Compute migration cost, expected benefit, and payback ratio for all rows.

    Args:
        df: DataFrame with predicted_col and file_size_col columns.

    Returns:
        DataFrame with added columns:
            migration_cost_sec, expected_benefit_sec, payback_ratio
    """
    result = df.copy()

    result["migration_cost_sec"] = result[file_size_col].apply(
        lambda s: compute_migration_cost_seconds(float(s))
    )
    result["expected_benefit_sec"] = result[predicted_col].apply(
        lambda a: compute_expected_benefit_seconds(float(a))
    )
    result["payback_ratio"] = result.apply(
        lambda r: compute_payback_ratio(r["expected_benefit_sec"], r["migration_cost_sec"]),
        axis=1,
    )

    return result


if __name__ == "__main__":
    print("Testing Payback Calculator...")

    # Synthetic test cases
    cases = [
        {"label": "Small hot file",  "size": 1_000_000,   "accesses": 50},
        {"label": "Large hot file",  "size": 200_000_000, "accesses": 50},
        {"label": "Small cold file", "size": 1_000_000,   "accesses": 1},
        {"label": "Large cold file", "size": 200_000_000, "accesses": 1},
        {"label": "Zero accesses",   "size": 50_000_000,  "accesses": 0},
    ]
    print(f"\n  {'Case':<20} {'Size(MB)':>10} {'Accesses':>10} {'Cost(s)':>10} {'Benefit(s)':>12} {'Payback':>10}")
    print("  " + "-" * 75)
    for c in cases:
        cost = compute_migration_cost_seconds(c["size"])
        benefit = compute_expected_benefit_seconds(c["accesses"])
        payback = compute_payback_ratio(benefit, cost)
        print(f"  {c['label']:<20} {c['size']/1e6:>10.1f} {c['accesses']:>10} {cost:>10.4f} {benefit:>12.6f} {payback:>10.4f}")
