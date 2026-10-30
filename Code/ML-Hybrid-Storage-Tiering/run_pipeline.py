"""
Integrated Pipeline & Testing Script for Phases 7–10.

Runs the full decision pipeline end-to-end, executes unit/integration tests
for heat-map, predictions, payback, and decision logic, and generates
evaluation visualizations.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parent))
import config
from ml.heatmap import (
    compute_heat_scores,
    compute_frequency_score,
    compute_recency_score,
    validate_heat_scores,
)
from ml.predictor import HotspotPredictor
from migration.payback import (
    compute_migration_cost_seconds,
    compute_expected_benefit_seconds,
    compute_payback_ratio,
    compute_payback_for_batch,
)
from migration.decision_engine import (
    run_decision_pipeline,
    make_migration_decisions,
    generate_decision_summary,
    DECISION_MIGRATE,
    DECISION_ALREADY_SSD,
    DECISION_NOT_HOT,
    DECISION_KEEP,
    DECISION_TABLE_COLUMNS,
)

FIG_DIR = config.FIGURES_DIR
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


# ================================================================
# UNIT / INTEGRATION TESTS
# ================================================================

def run_tests():
    """Execute all unit and integration tests for Phases 7–10."""
    passed = 0
    failed = 0
    total = 0

    def check(name, condition):
        nonlocal passed, failed, total
        total += 1
        status = "PASSED" if condition else "FAILED"
        if condition:
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] {name}")

    print("\n" + "=" * 70)
    print("UNIT & INTEGRATION TESTS")
    print("=" * 70)

    # --- Heat-map tests ---
    print("\n--- Heat-Map Tests ---")

    s = pd.Series([0, 5, 10, 20, 100])
    fs = compute_frequency_score(s)
    check("Frequency score range [0,1]", fs.min() >= 0 and fs.max() <= 1.0)
    check("Max freq gets score 1.0", np.isclose(fs.iloc[-1], 1.0))
    check("Min freq gets score 0.0", np.isclose(fs.iloc[0], 0.0))

    rec = pd.Series([1.0, 100.0, 1000.0, 10000.0, 43200.0])
    rs = compute_recency_score(rec)
    check("Recency score range [0,1]", rs.min() >= 0 and rs.max() <= 1.0)
    check("Most recent (small sec) -> high rec score", rs.iloc[0] > rs.iloc[-1])
    check("Least recent (large sec) -> low rec score", rs.iloc[-1] < rs.iloc[0])

    # Constant recency → min_max gives 0.0, inversion gives 1.0 (all equally "recent")
    const_rec = pd.Series([500.0, 500.0, 500.0])
    const_rs = compute_recency_score(const_rec)
    check("Constant recency -> all 1.0 (equally recent)", (const_rs == 1.0).all())

    # Heat score composition
    test_df = pd.DataFrame({
        "recent_access_count": [50, 0, 50, 0],
        "recency_seconds": [10.0, 40000.0, 40000.0, 10.0],
        "file_id": ["hot", "cold", "freq_only", "recent_only"],
    })
    scored = compute_heat_scores(test_df)
    check("Hot file has highest heat", scored.loc[0, "heat_score"] >= scored["heat_score"].max() - 0.01)
    check("Cold file has lowest heat", scored.loc[1, "heat_score"] <= scored["heat_score"].min() + 0.01)
    check("Heat scores in [0,1]", (scored["heat_score"] >= 0).all() and (scored["heat_score"] <= 1).all())

    # --- Payback tests ---
    print("\n--- Payback Calculation Tests ---")

    cost = compute_migration_cost_seconds(100 * 1024 * 1024)  # 100 MB
    check("100MB migration cost > 0", cost > 0)
    check("100MB cost ~1.0 sec at 100MB/s", np.isclose(cost, 1.0, atol=0.01))

    benefit = compute_expected_benefit_seconds(50)  # 50 accesses
    check("50 accesses benefit > 0", benefit > 0)
    expected_b = 50 * (config.HDD_READ_LATENCY_US - config.SSD_READ_LATENCY_US) / 1e6
    check("Benefit matches formula", np.isclose(benefit, expected_b, rtol=0.01))

    benefit_zero = compute_expected_benefit_seconds(0)
    check("Zero accesses -> zero benefit", benefit_zero == 0.0)

    payback = compute_payback_ratio(benefit, cost)
    check("Payback ratio > 0 for hot file", payback > 0)

    payback_zero_cost = compute_payback_ratio(1.0, 0.0)
    check("Zero cost -> inf payback", payback_zero_cost == float("inf"))

    payback_zero_both = compute_payback_ratio(0.0, 0.0)
    check("Zero cost + zero benefit -> 0.0 payback", payback_zero_both == 0.0)

    large_file_cost = compute_migration_cost_seconds(200_000_000)
    small_file_cost = compute_migration_cost_seconds(1_000_000)
    check("Larger file -> higher migration cost", large_file_cost > small_file_cost)

    # --- Decision logic tests ---
    print("\n--- Decision Logic Tests ---")

    def make_row(tier, pred_acc, pred_hot, payback_r):
        return pd.DataFrame([{
            "file_id": "test",
            "current_tier": tier,
            "predicted_future_access_count": pred_acc,
            "predicted_hotspot": pred_hot,
            "payback_ratio": payback_r,
            "migration_cost_sec": 1.0,
            "expected_benefit_sec": payback_r * 1.0,
        }])

    # CASE 1: High predicted + HDD + high payback → MIGRATE
    r1 = make_migration_decisions(make_row(0, 24.0, 1, 5.25))
    check("CASE 1: Hot HDD + high payback -> MIGRATE_TO_SSD", r1.iloc[0]["decision"] == DECISION_MIGRATE)

    # CASE 2: Low predicted → NOT_HOT
    r2 = make_migration_decisions(make_row(0, 2.0, 0, 0.01))
    check("CASE 2: Low predicted -> NOT_HOT", r2.iloc[0]["decision"] == DECISION_NOT_HOT)

    # CASE 3: Hot but low payback → KEEP
    r3 = make_migration_decisions(make_row(0, 10.0, 1, 0.3))
    check("CASE 3: Hot HDD + low payback -> KEEP_CURRENT_TIER", r3.iloc[0]["decision"] == DECISION_KEEP)

    # CASE 4: Already on SSD → ALREADY_ON_SSD
    r4 = make_migration_decisions(make_row(1, 50.0, 1, 10.0))
    check("CASE 4: Hot SSD -> ALREADY_ON_SSD", r4.iloc[0]["decision"] == DECISION_ALREADY_SSD)

    # CASE 5: Zero cost file
    r5 = make_migration_decisions(make_row(0, 20.0, 1, float("inf")))
    check("CASE 5: Zero-cost hot HDD -> MIGRATE (inf payback)", r5.iloc[0]["decision"] == DECISION_MIGRATE)

    # CASE 6: Very large file (cost > benefit)
    big_cost = compute_migration_cost_seconds(200_000_000)
    big_benefit = compute_expected_benefit_seconds(2)
    big_payback = compute_payback_ratio(big_benefit, big_cost)
    check("CASE 6: Large cold file payback < 1", big_payback < 1.0)

    # --- ML model reload test ---
    print("\n--- ML Model Reload Test ---")
    predictor = HotspotPredictor(threshold=config.HOTSPOT_THRESHOLD)
    check("Model loaded from disk", predictor.model is not None)
    check("Scaler loaded from disk", predictor.scaler is not None)

    test_features = {
        "recent_access_count": 30,
        "previous_access_count": 10,
        "recency_seconds": 100.0,
        "burstiness": 0.1,
        "file_size_bytes": 50_000_000,
        "read_ratio": 0.8,
        "write_ratio": 0.2,
        "average_request_size": 32768.0,
        "current_tier": 0,
    }
    pred_count, is_hot = predictor.predict_single(test_features)
    check("Prediction returns float", isinstance(pred_count, float))
    check("Prediction >= 0", pred_count >= 0)
    check("Hotspot decision is bool", isinstance(is_hot, (bool, np.bool_)))

    print(f"\n{'='*70}")
    print(f"TEST SUMMARY: {passed}/{total} passed, {failed}/{total} failed")
    print(f"{'='*70}")
    return passed, failed, total


# ================================================================
# FULL INTEGRATED PIPELINE
# ================================================================

def run_integrated_pipeline():
    """Run the full Phase 7-10 integrated pipeline on test data."""
    print("\n" + "=" * 80)
    print("INTEGRATED PIPELINE: PHASES 7–10")
    print("=" * 80)

    df_test = pd.read_csv(config.TEST_DATASET_PATH)
    last_ts = sorted(df_test["prediction_timestamp"].unique())[-1]
    df_slice = df_test[df_test["prediction_timestamp"] == last_ts].copy()

    print(f"\nPrediction timestamp: {last_ts}")
    print(f"Files to evaluate: {len(df_slice)}")

    # Run full pipeline
    result = run_decision_pipeline(df_slice)
    summary = generate_decision_summary(result)

    print(f"\n--- Decision Summary ---")
    for k, v in summary.items():
        print(f"  {k:<20}: {v}")

    # Show decision table for a sample
    avail_cols = [c for c in DECISION_TABLE_COLUMNS if c in result.columns]
    print(f"\n--- Sample Decision Table (top 15 by heat_score) ---")
    top_heat = result.nlargest(15, "heat_score")
    for _, row in top_heat.iterrows():
        tier_str = "SSD" if int(row["current_tier"]) == 1 else "HDD"
        hot_str = "YES" if int(row["predicted_hotspot"]) == 1 else "NO"
        print(
            f"  {row['file_id']:<12} Tier={tier_str:<4} Heat={row['heat_score']:.4f} "
            f"PredAcc={row['predicted_future_access_count']:>6.2f} Hot={hot_str:<4} "
            f"Cost={row['migration_cost_sec']:.4f}s Benefit={row['expected_benefit_sec']:.6f}s "
            f"Payback={row['payback_ratio']:.4f} -> {row['decision']}"
        )

    return result, summary


# ================================================================
# VISUALIZATIONS
# ================================================================

def generate_visualizations(df_full: pd.DataFrame):
    """Generate Phase 7-10 evaluation plots from the full test set pipeline output."""
    print("\n[VIZ] Generating Phase 7-10 visualizations...")

    # 1. Heat score distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df_full["heat_score"], bins=40, kde=True, color="#e74c3c", edgecolor="black", ax=ax)
    ax.set_title("File Heat Score Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Heat Score", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase7_1_heat_score_distribution.png", dpi=300)
    plt.close(fig)

    # 2. Predicted future accesses distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df_full["predicted_future_access_count"], bins=40, kde=True, color="#2980b9", edgecolor="black", ax=ax)
    ax.axvline(config.HOTSPOT_THRESHOLD, color="red", linestyle="--", lw=2, label=f"Hotspot threshold ({config.HOTSPOT_THRESHOLD})")
    ax.set_title("Predicted Future Access Count Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Future Accesses", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase8_2_predicted_accesses_distribution.png", dpi=300)
    plt.close(fig)

    # 3. Migration cost vs expected benefit scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    colors_d = {
        DECISION_MIGRATE: "#27ae60",
        DECISION_NOT_HOT: "#95a5a6",
        DECISION_KEEP: "#e67e22",
        DECISION_ALREADY_SSD: "#3498db",
    }
    for dec, color in colors_d.items():
        mask = df_full["decision"] == dec
        if mask.any():
            ax.scatter(
                df_full.loc[mask, "migration_cost_sec"],
                df_full.loc[mask, "expected_benefit_sec"],
                c=color, label=dec, alpha=0.6, s=25, edgecolor="none",
            )
    ax.set_xlabel("Migration Cost (seconds)", fontsize=11)
    ax.set_ylabel("Expected Benefit (seconds)", fontsize=11)
    ax.set_title("Migration Cost vs. Expected Benefit", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase9_3_cost_vs_benefit.png", dpi=300)
    plt.close(fig)

    # 4. Payback ratio distribution (cap at reasonable max for display)
    payback_clipped = df_full["payback_ratio"].clip(upper=10.0)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(payback_clipped, bins=40, kde=True, color="#8e44ad", edgecolor="black", ax=ax)
    ax.axvline(config.MIN_PAYBACK_RATIO, color="red", linestyle="--", lw=2, label=f"Min payback ({config.MIN_PAYBACK_RATIO})")
    ax.set_title("Payback Ratio Distribution (Clipped at 10)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Payback Ratio", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase10_4_payback_ratio_distribution.png", dpi=300)
    plt.close(fig)

    # 5. Decision distribution bar chart
    counts = df_full["decision"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_colors = [colors_d.get(d, "#555") for d in counts.index]
    ax.bar(counts.index, counts.values, color=bar_colors, edgecolor="black", alpha=0.85)
    for i, (dec, cnt) in enumerate(counts.items()):
        ax.text(i, cnt + len(df_full) * 0.005, str(cnt), ha="center", fontweight="bold")
    ax.set_title("Migration Decision Distribution", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of File-Window Instances", fontsize=11)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase9_5_decision_distribution.png", dpi=300)
    plt.close(fig)

    # 6. HDD vs SSD recommendation breakdown
    hdd_files = df_full[df_full["current_tier"] == config.TIER_HDD_ENCODING]
    ssd_files = df_full[df_full["current_tier"] == config.TIER_SSD_ENCODING]
    hdd_migrate = (hdd_files["decision"] == DECISION_MIGRATE).sum()
    hdd_keep = len(hdd_files) - hdd_migrate

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(["Migrate to SSD", "Keep on HDD"], [hdd_migrate, hdd_keep],
                color=["#27ae60", "#e74c3c"], edgecolor="black", alpha=0.85)
    axes[0].set_title("HDD Files: Migration Recommendations", fontweight="bold")
    axes[0].set_ylabel("Count")
    for i, v in enumerate([hdd_migrate, hdd_keep]):
        axes[0].text(i, v + 0.5, str(v), ha="center", fontweight="bold")

    axes[1].bar(["Already on SSD"], [len(ssd_files)], color=["#3498db"], edgecolor="black", alpha=0.85)
    axes[1].set_title("SSD Files: Status", fontweight="bold")
    axes[1].set_ylabel("Count")
    axes[1].text(0, len(ssd_files) + 0.5, str(len(ssd_files)), ha="center", fontweight="bold")

    fig.suptitle("Tier-Level Recommendation Breakdown", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase9_6_hdd_ssd_recommendations.png", dpi=300)
    plt.close(fig)

    print("[VIZ] All 6 Phase 7-10 visualizations saved to evaluation/figures/")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    # 1. Run unit tests
    test_passed, test_failed, test_total = run_tests()

    # 2. Run full pipeline on ALL test timestamps
    print("\n" + "=" * 80)
    print("FULL TEST-SET PIPELINE (All test timestamps)")
    print("=" * 80)

    df_test = pd.read_csv(config.TEST_DATASET_PATH)
    print(f"Loaded {len(df_test)} test rows across {df_test['prediction_timestamp'].nunique()} timestamps.")

    df_full = run_decision_pipeline(df_test)
    full_summary = generate_decision_summary(df_full)

    print(f"\n--- Full Test-Set Decision Summary ---")
    for k, v in full_summary.items():
        print(f"  {k:<20}: {v}")

    # 3. Single-timestamp detailed view
    result_single, summary_single = run_integrated_pipeline()

    # 4. Visualizations
    generate_visualizations(df_full)

    # 5. Final verification report
    print("\n" + "=" * 80)
    print("FINAL VERIFICATION REPORT — PHASES 7–10")
    print("=" * 80)
    print(f"  PHASE 7  Heat-map engine implemented             : YES")
    print(f"  PHASE 8  Trained ML model loaded from disk        : YES")
    print(f"           ML predictions generated                 : YES")
    print(f"           Hotspot classification generated          : YES")
    print(f"  PHASE 9  Migration decision engine implemented    : YES")
    print(f"  PHASE 10 Payback calculation implemented          : YES")
    print(f"  Actual file migration performed                   : NO")
    print(f"  Feedback mechanism implemented                    : NO")
    print(f"  Dashboard implemented                             : NO")
    print(f"  Unit tests: {test_passed}/{test_total} passed, {test_failed} failed")

    print(f"\n--- Sample Decision Table ---")
    avail_cols = [c for c in DECISION_TABLE_COLUMNS if c in result_single.columns]
    sample = result_single.nlargest(10, "heat_score")
    for _, row in sample.iterrows():
        tier_str = "SSD" if int(row["current_tier"]) == 1 else "HDD"
        hot_str = "YES" if int(row["predicted_hotspot"]) == 1 else "NO"
        print(
            f"  {row['file_id']:<12} Tier={tier_str:<4} "
            f"Heat={row['heat_score']:.4f} "
            f"PredAcc={row['predicted_future_access_count']:>6.2f} "
            f"Hot={hot_str:<4} "
            f"Cost={row['migration_cost_sec']:.4f}s "
            f"Benefit={row['expected_benefit_sec']:.6f}s "
            f"Payback={row['payback_ratio']:.4f} "
            f"-> {row['decision']}"
        )
    print("=" * 80)
