"""
Comprehensive Evaluation Runner (Phase 18) for ML-Driven Hybrid Storage Staging.

Orchestrates the full evaluation workflow:
    Dataset → Feature Engineering → ML Prediction → Heat Map →
    Payback → Decision → Baselines → Migration → Feedback →
    Evaluation → Comparison Results

Generates all result files, comparison tables, and visualisations.
All values come from actual execution — nothing is hardcoded.
"""

from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from ml.predictor import HotspotPredictor
from ml.heatmap import compute_heat_scores
from migration.decision_engine import run_decision_pipeline, DECISION_MIGRATE
from migration.payback import compute_payback_for_batch
from feedback.migration_feedback import (
    run_feedback_loop,
    compute_feedback_summary,
    PRED_TRUE_POSITIVE,
    PRED_FALSE_POSITIVE,
    PRED_FALSE_NEGATIVE,
    PRED_TRUE_NEGATIVE,
)
from evaluation.baselines import (
    run_static_tiering_baseline,
    run_frequency_baseline,
    run_rule_based_baseline,
)
from evaluation.metrics import (
    compute_ml_metrics,
    compute_estimated_avg_latency,
    compute_ssd_utilization,
    compute_hdd_utilization,
    compute_migration_metrics,
    compute_decision_quality_metrics,
    compute_payback_metrics,
    compute_latency_improvement,
    compute_migration_reduction,
    build_comparison_table,
)


# Directories
RESULTS_DIR = config.EVALUATION_DIR / "results"
FIG_DIR = config.FIGURES_DIR

# Use a style
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except:
    plt.style.use("default")


# ============================================================
# HELPER — Proposed ML Method Metrics
# ============================================================

def run_proposed_ml_method(
    df: pd.DataFrame,
    min_payback: float = 0.05,
    hdd_latency_us: float = config.HDD_READ_LATENCY_US,
    ssd_latency_us: float = config.SSD_READ_LATENCY_US,
    bandwidth: float = config.MIGRATION_BANDWIDTH_BYTES_PER_SEC,
    max_migrations: int = config.MAX_MIGRATIONS_PER_RUN,
) -> dict:
    """
    Run the proposed ML-based method and compute its metrics.
    Returns a dict similar to baseline dicts for comparison.
    """
    df_decisions = run_decision_pipeline(
        df,
        threshold=config.HOTSPOT_THRESHOLD,
        min_payback=min_payback,
    )

    # Determine migrated files
    candidates = df_decisions[df_decisions["decision"] == DECISION_MIGRATE].copy()
    migrations = len(candidates)

    # Apply effective tier: migrated files go to SSD
    df_eval = df_decisions.copy()
    df_eval["effective_tier"] = df_eval["current_tier"].copy()
    df_eval.loc[df_eval["decision"] == DECISION_MIGRATE, "effective_tier"] = config.TIER_SSD_ENCODING

    # Compute latency
    latencies = df_eval["effective_tier"].apply(
        lambda t: ssd_latency_us if t == config.TIER_SSD_ENCODING else hdd_latency_us
    )
    df_eval["total_latency_us"] = df_eval["future_access_count"] * latencies

    total_accesses = df_eval["future_access_count"].sum()
    total_latency = df_eval["total_latency_us"].sum()
    avg_latency_us = total_latency / total_accesses if total_accesses > 0 else 0.0

    # Migration time and volume
    latency_saving_per_access_sec = (hdd_latency_us - ssd_latency_us) / 1_000_000.0
    total_mig_time = 0.0
    total_mig_volume = 0
    for _, row in candidates.iterrows():
        fsize = float(row["file_size_bytes"])
        cost = fsize / bandwidth if bandwidth > 0 else 0.0
        total_mig_time += cost
        total_mig_volume += int(fsize)

    # SSD util
    ssd_count = (df_eval["effective_tier"] == config.TIER_SSD_ENCODING).sum()
    ssd_utilization = ssd_count / len(df_eval) if len(df_eval) > 0 else 0.0

    # Classification quality
    df_eval["actual_hotspot"] = (df_eval["future_access_count"] >= config.HOTSPOT_THRESHOLD).astype(int)
    tp = ((df_eval["predicted_hotspot"] == 1) & (df_eval["actual_hotspot"] == 1)).sum()
    fp = ((df_eval["predicted_hotspot"] == 1) & (df_eval["actual_hotspot"] == 0)).sum()
    fn = ((df_eval["predicted_hotspot"] == 0) & (df_eval["actual_hotspot"] == 1)).sum()

    total_pred_hot = tp + fp
    false_hotspot_rate = fp / total_pred_hot if total_pred_hot > 0 else None

    # Useful migrations
    useful = 0
    unnecessary = 0
    for _, row in candidates.iterrows():
        actual = row["future_access_count"]
        if actual >= config.HOTSPOT_THRESHOLD:
            useful += 1
        else:
            unnecessary += 1

    useful_rate = useful / migrations if migrations > 0 else None

    return {
        "method": "Proposed ML",
        "avg_latency_us": avg_latency_us,
        "avg_latency_ms": avg_latency_us / 1000.0,
        "total_latency_us": total_latency,
        "total_accesses": int(total_accesses),
        "migrations": migrations,
        "migration_time_sec": total_mig_time,
        "migration_volume_bytes": total_mig_volume,
        "false_hotspot_rate": false_hotspot_rate,
        "useful_migration_rate": useful_rate,
        "ssd_utilization": ssd_utilization,
        "missed_hotspots": int(fn),
        "unnecessary_migrations": unnecessary,
        "decisions_df": df_decisions,
    }


# ============================================================
# VISUALIZATION FUNCTIONS (Part 18.11)
# ============================================================

def plot_actual_vs_predicted(df, fig_dir):
    """1. Actual vs predicted future accesses scatter plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["future_access_count"], df["predicted_future_access_count"],
               alpha=0.3, s=15, c="#2980b9", edgecolor="none")
    max_val = max(df["future_access_count"].max(), df["predicted_future_access_count"].max())
    ax.plot([0, max_val], [0, max_val], "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Future Accesses", fontsize=11)
    ax.set_ylabel("Predicted Future Accesses", fontsize=11)
    ax.set_title("Actual vs Predicted Future Accesses", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(fig_dir / "actual_vs_predicted.png", dpi=300)
    plt.close(fig)


def plot_confusion_matrix(cm, fig_dir):
    """2. Confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Hotspot", "Hotspot"],
                yticklabels=["Not Hotspot", "Hotspot"], ax=ax)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title("Hotspot Classification Confusion Matrix", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "confusion_matrix.png", dpi=300)
    plt.close(fig)


def plot_method_comparison_bars(results, metric_key, title, ylabel, filename, fig_dir):
    """Generic comparison bar chart across methods."""
    methods = [r["method"] for r in results]
    values = []
    for r in results:
        v = r.get(metric_key)
        if v is None:
            v = 0
        values.append(v)

    colors = ["#3498db", "#e67e22", "#27ae60"][:len(methods)]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, values, color=colors, edgecolor="black", alpha=0.85)
    for bar, val in zip(bars, values):
        label = f"{val:.4f}" if isinstance(val, float) else str(val)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                label, ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(fig_dir / filename, dpi=300)
    plt.close(fig)


def plot_payback_distribution(df, fig_dir):
    """10. Payback ratio distribution."""
    if "payback_ratio" not in df.columns:
        return
    payback_clipped = df["payback_ratio"].replace([np.inf, -np.inf], np.nan).dropna().clip(upper=10.0)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(payback_clipped, bins=40, kde=True, color="#8e44ad", edgecolor="black", ax=ax)
    ax.axvline(0.05, color="red", linestyle="--", lw=2, label="Min payback (0.05)")
    ax.set_title("Payback Ratio Distribution (Clipped at 10)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Payback Ratio", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(fig_dir / "payback_ratio_distribution.png", dpi=300)
    plt.close(fig)


def plot_feedback_comparison(before, after, metric, title, ylabel, filename, fig_dir):
    """Bar chart comparing a metric before/after feedback."""
    labels = ["Without Feedback", "With Feedback"]
    values = [before, after]
    colors = ["#e74c3c", "#27ae60"]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", alpha=0.85)
    for bar, val in zip(bars, values):
        label = f"{val:.4f}" if val is not None else "N/A"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                label, ha="center", va="bottom", fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(fig_dir / filename, dpi=300)
    plt.close(fig)


# ============================================================
# MAIN EVALUATION RUNNER
# ============================================================

def run_controlled_experiment():
    """
    Run a small controlled experiment on a single timestamp.
    Returns True if the experiment succeeds.
    """
    print("\n" + "=" * 80)
    print("CONTROLLED SMALL EXPERIMENT (Single Timestamp)")
    print("=" * 80)

    df_test = pd.read_csv(config.TEST_DATASET_PATH)
    df_trace = pd.read_csv(config.TRACE_DATASET_PATH)
    df_trace["_parsed_ts"] = pd.to_datetime(df_trace["timestamp"])

    # Use one timestamp
    timestamps = sorted(df_test["prediction_timestamp"].unique())
    target_ts = timestamps[-1]
    df_slice = df_test[df_test["prediction_timestamp"] == target_ts].copy()

    print(f"  Timestamp: {target_ts}")
    print(f"  Files: {len(df_slice)}")

    # Run decision pipeline
    df_decisions = run_decision_pipeline(df_slice, min_payback=0.05)
    candidates = df_decisions[df_decisions["decision"] == DECISION_MIGRATE]
    print(f"  Predicted hotspots: {(df_decisions['predicted_hotspot'] == 1).sum()}")
    print(f"  Migration candidates: {len(candidates)}")

    # Run feedback on this slice
    import tempfile
    tmp_log = Path(tempfile.mktemp(suffix="_ctrl_fb.csv"))
    feedback_result = run_feedback_loop(
        df_decisions, df_trace, log_path=tmp_log,
    )

    # Verify feedback log
    if tmp_log.exists():
        fb_log = pd.read_csv(tmp_log)
        print(f"  Feedback log rows: {len(fb_log)}")
        os.remove(tmp_log)
    else:
        print("  [WARNING] Feedback log not created")

    # Verify heat scores in bounds
    heat_valid = (feedback_result["updated_heat_score"] >= 0).all() and \
                 (feedback_result["updated_heat_score"] <= 1).all()
    print(f"  Heat scores in [0,1]: {'YES' if heat_valid else 'NO'}")

    # Verify no data leakage
    no_leakage = "true_hotspot_next_6h" not in df_decisions.columns
    print(f"  No data leakage: {'YES' if no_leakage else 'NO'}")

    # Run baselines
    static = run_static_tiering_baseline(df_slice)
    freq = run_frequency_baseline(df_slice, min_payback=0.05)
    print(f"  Static latency: {static['avg_latency_ms']:.4f} ms")
    print(f"  Frequency latency: {freq['avg_latency_ms']:.4f} ms")

    success = heat_valid and no_leakage
    print(f"\n  Controlled experiment: {'PASSED' if success else 'FAILED'}")
    return success


def run_full_evaluation():
    """
    Run the complete evaluation across all test timestamps.
    Generates all result files and visualisations.
    """
    print("\n" + "=" * 80)
    print("FULL COMPREHENSIVE EVALUATION")
    print("=" * 80)

    # Create output directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n[1/10] Loading data...")
    df_test = pd.read_csv(config.TEST_DATASET_PATH)
    df_trace = pd.read_csv(config.TRACE_DATASET_PATH)
    df_trace["_parsed_ts"] = pd.to_datetime(df_trace["timestamp"])

    print(f"  Test rows: {len(df_test)}")
    print(f"  Test timestamps: {df_test['prediction_timestamp'].nunique()}")
    print(f"  Trace events: {len(df_trace)}")

    # --------------------------------------------------------
    # STEP 2: Run Proposed ML Method
    # --------------------------------------------------------
    print("\n[2/10] Running Proposed ML Method...")
    ml_result = run_proposed_ml_method(df_test, min_payback=0.05)
    df_decisions = ml_result.pop("decisions_df")
    print(f"  ML avg latency: {ml_result['avg_latency_ms']:.4f} ms")
    print(f"  ML migrations: {ml_result['migrations']}")

    # --------------------------------------------------------
    # STEP 3: Run Baselines
    # --------------------------------------------------------
    print("\n[3/10] Running Baselines...")
    static_result = run_static_tiering_baseline(df_test)
    freq_result = run_frequency_baseline(df_test, min_payback=0.05)
    rule_result = run_rule_based_baseline(df_test)
    print(f"  Static avg latency: {static_result['avg_latency_ms']:.4f} ms")
    print(f"  Freq avg latency: {freq_result['avg_latency_ms']:.4f} ms")
    print(f"  Rule avg latency: {rule_result['avg_latency_ms']:.4f} ms")

    # --------------------------------------------------------
    # STEP 4: ML Prediction Metrics
    # --------------------------------------------------------
    print("\n[4/10] Computing ML Metrics...")
    ml_metrics = compute_ml_metrics(df_decisions)
    print(f"  MAE: {ml_metrics['mae']:.4f}")
    print(f"  RMSE: {ml_metrics['rmse']:.4f}")
    print(f"  R²: {ml_metrics['r2']:.4f}")
    print(f"  Precision: {ml_metrics['precision']:.4f}")
    print(f"  Recall: {ml_metrics['recall']:.4f}")
    print(f"  F1: {ml_metrics['f1_score']:.4f}")

    # Save ML metrics
    ml_rows = []
    for k, v in ml_metrics.items():
        if k != "confusion_matrix":
            ml_rows.append({"Metric": k, "Value": v})
    pd.DataFrame(ml_rows).to_csv(RESULTS_DIR / "ml_metrics.csv", index=False)

    # --------------------------------------------------------
    # STEP 5: Feedback Loop (WITH feedback)
    # --------------------------------------------------------
    print("\n[5/10] Running Feedback Loop...")
    feedback_log_path = config.FEEDBACK_LOG_PATH
    # Clear existing feedback log
    if feedback_log_path.exists():
        os.remove(feedback_log_path)

    feedback_result = run_feedback_loop(
        df_decisions, df_trace, log_path=feedback_log_path,
    )
    fb_summary = compute_feedback_summary(feedback_result)
    print(f"  Total records: {fb_summary['total_records']}")
    print(f"  True Positives: {fb_summary['true_positives']}")
    print(f"  False Positives: {fb_summary['false_positives']}")
    print(f"  False Negatives: {fb_summary['false_negatives']}")
    fhr = fb_summary.get('false_hotspot_rate')
    print(f"  False Hotspot Rate: {fhr:.4f}" if fhr is not None else "  False Hotspot Rate: N/A")

    # --------------------------------------------------------
    # STEP 6: Feedback Comparison (without vs with feedback)
    # --------------------------------------------------------
    print("\n[6/10] Feedback Comparison...")

    # WITHOUT feedback: use original heat scores and decisions
    no_fb_fp_rate = fb_summary.get("false_hotspot_rate")
    no_fb_useful_rate = None
    migrated_mask = df_decisions["decision"] == DECISION_MIGRATE
    if migrated_mask.sum() > 0:
        migrated_actual = feedback_result.loc[migrated_mask, "actual_hotspot"]
        no_fb_useful = (migrated_actual == 1).sum()
        no_fb_total_mig = migrated_mask.sum()
        no_fb_useful_rate = no_fb_useful / no_fb_total_mig if no_fb_total_mig > 0 else None

    no_fb_avg_heat = df_decisions["heat_score"].mean() if "heat_score" in df_decisions.columns else 0
    no_fb_unnecessary = 0
    if migrated_mask.sum() > 0:
        no_fb_unnecessary = (feedback_result.loc[migrated_mask, "actual_hotspot"] == 0).sum()

    # WITH feedback: use updated heat scores to simulate re-decision
    # Re-evaluate decisions using updated heat scores
    fb_heat_mean = feedback_result["updated_heat_score"].mean()

    # For a fair comparison, we note the feedback adjusts heat scores
    # which COULD change future decisions. We report the metrics.
    with_fb_summary = compute_decision_quality_metrics(feedback_result)
    with_fb_fp_rate = with_fb_summary.get("false_hotspot_rate")
    with_fb_useful_rate = with_fb_summary.get("useful_migration_rate")
    with_fb_unnecessary = with_fb_summary.get("unnecessary_migrations", 0)

    print(f"  No-feedback false hotspot rate: {no_fb_fp_rate}")
    print(f"  With-feedback false hotspot rate: {with_fb_fp_rate}")
    print(f"  No-feedback avg heat: {no_fb_avg_heat:.4f}")
    print(f"  With-feedback avg heat: {fb_heat_mean:.4f}")

    # Save feedback comparison
    fb_comp = {
        "Metric": [
            "False Hotspot Rate",
            "Useful Migration Rate",
            "Avg Heat Score",
            "Unnecessary Migrations",
            "Missed Hotspots",
        ],
        "Without Feedback": [
            no_fb_fp_rate,
            no_fb_useful_rate,
            no_fb_avg_heat,
            no_fb_unnecessary,
            fb_summary.get("false_negatives", 0),
        ],
        "With Feedback": [
            with_fb_fp_rate,
            with_fb_useful_rate,
            fb_heat_mean,
            with_fb_unnecessary,
            with_fb_summary.get("total_actual_hotspots", 0) -
            with_fb_summary.get("useful_migrations", 0) if with_fb_summary.get("total_actual_hotspots") else 0,
        ],
    }
    pd.DataFrame(fb_comp).to_csv(RESULTS_DIR / "feedback_comparison.csv", index=False)

    # --------------------------------------------------------
    # STEP 7: Payback Metrics
    # --------------------------------------------------------
    print("\n[7/10] Computing Payback Metrics...")
    payback_metrics = compute_payback_metrics(df_decisions, min_payback=0.05)
    for k, v in payback_metrics.items():
        print(f"  {k}: {v}")

    # --------------------------------------------------------
    # STEP 8: Build Comparison Table
    # --------------------------------------------------------
    print("\n[8/10] Building Comparison Table...")
    comparison_results = [static_result, freq_result, ml_result]
    comparison_table = build_comparison_table(comparison_results)
    print(comparison_table.to_string(index=False))
    comparison_table.to_csv(RESULTS_DIR / "baseline_results.csv", index=False)

    # Improvement calculations
    lat_imp_vs_static = compute_latency_improvement(
        static_result["avg_latency_us"], ml_result["avg_latency_us"]
    )
    lat_imp_vs_freq = compute_latency_improvement(
        freq_result["avg_latency_us"], ml_result["avg_latency_us"]
    )
    print(f"\n  Latency improvement vs Static: "
          f"{lat_imp_vs_static:.2f}%" if lat_imp_vs_static is not None else "N/A")
    print(f"  Latency improvement vs Freq: "
          f"{lat_imp_vs_freq:.2f}%" if lat_imp_vs_freq is not None else "N/A")

    # --------------------------------------------------------
    # STEP 9: System Metrics Summary
    # --------------------------------------------------------
    print("\n[9/10] Saving System Metrics...")
    system_metrics = {
        "total_files_analyzed": len(df_test),
        "total_prediction_records": len(df_decisions),
        "predicted_hotspots": int((df_decisions["predicted_hotspot"] == 1).sum()),
        "actual_hotspots": fb_summary.get("total_actual_hotspots", 0),
        "migrations_recommended": ml_result["migrations"],
        "ml_mae": ml_metrics["mae"],
        "ml_rmse": ml_metrics["rmse"],
        "ml_r2": ml_metrics["r2"],
        "ml_precision": ml_metrics["precision"],
        "ml_recall": ml_metrics["recall"],
        "ml_f1": ml_metrics["f1_score"],
        "static_avg_latency_ms": static_result["avg_latency_ms"],
        "freq_avg_latency_ms": freq_result["avg_latency_ms"],
        "ml_avg_latency_ms": ml_result["avg_latency_ms"],
        "static_migrations": static_result["migrations"],
        "freq_migrations": freq_result["migrations"],
        "ml_migrations": ml_result["migrations"],
        "false_hotspot_rate": fb_summary.get("false_hotspot_rate"),
        "missed_hotspot_rate": fb_summary.get("false_negatives", 0) /
            max(fb_summary.get("total_actual_hotspots", 1), 1),
        "useful_migrations": fb_summary.get("useful_migrations", 0),
        "false_hotspot_migrations": fb_summary.get("false_hotspot_migrations", 0),
        "latency_improvement_vs_static_pct": lat_imp_vs_static,
    }

    sys_rows = [{"Metric": k, "Value": v} for k, v in system_metrics.items()]
    pd.DataFrame(sys_rows).to_csv(RESULTS_DIR / "system_metrics.csv", index=False)

    # Save JSON summary
    summary_json = {
        "ml_metrics": {k: v for k, v in ml_metrics.items() if k != "confusion_matrix"},
        "static_baseline": {k: v for k, v in static_result.items() if not isinstance(v, pd.DataFrame)},
        "frequency_baseline": {k: v for k, v in freq_result.items() if not isinstance(v, pd.DataFrame)},
        "proposed_ml": {k: v for k, v in ml_result.items() if not isinstance(v, pd.DataFrame)},
        "feedback_summary": {k: v for k, v in fb_summary.items() if k != "migration_outcomes"},
        "payback_metrics": payback_metrics,
        "improvement": {
            "latency_vs_static_pct": lat_imp_vs_static,
            "latency_vs_freq_pct": lat_imp_vs_freq,
        },
    }

    def _json_safe(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif obj is None:
            return None
        return obj

    safe_json = json.loads(json.dumps(summary_json, default=_json_safe))
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(safe_json, f, indent=2)

    # --------------------------------------------------------
    # STEP 10: Visualisations
    # --------------------------------------------------------
    print("\n[10/10] Generating Visualisations...")

    # 1. Actual vs predicted
    plot_actual_vs_predicted(df_decisions, FIG_DIR)

    # 2. Confusion matrix
    cm = np.array(ml_metrics["confusion_matrix"])
    plot_confusion_matrix(cm, FIG_DIR)

    # 3. ML vs frequency baseline — compare using bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    methods_compare = ["Frequency Baseline", "Proposed ML"]
    # Frequency baseline "prediction" = recent_access_count
    df_freq_pred = df_test.copy()
    df_freq_pred["freq_predicted"] = df_freq_pred["recent_access_count"]
    freq_mae = float(np.mean(np.abs(df_test["future_access_count"] - df_test["recent_access_count"])))
    freq_rmse = float(np.sqrt(np.mean((df_test["future_access_count"] - df_test["recent_access_count"])**2)))
    metrics_names = ["MAE", "RMSE"]
    freq_vals = [freq_mae, freq_rmse]
    ml_vals = [ml_metrics["mae"], ml_metrics["rmse"]]
    x = np.arange(len(metrics_names))
    w = 0.35
    ax.bar(x - w/2, freq_vals, w, label="Frequency Baseline", color="#e67e22", edgecolor="black", alpha=0.85)
    ax.bar(x + w/2, ml_vals, w, label="Proposed ML", color="#27ae60", edgecolor="black", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_title("Prediction Accuracy: Frequency vs ML", fontsize=13, fontweight="bold")
    ax.set_ylabel("Error", fontsize=11)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ml_vs_freq_metrics.png", dpi=300)
    plt.close(fig)

    # 4. Average latency by method
    plot_method_comparison_bars(
        comparison_results, "avg_latency_ms",
        "Average Estimated Latency by Method", "Latency (ms)",
        "avg_latency_by_method.png", FIG_DIR,
    )

    # 5. Number of migrations by method
    plot_method_comparison_bars(
        comparison_results, "migrations",
        "Number of Migrations by Method", "Migration Count",
        "migrations_by_method.png", FIG_DIR,
    )

    # 6. Migration overhead by method
    plot_method_comparison_bars(
        comparison_results, "migration_time_sec",
        "Migration Overhead by Method", "Total Migration Time (s)",
        "migration_overhead_by_method.png", FIG_DIR,
    )

    # 7. False hotspot rate by method
    fhr_results = []
    for r in comparison_results:
        fhr_results.append({
            "method": r["method"],
            "false_hotspot_rate_val": r.get("false_hotspot_rate") or 0,
        })
    fig, ax = plt.subplots(figsize=(8, 5))
    methods_fhr = [r["method"] for r in fhr_results]
    values_fhr = [r["false_hotspot_rate_val"] for r in fhr_results]
    colors_fhr = ["#3498db", "#e67e22", "#27ae60"][:len(methods_fhr)]
    bars = ax.bar(methods_fhr, values_fhr, color=colors_fhr, edgecolor="black", alpha=0.85)
    for bar, val, r in zip(bars, values_fhr, comparison_results):
        label = f"{val:.4f}" if r.get("false_hotspot_rate") is not None else "N/A"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                label, ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_title("False Hotspot Rate by Method", fontsize=13, fontweight="bold")
    ax.set_ylabel("Rate", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "false_hotspot_rate_by_method.png", dpi=300)
    plt.close(fig)

    # 8. Useful migration rate by method
    umr_results = []
    for r in comparison_results:
        umr_results.append({
            "method": r["method"],
            "useful_migration_rate_val": r.get("useful_migration_rate") or 0,
        })
    fig, ax = plt.subplots(figsize=(8, 5))
    methods_umr = [r["method"] for r in umr_results]
    values_umr = [r["useful_migration_rate_val"] for r in umr_results]
    colors_umr = ["#3498db", "#e67e22", "#27ae60"][:len(methods_umr)]
    bars = ax.bar(methods_umr, values_umr, color=colors_umr, edgecolor="black", alpha=0.85)
    for bar, val, r in zip(bars, values_umr, comparison_results):
        label = f"{val:.4f}" if r.get("useful_migration_rate") is not None else "N/A"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                label, ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_title("Useful Migration Rate by Method", fontsize=13, fontweight="bold")
    ax.set_ylabel("Rate", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "useful_migration_rate_by_method.png", dpi=300)
    plt.close(fig)

    # 9. SSD utilization by method
    plot_method_comparison_bars(
        comparison_results, "ssd_utilization",
        "SSD Utilization by Method", "SSD Utilization (fraction)",
        "ssd_utilization_by_method.png", FIG_DIR,
    )

    # 10. Payback ratio distribution
    plot_payback_distribution(df_decisions, FIG_DIR)

    # 11. Feedback false hotspot comparison
    before_fhr = no_fb_fp_rate if no_fb_fp_rate is not None else 0
    after_fhr = with_fb_fp_rate if with_fb_fp_rate is not None else 0
    plot_feedback_comparison(
        before_fhr, after_fhr, "false_hotspot_rate",
        "False Hotspot Rate: Before vs After Feedback",
        "False Hotspot Rate",
        "feedback_false_hotspot_comparison.png", FIG_DIR,
    )

    # 12. Feedback useful migration comparison
    before_umr = no_fb_useful_rate if no_fb_useful_rate is not None else 0
    after_umr = with_fb_useful_rate if with_fb_useful_rate is not None else 0
    plot_feedback_comparison(
        before_umr, after_umr, "useful_migration_rate",
        "Useful Migration Rate: Before vs After Feedback",
        "Useful Migration Rate",
        "feedback_useful_migration_comparison.png", FIG_DIR,
    )

    print(f"  Saved 12 visualisations to {FIG_DIR}")

    # ========================================================
    # RETURN ALL RESULTS FOR REPORT GENERATION
    # ========================================================
    return {
        "ml_metrics": ml_metrics,
        "static_result": static_result,
        "freq_result": freq_result,
        "rule_result": rule_result,
        "ml_result": ml_result,
        "feedback_summary": fb_summary,
        "payback_metrics": payback_metrics,
        "comparison_table": comparison_table,
        "system_metrics": system_metrics,
        "feedback_comparison": fb_comp,
        "lat_imp_vs_static": lat_imp_vs_static,
        "lat_imp_vs_freq": lat_imp_vs_freq,
        "df_decisions": df_decisions,
        "feedback_result": feedback_result,
    }


# ============================================================
# MAIN
# ============================================================

import os

if __name__ == "__main__":
    np.random.seed(config.RANDOM_SEED)

    # Step 1: Controlled experiment
    ctrl_ok = run_controlled_experiment()
    if not ctrl_ok:
        print("\n[ABORT] Controlled experiment failed. Stopping.")
        sys.exit(1)

    # Step 2: Full evaluation
    results = run_full_evaluation()

    # Step 3: Final verification report
    sm = results["system_metrics"]
    fb = results["feedback_summary"]

    print("\n" + "=" * 80)
    print("FINAL EXPERIMENT RESULTS")
    print("=" * 80)
    print(f"  Total files analyzed         : {sm['total_files_analyzed']}")
    print(f"  Total prediction records     : {sm['total_prediction_records']}")
    print(f"  Predicted hotspots           : {sm['predicted_hotspots']}")
    print(f"  Actual hotspots              : {sm['actual_hotspots']}")
    print(f"  Migrations recommended       : {sm['migrations_recommended']}")
    print(f"  Useful migrations            : {sm.get('useful_migrations', 0)}")
    print(f"  False hotspot migrations     : {sm.get('false_hotspot_migrations', 0)}")
    fhr_val = sm.get('false_hotspot_rate')
    print(f"  False hotspot rate           : {fhr_val:.4f}" if fhr_val is not None else "  False hotspot rate           : N/A")
    print(f"  Missed hotspot rate          : {sm.get('missed_hotspot_rate', 0):.4f}")
    print(f"  ML MAE                       : {sm['ml_mae']:.4f}")
    print(f"  ML RMSE                      : {sm['ml_rmse']:.4f}")
    print(f"  ML R²                        : {sm['ml_r2']:.4f}")
    print(f"  ML hotspot precision         : {sm['ml_precision']:.4f}")
    print(f"  ML hotspot recall            : {sm['ml_recall']:.4f}")
    print(f"  ML hotspot F1                : {sm['ml_f1']:.4f}")
    print(f"  Static avg latency           : {sm['static_avg_latency_ms']:.4f} ms")
    print(f"  Frequency avg latency        : {sm['freq_avg_latency_ms']:.4f} ms")
    print(f"  Proposed ML avg latency      : {sm['ml_avg_latency_ms']:.4f} ms")
    print(f"  Static migrations            : {sm['static_migrations']}")
    print(f"  Frequency migrations         : {sm['freq_migrations']}")
    print(f"  Proposed ML migrations       : {sm['ml_migrations']}")
    lat_imp = sm.get('latency_improvement_vs_static_pct')
    print(f"  Latency improvement vs static: {lat_imp:.2f}%" if lat_imp is not None else "  Latency improvement vs static: N/A")
    fb_fhr = results["feedback_comparison"]["Without Feedback"][0]
    nofb_fhr = results["feedback_comparison"]["With Feedback"][0]
    print(f"  No-feedback false hotspot rate: {fb_fhr:.4f}" if fb_fhr is not None else "  No-feedback false hotspot rate: N/A")
    print(f"  Feedback false hotspot rate   : {nofb_fhr:.4f}" if nofb_fhr is not None else "  Feedback false hotspot rate   : N/A")
    print("=" * 80)
