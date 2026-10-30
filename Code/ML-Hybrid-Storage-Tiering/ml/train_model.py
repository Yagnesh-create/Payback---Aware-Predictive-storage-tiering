"""
Model Training Script for ML-Driven Hybrid Storage Staging and Hotspot Migration.

Trains a genuine LinearRegression model on the prepared training data,
evaluates regression and classification metrics, compares against a
frequency-based baseline, and saves the model artifact.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from ml.data_preparation import (
    load_ml_dataset,
    perform_temporal_split,
    extract_X_y,
    scale_features,
)


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def compute_regression_metrics(y_true, y_pred, split_name: str) -> dict:
    """Compute MAE, RMSE, R² and print them."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n  {split_name.upper()} REGRESSION RESULTS")
    print(f"    MAE  : {mae:.4f}")
    print(f"    RMSE : {rmse:.4f}")
    print(f"    R²   : {r2:.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def compute_classification_metrics(
    y_true_count, y_pred_count, threshold: int, split_name: str, label: str = "ML Model"
) -> dict:
    """Derive hotspot binary labels from access counts and evaluate."""
    actual_hot = (y_true_count >= threshold).astype(int)
    pred_hot = (y_pred_count >= threshold).astype(int)

    acc = accuracy_score(actual_hot, pred_hot)
    prec = precision_score(actual_hot, pred_hot, zero_division=0)
    rec = recall_score(actual_hot, pred_hot, zero_division=0)
    f1 = f1_score(actual_hot, pred_hot, zero_division=0)
    cm = confusion_matrix(actual_hot, pred_hot)

    print(f"\n  {split_name.upper()} HOTSPOT CLASSIFICATION ({label}) [threshold >= {threshold}]")
    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1-Score  : {f1:.4f}")
    print(f"    Confusion Matrix:")
    print(f"      TN={cm[0,0]:>5}  FP={cm[0,1]:>5}")
    print(f"      FN={cm[1,0]:>5}  TP={cm[1,1]:>5}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
    }


# ================================================================
# MAIN TRAINING PIPELINE
# ================================================================

def train_and_evaluate():
    """Full training, evaluation, baseline comparison, and model saving pipeline."""

    print("=" * 80)
    print("PHASE 6: MACHINE LEARNING MODEL TRAINING")
    print("=" * 80)

    threshold = config.HOTSPOT_THRESHOLD
    FIG_DIR = config.FIGURES_DIR
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # STEP 1: Load and prepare data
    # ----------------------------------------------------------
    print("\n[STEP 1] Loading prepared split datasets...")
    df_train = pd.read_csv(config.TRAIN_DATASET_PATH)
    df_val = pd.read_csv(config.VAL_DATASET_PATH)
    df_test = pd.read_csv(config.TEST_DATASET_PATH)

    print(f"  Train rows : {len(df_train)}")
    print(f"  Val rows   : {len(df_val)}")
    print(f"  Test rows  : {len(df_test)}")

    # ----------------------------------------------------------
    # STEP 2: Extract X and y
    # ----------------------------------------------------------
    print("\n[STEP 2] Extracting feature matrices and target vectors...")

    feature_cols = config.FEATURE_COLUMNS
    target_col = config.TARGET_COLUMN

    X_train_raw = df_train[feature_cols].copy()
    y_train = df_train[target_col].copy()

    X_val_raw = df_val[feature_cols].copy()
    y_val = df_val[target_col].copy()

    X_test_raw = df_test[feature_cols].copy()
    y_test = df_test[target_col].copy()

    # Explicit leakage guard
    forbidden_cols = [target_col, "true_hotspot_next_6h", "eval_true_hotspot_next_6h"]
    for fc in forbidden_cols:
        for name, X in [("train", X_train_raw), ("val", X_val_raw), ("test", X_test_raw)]:
            assert fc not in X.columns, f"LEAKAGE: '{fc}' found in X_{name}!"

    print(f"  Features ({len(feature_cols)}): {feature_cols}")
    print(f"  Target: {target_col}")
    print("  Leakage check: PASSED")

    # ----------------------------------------------------------
    # STEP 3: Scale features (load existing scaler OR re-fit)
    # ----------------------------------------------------------
    print("\n[STEP 3] Scaling features...")
    if config.SCALER_PATH.exists():
        scaler = joblib.load(config.SCALER_PATH)
        print(f"  Loaded existing fitted scaler from: {config.SCALER_PATH}")
    else:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(X_train_raw)
        joblib.dump(scaler, config.SCALER_PATH)
        print(f"  Fitted new scaler on X_train and saved to: {config.SCALER_PATH}")

    X_train = scaler.transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)
    print("  Scaling applied to train, val, and test splits.")

    # ----------------------------------------------------------
    # STEP 4: Train the model — GENUINE model.fit()
    # ----------------------------------------------------------
    print("\n[STEP 4] Training LinearRegression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("  model.fit(X_train, y_train) EXECUTED SUCCESSFULLY.")

    # ----------------------------------------------------------
    # STEP 5: Generate predictions
    # ----------------------------------------------------------
    print("\n[STEP 5] Generating predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    # Clip negative predictions to 0 (access count cannot be negative)
    y_pred_train = np.maximum(y_pred_train, 0)
    y_pred_val = np.maximum(y_pred_val, 0)
    y_pred_test = np.maximum(y_pred_test, 0)
    print("  Predictions generated and clipped to >= 0.")

    # ----------------------------------------------------------
    # STEP 6: Regression metrics
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("REGRESSION EVALUATION")
    print("=" * 60)
    reg_train = compute_regression_metrics(y_train, y_pred_train, "Training")
    reg_val = compute_regression_metrics(y_val, y_pred_val, "Validation")
    reg_test = compute_regression_metrics(y_test, y_pred_test, "Test")

    # ----------------------------------------------------------
    # STEP 7: Hotspot classification metrics
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"HOTSPOT CLASSIFICATION (threshold >= {threshold} accesses)")
    print("=" * 60)
    cls_train = compute_classification_metrics(
        y_train, y_pred_train, threshold, "Training", "ML Model"
    )
    cls_val = compute_classification_metrics(
        y_val, y_pred_val, threshold, "Validation", "ML Model"
    )
    cls_test = compute_classification_metrics(
        y_test, y_pred_test, threshold, "Test", "ML Model"
    )

    # ----------------------------------------------------------
    # STEP 8: Baseline comparison — Recent Frequency Baseline
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("BASELINE: RECENT FREQUENCY")
    print("=" * 60)
    print("  Rule: Predict hotspot if recent_access_count >= threshold.")
    print("  This baseline uses only the most recent 6h access count")
    print("  as the predicted future_access_count (naive persistence).")

    # Baseline prediction: recent_access_count as the forecast
    baseline_pred_train = df_train["recent_access_count"].values.astype(float)
    baseline_pred_val = df_val["recent_access_count"].values.astype(float)
    baseline_pred_test = df_test["recent_access_count"].values.astype(float)

    base_reg_test = compute_regression_metrics(y_test, baseline_pred_test, "Baseline Test")
    base_cls_test = compute_classification_metrics(
        y_test, baseline_pred_test, threshold, "Baseline Test", "Recent Freq Baseline"
    )

    # ----------------------------------------------------------
    # STEP 9: Model coefficient analysis
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("MODEL COEFFICIENTS (LinearRegression)")
    print("=" * 60)
    print("  NOTE: Coefficients correspond to StandardScaler-normalized features.")
    print("  A positive coefficient means the model learned a positive association")
    print("  between increases in that (scaled) feature and future_access_count.")
    print("  This does NOT imply causation.\n")
    print(f"  {'Feature':<25} {'Coefficient':>12}")
    print("  " + "-" * 40)
    for fname, coef in zip(feature_cols, model.coef_):
        print(f"  {fname:<25} {coef:>12.6f}")
    print(f"\n  {'Intercept':<25} {model.intercept_:>12.6f}")

    # ----------------------------------------------------------
    # STEP 10: Save the trained model
    # ----------------------------------------------------------
    print("\n[STEP 10] Saving trained model...")
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    print(f"  Saved trained LinearRegression to: {config.MODEL_PATH}")

    # ----------------------------------------------------------
    # STEP 11: Reload model and verify predictions
    # ----------------------------------------------------------
    print("\n[STEP 11] Reloading model from disk and verifying predictions...")
    reloaded_model = joblib.load(config.MODEL_PATH)
    reloaded_scaler = joblib.load(config.SCALER_PATH)
    print("  Model and scaler reloaded from disk.")

    # Pick 10 sample rows from test set for demonstration
    sample_indices = np.linspace(0, len(df_test) - 1, 10, dtype=int)
    sample_df = df_test.iloc[sample_indices].copy()
    X_sample = sample_df[feature_cols].values
    X_sample_scaled = reloaded_scaler.transform(X_sample)
    preds_reloaded = np.maximum(reloaded_model.predict(X_sample_scaled), 0)

    print(f"\n  {'File ID':<12} {'Pred. Timestamp':<22} {'Actual':>8} {'Predicted':>10} {'Act.Hot':>8} {'Pred.Hot':>9}")
    print("  " + "-" * 75)
    for i, (idx, row) in enumerate(sample_df.iterrows()):
        actual = int(row[target_col])
        predicted = round(float(preds_reloaded[i]), 2)
        actual_hot = "YES" if actual >= threshold else "NO"
        pred_hot = "YES" if predicted >= threshold else "NO"
        print(f"  {row['file_id']:<12} {row['prediction_timestamp']:<22} {actual:>8} {predicted:>10.2f} {actual_hot:>8} {pred_hot:>9}")

    # ----------------------------------------------------------
    # STEP 12: Generate visualizations
    # ----------------------------------------------------------
    print("\n[STEP 12] Generating evaluation visualizations...")

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot 1: Actual vs Predicted (Test)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(y_test, y_pred_test, alpha=0.35, s=15, color="#2980b9", edgecolor="none")
    max_val = max(y_test.max(), y_pred_test.max())
    ax.plot([0, max_val], [0, max_val], "r--", lw=2, label="Perfect Prediction")
    ax.set_xlabel("Actual Future Access Count", fontsize=11)
    ax.set_ylabel("Predicted Future Access Count", fontsize=11)
    ax.set_title("Actual vs. Predicted Future Access Count (Test Set)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ml_1_actual_vs_predicted.png", dpi=300)
    plt.close(fig)

    # Plot 2: Residual distribution (Test)
    residuals = y_test.values - y_pred_test
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(residuals, bins=50, kde=True, color="#8e44ad", edgecolor="black", ax=ax)
    ax.axvline(0, color="red", linestyle="--", lw=2)
    ax.set_xlabel("Prediction Residual (Actual - Predicted)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Prediction Residual Distribution (Test Set)", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ml_2_residual_distribution.png", dpi=300)
    plt.close(fig)

    # Plot 3: Confusion matrix heatmap (Test)
    cm_test = cls_test["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_test,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Cold (0)", "Hotspot (1)"],
        yticklabels=["Cold (0)", "Hotspot (1)"],
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title(f"Hotspot Confusion Matrix — Test Set (threshold >= {threshold})", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ml_3_confusion_matrix.png", dpi=300)
    plt.close(fig)

    # Plot 4: Feature coefficient bar chart
    coefs = model.coef_
    sorted_idx = np.argsort(np.abs(coefs))[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#27ae60" if c >= 0 else "#c0392b" for c in coefs[sorted_idx]]
    ax.barh(
        [feature_cols[i] for i in sorted_idx],
        coefs[sorted_idx],
        color=colors,
        edgecolor="black",
        alpha=0.85,
    )
    ax.set_xlabel("Coefficient Value (Scaled Features)", fontsize=11)
    ax.set_title("LinearRegression Feature Coefficients", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ml_4_feature_coefficients.png", dpi=300)
    plt.close(fig)

    # Plot 5: Metric comparison across splits
    split_names = ["Train", "Validation", "Test"]
    maes = [reg_train["mae"], reg_val["mae"], reg_test["mae"]]
    rmses = [reg_train["rmse"], reg_val["rmse"], reg_test["rmse"]]
    r2s = [reg_train["r2"], reg_val["r2"], reg_test["r2"]]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    bar_color = ["#3498db", "#e67e22", "#e74c3c"]

    axes[0].bar(split_names, maes, color=bar_color, edgecolor="black", alpha=0.85)
    axes[0].set_title("MAE by Split", fontweight="bold")
    axes[0].set_ylabel("MAE")
    for i, v in enumerate(maes):
        axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold", fontsize=9)

    axes[1].bar(split_names, rmses, color=bar_color, edgecolor="black", alpha=0.85)
    axes[1].set_title("RMSE by Split", fontweight="bold")
    axes[1].set_ylabel("RMSE")
    for i, v in enumerate(rmses):
        axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold", fontsize=9)

    axes[2].bar(split_names, r2s, color=bar_color, edgecolor="black", alpha=0.85)
    axes[2].set_title("R² by Split", fontweight="bold")
    axes[2].set_ylabel("R²")
    for i, v in enumerate(r2s):
        axes[2].text(i, v + 0.005, f"{v:.4f}", ha="center", fontweight="bold", fontsize=9)

    fig.suptitle("Regression Metrics Across Train / Validation / Test Splits", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ml_5_split_metric_comparison.png", dpi=300)
    plt.close(fig)

    print("  All 5 evaluation plots saved to evaluation/figures/")

    # ----------------------------------------------------------
    # FINAL VERIFICATION REPORT
    # ----------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL VERIFICATION REPORT")
    print("=" * 80)
    print(f"  MODEL                    : LinearRegression")
    print(f"  TRAINING ROWS            : {len(df_train)}")
    print(f"  VALIDATION ROWS          : {len(df_val)}")
    print(f"  TEST ROWS                : {len(df_test)}")
    print(f"  FEATURES                 : {feature_cols}")
    print(f"  TARGET                   : {target_col}")
    print(f"  TRAINING COMPLETED       : YES")
    print(f"  model.fit() EXECUTED     : YES")
    print(f"  MODEL SAVED              : YES ({config.MODEL_PATH})")
    print(f"  MODEL RELOADED           : YES")
    print(f"  TEST PREDICTIONS GENERATED: YES")
    print(f"  TEST MAE                 : {reg_test['mae']:.4f}")
    print(f"  TEST RMSE                : {reg_test['rmse']:.4f}")
    print(f"  TEST R²                  : {reg_test['r2']:.4f}")
    print(f"  HOTSPOT THRESHOLD        : >= {threshold} accesses")
    print(f"  ML HOTSPOT F1 (Test)     : {cls_test['f1']:.4f}")
    print(f"  BASELINE F1 (Test)       : {base_cls_test['f1']:.4f}")
    print("=" * 80)

    return {
        "model": model,
        "scaler": scaler,
        "reg_train": reg_train,
        "reg_val": reg_val,
        "reg_test": reg_test,
        "cls_test": cls_test,
        "base_cls_test": base_cls_test,
        "base_reg_test": base_reg_test,
    }


if __name__ == "__main__":
    results = train_and_evaluate()
