# Model Training Report: ML-Driven Hybrid Storage Staging & Hotspot Migration

---

## 1. ML Problem Definition
**Task**: Temporal supervised regression — predict the number of I/O accesses a file will receive during the next 6-hour future window, using only features derived from the file's historical access behavior observed before the prediction point.

**Downstream Decision**: The continuous regression output is thresholded to derive a binary hotspot classification ($\ge 8$ future accesses = hotspot), which later informs the cost-benefit migration decision engine.

---

## 2. Input Features (9 Features)
All features are strictly computed from data available **before** the prediction timestamp $t$:

| # | Feature | Window | Description |
| :- | :--- | :--- | :--- |
| 1 | `recent_access_count` | $[t-6\text{h}, t)$ | Total I/O accesses in last 6 hours |
| 2 | `previous_access_count` | $[t-12\text{h}, t-6\text{h})$ | Total I/O accesses in the preceding 6 hours |
| 3 | `recency_seconds` | $[0, t)$ | Seconds since last access |
| 4 | `burstiness` | $[t-6\text{h}, t)$ | Goh-Barabasi burst coefficient $\in [-1, 1]$ |
| 5 | `file_size_bytes` | Static | File capacity from metadata |
| 6 | `read_ratio` | $[t-6\text{h}, t)$ | Fraction of operations that are reads |
| 7 | `write_ratio` | $[t-6\text{h}, t)$ | Fraction of operations that are writes |
| 8 | `average_request_size` | $[t-6\text{h}, t)$ | Mean I/O payload size in bytes |
| 9 | `current_tier` | At $t$ | Current storage tier (0=HDD, 1=SSD) |

**Excluded**: `true_hotspot_next_6h` (ground-truth future label — used only for evaluation, never as input).

---

## 3. Target Variable
- **`future_access_count`**: Total number of I/O accesses to the file during the future prediction window $[t, t+6\text{h})$.
- **Nature**: Non-negative integer.
- **Range**: 0 to 133 (training set).

---

## 4. Dataset Used
- **Source**: `data/ml_training_dataset.csv` (10,800 rows × 13 columns, generated from 28,000 raw I/O trace events across 200 files and 14 days).
- **Dataset Type**: Controlled synthetic trace dataset created for this project (random seed 42). Not live production data.

---

## 5. Temporal Split
Data was partitioned strictly chronologically (no shuffling) to simulate real-world deployment where only past data is available:

| Split | Rows | Percentage | Timestamps | Period |
| :--- | :--- | :--- | :--- | :--- |
| **Training** | 7,600 | 70.37% | 38 | Jan 01 12:00 → Jan 10 18:00 |
| **Validation** | 1,600 | 14.81% | 8 | Jan 11 00:00 → Jan 12 18:00 |
| **Test** | 1,600 | 14.81% | 8 | Jan 13 00:00 → Jan 14 18:00 |

**Temporal Disjointness**: $\max(\text{train}) < \min(\text{val}) < \min(\text{test})$ — verified and enforced by assertions.

---

## 6. Model Selected
**`sklearn.linear_model.LinearRegression`** — Ordinary Least Squares (OLS) regression.

---

## 7. Why Linear Regression Was Selected
1. **Interpretability**: Coefficients directly reveal how each feature contributes to the prediction — essential for a college viva where every decision must be defensible.
2. **Simplicity**: No hyperparameter tuning required; deterministic closed-form solution.
3. **Baseline Establishment**: Provides a clear, well-understood baseline for future model comparisons (e.g., if Decision Trees or Random Forest are added later).
4. **Reproducibility**: Identical results on every run without random initialization.
5. **Project Constraint**: Per project specification, advanced research frameworks (GNNs, RL, etc.) are explicitly excluded.

---

## 8. Training Procedure
1. Loaded prepared split CSVs from `data/train.csv`, `data/validation.csv`, `data/test.csv`.
2. Extracted feature matrices $X$ and target vectors $y$, with explicit leakage assertions.
3. Loaded the previously fitted `StandardScaler` from `models/feature_scaler.pkl` (fitted exclusively on $X_{\text{train}}$ during Phase 5).
4. Applied Z-score scaling to all three splits using the training scaler.
5. Called `model.fit(X_train_scaled, y_train)` — genuine training on 7,600 samples.
6. Clipped predictions to $\ge 0$ (access counts are non-negative).

---

## 9. Validation Procedure
- Predictions generated on the scaled validation set.
- Validation metrics compared against training metrics to detect potential overfitting.
- Result: Validation MAE (2.1201) closely matches training MAE (2.0337), indicating stable generalization with no significant overfitting.

---

## 10. Test Procedure
- Predictions generated on the **held-out** test set (unseen future timestamps).
- Test metrics serve as the final unbiased performance evaluation.
- Test MAE (2.0826), RMSE (4.2676), and R² (0.4817) are consistent with validation performance.

---

## 11. Regression Metrics

| Metric | Training | Validation | Test |
| :--- | :--- | :--- | :--- |
| **MAE** | 2.0337 | 2.1201 | **2.0826** |
| **RMSE** | 4.0137 | 4.4089 | **4.2676** |
| **R²** | 0.4938 | 0.4414 | **0.4817** |

**Interpretation**:
- The model explains approximately **48%** of the variance in future access counts on unseen test data.
- Mean absolute prediction error is approximately **2.08 accesses** per file-window.
- Consistent performance across all three splits confirms no overfitting.

---

## 12. Hotspot Classification Metrics (Threshold ≥ 8 Accesses)

### ML Model (Test Set):
| Metric | Value |
| :--- | :--- |
| **Accuracy** | **0.9369** |
| **Precision** | **0.6458** |
| **Recall** | **0.4806** |
| **F1-Score** | **0.5511** |

**Confusion Matrix (Test)**:
```
            Predicted Cold    Predicted Hotspot
Actual Cold     TN = 1437         FP = 34
Actual Hot      FN = 67           TP = 62
```

- **False Hotspot Rate**: 34 files unnecessarily flagged (2.3% of cold files).
- **Missed Hotspots**: 67 files that were actually hot but predicted cold (51.9% miss rate).

---

## 13. Baseline Comparison: Recent Frequency Baseline

**Baseline Rule**: Use `recent_access_count` directly as the predicted `future_access_count` (naive persistence assumption — "what happened in the last 6h will happen in the next 6h").

| Metric | ML Model (Test) | Baseline (Test) | ML Advantage |
| :--- | :--- | :--- | :--- |
| **MAE** | 2.0826 | 2.2881 | ML is 8.98% lower (better) |
| **RMSE** | 4.2676 | 5.0234 | ML is 15.05% lower (better) |
| **R²** | 0.4817 | 0.2819 | ML explains 70.87% more variance |
| **F1-Score** | 0.5511 | 0.5159 | ML is 6.82% higher (better) |

**Conclusion**: The trained Linear Regression model consistently outperforms the naive frequency baseline across all regression and classification metrics, demonstrating that multi-feature ML modeling provides measurable predictive value over simple heuristic approaches.

---

## 14. Learned Coefficients

Coefficients correspond to **StandardScaler-normalized** feature values. A positive coefficient indicates a learned positive association (not causation) between the scaled feature and predicted future access count.

| Feature | Coefficient | Interpretation |
| :--- | :--- | :--- |
| `recent_access_count` | **+4.0452** | Strongest positive predictor — temporal locality |
| `current_tier` | **+1.1283** | Files on SSD predicted to have higher access |
| `previous_access_count` | **−1.3425** | Negative after controlling for recent — captures decaying trends |
| `recency_seconds` | **−0.1680** | More recently accessed → higher predicted access |
| `burstiness` | **+0.1324** | Bursty access patterns weakly associated with future activity |
| `read_ratio` | **+0.1235** | Read-heavy files slightly predicted to be more active |
| `average_request_size` | **+0.1169** | Larger I/O requests weakly associated with continued activity |
| `file_size_bytes` | **−0.0982** | Larger files weakly predicted to have lower access density |
| `write_ratio` | **−0.0383** | Write-heavy files weakly predicted to have lower future access |
| *Intercept* | **+2.5100** | Baseline predicted access count at the population mean |

---

## 15. Model File Locations
- **Trained Model**: `models/hotspot_predictor.pkl` (genuine `LinearRegression` object from `model.fit()`)
- **Feature Scaler**: `models/feature_scaler.pkl` (StandardScaler fitted on training data)
- **Training Script**: `ml/train_model.py` (fully reproducible)
- **Prediction Module**: `ml/predictor.py` (loads model from disk, accepts feature input)

---

## 16. Leakage Prevention Checks
All passed:
- `true_hotspot_next_6h` not in $X_{\text{train}}, X_{\text{val}}, X_{\text{test}}$ ✓
- `eval_true_hotspot_next_6h` not in $X_{\text{train}}, X_{\text{val}}, X_{\text{test}}$ ✓
- `future_access_count` not in $X_{\text{train}}, X_{\text{val}}, X_{\text{test}}$ ✓
- Scaler fitted exclusively on training data ✓
- Test timestamps strictly after validation timestamps, which are strictly after training timestamps ✓
- Model reloaded from disk and verified producing identical predictions ✓

---

## 17. Limitations & Honest Assessment
1. **R² ≈ 0.48**: The model explains only about half the variance in future access counts. This is expected for a simple linear model on a noisy temporal prediction problem.
2. **Recall ≈ 0.48**: The model misses approximately half of actual hotspots. Increasing recall would require lowering the threshold (at the cost of precision) or using a more expressive model.
3. **Linear Assumption**: LinearRegression assumes a linear relationship between features and target. Non-linear access patterns (e.g., diurnal cycles, application-specific spikes) are not fully captured.
4. **Synthetic Data**: The dataset is a controlled synthetic trace. Real-world production workloads may exhibit different distributions, drift patterns, and anomalies.
5. **Single Model**: No ensemble methods, feature selection, or hyperparameter tuning have been applied. These could potentially improve performance in subsequent phases if approved.

**This model is a college project demonstration and is not production-ready.**

---

## Generated Evaluation Visualizations
Saved in `evaluation/figures/`:
1. `ml_1_actual_vs_predicted.png` — Scatter plot of actual vs. predicted future accesses
2. `ml_2_residual_distribution.png` — Histogram of prediction residuals
3. `ml_3_confusion_matrix.png` — Hotspot classification confusion matrix heatmap
4. `ml_4_feature_coefficients.png` — Horizontal bar chart of learned feature coefficients
5. `ml_5_split_metric_comparison.png` — Side-by-side MAE / RMSE / R² across splits
