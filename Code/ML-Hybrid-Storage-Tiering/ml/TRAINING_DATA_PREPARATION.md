# Training Data Preparation & Temporal Splitting Documentation

This document describes the temporal dataset partitioning, feature/target matrix extraction, data-leakage prevention mechanisms, and feature scaling methodology for training the hybrid storage hotspot prediction model.

---

## 1. Machine Learning Feature Set
The input feature matrix $X$ consists strictly of **9 pre-prediction historical features**:

| # | Feature Name | Representation | Data Type | Physical Meaning |
| :- | :--- | :--- | :--- | :--- |
| 1 | `recent_access_count` | $[t - 6\text{h}, t)$ | `int64` | Immediate temporal access intensity |
| 2 | `previous_access_count` | $[t - 12\text{h}, t - 6\text{h})$ | `int64` | Baseline prior access intensity |
| 3 | `recency_seconds` | $[0, t)$ | `float64` | Seconds since most recent access (LRU factor) |
| 4 | `burstiness` | $[t - 6\text{h}, t)$ | `float64` | Goh-Barabasi burst coefficient $\in [-1, 1]$ |
| 5 | `file_size_bytes` | Static metadata | `int64` | Storage capacity footprint |
| 6 | `read_ratio` | $[t - 6\text{h}, t)$ | `float64` | Fraction of accesses that are reads $\in [0, 1]$ |
| 7 | `write_ratio` | $[t - 6\text{h}, t)$ | `float64` | Fraction of accesses that are writes $\in [0, 1]$ |
| 8 | `average_request_size` | $[t - 6\text{h}, t)$ | `float64` | Average I/O request payload in bytes |
| 9 | `current_tier` | At $t$ | `int64` | Current storage tier (`0` = HDD, `1` = SSD) |

---

## 2. Target Variable
- **`future_access_count`**: Total number of I/O operations directed to the file during the future 6-hour prediction window $[t, t + 6\text{h})$.
- **Evaluation-Only Column**: `eval_true_hotspot_next_6h` is saved in the CSVs for post-prediction classification validation, but is strictly isolated from $X$ during ML training.

---

## 3. Why Temporal Splitting is Mandatory
In storage systems and time-series forecasting, standard random shuffling (`train_test_split(shuffle=True)`) causes catastrophic **lookahead data leakage**:
1. If future access events appear in the training set while past events appear in the test set, the model memorizes future access trajectories rather than learning real-world predictive temporal dynamics.
2. In production OS tiering, the kernel can only evaluate past access traces to schedule migrations for the future.
3. Therefore, data must be partitioned strictly chronologically: **Training on the Past $\to$ Tuning on the Intermediate Future $\to$ Testing on the Unseen Distant Future**.

---

## 4. Chronological 70% / 15% / 15% Split

The 54 discrete prediction timestamps (each encompassing all 200 files = 10,800 total rows) are partitioned chronologically:

```
PAST                                                                           FUTURE
[========================================|==================|==================]
               TRAIN (70.37%)                VAL (14.81%)        TEST (14.81%)
            7,600 rows (38 timestamps)     1,600 rows (8 ts)   1,600 rows (8 ts)
```

| Dataset Split | Rows | % of Dataset | Prediction Timestamps | Start Timestamp | End Timestamp |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Training** | **7,600** | **70.37%** | 38 points | `2026-01-01 12:00:00` | `2026-01-10 18:00:00` |
| **Validation** | **1,600** | **14.81%** | 8 points | `2026-01-11 00:00:00` | `2026-01-12 18:00:00` |
| **Test** | **1,600** | **14.81%** | 8 points | `2026-01-13 00:00:00` | `2026-01-14 18:00:00` |
| **Total** | **10,800** | **100.0%** | 54 points | `2026-01-01 12:00:00` | `2026-01-14 18:00:00` |

### Strict Temporal Disjointness Verification:
- $\max(\text{train\_timestamp}) = \text{2026-01-10 18:00:00} < \min(\text{val\_timestamp}) = \text{2026-01-11 00:00:00}$ `[PASSED]`
- $\max(\text{val\_timestamp}) = \text{2026-01-12 18:00:00} < \min(\text{test\_timestamp}) = \text{2026-01-13 00:00:00}$ `[PASSED]`

---

## 5. Temporal Split Target Statistics

| Split | Min Accesses | Max Accesses | Mean Accesses | Median Accesses | Standard Deviation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | `0` | `133` | `2.5100` | `1.0` | `5.7203` |
| **Validation** | `0` | `100` | `2.5000` | `1.0` | `5.5786` |
| **Test** | `0` | `105` | `2.5000` | `1.0` | `5.8845` |

The consistent mean (~$2.50$) and median ($1.0$) confirm stable, non-drifting workload dynamics across all three periods.

---

## 6. Feature Scaling Methodology
- **Algorithm**: `sklearn.preprocessing.StandardScaler` (Z-score standardization: $z = \frac{x - \mu}{\sigma}$).
- **Why Scaling is Necessary for Linear Regression**:
  - `file_size_bytes` spans $10^6$ to $2 \times 10^8$ bytes.
  - `recency_seconds` spans $1$ to $8 \times 10^5$ seconds.
  - `read_ratio` and `write_ratio` are bounded in $[0, 1]$.
  - `burstiness` is bounded in $[-1, 1]$.
  - Unscaled features would cause extreme numerical conditioning issues and make regression weight interpretation meaningless.
- **Strict Leakage Guard**:
  - The `StandardScaler` is fitted **exclusively** on `X_train` (`scaler.fit(X_train)`).
  - Validation and test splits are transformed using the training parameters (`scaler.transform(X_val)`, `scaler.transform(X_test)`).
  - The fitted scaler is persisted to [`models/feature_scaler.pkl`](file:///d:/THEJA%20VIT/OS%20operating%20system/Lab%20Project/Code/ML-Hybrid-Storage-Tiering/models/feature_scaler.pkl).

---

## 7. Data Leakage Checklist & Safeguards
1. **Target Exclusion**: `future_access_count` was asserted not present in $X_{\text{train}}, X_{\text{val}}, X_{\text{test}}$.
2. **Ground Truth Exclusion**: Neither `true_hotspot_next_6h` nor `eval_true_hotspot_next_6h` exist in $X$.
3. **Timestamp Exclusion**: `prediction_timestamp` and `file_id` are removed from the feature matrix before scaling/fitting.
4. **Scaler Isolation**: Means and standard deviations were derived solely from the 7,600 training samples.

---

## 8. Generated Files and Artifacts
- **[`data/train.csv`](file:///d:/THEJA%20VIT/OS%20operating%20system/Lab%20Project/Code/ML-Hybrid-Storage-Tiering/data/train.csv)**: 7,600 rows (70.37%).
- **[`data/validation.csv`](file:///d:/THEJA%20VIT/OS%20operating%20system/Lab%20Project/Code/ML-Hybrid-Storage-Tiering/data/validation.csv)**: 1,600 rows (14.81%).
- **[`data/test.csv`](file:///d:/THEJA%20VIT/OS%20operating%20system/Lab%20Project/Code/ML-Hybrid-Storage-Tiering/data/test.csv)**: 1,600 rows (14.81%).
- **[`data/data_split_metadata.json`](file:///d:/THEJA%20VIT/OS%20operating%20system/Lab%20Project/Code/ML-Hybrid-Storage-Tiering/data/data_split_metadata.json)**: Full configuration and split boundaries JSON.
- **[`models/feature_scaler.pkl`](file:///d:/THEJA%20VIT/OS%20operating%20system/Lab%20Project/Code/ML-Hybrid-Storage-Tiering/models/feature_scaler.pkl)**: Fitted `StandardScaler` artifact.
- **[`ml/data_preparation.py`](file:///d:/THEJA%20VIT/OS%20operating%20system/Lab%20Project/Code/ML-Hybrid-Storage-Tiering/ml/data_preparation.py)**: Reusable data preparation module.

---

## 9. Reproducibility
Running `python ml/data_preparation.py` dynamically regenerates the identical split partitions and scalar normalization parameters deterministically.
