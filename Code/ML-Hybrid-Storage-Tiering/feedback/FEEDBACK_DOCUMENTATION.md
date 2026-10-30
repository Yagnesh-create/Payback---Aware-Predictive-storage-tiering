# Phase 16 — Migration Feedback & Adaptive Heat-Map Documentation

## 1. Purpose

The feedback mechanism closes the prediction–migration loop by comparing what the ML model **predicted** with what **actually happened**, then adjusting the heat-map scores to improve future migration decisions.

```
Prediction → Migration → Observe actual behaviour → Compare →
Feedback signal → Bounded adjustment → Update heat score →
Improved future decisions
```

Without feedback, the system makes decisions based solely on the ML model's static predictions. With feedback, the decision layer can adapt to prediction errors over time.

---

## 2. Migration Success vs. Prediction Success

These are **two fundamentally different concepts** that must not be confused:

### Migration Execution Success
The file was physically/simulated successfully copied from the source tier to the destination tier, and its SHA-256 checksum was verified.

### Prediction Success
The file actually received enough accesses after migration to justify the hotspot prediction.

**Example:**
```
Migration execution:   SUCCESS  (file copied, checksum verified)
Predicted accesses:    20
Actual future accesses: 2

→ Migration operation succeeded, but the hotspot prediction was poor.
```

---

## 3. Actual Future Access Calculation

For a migration decision at `prediction_timestamp`, the system counts the number of actual accesses to the file during the subsequent 6-hour window using the **raw storage access trace** (`storage_access_trace.csv`).

```
actual_future_access_count = COUNT(
    trace events WHERE
        file_id = target_file AND
        timestamp > prediction_timestamp AND
        timestamp <= prediction_timestamp + 6 hours
)
```

This count is derived directly from the raw access trace, **not** from the ML prediction or the pre-computed `eval_true_hotspot_next_6h` label. The ground-truth label may be used for consistency checking but is not the primary source.

---

## 4. Prediction Outcomes

For each prediction record, the system computes:

| Metric | Formula |
|--------|---------|
| `prediction_error` | `actual_future_access_count - predicted_future_access_count` |
| `absolute_prediction_error` | `abs(prediction_error)` |
| `predicted_hotspot` | `predicted_future_access_count >= HOTSPOT_THRESHOLD` |
| `actual_hotspot` | `actual_future_access_count >= HOTSPOT_THRESHOLD` |

### Classification:

| Outcome | Predicted Hotspot | Actual Hotspot |
|---------|-------------------|----------------|
| **True Positive (TP)** | Yes | Yes |
| **True Negative (TN)** | No | No |
| **False Positive (FP)** | Yes | No |
| **False Negative (FN)** | No | Yes |

---

## 5. False Hotspot Definition

A **false hotspot** occurs when:
- The ML model predicted the file would be a hotspot (`predicted_accesses >= threshold`)
- But the file did **not** actually become a hotspot (`actual_accesses < threshold`)

**False Hotspot Rate:**
```
FP / (TP + FP)
```
Where the denominator is the total number of predicted hotspots.

**False Hotspot Migration Rate:**
```
Migrated-but-not-hotspot / Total-migrations
```

Both metrics return "N/A" if the denominator is zero.

---

## 6. Missed Hotspot Definition

A **missed hotspot** occurs when:
- The ML model predicted the file would **not** be a hotspot
- But the file **did** become a hotspot

This means a performance-critical file was left on the slower tier.

---

## 7. Migration-Specific Outcome Categories

| Category | Condition |
|----------|-----------|
| **SUCCESSFUL_USEFUL_MIGRATION** | Migrated + predicted hot + actually hot |
| **FALSE_HOTSPOT_MIGRATION** | Migrated + predicted hot + NOT actually hot |
| **MISSED_HOTSPOT** | Not migrated + actually hot |
| **CORRECT_NON_MIGRATION** | Not migrated + NOT actually hot |

---

## 8. Feedback Signal

The feedback signal captures the prediction error direction and magnitude:

```
feedback_signal = actual_future_access_count - predicted_future_access_count
```

- **Positive signal**: model under-predicted (file was hotter than expected)
- **Negative signal**: model over-predicted (file was colder than expected)

---

## 9. Tanh / Bounded Feedback

To prevent a single extreme workload event from completely dominating the heat score, the feedback signal is transformed using `tanh`:

```
bounded_feedback = tanh(feedback_signal / FEEDBACK_SCALE)
```

Where `FEEDBACK_SCALE` (default: 5.0) controls how quickly the bounding saturates.

Properties of this transformation:
- Output is always in [-1, +1]
- Small errors pass through approximately linearly
- Large errors are compressed (saturate toward ±1)
- Zero error produces zero feedback

---

## 10. Feedback Weight

The `FEEDBACK_WEIGHT` parameter (default: 0.1) controls how strongly feedback adjusts the heat score:

```
updated_heat = old_heat + FEEDBACK_WEIGHT × bounded_feedback
```

A weight of 0.1 means even the maximum bounded feedback (±1) can only shift the heat score by ±0.1 in a single feedback cycle.

---

## 11. Heat-Score Update

After applying the weighted bounded feedback, the heat score is **clipped** to ensure it remains in [0, 1]:

```
updated_heat = clamp(old_heat + FEEDBACK_WEIGHT × bounded_feedback, 0.0, 1.0)
```

This guarantees:
- `heat_score >= 0` (cold files never get negative heat)
- `heat_score <= 1` (hot files never exceed maximum heat)

---

## 12. Why the Model Is NOT Retrained

**Critical design decision**: Phase 16 does NOT automatically retrain the LinearRegression model.

Reasons:
1. **Reproducibility**: Keeping the model fixed allows clear distinction between ML prediction quality and adaptive decision feedback.
2. **Separation of concerns**: The feedback mechanism operates at the heat-map/decision layer, not the ML layer.
3. **Experiment integrity**: Retraining on test data would constitute data leakage.
4. **Explainability**: It is clearer to show that the base ML predictions remain constant while only the decision heuristic adapts.

The model file `models/hotspot_predictor.pkl` is never modified during feedback processing.

---

## 13. Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FEEDBACK_WEIGHT` | 0.1 | Strength of heat score adjustment |
| `FEEDBACK_SCALE` | 5.0 | Normalisation scale for tanh bounding |
| `HOTSPOT_THRESHOLD` | 8 | Access count threshold for hotspot classification |

These are **experimental configuration parameters**, not claimed as optimal values.

---

## 14. Feedback Log

Every feedback computation is logged to `logs/feedback_log.csv` with 17 fields:

```
feedback_id, timestamp, file_id, prediction_timestamp,
predicted_future_access_count, actual_future_access_count,
prediction_error, absolute_prediction_error,
predicted_hotspot, actual_hotspot, prediction_outcome,
migration_status, migration_was_useful,
old_heat_score, feedback_signal, bounded_feedback, updated_heat_score
```

---

## 15. Limitations

1. **Heuristic nature**: The bounded feedback mechanism is a reasonable engineering heuristic, not a theoretically optimal adaptation algorithm.
2. **Single-pass**: Feedback is applied once per prediction cycle, not iteratively refined.
3. **No model retraining**: The ML model's weights are unchanged; only the heat-map decision layer adapts.
4. **Scale sensitivity**: The `FEEDBACK_SCALE` and `FEEDBACK_WEIGHT` parameters affect behaviour and have not been exhaustively tuned.
5. **Stale feedback**: In a real system, feedback from much older predictions may be less relevant, but this implementation does not discount by age.
6. **Simulated environment**: Feedback is evaluated using simulated storage, not real hardware I/O patterns.
