# Migration Decision Engine & Payback Documentation

## 1. Purpose

The migration decision engine answers: **"Should this file actually be migrated
to SSD?"** It combines ML predictions, heat scores, and a cost/benefit payback
analysis to produce one of four recommendations:

| Decision | Meaning |
| :------- | :------ |
| `MIGRATE_TO_SSD` | Predicted hotspot, currently on HDD, payback ratio ≥ threshold |
| `ALREADY_ON_SSD` | File is already on the faster tier |
| `NOT_HOT` | Not predicted to be a hotspot |
| `KEEP_CURRENT_TIER` | Predicted hotspot but migration cost exceeds expected benefit |

> **IMPORTANT**: A predicted hotspot does NOT automatically mean migration.
> The payback ratio must confirm that the expected benefit justifies the cost.

---

## 2. Decision Logic (Policy Rules)

```
IF current_tier == SSD:
    decision = ALREADY_ON_SSD

ELIF predicted_hotspot == FALSE:
    decision = NOT_HOT

ELIF payback_ratio >= MIN_PAYBACK_RATIO:
    decision = MIGRATE_TO_SSD

ELSE:
    decision = KEEP_CURRENT_TIER
```

Order of evaluation matters: tier check → hotspot check → payback check.

---

## 3. Hotspot Threshold

A file is classified as a predicted hotspot when:

```
predicted_future_access_count >= HOTSPOT_THRESHOLD
```

| Parameter | Value | Source |
| :-------- | :---- | :----- |
| `HOTSPOT_THRESHOLD` | **8** | `config.py` |

**Interpretation**: 8 or more predicted accesses in the next 6-hour window.

---

## 4. Migration Cost Calculation

**Concept**: The cost of migrating a file is the time required to transfer it
between storage tiers.

```
migration_cost_seconds = file_size_bytes / migration_bandwidth_bytes_per_sec
```

| Parameter | Value | Source |
| :-------- | :---- | :----- |
| `MIGRATION_BANDWIDTH_BYTES_PER_SEC` | 100 MB/s (104,857,600 B/s) | `config.py` |

**Example**: A 100 MB file costs 100,000,000 / 104,857,600 ≈ **0.954 seconds**
to migrate.

> **NOTE**: This bandwidth is a **SIMULATION PARAMETER**, not a measurement
> from physical hardware. It represents a plausible order-of-magnitude estimate
> for a SATA SSD sequential write operation.

---

## 5. Expected Performance Benefit

**Concept**: The total latency saved by serving predicted future accesses from
SSD instead of HDD.

```
latency_saving_per_access = HDD_latency - SSD_latency
expected_benefit_seconds = predicted_future_accesses × latency_saving_per_access
```

| Parameter | Value | Unit | Source |
| :-------- | :---- | :--- | :----- |
| `HDD_READ_LATENCY_US` | 10,000 | microseconds (10 ms) | `config.py` |
| `SSD_READ_LATENCY_US` | 200 | microseconds (0.2 ms) | `config.py` |
| Latency saving per access | 9,800 | microseconds (9.8 ms) | Derived |

**Example**: 50 predicted accesses → 50 × 9,800 / 1,000,000 = **0.490 seconds**
of expected benefit.

> **NOTE**: These latency values are **SIMULATION PARAMETERS**, not measurements
> from physical HDD and SSD hardware. They represent realistic order-of-magnitude
> estimates for mechanical disk seek time vs. flash memory access.

---

## 6. Payback Ratio

```
payback_ratio = expected_benefit_seconds / migration_cost_seconds
```

| Interpretation | Meaning |
| :------------- | :------ |
| `payback_ratio > 1.0` | Benefit exceeds cost — migration is worthwhile |
| `payback_ratio = 1.0` | Benefit exactly equals cost — break-even |
| `payback_ratio < 1.0` | Cost exceeds benefit — migration is not justified |
| `payback_ratio = inf` | Zero migration cost with positive benefit (edge case) |

**Minimum threshold**: `MIN_PAYBACK_RATIO = 1.0` (configurable in `config.py`).

> **NOTE**: 1.0 is an **initial policy threshold**, NOT a scientifically
> optimal value. It simply requires that benefit ≥ cost.

**Edge case handling**:
- If `migration_cost = 0` and `benefit > 0` → payback = `inf` (safe)
- If both are zero → payback = `0.0` (no benefit, no cost)

---

## 7. Why Zero Migrations Occurred in the Test Set

The test dataset produced **0 MIGRATE_TO_SSD** decisions and **5 KEEP_CURRENT_TIER**
decisions. This is **correct behavior**, not a bug, because:

1. The dataset's hot files are predominantly already on SSD (314 of 1,600 test rows).
2. The 5 hot HDD files have predicted access counts of 8–12.
3. At 12 predicted accesses, the expected benefit is only:
   `12 × 9,800 μs = 117,600 μs = 0.1176 seconds`
4. But their file sizes (33–200 MB) produce migration costs of 0.26–1.27 seconds.
5. Therefore payback ratios are 0.07–0.30, well below the threshold of 1.0.

**Root cause**: The per-access latency saving (~9.8 ms = 0.0098 sec) is small
relative to the sequential file transfer cost. A file needs *thousands* of
predicted accesses for the per-access savings to outweigh a multi-second transfer.

This is physically realistic: in real storage systems, SSD migration is only
justified for files with very high sustained access rates or across many
prediction windows (amortized benefit).

---

## 8. Decision Output Schema

| Column | Type | Description |
| :----- | :--- | :---------- |
| `file_id` | str | Unique file identifier |
| `prediction_timestamp` | str | Prediction point timestamp |
| `current_tier` | int | 0=HDD, 1=SSD |
| `heat_score` | float | Composite historical heat [0, 1] |
| `predicted_future_access_count` | float | ML-predicted accesses (next 6h) |
| `predicted_hotspot` | int | 1 if predicted ≥ threshold, else 0 |
| `migration_cost_sec` | float | Estimated transfer time (seconds) |
| `expected_benefit_sec` | float | Estimated latency savings (seconds) |
| `payback_ratio` | float | benefit / cost |
| `decision` | str | One of the four decision labels |
| `decision_reason` | str | Human-readable explanation |

---

## 9. Simulation Assumptions Summary

| Assumption | Value | Status |
| :--------- | :---- | :----- |
| HDD per-access latency | 10 ms | Simulated |
| SSD per-access latency | 0.2 ms | Simulated |
| Migration bandwidth | 100 MB/s | Simulated |
| Hotspot threshold | 8 accesses / 6h | Configurable |
| Minimum payback ratio | 1.0 | Configurable |
| Heat frequency weight (α) | 0.7 | Configurable |
| Heat recency weight (β) | 0.3 | Configurable |

All values are clearly labeled as simulation parameters. None are claimed to be
measured from physical hardware.

---

## 10. Limitations

1. **Single-window payback**: The payback ratio only considers one 6-hour window.
   In practice, a hot file may remain hot across many windows, so the amortized
   benefit would be higher. Multi-window payback amortization is not implemented.
2. **No SSD capacity constraint**: The decision engine does not check whether
   SSD has capacity to accept the migrated file. This would be needed in a
   production system.
3. **No priority ranking**: When multiple files qualify for migration, the engine
   does not rank them by priority. A priority queue (e.g., sort by payback ratio
   descending) would improve resource allocation.
4. **Linear model precision**: The ML model's R² ≈ 0.48 means predictions are
   noisy. Overconfident migration decisions could waste resources.
5. **No write-amplification cost**: The migration cost only considers transfer
   time, not SSD wear leveling or write amplification.

---

## 11. File Locations

- Payback calculator: `migration/payback.py`
- Decision engine: `migration/decision_engine.py`
- Configuration: `config.py`
- Integrated pipeline: `run_pipeline.py`
